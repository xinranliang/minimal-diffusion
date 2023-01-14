import os
import time
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
from PIL import Image
from torch.nn.functional import adaptive_avg_pool2d

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x
#from models import lenet
from models.inception import InceptionV3
from models.lenet import LeNet5


def get_activations(files, model, batch_size=50, dims=2048,
                    cuda=False, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    is_numpy = True if type(files[0]) == np.ndarray else False

    if len(files) % batch_size != 0:
        print(('Warning: number of images is not a multiple of the '
               'batch size. Some samples are going to be ignored.'))
    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    n_batches = len(files) // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, dims))

    for i in tqdm(range(n_batches)):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches), end='', flush=True)
        start = i * batch_size
        end = start + batch_size
        if is_numpy:
            images = np.copy(files[start:end]) + 1
            images /= 2.
        else:
            images = [np.array(Image.open(str(f))) for f in files[start:end]]
            images = np.stack(images).astype(np.float32) / 255.
            # Reshape to (n_images, 3, height, width)
            images = images.transpose((0, 3, 1, 2))

        batch = torch.from_numpy(images).type(torch.FloatTensor)
        if cuda:
            batch = batch.cuda()

        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)

    if verbose:
        print('done', np.min(images))

    return pred_arr


# https://github.com/kynkaat/improved-precision-and-recall-metric (adapted for pytorch)
def batch_pairwise_distances(U, V):
    """Compute pairwise distances between two batches of feature vectors."""
    # Squared norms of each row in U and V.
    norm_u = np.sum(np.square(U), axis=1)
    norm_v = np.sum(np.square(V), axis=1) 
    
    # norm_u as a column and norm_v as a row vectors.
    norm_u = np.reshape(norm_u, [-1, 1])
    norm_v = np.reshape(norm_v, [1, -1])

    # Pairwise squared Euclidean distances.
    D = np.maximum(norm_u - 2 * U.dot(V.T) + norm_v, 0.0)
    return D


def _compute_activations(path, model, batch_size, dims, cuda, model_type):
    if not type(path) == np.ndarray:
        import glob
        jpg = os.path.join(path, '*.jpg')
        png = os.path.join(path, '*.png')
        path = glob.glob(jpg) + glob.glob(png)
        if len(path) > 25000:
            import random
            random.shuffle(path)
            path = path[:25000]
    if model_type == 'inception':
        act = get_activations(path, model, batch_size, dims, cuda)
    elif model_type == 'lenet':
        act = extract_lenet_features(path, model, cuda)

    return act


class ManifoldEstimator():
    """Estimates the manifold of given feature vectors."""

    def __init__(self,
                 distance_block,
                 features,
                 row_batch_size=25000,
                 col_batch_size=50000,
                 nhood_sizes=[3],
                 clamp_to_percentile=None,
                 eps=1e-5):
        """Estimate the manifold of given feature vectors.
        
            Args:
                distance_block: DistanceBlock object that distributes pairwise distance
                    calculation to multiple GPUs.
                features (np.array/tf.Tensor): Matrix of feature vectors to estimate their manifold.
                row_batch_size (int): Row batch size to compute pairwise distances
                    (parameter to trade-off between memory usage and performance).
                col_batch_size (int): Column batch size to compute pairwise distances.
                nhood_sizes (list): Number of neighbors used to estimate the manifold.
                clamp_to_percentile (float): Prune hyperspheres that have radius larger than
                    the given percentile.
                eps (float): Small number for numerical stability.
        """
        num_images = features.shape[0]
        self.nhood_sizes = nhood_sizes
        self.num_nhoods = len(nhood_sizes)
        self.eps = eps
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self._ref_features = features
        self._distance_block = distance_block

        # Estimate manifold of features by calculating distances to k-NN of each sample.
        self.D = np.zeros([num_images, self.num_nhoods], dtype=np.float32)
        distance_batch = np.zeros([row_batch_size, num_images],
                                  dtype=np.float32)
        seq = np.arange(max(self.nhood_sizes) + 1, dtype=np.int32)

        for begin1 in range(0, num_images, row_batch_size):
            end1 = min(begin1 + row_batch_size, num_images)
            row_batch = features[begin1:end1]

            for begin2 in range(0, num_images, col_batch_size):
                end2 = min(begin2 + col_batch_size, num_images)
                col_batch = features[begin2:end2]

                # Compute distances between batches.
                distance_batch[
                    0:end1 - begin1,
                    begin2:end2] = self._distance_block(
                        row_batch, col_batch)

            # Find the k-nearest neighbor from the current batch.
            self.D[begin1:end1, :] = np.partition(distance_batch[0:end1 -
                                                                 begin1, :],
                                                  seq,
                                                  axis=1)[:, self.nhood_sizes]

        if clamp_to_percentile is not None:
            max_distances = np.percentile(self.D, clamp_to_percentile, axis=0)
            self.D[self.D > max_distances] = 0

    def evaluate(self,
                 eval_features,
                 return_realism=False,
                 return_neighbors=False):
        """Evaluate if new feature vectors are at the manifold."""
        num_eval_images = eval_features.shape[0]
        num_ref_images = self.D.shape[0]
        distance_batch = np.zeros([self.row_batch_size, num_ref_images],
                                  dtype=np.float32)
        batch_predictions = np.zeros([num_eval_images, self.num_nhoods],
                                     dtype=np.int32)
        max_realism_score = np.zeros([
            num_eval_images,
        ], dtype=np.float32)
        nearest_indices = np.zeros([
            num_eval_images,
        ], dtype=np.int32)

        for begin1 in range(0, num_eval_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_eval_images)
            feature_batch = eval_features[begin1:end1]

            for begin2 in range(0, num_ref_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_ref_images)
                ref_batch = self._ref_features[begin2:end2]

                distance_batch[
                    0:end1 - begin1,
                    begin2:end2] = self._distance_block(
                        feature_batch, ref_batch)

            # From the minibatch of new feature vectors, determine if they are in the estimated manifold.
            # If a feature vector is inside a hypersphere of some reference sample, then
            # the new sample lies at the estimated manifold.
            # The radii of the hyperspheres are determined from distances of neighborhood size k.
            samples_in_manifold = distance_batch[0:end1 - begin1, :,
                                                 None] <= self.D
            batch_predictions[begin1:end1] = np.any(samples_in_manifold,
                                                    axis=1).astype(np.int32)

            max_realism_score[begin1:end1] = np.max(
                self.D[:, 0] / (distance_batch[0:end1 - begin1, :] + self.eps),
                axis=1)
            nearest_indices[begin1:end1] = np.argmin(distance_batch[0:end1 -
                                                                    begin1, :],
                                                     axis=1)

        if return_realism and return_neighbors:
            return batch_predictions, max_realism_score, nearest_indices
        elif return_realism:
            return batch_predictions, max_realism_score
        elif return_neighbors:
            return batch_predictions, nearest_indices

        return batch_predictions


def knn_precision_recall_features(real_features,
                                  fake_features,
                                  nhood_sizes=[3],
                                  row_batch_size=10000,
                                  col_batch_size=10000,
                                  num_gpus=1):
    """Calculates k-NN precision and recall for two sets of feature vectors.
    
        Args:
            real_features (np.array/tf.Tensor): Feature vectors of real images.
            fake_features (np.array/tf.Tensor): Feature vectors of fake images.
            nhood_sizes (list): Number of neighbors used to estimate the manifold.
            row_batch_size (int): Row batch size to compute pairwise distances
                (parameter to trade-off between memory usage and performance).
            col_batch_size (int): Column batch size to compute pairwise distances.
            num_gpus (int): Number of GPUs used to evaluate precision and recall.
        Returns:
            State (dict): Dict that contains precision and recall calculated from
            ref_features and eval_features.
    """
    print("Ensure that first set of features are for real images and second for generated images.")
    state = dict()
    num_images = real_features.shape[0]
    num_features = real_features.shape[1]

    # Initialize DistanceBlock and ManifoldEstimators.
    real_manifold = ManifoldEstimator(batch_pairwise_distances, real_features,
                                     row_batch_size, col_batch_size,
                                     nhood_sizes)
    fake_manifold = ManifoldEstimator(batch_pairwise_distances, fake_features,
                                      row_batch_size, col_batch_size,
                                      nhood_sizes)

    # Evaluate precision and recall using k-nearest neighbors.
    print('Evaluating k-NN precision and recall with %i samples...' %
          num_images)

    # Precision: How many points from fake_features are in real_features manifold.
    precision = real_manifold.evaluate(fake_features)
    state['precision'] = precision.mean(axis=0).item()

    # Recall: How many points from real_features are in fake_features manifold.
    recall = fake_manifold.evaluate(real_features)
    state['recall'] = recall.mean(axis=0).item()

    return state


def calculate_precision_recall_given_paths(paths, batch_size, cuda, dims, bootstrap=True, n_bootstraps=10, model_type='inception'):
    """Calculates the Precision and Recall of two paths"""
    pths = []
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)
        if os.path.isdir(p):
            pths.append(p)
        elif p.endswith('.npy'):
            np_imgs = np.load(p).astype(np.float64)
            if np_imgs.shape[0] > 25000:
                np_imgs = np_imgs[:10000]
            pths.append(np_imgs)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    if model_type == 'inception':
        model = InceptionV3([block_idx])
    elif model_type == 'lenet':
        model = LeNet5()
        model.load_state_dict(torch.load('./models/lenet.pth'))
    if cuda:
       model.cuda()

    real_feat = _compute_activations(pths[0], model, batch_size, dims, cuda, model_type)
    fake_feat = _compute_activations(pths[1], model, batch_size, dims, cuda, model_type)

    results = knn_precision_recall_features(real_feat, fake_feat)

    print("Precision: {}".format(results["precision"]))
    print("Recall: {}".format(results["recall"]))

    return results


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--true', type=str, required=True,
                        help=('Path to the true images'))
    parser.add_argument('--fake', type=str, nargs='+', required=True,
                        help=('Path to the generated images'))
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size to use')
    parser.add_argument('--dims', type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=('Dimensionality of Inception features to use. '
                              'By default, uses pool3 features'))
    parser.add_argument('-c', '--gpu', default='', type=str,
                        help='GPU to use (leave blank for CPU only)')
    parser.add_argument('--model', default='inception', type=str,
                        help='inception or lenet')
    args = parser.parse_args()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    paths = [args.true] + args.fake

    results = calculate_precision_recall_given_paths(paths, args.batch_size, args.gpu != '', args.dims, model_type=args.model)