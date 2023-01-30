import os
import time
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
from PIL import Image
from torch.nn.functional import adaptive_avg_pool2d

from cleanfid.fid import get_folder_features
from cleanfid.features import build_feature_extractor

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x


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


def compute_precision_recall(fdir1, fdir2, mode, batch_size, num_workers, model_name = "inception_v3", device=torch.device("cuda"), num_gen=50000, num_eval=10000, use_dataparallel=True):
    if model_name == "inception_v3":
        feat_model = build_feature_extractor(mode, device, use_dataparallel=use_dataparallel)
    
    np_feats1 = get_folder_features(fdir1, model=feat_model, num_workers=num_workers, batch_size=batch_size, device=device, mode=mode)
    np_feats2 = get_folder_features(fdir2, model=feat_model, num_workers=num_workers, batch_size=batch_size, device=device, mode=mode)
    
    print("Taking 10,000 image features at random")
    rand_idx = np.random.choice(num_gen, num_eval, replace=False)
    results = knn_precision_recall_features(np_feats1[rand_idx], np_feats2[rand_idx])
    return results
