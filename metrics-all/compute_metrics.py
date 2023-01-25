import numpy as np
import cv2
import os
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from cleanfid import fid
from cleanfid.fid import get_folder_features
from cleanfid.features import get_reference_statistics, build_feature_extractor
from precision_recall import knn_precision_recall_features


def array_to_image(path, folder="./logs/temp"):
    if path.endswith(".npy"):
        images = np.load(path)
    elif path.endswith("npz"):
        images = np.load(path)["arr_0"]
    else:
        raise ValueError(f"Unrecognized file type: {path}")
    
    if images.min() >= 0 and images.max() <= 1:
        images = (images * 255).astype("uint8")
    elif images.min() >= -1 and images.max() <= 1:
        images = (127.5 * (images + 1)).astype("uint8")
    else:
        assert images.min() >= 0 and images.max() <= 255
    
    assert len(images.shape) == 4, "Images must be a batch"
    num_images = images.shape[0]
    for idx in range(num_images):
        cv2.imwrite(
            os.path.join(folder, "sample_%05d.png" % (idx)),
            images[idx, :, :, ::-1]
        )

def compute_precision_recall(args):
    feat_model = build_feature_extractor(args.mode, args.device)
    # features
    fake_features = get_folder_features("./logs/temp/", feat_model, batch_size = args.batch_size, num_workers = args.num_gpus * 4, mode=args.mode, device=args.device)
    real_features = get_reference_statistics(args.dataset, args.resolution, args.mode, split="train", metric="FID")["feats"]
    # check shape of features
    print(f"Extract real features {real_features.shape}")
    print(f"Extract fake features {fake_features.shape}")

    # compute precision recall
    data_dict = knn_precision_recall_features(real_features, fake_features, num_gpus=args.num_gpus)

    return data_dict


def main(args):
    # construct image folder
    array_to_image(args.fake)

    # compute fid
    fid_score = fid.compute_fid("./logs/temp/", mode=args.mode, dataset_name=args.dataset, dataset_res=args.resolution, dataset_split="train", batch_size = args.batch_size, num_workers = args.num_gpus * 4)
    print(f"clean-fid score: {fid_score:.3f}")

    # compute precision recall
    """score_dict = compute_precision_recall(args)
    precision_score = score_dict["precision"]
    recall_score = score_dict["precision"]
    print(f"precision score: {precision_score}")
    print(f"recall score: {recall_score}")"""


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, required=True, help='real dataset to evaluate')
    parser.add_argument('--fake', type=str, required=True, help=('Path to the generated images'))
    parser.add_argument('--resolution', type=int, required=True, help='image resolution to compute metrics')
    parser.add_argument('--mode', type=str, default="clean", choices=["clean", "legacy_pytorch", "legacy_tensorflow"])
    parser.add_argument('--batch-size', type=int, default=50, help="batch size to extract features")
    parser.add_argument("--num-gpus", type=int, help='number of GPUs for evaluation')
    args = parser.parse_args()

    if args.num_gpus > 0 and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    print(args)
    
    main(args)