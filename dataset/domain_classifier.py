import os
import pickle
import tempfile
import glob
from shutil import copyfile
import argparse

import torchvision
from tqdm.auto import tqdm

import numpy as np
import random
from PIL import Image
import cv2
import scipy, scipy.io
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

import torch
import torch.nn as nn 
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from utils import logger

class Domain_CifarImageNet(datasets.ImageFolder):
    def __init__(
        self,
        root, 
        transform,
        target_transform
    ):
        super().__init__(root, transform, target_transform)

        # return labels
            # cifar10: 0
            # imagenet: 1
        self.num_classes = 2
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        # if self.target_transform is not None:
            # target = self.target_transform(target)
        image_name = path.split("/")[-1]
        # extract cifar10
        if "cifar10" in image_name:
            label = 0
        else:
            label = 1

        return sample, label


class DomainClassifier(nn.Module):
    def __init__(self, num_classes, arch, pretrained, learning_rate, weight_decay, device):
        super().__init__()

        self.device = device
        self.cnn = self.get_model(arch, pretrained)
        # don't need last fc layer
        for param in self.cnn.fc.parameters():
            param.requires_grad = False
        self.head = nn.Linear(2048, num_classes).to(self.device)

        # loss function
        self.loss_fn = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)

        # optimizer
        self.opt = torch.optim.Adam(list(self.cnn.parameters()) + list(self.head.parameters()), lr=learning_rate, weight_decay=weight_decay)

        self.train()
    
    def get_model(self, arch, pretrained):
        """
        store model arch dict
        """
        self.arch_dict = {
            "resnet18": resnet18(weights=None).to(self.device),
            "resnet18-pretrain": resnet18(weights=ResNet18_Weights.DEFAULT).to(self.device),
            "resnet50": resnet50(weights=None).to(self.device),
            "resnet50-pretrain": resnet50(weights=ResNet50_Weights.DEFAULT).to(self.device),
        }
        if pretrained:
            key = arch + "-pretrain"
        else:
            key = arch 
        return self.arch_dict[key]
    
    def save(self, logdir, step, final=False):
        checkpoint = {
                "cnn": self.cnn.state_dict(),
                "head": self.head.state_dict(),
                "opt": self.opt.state_dict()
            }
        if not final:
            torch.save(
                checkpoint, os.path.join(logdir, "model_param_{}".format(int(step)))
            )
        else:
            torch.save(
                checkpoint, os.path.join(logdir, "model_param_final")
            )
    
    def load(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        self.cnn.load_state_dict(checkpoint["cnn"])
        self.head.load_state_dict(checkpoint["head"])
        self.opt.load_state_dict(checkpoint["opt"])

    def forward(self, x):
        # forward resnet to get last layer feature
        x = self.cnn.conv1(x)
        x = self.cnn.bn1(x)
        x = self.cnn.relu(x)
        x = self.cnn.maxpool(x)

        x = self.cnn.layer1(x)
        x = self.cnn.layer2(x)
        x = self.cnn.layer3(x)
        x = self.cnn.layer4(x)

        x = self.cnn.avgpool(x)
        x = torch.flatten(x, 1)

        # linear head for classification
        x = self.head(x)

        return x 
    
    def get_error(self, image, label):
        # compute loss
        image, label = image.to(self.device), label.to(self.device)
        pred_logits = self.forward(image)
        pred_loss = self.loss_fn(pred_logits, label)

        # compute accuracy
        pred_probs = self.softmax(pred_logits)
        pred_target = torch.argmax(pred_probs, dim=-1)
        pred_acc = torch.sum(pred_target == label) / label.shape[0]

        return pred_loss, pred_acc
    
    def update(self, image, label):
        pred_loss, pred_acc = self.get_error(image, label)

        self.opt.zero_grad()
        pred_loss.backward()
        self.opt.step()

        return pred_loss, pred_acc


def get_args():
    parser = argparse.ArgumentParser("Training domain classifier on images")

    parser.add_argument("--arch", type=str, default="resnet50", help="model architecture used for training")
    parser.add_argument("--pretrained", action="store_true", help="whether to initialize with pretrained weights")

    parser.add_argument("--dataset", type=str, help="specify what dataset source")
    parser.add_argument("--num-domains", type=int, help="number of output classes")

    parser.add_argument("--mode", type=str, choices=["train", "eval"], help="whether to train or evaluate model")

    parser.add_argument("--num-gpus", type=int, help="number of gpus used for training")
    parser.add_argument("--num-epochs", type=int, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, help="batch size for model training")
    parser.add_argument("--learning-rate", type=float, help="learning rate value")
    parser.add_argument("--weight-decay", type=float, help="weight decay value")

    parser.add_argument("--date", type=str, help="for logging")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--seed", default=112233, type=int)

    args = parser.parse_args()
    return args 


def train(args):
    # set up dataset
    if args.dataset == "cifar10-imagenet":
        trainsform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404], std=[0.24205776, 0.23828046, 0.25874835])
            ]
        )
        domain_dataset = Domain_CifarImageNet(
            root="/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/train",
            transform=trainsform_train,
            target_transform=None
        )
        domain_loader = DataLoader(
            domain_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_gpus * 4,
        )
    
    # set up model
    model = DomainClassifier(
        num_classes=args.num_domains,
        arch=args.arch,
        pretrained=args.pretrained,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device
    )
    if args.num_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    # set up logger
    ckpt_dir = os.path.join(args.log_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    train_logger = logger(args.num_epochs * len(domain_loader), ["tb", "csv"], args.log_dir)

    model.train()
    step = 0
    for epoch in range(args.num_epochs):
        for _, (image, label) in enumerate(domain_loader):
            if args.num_gpus > 1:
                pred_loss, pred_acc = model.module.update(image, label)
            else:
                pred_loss, pred_acc = model.update(image, label)
            
            data_dict = {"pred_loss": pred_loss, "pred_acc": pred_acc}
            if args.local_rank == 0:
                train_logger.log(data_dict, step)
            step += 1

        if args.local_rank == 0:
            model.save(ckpt_dir, epoch)
    if args.local_rank == 0:
        model.save(ckpt_dir, epoch, final=True)
    return


def main():
    args = get_args()

    # distribute data parallel
    torch.backends.cudnn.benchmark = True
    args.device = "cuda:{}".format(args.local_rank)
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)
    if args.local_rank == 0:
        print(args)
    if args.num_gpus > 1:
        if args.local_rank == 0:
            print(f"Using distributed training on {args.num_gpus} gpus.")
        args.batch_size = args.batch_size // args.num_gpus
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    
    args.log_dir = os.path.join(
        "/n/fs/xl-diffbia/projects/minimal-diffusion/logs", args.date, args.dataset, "domain_classifier",
        "bs{}_lr{}_decay{}".format(arhs.batch_size, args.learning_rate, args.weight_decay)
    )
    os.makedirs(args.log_dir, exist_ok=True)

    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
