import os 
import numpy as np 
import random 
import argparse
from PIL import Image
import cv2

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn 
from torch.nn.parallel import DistributedDataParallel

from dataset.utils import logger, ArraytoImage


class SimpleClassifier(nn.Module):
    def __init__(self, learning_rate, weight_decay, device):
        super().__init__()
        self.device = device

        # conn blocks
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)

        # fc layer
        self.fc = nn.Sequential(
            nn.Linear(64 * 2 * 2, 128),
            self.relu,
            nn.Linear(128, 10)
        )

        self.to(self.device)

        # opt
        self.opt = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # loss
        self.loss_fn = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)

        self.train()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = self.flatten(x)
        x = self.fc(x)
        return x
    
    def error(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        pred_logits = self.forward(x)
        pred_loss = self.loss_fn(pred_logits, y)

        # compute accuracy
        pred_probs = self.softmax(pred_logits)
        pred_y = torch.argmax(pred_probs, dim=-1)
        pred_acc = torch.sum(pred_y == y) / y.shape[0]

        return pred_loss, pred_acc
    
    def update(self, x, y):
        pred_loss, pred_acc = self.error(x, y)

        self.opt.zero_grad()
        pred_loss.backward()
        self.opt.step()

        return pred_loss, pred_acc
    
    def predict(self, x):
        pred_logits = self.forward(x)
        pred_probs = self.softmax(pred_logits)
        pred_y = torch.argmax(pred_probs, dim=-1)
        return pred_y
    
    def save(self, log_dir, step, final=False):
        checkpoint = {
            "param": self.state_dict(),
            "optim": self.opt.state_dict(),
        }
        if not final:
            torch.save(
                checkpoint, os.path.join(log_dir, "model_param_{}.pth".format(int(step)))
            )
        else:
            torch.save(
                checkpoint, os.path.join(log_dir, "model_param_final.pth")
            )
    
    def load(self, ckpt_path):
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        self.load_state_dict(ckpt_dict["param"])
        self.opt.load_state_dict(ckpt_dict["optim"])
        print(f"Loading model checkpoint from {ckpt_path}.")


def get_args():
    parser = argparse.ArgumentParser("Training simple classifier on CIFAR10")
    parser.add_argument("--num-classes", type=int, help="number of conditioning classes")

    parser.add_argument("--mode", type=str, choices=["train", "test-real", "confusion-matrix"], help="whether to train or test model or plot confusion matrix")
    parser.add_argument("--ckpt-path", type=str, required=False, help="path directory to pretrained checkpoint for evaluation")

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
    # transformation
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    # dataset and dataloader
    train_dataset = datasets.CIFAR10(
        root="/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10",
        train=True,
        transform=transform_train,
        target_transform=None,
        download=False
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_gpus * 4
    )
    test_dataset = datasets.CIFAR10(
        root="/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10",
        train=False,
        transform=transform_test,
        target_transform=None,
        download=False
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_gpus * 4
    )

    # model
    simple_model = SimpleClassifier(
        learning_rate = args.learning_rate,
        weight_decay = args.weight_decay,
        device=args.device
    )
    if args.num_gpus > 1:
        simple_model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    # set up logger
    ckpt_dir = os.path.join(args.log_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    train_logger = logger(args.num_epochs * len(train_dataloader), ["tb", "csv"], args.log_dir)

    simple_model.train()
    step = 0
    for epoch in range(args.num_epochs):
        for _, (image, label) in enumerate(train_dataloader):
            if args.num_gpus > 1:
                pred_loss, pred_acc = simple_model.module.update(image, label)
            else:
                pred_loss, pred_acc = simple_model.update(image, label)
            
            data_dict = {"pred_loss": pred_loss.detach().cpu().numpy(), "pred_acc": pred_acc.detach().cpu().numpy()}
            if args.local_rank == 0:
                train_logger.log(data_dict, step)
            step += 1
        
        if args.local_rank == 0 and (epoch + 1) % 5 == 0:
            model.save(ckpt_dir, epoch)
    if args.local_rank == 0:
        model.save(ckpt_dir, epoch, final=True)
    
    # evaluate
    simple_model.eval()
    accs = []
    for image, label in iter(test_dataloader):
        with torch.no_grad():
            _, acc = simple_model.error(image, label)
            accs.append(acc.detach().cpu().numpy())
    final_accuracy = np.array(accs, dtype=float).mean()
    print(f"Final test accuracy at epoch {epoch + 1}: {final_accuracy:.3f}")

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
    
    if args.mode == "train":
        args.log_dir = os.path.join(
            "/n/fs/xl-diffbia/projects/minimal-diffusion/logs", args.date, "cifar-superclass/simple_classifier",
            "bs{}_lr{}_decay{}".format(args.batch_size, args.learning_rate, args.weight_decay)
        )
        os.makedirs(args.log_dir, exist_ok=True)
    
    if args.mode == "train":
        train(args)


if __name__ == "__main__":
    main()
