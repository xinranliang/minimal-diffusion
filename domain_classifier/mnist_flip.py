import os
import pickle
import argparse
import numpy as np
import random
from PIL import Image
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn 
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from dataset.utils import logger, ArrayToImageLabel
import model.unets as unets
from model.diffusion import GuassianDiffusion

flip_classes = {
    2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 9: 6
}
flip_classes_inverse = {
    0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 9
}


class Domain_MNIST_Flip(datasets.MNIST):
    def __init__(
        self,
        root,
        transform_left,
        transform_right,
        target_transform,
        train,
        download
    ):
        super().__init__(root, train, None, target_transform, download)

        self.transform_left = transform_left
        self.transform_right = transform_right

        self.index_left = []
        self.index_right = []

        # all indexes by classes
        for index in range(len(self.data)):
            label = int(self.targets[index])
            if label in flip_classes.keys():
                self.index_left.append(index)
                self.index_right.append(index)
        # sort each by increasing order
        assert len(self.index_left) == len(self.index_right)
        self.index_left.sort()
        self.index_right.sort()
    
    def __len__(self):
        return len(self.index_left) + len(self.index_right)
    
    def __getitem__(self, index):
        if index < len(self.index_left):
            image, target = super().__getitem__(self.index_left[index])
            image = self.transform_left(image)
            label = 0
        else:
            image, target = super().__getitem__(self.index_right[index - len(self.index_left)])
            image = self.transform_right(image)
            label = 1

        if self.target_transform is not None:
            target = self.target_transform(target)
        target = flip_classes[target]

        return image, target, label



class DomainClassifier(nn.Module):
    def __init__(
        self,
        num_classes, # class-conditional domain classifier
        num_domains, # number of output dimension
        learning_rate,
        weight_decay,
        device
    ):
        super().__init__()
        self.device = device

        # conn blocks
        self.relu = nn.ReLU(inplace=True)
        # self.pool = nn.MaxPool2d(2)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.norm1 = nn.BatchNorm2d(32, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.norm2 = nn.BatchNorm2d(64, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, 1)
        self.norm3 = nn.BatchNorm2d(128, affine=False)

        # embedding to get repr of class-conditioning
        self.emb1 = nn.Embedding(num_classes, 32)
        self.emb2 = nn.Linear(32, 64)
        self.emb3 = nn.Linear(64, 128)

        # fc layer
        self.fc = nn.Sequential(
            nn.Linear(128, 32),
            self.relu,
            nn.Linear(32, num_domains)
        )

        self.to(self.device)

        # opt
        self.opt = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # loss function
        self.loss_fn = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)

        self.train()

        # print(sum(p.numel() for p in self.parameters() if p.requires_grad))
    

    def forward(self, x, c):
        # input shape: bs x c x h x w
        x = self.conv1(x)
        c = self.emb1(c)
        x = self.norm1(x) + c.unsqueeze(-1).unsqueeze(-1)
        x = self.relu(x)

        x = self.conv2(x)
        c = self.emb2(c)
        x = self.norm2(x) + c.unsqueeze(-1).unsqueeze(-1)
        x = self.relu(x)

        x = self.conv3(x)
        c = self.emb3(c)
        x = self.norm3(x) + c.unsqueeze(-1).unsqueeze(-1)
        x = self.relu(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x
    
    def error(self, x, c, y):
        x, c, y = x.to(self.device), c.to(self.device), y.to(self.device)
        pred_logits = self.forward(x, c)
        pred_loss = self.loss_fn(pred_logits, y)

        # compute accuracy
        pred_probs = self.softmax(pred_logits)
        pred_y = torch.argmax(pred_probs, dim=-1)
        pred_acc = torch.sum(pred_y == y) / y.shape[0]

        return pred_loss, pred_acc
    
    def predict(self, x, c):
        x, c = x.to(self.device), c.to(self.device)
        pred_logits = self.forward(x, c)
        # compute accuracy
        pred_probs = self.softmax(pred_logits)
        pred_y = torch.argmax(pred_probs, dim=-1)
        return pred_y
    
    def update(self, x, c, y):
        pred_loss, pred_acc = self.error(x, c, y)

        self.opt.zero_grad()
        pred_loss.backward()
        self.opt.step()

        return pred_loss, pred_acc
    
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
    parser = argparse.ArgumentParser("Training domain classifier on images")

    parser.add_argument("--dataset", type=str, help="specify what dataset source")
    parser.add_argument("--num-classes", type=int, help="number of conditioning classes")
    parser.add_argument("--num-domains", type=int, help="number of output classes")

    parser.add_argument("--mode", type=str, choices=["train", "test-real", "test-fake"], help="whether to train or test model")
    parser.add_argument("--train-split", type=float, default=0.8, help="portion of dataset splitted to training")
    parser.add_argument("--test-split", type=float, default=0.2, help="portion of dataset splitted to evaluation")

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
    # set up dataset
    if args.dataset == "mnist-subset":
        transform_left = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        28, scale=(0.8, 1.0), ratio=(0.8, 1.2)
                    ),
                    transforms.ToTensor(),
                ]
        )
        transform_right = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=1.0),
                    transforms.RandomResizedCrop(
                        28, scale=(0.8, 1.0), ratio=(0.8, 1.2)
                    ),
                    transforms.ToTensor(),
                ]
        )
        domain_dataset_train = Domain_MNIST_Flip(
            root = "/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/mnist",
            transform_left = transform_left,
            transform_right = transform_right,
            target_transform = None,
            train=True,
            download=True
        )
        domain_loader_train = DataLoader(
            domain_dataset_train, 
            batch_size = args.batch_size,
            shuffle=True,
            num_workers = args.num_gpus * 4
        )
        domain_dataset_test = Domain_MNIST_Flip(
            root = "/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/mnist",
            transform_left = transform_left,
            transform_right = transform_right,
            target_transform = None,
            train=False,
            download=True
        )
        domain_loader_test = DataLoader(
            domain_dataset_test, 
            batch_size = args.batch_size,
            shuffle=False,
            num_workers = args.num_gpus * 4
        )
    
    # set up model
    model = DomainClassifier(
        num_classes = 7,
        num_domains = 2,
        learning_rate = args.learning_rate,
        weight_decay = args.weight_decay,
        device = args.device
    )
    if args.num_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    # set up logger
    ckpt_dir = os.path.join(args.log_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    train_logger = logger(args.num_epochs * len(domain_loader_train), ["tb", "csv"], args.log_dir)

    model.train()
    step = 0
    for epoch in range(args.num_epochs):
        for _, (image, target, label) in enumerate(domain_loader_train):
            if args.num_gpus > 1:
                pred_loss, pred_acc = model.module.update(image, target, label)
            else:
                pred_loss, pred_acc = model.update(image, target, label)
            
            data_dict = {"pred_loss": pred_loss.detach().cpu().numpy(), "pred_acc": pred_acc.detach().cpu().numpy()}
            if args.local_rank == 0:
                train_logger.log(data_dict, step)
            step += 1

        if args.local_rank == 0 and (epoch + 1) % 5 == 0:
            model.save(ckpt_dir, epoch)
    if args.local_rank == 0:
        model.save(ckpt_dir, epoch, final=True)
    
    # evaluate
    model.eval()
    accs = []
    for image, target, label in iter(domain_loader_test):
        with torch.no_grad():
            _, acc = model.error(image, target, label)
            accs.append(acc.detach().cpu().numpy())
    final_accuracy = np.array(accs, dtype=float).mean()
    print(f"Final test accuracy at epoch {epoch + 1}: {final_accuracy:.3f}")

    return


def test_real(args):
    # set up dataset
    if args.dataset == "mnist-subset":
        transform_left = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        28, scale=(0.8, 1.0), ratio=(0.8, 1.2)
                    ),
                    transforms.ToTensor(),
                ]
        )
        transform_right = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=1.0),
                    transforms.RandomResizedCrop(
                        28, scale=(0.8, 1.0), ratio=(0.8, 1.2)
                    ),
                    transforms.ToTensor(),
                ]
        )
        domain_dataset_train = Domain_MNIST_Flip(
            root = "/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/mnist",
            transform_left = transform_left,
            transform_right = transform_right,
            target_transform = None,
            train=True,
            download=False
        )
        domain_loader_train = DataLoader(
            domain_dataset_train, 
            batch_size = args.batch_size,
            shuffle=False,
            num_workers = args.num_gpus * 4
        )
        domain_dataset_test = Domain_MNIST_Flip(
            root = "/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/mnist",
            transform_left = transform_left,
            transform_right = transform_right,
            target_transform = None,
            train=False,
            download=False
        )
        domain_loader_test = DataLoader(
            domain_dataset_test, 
            batch_size = args.batch_size,
            shuffle=False,
            num_workers = args.num_gpus * 4
        )
    
    # set up model
    model = DomainClassifier(
        num_classes = 7,
        num_domains = 2,
        learning_rate = args.learning_rate,
        weight_decay = args.weight_decay,
        device = args.device
    )
    if args.num_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    if args.ckpt_path:
        model.load(args.ckpt_path)
    model.eval()

    # evaluate
    train_accs = []
    num_train = 0
    for image, target, label in iter(domain_loader_train):
        with torch.no_grad():
            _, train_acc = model.error(image, target, label)
            train_accs.append(train_acc.detach().cpu().numpy())
            num_train += label.shape[0]
    train_accuracy = np.array(train_accs, dtype=float).mean()
    print(f"Training set accuracy: {train_accuracy:.3f} over {num_train} images.")

    test_accs = []
    num_test = 0
    for image, target, label in iter(domain_loader_test):
        with torch.no_grad():
            _, test_acc = model.error(image, target, label)
            test_accs.append(test_acc.detach().cpu().numpy())
            num_test += label.shape[0]
    test_accuracy = np.array(test_accs, dtype=float).mean()
    print(f"Training set accuracy: {test_accuracy:.3f} over {num_test} images.")

    return


def test_fake(args):
    if args.dataset == "mnist-subset":
        # Creat model and diffusion process
        myunet = unets.__dict__["UNetSmall"](
            image_size=28,
            in_channels=1,
            out_channels=1,
            num_classes=7,
            prob_drop_cond=0.1
        ).to(args.device)
        if args.local_rank == 0:
            print(
                "We are assuming that model input/ouput pixel range is [-1, 1]. Please adhere to it."
            )
        mydiffusion = GuassianDiffusion(1000, args.device)
        ckpt_list = [
            "./logs/2023-04-08/mnist-subset/left0.5_right0.5/UNetSmall_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_99_ema_0.9995.pth",
            # "./logs/2023-04-09/mnist-subset/left0.5_right0.5/UNetSmall_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_256_dropprob_0.1/ckpt/epoch_99_ema_0.9995.pth"
        ]
        for ckpt_path in ckpt_list:
            print(f"Loading pretrained model from {ckpt_path}")
            file_load = torch.load(ckpt_path, map_location=args.device)
            myunet.load_state_dict(file_load)
            print(f"Loaded pretrained model from {ckpt_path}")

            samples, labels, num_samples = [], [], 0

            with tqdm(total=len(flip_classes_inverse.keys())) as pbar:
                for class_label in flip_classes_inverse.keys():
                    # initialize noise
                    xT = (
                            torch.randn(10, 1, 28, 28)
                            .float()
                            .to(args.device)
                        )
                    # specify class label
                    y = torch.zeros(size=(len(xT),), dtype=torch.int64).to(args.device)
                    y.fill_(class_label)

                    gen_images = mydiffusion.sample_from_reverse_process(
                        myunet, xT, 250, {"y": y}, False, 0.0
                    )
                    labels.append(y.detach().cpu().numpy())
                    samples.append(gen_images.detach().cpu().numpy())
                    num_samples += len(xT)
                    pbar.update(1)
                    
            samples = np.concatenate(samples).transpose(0, 2, 3, 1) # shape = num_samples x height x width x n_channel
            samples = (127.5 * (samples + 1)).astype(np.uint8)
            labels = np.concatenate(labels)

            assert samples.shape[0] == labels.shape[0], "samples and labels must in same number"
            for index in range(samples.shape[0]):
                folder_path = os.path.join("./logs/2023-04-08/mnist-subset/domain_classifier/synthetic_testset", "class_{}".format(flip_classes_inverse[labels[index]]))
                os.makedirs(folder_path, exist_ok=True)
                cv2.imwrite(
                    os.path.join(folder_path, "sample_{}.png".format(index)), samples[index]
                )
            
            # set up dataset
            transform_test = transforms.Compose(
                    [
                        # transforms.Grayscale(num_output_channels=1),
                        transforms.RandomResizedCrop(
                            28, scale=(0.8, 1.0), ratio=(0.8, 1.2)
                        ),
                        transforms.ToTensor(),
                    ]
            )
            # domain_dataset = datasets.ImageFolder(
                # root = "./logs/2023-04-08/mnist-subset/domain_classifier/synthetic_testset",
                # transform = transform_test,
                # target_transform = None
            # )
            domain_dataset = ArrayToImageLabel(
                samples = samples,
                labels = labels,
                mode = "L",
                transform = transform_test,
                target_transform = None
            )
            domain_dataloader = DataLoader(
                domain_dataset,
                batch_size = 10,
                shuffle = False,
                num_workers = args.num_gpus * 4
            )
    
    # set up model
    model = DomainClassifier(
        num_classes = 7,
        num_domains = 2,
        learning_rate = args.learning_rate,
        weight_decay = args.weight_decay,
        device = args.device
    )
    if args.num_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    if args.ckpt_path:
        model.load(args.ckpt_path)
    model.eval()
    
    for image, label in iter(domain_dataloader):
        with torch.no_grad():
            syn_pred = model.predict(image, label)
            print(label.detach().cpu().numpy())
            print(syn_pred.detach().cpu().numpy())
    
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
            "/n/fs/xl-diffbia/projects/minimal-diffusion/logs", args.date, args.dataset, "domain_classifier",
            "bs{}_lr{}_decay{}".format(args.batch_size, args.learning_rate, args.weight_decay)
        )
        os.makedirs(args.log_dir, exist_ok=True)
    
    # set up dataset
    if args.dataset == "mnist-subset" and (args.mode == "train" or args.mode == "test-real"):
        with open(
            os.path.join("/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/mnist/MNIST_FLIP/flip_index_split/domain_classifier",
            "train{}_test{}_index.pkl".format(args.train_split, args.test_split)), "rb"
        ) as f:
            file_load = pickle.load(f)

        train_idx, test_idx = file_load["train_index"], file_load["test_index"]
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)
    
    if args.mode == "train":
        train(args)
    elif args.mode == "test-real":
        test_real(args)
    elif args.mode == "test-fake":
        test_fake(args)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()