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
from torch.utils.data.sampler import SubsetRandomSampler

import torch
import torch.nn as nn 
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from dataset.utils import logger, ArraytoImage


def count_cifar_imgnet(imagelabel_loader, classifier, return_type, classwise=False):
    if not classwise:
        num_cifar, num_imgnet = 0, 0
        prob_list = []
        for image, _ in iter(imagelabel_loader):
            with torch.no_grad():
                syn_pred = classifier.predict(image)
                syn_pred_prob = syn_pred.detach().cpu().numpy() # num_samples x 2
                # return predicted probability histogram
                prob_list.extend(np.squeeze(syn_pred_prob[:, 0])) # (num_samples, )
                # return binary predicted counts
                syn_pred_count = torch.argmax(syn_pred, dim=-1)
                syn_pred_count = syn_pred_count.detach().cpu().numpy() # (num_samples, )
                num_cifar += sum(syn_pred_count == 0)
                num_imgnet += sum(syn_pred_count == 1)
        
        return_results = {}
        if "percent" in return_type:
            return_results["percent"] = {"num_cifar": num_cifar, "num_imgnet": num_imgnet}
        if "histogram" in return_type:
            return_results["histogram"] = np.array(prob_list, dtype=float)
        return return_results


def eval_cifar_imgnet_domain(imagelabel_loader, classifier, return_type):
    pred_labels = []
    true_labels = []
    for image, domain_class_label in iter(imagelabel_loader):
        # transform true label from class conditioning
        # [0, 9] in cifar; [10, 19] in imgnet
        domain_class_label = domain_class_label.detach().cpu().numpy()
        domain_class_label = np.where(domain_class_label < 10, 0, np.where(domain_class_label >= 10, 1, domain_class_label))
        true_labels.append(domain_class_label)

        with torch.no_grad():
            syn_pred = classifier.predict(image)
            syn_pred_prob = syn_pred.detach().cpu().numpy() # num_samples x 2
            syn_pred_value = torch.argmax(syn_pred, dim=-1).detach().cpu().numpy()
            pred_labels.append(syn_pred_value)
        
    pred_labels = np.concatenate(pred_labels)
    true_labels = np.concatenate(true_labels)
    accuracy = sum(pred_labels == true_labels) / 2500
    return_results = {}
    if "true" in return_type:
        return_results["true"] = {"num_cifar": sum(true_labels == 0), "num_imgnet": sum(true_labels == 1)}
    if "pred" in return_type:
        return_results["pred"] = {"num_cifar": sum(pred_labels == 0), "num_imgnet": sum(pred_labels == 1)}
    if "full" in return_type:
        return_results["full"] = {"pred_labels": pred_labels, "true_labels": true_labels, "accuracy": accuracy}
    return return_results


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
                checkpoint, os.path.join(logdir, "model_param_{}.pth".format(int(step)))
            )
        else:
            torch.save(
                checkpoint, os.path.join(logdir, "model_param_final.pth")
            )
    
    def load(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        self.cnn.load_state_dict(checkpoint["cnn"])
        self.head.load_state_dict(checkpoint["head"])
        self.opt.load_state_dict(checkpoint["opt"])
        print(f"Loading model checkpoint from {ckpt_path}.")

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
    
    def predict(self, image, label=None, adapt=False):
        # training time provided with true labels, return binary prediction accuracy
        # inference time provided with image only, return predicted probability of two classes
        if label is None:
            image = image.to(self.device)
        else:
            image, label = image.to(self.device), label.to(self.device)

        pred_logits = self.forward(image)
        # compute accuracy
        if adapt:
            pred_logits[:, 0] = pred_logits[:, 0] - torch.log(torch.tensor(50 / 260, dtype=torch.float, device=self.device))
            pred_logits[:, 1] = pred_logits[:, 1] - torch.log(torch.tensor(210 / 260, dtype=torch.float, device=self.device))
        pred_probs = self.softmax(pred_logits)
        # pred_target = torch.argmax(pred_probs, dim=-1)

        if label is not None:
            pred_target = torch.argmax(pred_probs, dim=-1)
            pred_acc = torch.sum(pred_target == label) / label.shape[0]
            return pred_acc
        else:
            return pred_probs
    
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

    parser.add_argument("--mode", type=str, choices=["train", "test-real", "test-fake", "test-gan"], help="whether to train or test model")
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


def train(args, train_sampler, test_sampler):
    # set up dataset
    if args.dataset == "cifar10-imagenet":
        # train mode
        trainsform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404], std=[0.24205776, 0.23828046, 0.25874835])
            ]
        )
        domain_dataset_train = Domain_CifarImageNet(
            root="/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/train",
            transform=trainsform_train,
            target_transform=None
        )
        domain_loader_train = DataLoader(
            domain_dataset_train,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_gpus * 4,
        )
        # test_mode
        trainsform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404], std=[0.24205776, 0.23828046, 0.25874835])
            ]
        )
        domain_dataset_test = Domain_CifarImageNet(
            root="/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/train",
            transform=trainsform_test,
            target_transform=None
        )
        domain_loader_test = DataLoader(
            domain_dataset_test,
            batch_size=args.batch_size,
            sampler=test_sampler,
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
    train_logger = logger(args.num_epochs * len(domain_loader_train), ["tb", "csv"], args.log_dir)

    model.train()
    step = 0
    for epoch in range(args.num_epochs):
        for _, (image, label) in enumerate(domain_loader_train):
            if args.num_gpus > 1:
                pred_loss, pred_acc = model.module.update(image, label)
            else:
                pred_loss, pred_acc = model.update(image, label)
            
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
    for image, label in iter(domain_loader_test):
        with torch.no_grad():
            _, acc = model.get_error(image, label)
            accs.append(acc.detach().cpu().numpy())
    final_accuracy = np.array(accs, dtype=float).mean()
    print(f"Final test accuracy at epoch {epoch + 1}: {final_accuracy:.3f}")

    return


def test_real(args, train_sampler, test_sampler):
    # set up dataset
    if args.dataset == "cifar10-imagenet":
        data_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404], std=[0.24205776, 0.23828046, 0.25874835])
            ]
        )
        domain_dataset = Domain_CifarImageNet(
            root="/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/train",
            transform=data_transform,
            target_transform=None
        )
        domain_loader_train = DataLoader(
            domain_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_gpus * 4,
        )
        domain_loader_test = DataLoader(
            domain_dataset,
            batch_size=args.batch_size,
            sampler=test_sampler,
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
    if args.ckpt_path:
        model.load(args.ckpt_path)
    model.eval()

    # evaluate
    train_accs = []
    num_train = 0
    for image, label in iter(domain_loader_train):
        with torch.no_grad():
            _, train_acc = model.get_error(image, label)
            train_accs.append(train_acc.detach().cpu().numpy())
            num_train += label.shape[0]
    train_accuracy = np.array(train_accs, dtype=float).mean()
    print(f"Training set accuracy: {train_accuracy:.3f} over {num_train} images.")

    test_accs = []
    num_test = 0
    for image, label in iter(domain_loader_test):
        with torch.no_grad():
            _, test_acc = model.get_error(image, label)
            test_accs.append(test_acc.detach().cpu().numpy())
            num_test += label.shape[0]
    test_accuracy = np.array(test_accs, dtype=float).mean()
    print(f"Test set accuracy: {test_accuracy:.3f} over {num_test} images.")

    return


def test_fake(args, adapt):
    # set up dataset
    if args.dataset == "cifar10-imagenet":
        data_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404], std=[0.24205776, 0.23828046, 0.25874835])
            ]
        )
        domain_dataset = ArraytoImage(
            paths=[
                "./logs/2023-04-06/cifar10-imagenet/cifar50000_imagenet0/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/cifar50_imagenet0_cond_ema_num50000_guidance0.0.npz",
                "./logs/2023-04-07/cifar10-imagenet/cifar50000_imagenet0/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/cifar50_imagenet0_cond_ema_num50000_guidance0.0.npz",
                "./logs/2023-04-06/cifar10-imagenet/cifar0_imagenet210000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/cifar0_imagenet210_cond_ema_num50000_guidance0.0.npz",
                "./logs/2023-04-07/cifar10-imagenet/cifar0_imagenet210000/UNet_diffusionstep_1000_samplestep_250_condition_True_lr_0.0001_bs_5000_dropprob_0.1/samples_ema/cifar0_imagenet210_cond_ema_num50000_guidance0.0.npz"
                ],
            transform=data_transform,
            target_transform=None
        )
        domain_dataloader = DataLoader(
            domain_dataset,
            batch_size=args.batch_size,
            shuffle=False,
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
    if args.ckpt_path:
        model.load(args.ckpt_path)
    model.eval()

    syn_accs = []
    num_syn = 0
    for image, _, label in iter(domain_dataloader):
        with torch.no_grad():
            # label = torch.zeros((image.shape[0], ), dtype=torch.long)
            syn_acc = model.predict(image, label, adapt=adapt)
            syn_accs.append(syn_acc.detach().cpu().numpy())
            num_syn += image.shape[0]
    synthetic_accuracy = np.array(syn_accs, dtype=float).mean()
    if adapt:
        print(f"Test set accuracy w/ adaptation: {synthetic_accuracy:.3f} over {num_syn} images.")
    else:
        print(f"Test set accuracy w/o adaptation: {synthetic_accuracy:.3f} over {num_syn} images.")

    return


def test_gan(args):
    if args.dataset == "cifar10-imagenet":
        data_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404], std=[0.24205776, 0.23828046, 0.25874835])
            ]
        )
        domain_dataset = datasets.ImageFolder(
            root = "/n/fs/xl-diffbia/projects/stylegan2-ada/logs/samples",
            transform = data_transform,
            target_transform = None
        )
        domain_dataloader = DataLoader(
            domain_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_gpus * 4,
        )
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
    if args.ckpt_path:
        model.load(args.ckpt_path)
    model.eval()

    syn_accs = []
    num_syn = 0
    for image, label in iter(domain_dataloader):
        with torch.no_grad():
            syn_acc = model.predict(image, label, adapt=False)
            syn_accs.append(syn_acc.detach().cpu().numpy())
            num_syn += image.shape[0]
    synthetic_accuracy = np.array(syn_accs, dtype=float).mean()
    print(f"Test set accuracy w/o adaptation: {synthetic_accuracy:.3f} over {num_syn} images.")

    syn_accs = []
    num_syn = 0
    for image, label in iter(domain_dataloader):
        with torch.no_grad():
            syn_acc = model.predict(image, label, adapt=True)
            syn_accs.append(syn_acc.detach().cpu().numpy())
            num_syn += image.shape[0]
    synthetic_accuracy = np.array(syn_accs, dtype=float).mean()
    print(f"Test set accuracy w/ adaptation: {synthetic_accuracy:.3f} over {num_syn} images.")
    
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
    if args.dataset == "cifar10-imagenet" and (args.mode == "train" or args.mode == "test-real"):
        with open(
            os.path.join(
                "/n/fs/xl-diffbia/projects/minimal-diffusion/datasets/cifar10-imagenet/index_split/domain_classifier",
                "train{}_test{}_index.pkl".format(args.train_split, args.test_split)
            ), "rb"
        ) as f:
            file_load = pickle.load(f)

        train_idx, test_idx = file_load["train_index"], file_load["test_index"]
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)

    if args.mode == "train":
        train(args, train_sampler, test_sampler)
    elif args.mode == "test-real":
        test_real(args, train_sampler, test_sampler)
    elif args.mode == "test-fake":
        test_fake(args, adapt=False)
        test_fake(args, adapt=True)
    elif args.mode == "test-gan":
        test_gan(args)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
