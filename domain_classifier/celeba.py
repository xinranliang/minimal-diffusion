import argparse
import os
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn 
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from dataset.utils import logger
from dataset.celeba import root_path

class Domain_CelebA(datasets.CelebA):
    def __init__(
        self, 
        root,
        split,
        target_type,
        transform,
        target_transform=None,
        download=False
    ):
        super().__init__(root, split, target_type, transform, target_transform, download)

    def __getitem__(self, index):
        image, attr_target = super().__getitem__(index)
        # attribute target shape = (40,)
        gender_label = torch.select(attr_target, 0, self.attr_names.index("Male")).item() # Male = 1, Female = 0
        return image, attr_target, gender_label


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
    
    def predict(self, image, label=None):
        if label is None:
            image = image.to(self.device)
        else:
            image, label = image.to(self.device), label.to(self.device)

        pred_logits = self.forward(image)
        # compute accuracy
        pred_probs = self.softmax(pred_logits)
        pred_target = torch.argmax(pred_probs, dim=-1)

        if label is not None:
            pred_acc = torch.sum(pred_target == label) / label.shape[0]
            return pred_acc
        else:
            return pred_target
    
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

    parser.add_argument("--mode", type=str, choices=["train", "test-real", "test-fake"], help="whether to train or test model")

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
    if args.dataset == "celeba":
        # train mode
        trainsform_train = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalized by imagenet mean std
            ]
        )
        # eval mode
        transform_eval = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalized by imagenet mean std
            ]
        )

        # dataset
        domain_dataset_train = Domain_CelebA(
            root = root_path,
            split = "train",
            target_type="attr",
            transform=trainsform_train,
            target_transform=None
        )
        domain_dataset_val = Domain_CelebA(
            root = root_path,
            split = "valid",
            target_type="attr",
            transform=transform_eval,
            target_transform=None
        )
        domain_dataset_test = Domain_CelebA(
            root = root_path,
            split = "test",
            target_type="attr",
            transform=transform_eval,
            target_transform=None
        )
        # get domain label distribution
        """if args.local_rank == 0:
            get_domain_dist(domain_dataset_train, "train")
            get_domain_dist(domain_dataset_val, "val")
            get_domain_dist(domain_dataset_test, "test")"""
        # dataloader
        if args.num_gpus > 1:
            domain_loader_train = DataLoader(
                domain_dataset_train,
                batch_size=args.batch_size,
                shuffle=False,
                sampler=DistributedSampler(domain_dataset_train),
                num_workers=args.num_gpus * 4,
            )
        else:
            domain_loader_train = DataLoader(
                domain_dataset_train,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_gpus * 4,
            )
        domain_loader_val = DataLoader(
            domain_dataset_val,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_gpus * 4,
        )
        domain_loader_test = DataLoader(
            domain_dataset_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_gpus * 4,
        )

    # model
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
    
    # logger
    ckpt_dir = os.path.join(args.log_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    train_logger = logger(args.num_epochs * len(domain_loader_train), ["tb", "csv"], args.log_dir)

    model.train()
    step = 0
    for epoch in range(args.num_epochs):
        if args.num_gpus > 1:
            domain_loader_train.sampler.set_epoch(epoch)
        for _, (image, attr, label) in enumerate(domain_loader_train):
            if args.num_gpus > 1:
                pred_loss, pred_acc = model.module.update(image, label)
            else:
                pred_loss, pred_acc = model.update(image, label)
            
            data_dict = {"pred_loss": pred_loss.detach().cpu().numpy(), "pred_acc": pred_acc.detach().cpu().numpy()}
            if args.local_rank == 0:
                train_logger.log(data_dict, step)
            step += 1

        if args.local_rank == 0 and (epoch + 1) % 5 == 0:
            if isinstance(model, DistributedDataParallel):
                model.module.save(ckpt_dir, epoch)
            else:
                model.save(ckpt_dir, epoch)
            validate(model, domain_loader_val, epoch)
            model.train() # back to training mode
    
    if args.local_rank == 0:
        if isinstance(model, DistributedDataParallel):
            model.module.save(ckpt_dir, epoch, final=True)
        else:
            model.save(ckpt_dir, epoch, final=True)
        test(model, domain_loader_test)


def validate(model, val_dataloader, epoch):
    model.eval()
    with torch.no_grad():
        accs = []
        for image, attr, label in iter(val_dataloader):
            with torch.no_grad():
                if isinstance(model, DistributedDataParallel):
                    _, acc = model.module.get_error(image, label)
                else:
                    _, acc = model.get_error(image, label)
                accs.append(acc.detach().cpu().numpy())
        final_accuracy = np.array(accs, dtype=float).mean()
    print(f"Mean validation accuracy at epoch {epoch + 1}: {final_accuracy:.3f}")

def test(model, test_dataloader):
    model.eval()
    with torch.no_grad():
        accs = []
        for image, attr, label in iter(test_dataloader):
            with torch.no_grad():
                if isinstance(model, DistributedDataParallel):
                    _, acc = model.module.get_error(image, label)
                else:
                    _, acc = model.get_error(image, label)
                accs.append(acc.detach().cpu().numpy())
        final_accuracy = np.array(accs, dtype=float).mean()
    print(f"Final test accuracy: {final_accuracy:.3f}")


def get_domain_dist(domain_dataset, split):
    count_labels = defaultdict(int)
    for image, attr_label, domain_label in domain_dataset:
        count_labels[domain_label] += 1
    
    print("Split {} has {} Female samples and {} Male samples".format(split, count_labels[0], count_labels[1]))


def main(auto_rank, world_size):
    args = get_args()

    # distribute data parallel
    torch.backends.cudnn.benchmark = True
    args.local_rank = auto_rank
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
        torch.distributed.init_process_group(backend="nccl", init_method="env://", rank=args.local_rank, world_size=world_size)
    
    if args.mode == "train":
        args.log_dir = os.path.join(
            "/n/fs/xl-diffbia/projects/minimal-diffusion/logs", args.date, args.dataset, "domain_classifier",
            "bs{}_lr{}_decay{}".format(args.batch_size, args.learning_rate, args.weight_decay)
        )
        os.makedirs(args.log_dir, exist_ok=True)
    
    if args.mode == "train":
        train(args)
    
    return


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "8888"
        torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size)
    else:
        main(auto_rank = 0, world_size = 1)