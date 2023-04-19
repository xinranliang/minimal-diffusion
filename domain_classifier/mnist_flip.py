import torch
import torch.nn as nn 

from torchvision import datasets, transforms

flip_classes = {
    2: 0,
    3: 1,
    4: 2,
    5: 3,
    6: 4,
    7: 5,
    9: 6
}

class CondNorm(nn.Module):
    def __init__(self, num_channel):
        super().__init__()
        self.num_channel = num_channel
    
    def mean(self, x):
        # over spatial dimension
        # input shape: bs x c x h x w
        return torch.sum(x, (0, 1)) / (x.shape[0] * x.shape[1]) # h x w
    
    def std(self, x):
        return torch.sqrt(torch.var(x, dim=(0, 1)) + 1e-5)
    
    def forward(self, x, c):
        return (x - self.mean(x)) / self.std(x) + c.unsqueeze(-1).unsqueeze(-1)


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
        super().__init__(root, train, None, target_transform, False)

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

        if target_transform is not None:
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

        print(sum(p.numel() for p in self.parameters() if p.requires_grad))
    

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
        pred_logits = self.dorward(x, c)
        pred_loss = self.loss_fn(pred_logits, y)

        # compute accuracy
        pred_probs = self.softmax(pred_logits)
        pred_y = torch.argmax(pred_probs, dim=-1)
        pred_acc = torch.sum(pred_y == y) / y.shape[0]

        return pred_loss, pred_acc
    
    def update(self, x, c, y):
        pred_loss, pred_acc = self.error(x, c, y)

        self.opt.zero_grad()
        pred_loss.backward()
        self.opt.step()

        return pred_loss, pred_acc


if __name__ == "__main__":
    image = torch.rand((4, 1, 28, 28), dtype=torch.float).cuda() # bs x c x h x w
    condition = torch.empty(4, dtype=torch.long).random_(10)
    model = DomainClassifier(10, 2, 0.001, 0.0001, "cuda")
    model.forward(image, condition)
