import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import json
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from optimiser_wrapper import CustomOptimizer


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_blocks=[3, 3, 3], num_classes=10):
        super().__init__()
        self.in_channels = 16

        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.stage1 = self._make_stage(16, num_blocks[0], stride=1)
        self.stage2 = self._make_stage(32, num_blocks[1], stride=2)
        self.stage3 = self._make_stage(64, num_blocks[2], stride=2)

        self.fc = nn.Linear(64, num_classes)

    def _make_stage(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(Block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def resnet20():
    return ResNet([3, 3, 3])


def resnet32():
    return ResNet([5, 5, 5])


def resnet56():
    return ResNet([9, 9, 9])


def get_data(bs=128):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    train = datasets.CIFAR10(
        "./data", train=True, download=True, transform=train_transform
    )
    test = datasets.CIFAR10(
        "./data", train=False, download=True, transform=test_transform
    )

    return DataLoader(train, bs, shuffle=True, num_workers=4), DataLoader(
        test, bs, num_workers=4
    )


def train_epoch(model, loader, opt, crit, device):
    model.train()
    loss_sum, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item()
        correct += out.argmax(1).eq(y).sum().item()
        total += y.size(0)
    return loss_sum / len(loader), 100 * correct / total


def test_epoch(model, loader, crit, device):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += crit(out, y).item()
            correct += out.argmax(1).eq(y).sum().item()
            total += y.size(0)
    return loss_sum / len(loader), 100 * correct / total


def adjust_lr(opt, epoch, miletones=[80, 120], gamma=0.1, base_lr=0.1):
    lr = base_lr
    for m in miletones:
        if epoch >= m:
            lr *= gamma
    opt.lr = lr
    return lr


def train(opt_name="sgd", base_lr=0.1, epochs=160, device="cuda", seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model = resnet20().to(device)
    opt = CustomOptimizer(model.parameters(), opt_type=opt_name, lr=base_lr)
    crit = nn.CrossEntropyLoss()
    tr_loader, te_loader = get_data()

    hist = {"tr_loss": [], "tr_acc": [], "te_loss": [], "te_acc": [], "lr": []}
    best_acc = 0

    for ep in range(epochs):
        current_lr = adjust_lr(opt, ep, base_lr=base_lr)
        tr_loss, tr_acc = train_epoch(model, tr_loader, opt, crit, device)
        te_loss, te_acc = test_epoch(model, te_loader, crit, device)

        hist["tr_loss"].append(tr_loss)
        hist["tr_acc"].append(tr_acc)
        hist["te_loss"].append(te_loss)
        hist["te_acc"].append(te_acc)
        hist["lr"].append(current_lr)

        if te_acc > best_acc:
            best_acc = te_acc

        if (ep + 1) % 20 == 0 or ep == 0:
            print(
                f"Ep {ep+1:3d} | Tr:{tr_loss:.3f}({tr_acc:.1f}%) | Te:{te_loss:.3f}({te_acc:.1f}%) | Best:{best_acc:.2f}%"
            )

        print(f"Final: Test{te_acc:.2f}% | Best: {best_acc:.2f}%")
        os.makedirs("results", exist_ok=True)
        with open(f"results/resnet20_{opt_name}.json", "w") as f:
            json.dump(hist, f, indent=2)

        return hist, best_acc


if __name__ == "__main__":
    hist, best_acc = train(opt_name="sgd", base_lr=0.1, epochs=160)
