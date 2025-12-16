# train_cifar_final.py - Production-ready CNN baseline
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import time
import json
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from optimiser_wrapper import CustomOptimizer


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def get_data(bs=256):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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


def adjust_lr(opt, epoch, milestones=[60, 120, 160], gamma=0.1, base_lr=0.1):
    lr = base_lr
    for m in milestones:
        if epoch >= m:
            lr *= gamma
    opt.lr = lr
    return lr


def save_checkpoint(model, opt, epoch, best_acc, path="checkpoints/best.pt"):
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(
        {"epoch": epoch, "model_state": model.state_dict(), "best_acc": best_acc}, path
    )


def train(opt_name="sgd", base_lr=0.1, epochs=200, device="cuda", seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model = CNN().to(device)
    opt = CustomOptimizer(model.parameters(), opt_type=opt_name, lr=base_lr)
    crit = nn.CrossEntropyLoss()
    tr_loader, te_loader = get_data()

    hist = {"tr_loss": [], "tr_acc": [], "te_loss": [], "te_acc": [], "lr": []}
    best_acc = 0

    print(f"\n{'='*70}")
    print(f"Training with BatchNorm | {opt_name.upper()}")
    print(
        f"Base LR: {base_lr} | Epochs: {epochs} | Device: {torch.cuda.get_device_name(0)}"
    )
    print(f"{'='*70}\n")

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
            save_checkpoint(model, opt, ep, best_acc)

        if (ep + 1) % 20 == 0 or ep == 0:
            print(
                f"Ep {ep+1:3d} | Tr:{tr_loss:.3f}({tr_acc:.1f}%) | Te:{te_loss:.3f}({te_acc:.1f}%) | LR:{current_lr:.4f} | Best:{best_acc:.2f}%"
            )

    print(f"\n{'='*70}")
    print(f"Final: Test {te_acc:.2f}% | Best: {best_acc:.2f}%")
    print(f"{'='*70}\n")

    # Save results
    os.makedirs("results", exist_ok=True)
    with open(f"results/cnn_final_{opt_name}.json", "w") as f:
        json.dump(hist, f, indent=2)

    return hist, best_acc


def plot_results(hist, opt_name, best_acc):
    os.makedirs("viz", exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    ax[0, 0].plot(hist["tr_loss"], "b-", lw=2, label="Train")
    ax[0, 0].plot(hist["te_loss"], "r-", lw=2, label="Test")
    ax[0, 0].set_title("Loss", fontweight="bold")
    ax[0, 0].legend()
    ax[0, 0].grid(alpha=0.3)
    ax[0, 0].set_xlabel("Epoch")

    ax[0, 1].plot(hist["tr_acc"], "b-", lw=2, label="Train")
    ax[0, 1].plot(hist["te_acc"], "r-", lw=2, label="Test")
    ax[0, 1].set_title("Accuracy (%)", fontweight="bold")
    ax[0, 1].legend()
    ax[0, 1].grid(alpha=0.3)
    ax[0, 1].set_xlabel("Epoch")

    ax[1, 0].plot(hist["lr"], "g-", lw=2)
    ax[1, 0].set_title("Learning Rate", fontweight="bold")
    ax[1, 0].grid(alpha=0.3)
    ax[1, 0].set_xlabel("Epoch")
    ax[1, 0].set_yscale("log")

    ax[1, 1].axis("off")
    summary = f"""
    CNN with BatchNorm - {opt_name.upper()}
    
    Final Test Accuracy: {hist['te_acc'][-1]:.2f}%
    Best Test Accuracy:  {best_acc:.2f}%
    
    Final Train Accuracy: {hist['tr_acc'][-1]:.2f}%
    Train/Test Gap: {hist['tr_acc'][-1] - hist['te_acc'][-1]:.2f}%
    
    Total Epochs: {len(hist['tr_loss'])}
    """
    ax[1, 1].text(
        0.1, 0.5, summary, fontsize=12, family="monospace", verticalalignment="center"
    )

    fig.suptitle(f"Baseline - {opt_name.upper()}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"viz/cnn_final_{opt_name}.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    hist, best_acc = train(opt_name="sgd", base_lr=0.1, epochs=200)
    plot_results(hist, "sgd", best_acc)
