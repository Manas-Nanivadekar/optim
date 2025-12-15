import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from optimiser_wrapper import CustomOptimizer


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def get_data(batch_size=128):
    tranform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_data = datasets.CIFAR10(
        "./data", train=True, download=True, transform=tranform
    )
    test_data = datasets.CIFAR10(
        "./data", train=False, download=True, transform=tranform
    )

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backwards()
        optimizer.step()

        total_loss += loss.item()
        _, pred = outputs.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), 100.0 * correct / total


def test_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), 100.0 * correct / total


def train_optimizer(opt_name, lr=0.001, epochs=10, device="cuda", seed=42):
    print(f"Training {opt_name.upper()} (lr={lr})")

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = CustomOptimizer(model.parameters(), opt_type=opt_name, lr=lr)

    train_loader, test_loader = get_data()

    hist = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for ep in range(epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = test_epoch(model, test_loader, criterion, device)

        hist["train_loss"].append(tr_loss)
        hist["train_acc"].append(tr_acc)
        hist["test_loss"].append(te_loss)
        hist["test_acc"].append(te_acc)

        print(
            f"Epoch {ep+1:2d}/{epochs} | Train: {tr_loss:.3f} ({tr_acc:.1f}%) | Test: {te_loss:.3f} ({te_acc:.1f}%)"
        )

    return hist


def compare(opts=["sgd", "momentum", "rmsprop", "adam"], lr=0.001, epochs=10):
    device = "cuda"

    results = {}
    for opt in opts:
        results[opt] = train_optimizer(opt, lr, epochs, device)

    os.makedirs("viz", exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    colors = {"sgd": "r", "momentum": "b", "rmsprop": "g", "adam": "m"}
    for opt, h in results.items():
        ax[0, 0].plot(h["train_loss"], label=opt.upper(), c=colors[opt], lw=2)
        ax[0, 1].plot(h["train_acc"], label=opt.upper(), c=colors[opt], lw=2)
        ax[1, 0].plot(h["test_loss"], label=opt.upper(), c=colors[opt], lw=2)
        ax[1, 1].plot(h["test_acc"], label=opt.upper(), c=colors[opt], lw=2)

    titles = ["Train Loss", "Train Accuracy (%)", "Test Loss", "Test Accuracy (%)"]
    for a, t in zip(ax.flat, titles):
        a.set_title(t, fontweight="bold")
        a.legend()
        a.grid(alpha=0.3)
        a.set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(f"viz/cifar_lr{lr}_ep{epochs}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: viz/cifar_lr{lr}_ep{epochs}.png")

    print(f"\n{'='*60}\nFINAL RESULTS (Epoch {epochs})\n{'='*60}")
    for opt in opts:
        te_acc = results[opt]["test_acc"][-1]
        print(f"{opt.upper():10s} Test Acc: {te_acc:.2f}%")

    return results


if __name__ == "__main__":
    compare(lr=0.001, epochs=10)
