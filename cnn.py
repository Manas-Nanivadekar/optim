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


def get_data(bs=256, augment=True):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    if augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        train_transform = transforms.Compose([transforms.ToTensor(), normalize])

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


def grad_norm(model):
    return (
        sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None)
        ** 0.5
    )


def train_epoch(model, loader, opt, crit, device):
    model.train()
    loss_sum, correct, total, gnorms = 0, 0, 0, []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        gnorms.append(grad_norm(model))
        opt.step()
        loss_sum += loss.item()
        correct += out.argmax(1).eq(y).sum().item()
        total += y.size(0)
    return loss_sum / len(loader), 100 * correct / total, sum(gnorms) / len(gnorms)


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


def train(opt_name, base_lr=0.1, epochs=200, device="cuda", seed=42, use_schedule=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = CNN().to(device)
    opt = CustomOptimizer(model.parameters(), opt_type=opt_name, lr=base_lr)
    crit = nn.CrossEntropyLoss()
    tr_loader, te_loader = get_data(augment=True)

    hist = {
        "tr_loss": [],
        "tr_acc": [],
        "te_loss": [],
        "te_acc": [],
        "gnorm": [],
        "time": [],
        "lr": [],
    }

    schedule_label = "w/ schedule" if use_schedule else "fixed LR"
    print(f"\n{opt_name.upper()} (base_lr={base_lr}, {schedule_label})")

    best_acc = 0
    for ep in range(epochs):
        if use_schedule:
            current_lr = adjust_lr(opt, ep, base_lr=base_lr)
        else:
            current_lr = base_lr

        t0 = time.time()
        tr_loss, tr_acc, gn = train_epoch(model, tr_loader, opt, crit, device)
        te_loss, te_acc = test_epoch(model, te_loader, crit, device)
        et = time.time() - t0

        hist["tr_loss"].append(tr_loss)
        hist["tr_acc"].append(tr_acc)
        hist["te_loss"].append(te_loss)
        hist["te_acc"].append(te_acc)
        hist["gnorm"].append(gn)
        hist["time"].append(et)
        hist["lr"].append(current_lr)

        if te_acc > best_acc:
            best_acc = te_acc

        if (ep + 1) % 20 == 0 or ep == 0:
            print(
                f"Ep {ep+1:3d} | Tr:{tr_loss:.3f}({tr_acc:.1f}%) | Te:{te_loss:.3f}({te_acc:.1f}%) | LR:{current_lr:.4f} | Best:{best_acc:.1f}%"
            )

    print(f"Final: Test {te_acc:.2f}% | Best: {best_acc:.2f}%")
    return hist


def plot(results, title, filename):
    os.makedirs("viz", exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))

    for label, h in results.items():
        ax[0, 0].plot(h["tr_loss"], label=label, lw=2)
        ax[0, 1].plot(h["tr_acc"], label=label, lw=2)
        ax[0, 2].plot(h["te_acc"], label=label, lw=2)
        ax[1, 0].plot(h["te_loss"], label=label, lw=2)
        ax[1, 1].plot(h["gnorm"], label=label, lw=2)
        ax[1, 2].plot(h["lr"], label=label, lw=2)

    titles = [
        "Train Loss",
        "Train Acc (%)",
        "Test Acc (%)",
        "Test Loss",
        "Grad Norm",
        "Learning Rate",
    ]
    for a, t in zip(ax.flat, titles):
        a.set_title(t, fontweight="bold")
        a.legend()
        a.grid(alpha=0.3)
        a.set_xlabel("Epoch")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"viz/{filename}.png", dpi=150)
    plt.close()


def compare_schedule(opt="sgd", base_lr=0.1, epochs=200):
    device = "cuda"
    print(f"\nCIFAR-10 LR Scheduling | {torch.cuda.get_device_name(0)}")

    results = {
        "Fixed LR": train(opt, base_lr, epochs, device, use_schedule=False),
        "Scheduled LR": train(opt, base_lr, epochs, device, use_schedule=True),
    }

    plot(
        results,
        f"{opt.upper()} - LR Scheduling Comparison",
        f"schedule_comparison_{opt}",
    )

    print(f"Final Result (Epoch {epochs})")
    for label, h in results.items():
        print(
            f"{label:15s} Test Acc: {h['te_acc'][-1]:6.2f}%  (Best: {max(h['te_acc']):.2f}%)"
        )

    return results


if __name__ == "__main__":
    compare_schedule(opt="sgd", base_lr=0.1, epochs=200)
