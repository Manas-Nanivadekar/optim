# train_cifar.py
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


def get_data(bs=256):
    t = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)]
    )
    train = datasets.CIFAR10("./data", train=True, download=True, transform=t)
    test = datasets.CIFAR10("./data", train=False, download=True, transform=t)
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


def train(opt_name, lr=0.001, epochs=20, device="cuda", seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = CNN().to(device)
    opt = CustomOptimizer(model.parameters(), opt_type=opt_name, lr=lr)
    crit = nn.CrossEntropyLoss()
    tr_loader, te_loader = get_data()

    hist = {
        "tr_loss": [],
        "tr_acc": [],
        "te_loss": [],
        "te_acc": [],
        "gnorm": [],
        "time": [],
    }
    print(f"\n{opt_name.upper()} (lr={lr})")

    for ep in range(epochs):
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

        print(
            f"Ep {ep+1:2d} | Tr:{tr_loss:.3f}({tr_acc:.1f}%) | Te:{te_loss:.3f}({te_acc:.1f}%) | GN:{gn:.2f} | {et:.1f}s"
        )

    return hist


def plot(results, lr, epochs):
    os.makedirs("viz", exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    cols = {"sgd": "r", "momentum": "b", "rmsprop": "g", "adam": "m"}

    for opt, h in results.items():
        ax[0, 0].plot(h["te_loss"], label=opt.upper(), c=cols[opt], lw=2)
        ax[0, 1].plot(h["te_acc"], label=opt.upper(), c=cols[opt], lw=2)
        ax[0, 2].plot(h["gnorm"], label=opt.upper(), c=cols[opt], lw=2)
        ax[1, 0].plot(h["time"], label=opt.upper(), c=cols[opt], lw=2)

    # Convergence speed
    conv = {
        opt: next((i + 1 for i, a in enumerate(h["te_acc"]) if a >= 50), epochs)
        for opt, h in results.items()
    }
    ax[1, 1].bar(conv.keys(), conv.values(), color=[cols[k] for k in conv])

    # Final accuracy
    final = {opt: h["te_acc"][-1] for opt, h in results.items()}
    ax[1, 2].bar(final.keys(), final.values(), color=[cols[k] for k in final])

    titles = [
        "Test Loss",
        "Test Acc (%)",
        "Grad Norm",
        "Epoch Time (s)",
        "Epochs to 50%",
        f"Final Acc (ep{epochs})",
    ]
    for a, t in zip(ax.flat, titles):
        a.set_title(t, fontweight="bold")
        if "bar" not in str(type(a.containers)):
            a.legend()
            a.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"viz/cifar_lr{lr}_ep{epochs}.png", dpi=150)
    plt.close()


def compare(opts=["sgd", "momentum", "rmsprop", "adam"], lr=0.001, epochs=20):
    device = "cuda"
    print(f"\nCIFAR-10 | {torch.cuda.get_device_name(0)} | lr={lr} | epochs={epochs}")

    results = {opt: train(opt, lr, epochs, device) for opt in opts}
    plot(results, lr, epochs)

    os.makedirs("results", exist_ok=True)
    with open(f"results/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}\nFINAL RESULTS\n{'='*60}")
    for opt in opts:
        acc = results[opt]["te_acc"][-1]
        t = sum(results[opt]["time"])
        print(f"{opt.upper():10s} {acc:6.2f}%  {t:6.1f}s")

    return results


if __name__ == "__main__":
    compare(lr=0.001, epochs=20)
