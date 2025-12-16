import torch
import torch.nn as nn
import numpy as np
from cnn import CNN, get_data
from optimiser_wrapper import CustomOptimizer


def single_batch(lr=0.01, steps=100):
    device = "cpu"
    train_loader, _ = get_data(batch_size=16)
    imgs, labels = next(iter(train_loader))

    model = CNN()
    citerion = nn.CrossEntropyLoss()
    optimizer = CustomOptimizer(model.parameters(), opt_type="adam", lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = citerion(outputs, labels)
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            _, pred = outputs.max(1)
            acc = 100.0 * pred.eq(labels).sum().item() / labels.size(0)
            print(f"Step {step:3d} | Loss: {loss.item():.4f} | Acc: {acc:.1f}%")

    _, pred = outputs.max(1)
    final_acc = 100.0 * pred.eq(labels).sum().item() / labels.size(0)

    if final_acc < 90:
        print(f"\nFAIL: Only {final_acc:.1f}% accuracy")
        print("Check: architecture, optimizer, learning rate")
        return False
    else:
        print(f"\nPass: {final_acc:.1f}% accuracy")
        return True


def gradient_flow():
    train_loader, _ = get_data(batch_size=16)
    imgs, labels = next(iter(train_loader))

    model = CNN()
    criterion = nn.CrossEntropyLoss()

    outputs = model(imgs)
    loss = criterion(outputs, labels)
    loss.backward()

    ok = True
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            status = "Sucess" if grad_norm > 0 else "Fail"
            print(f"status: {status}")
            if grad_norm == 0:
                ok = False
        else:
            print(f"{name} | No gradient")
            ok = False

    return ok


def loss_curve_check(lr=0.01, batches=20):
    train_loader, _ = get_data(batch_size=32)
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = CustomOptimizer(model.parameters(), opt_type="adam", lr=lr)

    print(f"Loss Curve (lr={lr}, {batches} batches)")

    losses = []

    for i, (imgs, labels) in enumerate(train_loader):
        if i >= batches:
            break

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if i % 5 == 0:
            print(f"Batch {i:2d} | Loss: {loss.item():.4f}")

    losses = np.array(losses)

    if np.any(np.isnan(losses)):
        print("Fail: Loss became NaN")
        return False
    elif losses[-1] > losses[0]:
        print(f"Warning: Loss increased ({losses[0]:.3f} -> {losses[-1]:.3f})")
        return False
    else:
        improvement = (losses[0] - losses[-1]) / losses[0] * 100
        print(f"Pass: Loss decreased by : {improvement:.1f}")
        return True


def compare_optimizers():
    for opt in ["sgd", "momentum", "rmsprop", "adam"]:
        print(f"Testing: {opt.upper()}")
        train_loader, _ = get_data(batch_size=16)
        imgs, labels = next(iter(train_loader))

        model = CNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = CustomOptimizer(model.parameters(), opt_type=opt, lr=0.001)

        initial_loss = None

        for step in range(50):
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            if step == 0:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        improvement = (initial_loss - final_loss) / initial_loss * 100

        status = "Sucess" if improvement > 50 else "Fail"
        print(
            f"{status} {opt:10s}: {initial_loss:.3f} -> {final_loss:.3f} ({improvement:+.1f}%)"
        )


if __name__ == "__main__":
    print("Test 1/4: Checking gradient flow...")
    grad_ok = gradient_flow()

    # Test 2: Single batch overfit
    print("\nTest 2/4: Can model memorize data?")
    overfit_ok = single_batch(lr=0.01, steps=100)

    # Test 3: Loss behavior
    print("\nTest 3/4: Does loss decrease?")
    loss_ok = loss_curve_check(lr=0.001, batches=20)

    # Test 4: Optimizer comparison
    print("\nTest 4/4: Quick optimizer check")
    compare_optimizers()

    if grad_ok and overfit_ok and loss_ok:
        print("Checks Passed")
        print("Ready for full training run")
    else:
        print("Some checks failed")
        print("Fix issues before training")
