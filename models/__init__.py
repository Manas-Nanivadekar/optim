from .cnn import CNN
from .resnet import ResNet20, ResNet32, ResNet56

__all__ = ["CNN", "ResNet20", "ResNet32", "ResNet56"]


def get_model(name, num_classes=10):
    models = {
        "cnn": CNN,
        "resnet20": ResNet20,
        "resnet32": ResNet32,
        "resnet56": ResNet56,
    }

    if name not in models:
        raise ValueError(f"Unknown model: {name}. Choose from {list(models.keys())}")

    return models[name](num_classes=num_classes)
