# config.py
from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    # Model
    model: str = "resnet20"
    num_classes: int = 10

    # Optimizer
    optimizer: str = "sgd"
    base_lr: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9

    # Training
    epochs: int = 160
    batch_size: int = 128
    warmup_epochs: int = 5

    # LR Schedule
    lr_schedule: str = "step"  # step, cosine
    lr_milestones: List[int] = None  # [80, 120] for step schedule
    lr_gamma: float = 0.1

    # Regularization
    grad_clip_norm: float = 1.0

    # Data augmentation
    use_augmentation: bool = True

    # System
    device: str = "cuda"
    seed: int = 42
    num_workers: int = 4

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    results_dir: str = "results"

    # Experiment tracking
    experiment_name: str = None
    save_freq: int = 20  # Save checkpoint every N epochs

    def __post_init__(self):
        # Set default milestones if not provided
        if self.lr_milestones is None:
            self.lr_milestones = [80, 120]

        # Auto-generate experiment name
        if self.experiment_name is None:
            self.experiment_name = f"{self.model}_{self.optimizer}_lr{self.base_lr}"


# Predefined configurations for different experiments
CONFIGS = {
    "cnn_baseline": Config(
        model="cnn", base_lr=0.1, epochs=200, lr_milestones=[60, 120, 160]
    ),
    "resnet20": Config(
        model="resnet20", base_lr=0.1, epochs=160, lr_milestones=[80, 120]
    ),
    "resnet32": Config(
        model="resnet32", base_lr=0.1, epochs=160, lr_milestones=[80, 120]
    ),
    "resnet20_adam": Config(
        model="resnet20",
        optimizer="adam",
        base_lr=0.001,
        epochs=160,
        lr_milestones=[80, 120],
    ),
}


def get_config(name: str = None) -> Config:
    """Get config by name or return default"""
    if name and name in CONFIGS:
        return CONFIGS[name]
    return Config()
