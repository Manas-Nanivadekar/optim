# train.py
import torch
import torch.nn as nn
import os
import time
from config import Config
from models import get_model
from data import get_cifar10_loaders
from optimizers.wrapper import CustomOptimizer
from utils import save_checkpoint, load_checkpoint, Logger, AverageMeter, accuracy


class Trainer:

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)

        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)

        self.model = get_model(config.model, config.num_classes).to(self.device)

        self.train_loader, self.test_loader = get_cifar10_loaders(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            augment=config.use_augmentation,
        )

        self.optimizer = CustomOptimizer(
            self.model.parameters(),
            opt_type=config.optimizer,
            lr=config.base_lr,
            weight_decay=config.weight_decay,
        )

        self.criterion = nn.CrossEntropyLoss()

        self.logger = Logger(config.log_dir, config.experiment_name)

        self.current_epoch = 0
        self.best_acc = 0.0
        self.train_history = {"loss": [], "acc": []}
        self.test_history = {"loss": [], "acc": []}

    def adjust_learning_rate(self, epoch):
        if epoch < self.config.warmup_epochs:
            lr = self.config.base_lr * (epoch + 1) / self.config.warmup_epochs
        else:
            if self.config.lr_schedule == "step":
                lr = self.config.base_lr
                for milestone in self.config.lr_milestones:
                    if epoch >= milestone:
                        lr *= self.config.lr_gamma
            elif self.config.lr_schedule == "cosine":
                import math

                lr = (
                    self.config.base_lr
                    * 0.5
                    * (1 + math.cos(math.pi * epoch / self.config.epochs))
                )

            else:
                lr = self.config.base_lr

        self.optimizer.lr = lr
        return lr

    def train_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()
        top1 = AverageMeter()

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            loss.backward()

            if self.config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip_norm
                )

            self.optimizer.step()

            acc = accuracy(outputs, targets)[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc, inputs.size(0))

        return losses.avg, top1.avg

    def validate(self):
        self.model.eval()
        losses = AverageMeter()
        top1 = AverageMeter()

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                acc = accuracy(outputs, targets)[0]
                losses.update(loss.item(), inputs.size(0))
                top1.update(acc, inputs.size(0))

        return losses.avg, top1.avg

    def train(self):
        print(
            f"Training {self.config.model.upper()} with {self.config.optimizer.upper()}"
        )
        print(f"Experiment: {self.config.experiment_name}")

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch

            current_lr = self.adjust_learning_rate(epoch)

            start_time = time.time()
            train_loss, train_acc = self.train_epoch(epoch)

            test_loss, test_acc = self.validate()
            epoch_time = time.time() - start_time

            self.train_history["loss"].append(train_loss)
            self.train_history["acc"].append(train_acc)
            self.test_history["loss"].append(test_loss)
            self.test_history["acc"].append(test_acc)

            self.logger.log_scalar("train/loss", train_loss, epoch)
            self.logger.log_scalar("train/acc", train_acc, epoch)
            self.logger.log_scalar("test/loss", test_loss, epoch)
            self.logger.log_scalar("test/acc", test_acc, epoch)
            self.logger.log_scalar("lr", current_lr, epoch)

            is_best = test_acc > self.best_acc
            if is_best:
                self.best_acc = test_acc

            if (epoch + 1) % self.config.save_freq == 0 or is_best:
                state = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state,
                    "best_acc": self.best_acc,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "config": self.config,
                }
                save_checkpoint(
                    state,
                    is_best,
                    self.config.checkpoint_dir,
                    f"{self.config.experiment_name}_epoch{epoch+1}.pt",
                )

            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                    f"LR: {current_lr:.4f} | "
                    f"Train: {train_loss:.3f} ({train_acc:.1f}%) | "
                    f"Test: {test_loss:.3f} ({test_acc:.1f}%) | "
                    f"Best: {self.best_acc:.2f}% | "
                    f"Time: {epoch_time:.1f}s"
                )

        print(f"Training Complete")
        print(f"Best Test Accuracy: {self.best_acc:.2f}%")

        self.logger.close()
        return self.best_acc
