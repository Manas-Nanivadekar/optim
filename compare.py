import torch
import subprocess
import time
import json
import matplotlib.pyplot as plt
import os
from pathlib import Path
from config import Config, CONFIGS


class ExperimentRunner:
    def __init__(self, experiments, available_gpus=[0, 1, 2]):
        self.experiments = experiments
        self.available_gpus = available_gpus
        self.results = {}

    def run_experiment(self, exp_name, config, gpu_id):
        print(f"\n{'='*70}")
        print(f"Starting {exp_name} on GPU {gpu_id}")
        print(f"{'='*70}\n")

        from train import Trainer

        config.device = f"cuda:{gpu_id}"
        config.experiment_name = exp_name

        trainer = Trainer(config)
        best_acc = trainer.train()

        self.results[exp_name] = {
            "best_acc": best_acc,
            "config": config,
            "train_history": trainer.train_history,
            "test_history": trainer.test_history,
        }

        return best_acc

    def run_sequential(self):
        for exp_name, config in self.experiments:
            gpu_id = self.available_gpus[0]
            self.run_experiment(exp_name, config, gpu_id)

    def run_parallel(self):
        import multiprocessing as mp

        processes = []
        gpu_idx = 0

        for exp_name, config in self.experiments:
            gpu_id = self.available_gpus[gpu_idx % len(self.available_gpus)]

            p = mp.Process(target=self.run_experiment, args=(exp_name, config, gpu_id))
            p.start()
            processes.append(p)

            gpu_idx += 1
            time.sleep(2)

        for p in processes:
            p.join()

    def load_results(self, experiment_names):
        for exp_name in experiment_names:
            result_file = f"results/{exp_name}.json"
            if os.path.exists(result_file):
                with open(result_file, "r") as f:
                    data = json.load(f)
                    self.results[exp_name] = data

    def compare_results(self):
        if not self.results:
            print("No results to compare!")
            return

        os.makedirs("viz", exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        colors = ["r", "b", "g", "m", "c", "y"]

        for idx, (exp_name, data) in enumerate(self.results.items()):
            color = colors[idx % len(colors)]

            if "train_history" in data and "test_history" in data:
                train_hist = data["train_history"]
                test_hist = data["test_history"]

                axes[0, 0].plot(train_hist["loss"], c=color, label=exp_name, lw=2)

                axes[0, 1].plot(train_hist["acc"], c=color, label=exp_name, lw=2)

                axes[1, 0].plot(test_hist["loss"], c=color, label=exp_name, lw=2)

                axes[1, 1].plot(test_hist["acc"], c=color, label=exp_name, lw=2)

        exp_names = list(self.results.keys())
        best_accs = [self.results[name]["best_acc"] for name in exp_names]
        axes[0, 2].bar(range(len(exp_names)), best_accs, color=colors[: len(exp_names)])
        axes[0, 2].set_xticks(range(len(exp_names)))
        axes[0, 2].set_xticklabels(exp_names, rotation=45, ha="right")
        axes[0, 2].set_ylabel("Best Accuracy (%)")
        axes[0, 2].set_title("Best Test Accuracy", fontweight="bold")
        axes[0, 2].grid(alpha=0.3, axis="y")

        axes[1, 2].axis("off")
        summary_text = "Final Results:\n" + "=" * 40 + "\n"
        for name, data in self.results.items():
            summary_text += f"{name:20s}: {data['best_acc']:.2f}%\n"
        axes[1, 2].text(
            0.1,
            0.5,
            summary_text,
            fontsize=10,
            family="monospace",
            verticalalignment="center",
        )

        titles = [
            "Train Loss",
            "Train Accuracy (%)",
            "Best Accuracy",
            "Test Loss",
            "Test Accuracy (%)",
            "Summary",
        ]
        for ax, title in zip(axes.flat, titles):
            if title != "Summary" and title != "Best Accuracy":
                ax.set_title(title, fontweight="bold")
                ax.legend()
                ax.grid(alpha=0.3)
                ax.set_xlabel("Epoch")

        plt.tight_layout()
        plt.savefig("viz/comparison.png", dpi=150, bbox_inches="tight")
        plt.close()

        print(f"\n{'='*70}")
        print("Comparison plot saved to viz/comparison.png")
        print(f"{'='*70}\n")


def main():
    experiments = [
        ("cnn_baseline", CONFIGS["cnn_baseline"]),
        ("resnet20_sgd", CONFIGS["resnet20"]),
        ("resnet32_sgd", CONFIGS["resnet32"]),
        ("resnet56_sgd", CONFIGS["resnet56"]),
    ]

    runner = ExperimentRunner(experiments, available_gpus=[0, 1])

    print("Running experiments in parallel")
    runner.run_parallel()

    print("\nGenerating comparison plots...")
    runner.compare_results()


if __name__ == "__main__":
    main()
