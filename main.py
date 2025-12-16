from .train import Trainer


def main():
    from config import get_config

    config = get_config("resnet20")

    trainer = Trainer(config)
    best_acc = trainer.train()

    print(f"Final result: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
