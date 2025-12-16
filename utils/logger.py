from torch.utils.tensorboard import SummaryWriter
import os


class Logger:
    def __init__(self, log_dir, experiment_name):
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag, value_dict, step):
        self.writer.add_scalars(tag, value_dict, step)

    def close(self):
        self.writer.close()
