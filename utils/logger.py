import logging
from torch.utils.tensorboard import SummaryWriter

def get_logger(log_file="train.log"):
    logger = logging.getLogger("Trainer")
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

class TensorboardLogger:
    def __init__(self, log_dir="runs"):
        self.writer = SummaryWriter(log_dir)

    def log_metrics(self, metrics, step, prefix="train"):
        for key, value in metrics.items():
            self.writer.add_scalar(f"{prefix}/{key}", value, step)

    def close(self):
        self.writer.close()
