"""
evaluation/utils.py — Logging, seeding, AverageMeter.
"""

import os
import random
import logging
import numpy as np
import torch


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def get_logger(name: str, log_dir: str, exp_name: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name + "_" + exp_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
        ch  = logging.StreamHandler()
        ch.setFormatter(fmt)
        fh  = logging.FileHandler(os.path.join(log_dir, f"{exp_name}.log"))
        fh.setFormatter(fmt)
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger


class AverageMeter:
    def __init__(self, name=""):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = self.count = 0.0

    def update(self, val, n=1):
        self.sum   += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)
