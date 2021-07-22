import os
import sys
import logging

import torch
import torch.distributed as dist

def get_device() :
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"your device is {device}")

    return device

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def setup_logger(name, save_dir, distributed_rank, filename="log.txt", mode='w') :
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if distributed_rank > 0 : return logger

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fh = logging.FileHandler(os.path.join(save_dir, filename), mode=mode)  # 'a+' for add, 'w' for overwrite
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
