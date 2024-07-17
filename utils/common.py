import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    return seed

def scaler_format(x, precision=4):
    return f'{x:.{precision}f}'