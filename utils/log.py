import numpy as np
from tqdm import tqdm
import random
import torch

progress_items = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']

def log(*args) -> None:
    print(*args)

class Logger:
    head_color = 94
    info_prefix = f'\033[{head_color}m[MIDSB]\033[0m'
    warning_prefix = "[MIDSB] Warning:"
    error_prefix = "[MIDSB] Error:"

    def __init__(self):
        pass

    @classmethod
    def log(cls, *msgs: str) -> None:
        """Logs a message. Accepts multiple arguments."""
        assert msgs, "No messages provided"
        message = ' '.join(map(str, msgs))
        print(message)

    @classmethod
    def info(cls, *msgs: str) -> None:
        """Logs an info message. Accepts multiple arguments."""
        assert msgs, "No messages provided"
        message = ' '.join(map(str, msgs))
        print(f"{cls.info_prefix} {message}")

    @classmethod
    def warning(cls, *msgs: str) -> None:
        """Logs a warning message. Accepts multiple arguments."""
        assert msgs, "No messages provided"
        message = ' '.join(map(str, msgs))
        print(f"{cls.warning_prefix} {message}")

    @classmethod
    def error(cls, *msgs: str) -> None:
        """Logs an error message. Accepts multiple arguments."""
        assert msgs, "No messages provided"
        message = ' '.join(map(str, msgs))
        print(f"{cls.error_prefix} {message}")



def progress_visualize(data_list: list, total_length: int = None, display_length: int = 10, sample_evenly: bool = True):
    if len(data_list) == 0:
        index = [0] * display_length
        progress = ''.join([progress_items[idx] for idx in index])
        return progress
    if total_length is None:
        total_length = len(data_list)

    if sample_evenly:
        sample_index = np.linspace(0, total_length - 1, display_length)
        sample_index = sample_index.astype(int)
        sampled_data_list = [data_list[i] for i in sample_index if i < len(data_list)]
    else:
        if len(data_list) > display_length:
            sampled_data_list = data_list[-display_length:]
        else:
            sampled_data_list = data_list

    non_tensor_data = []
    for data in sampled_data_list:
        if torch.is_tensor(data):
            data = data.cpu().item()
        non_tensor_data.append(data)
    np_data = np.array(non_tensor_data)

    norm_data = (np_data - np_data.min()) / np_data.max() * 7
    index = [round(data) + 1 for data in norm_data]

    index += [0] * (display_length - len(index))
    progress = ''.join([progress_items[idx] for idx in index])

    return progress


class ProcessBar():
    def __init__(self, data_iter, total_length, display_length=50, ncols=200, sample_evenly=True):
        self.iter = tqdm(data_iter, leave=False, ncols=ncols,
                         unit='batch', colour='#507fff',
                         bar_format='{desc} {n_fmt}/{total_fmt} [{elapsed}<{remaining},{rate_fmt}{postfix}]')
        self.total_length, self.display_length, self.sample_evenly = total_length, display_length, sample_evenly

    def get_iter(self):
        return self.iter

    def update(self, epoch, history, postfix: dict = None):
        self.iter.set_description_str(f'Epoch {epoch} - Training |{progress_visualize(history, self.total_length, self.display_length, self.sample_evenly)}|', refresh=False)
        if postfix is not None:
            self.iter.set_postfix(postfix, refresh=False)
        self.iter.refresh()