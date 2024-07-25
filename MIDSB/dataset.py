from glob import glob
import os

import numpy as np
import torch
from torch import nn
import torchaudio
from utils.config import read_config_from_yaml
from torch.utils.data import Dataset
import torch.nn.functional as F

def get_window(window_type, window_length):
    if window_type == 'sqrthann':
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    elif window_type == 'hann':
        return torch.hann_window(window_length, periodic=True)
    else:
        raise NotImplementedError(f"Window type {window_type} not implemented!")

class STFTUtil():
    n_fft = None
    num_frames = None
    hop_length = None
    spec_abs_exponent = None
    spec_factor = None
    window = None
    windows = None

    @classmethod
    def initial(cls, config=None):
        if config is None:
            config = read_config_from_yaml(config_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../config/data_representation.yml"))

        cls.n_fft = config.data.get('n_fft')
        cls.num_frames = config.data.get('num_frames')
        cls.hop_length = config.data.get('hop_length')
        cls.spec_abs_exponent = config.data.get('spec_abs_exponent')
        cls.spec_factor = config.data.get('spec_factor')
        cls.window = get_window(config.data.get('window'), cls.n_fft)
        cls.windows = {}

    @classmethod
    def _get_window(cls, x):
        window = cls.windows.get(x.device, None)
        if window is None:
            window = cls.window.to(x.device)
            cls.windows[x.device] = window
        return window

    @classmethod
    def stft(cls, x, transform=True, config=None):
        if cls.num_frames is None:
            cls.initial(config=config)

        window = cls._get_window(x)
        X = torch.stft(x, n_fft=cls.n_fft, hop_length=cls.hop_length, window=window, center=True, return_complex=True)
        if transform:
            X = cls.magnitude_warping(X)
        return X

    @classmethod
    def istft(cls, X, transform=True, length=None, config=None):
        if cls.num_frames is None:
            cls.initial(config=config)
        window = cls._get_window(X)
        if transform:
            X = cls.invert_magnitude_warping(X)
        x = torch.istft(X, n_fft=cls.n_fft, hop_length=cls.hop_length, window=window, center=True, length=length)
        return x

    @classmethod
    def magnitude_warping(cls, spec):
        if cls.spec_abs_exponent != 1:
            e = cls.spec_abs_exponent
            spec = spec.abs() ** e * torch.exp(1j * spec.angle())
        return spec * cls.spec_factor

    @classmethod
    def invert_magnitude_warping(cls, spec):
        spec = spec / cls.spec_factor
        if cls.spec_abs_exponent != 1:
            e = cls.spec_abs_exponent
            spec = spec.abs() ** (1 / e) * torch.exp(1j * spec.angle())
        return spec

    @classmethod
    def to_stft(cls, audio, device='cpu'):
        audio_length = audio.size(-1)
        audio = audio.view(1, -1)

        normfac = audio.abs().max().item()
        audio = audio.to(device)

        normlized_audio = audio / normfac
        x = STFTUtil.stft(normlized_audio)
        x = pad_spec(x.unsqueeze(0))

        def invert(x_):
            x_ = STFTUtil.istft(x_.squeeze(), length=audio_length)
            x_ = x_ * normfac
            x_ = x_.squeeze().cpu()
            return x_

        return x, invert

def pad_spec(Y):
    T = Y.size(3)
    if T%64 !=0:
        num_pad = 64-T%64
    else:
        num_pad = 0
    pad2d = torch.nn.ZeroPad2d((0, num_pad, 0,0))
    return pad2d(Y)


class AudioFolder(Dataset):
    def __init__(self, audio_path_or_pt_file, sample_rate=16000, return_path=False, reverse=False):
        assert os.path.exists(audio_path_or_pt_file), audio_path_or_pt_file

        self.audio_files = None
        self.audio_paths = None

        self.reverse = reverse

        if os.path.isdir(audio_path_or_pt_file):
            self.audio_paths = sorted(glob(audio_path_or_pt_file + '/*.wav'))
            self.sample_rate = sample_rate
        elif audio_path_or_pt_file.endswith('.pt'):
            self.audio_files = torch.load(audio_path_or_pt_file)
            self.sample_rate = self.audio_files['sample_rate']
            if self.sample_rate != sample_rate:
                raise ValueError(f"The sample rate of audio is not equal to the sample rate expected.")
        else:
            raise NotImplementedError

        self.return_path = return_path

    def __len__(self):
        if self.audio_files is None:
            return len(self.audio_paths)
        return len(self.audio_files)

    def __getitem__(self, index):
        if self.reverse:
            index = len(self) - index - 1
        if self.audio_files is None:
            audio_file = self.audio_paths[index]
            audio, sr = torchaudio.load(audio_file)
            if sr != self.sample_rate:
                audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(audio)

            if self.return_path:
                return audio, os.path.basename(audio_file)[:-4]
            else:
                return audio
        else:
            if self.return_path:
                return self.audio_files['audio'][index], self.audio_files['path'][index]
            else:
                return self.audio_files['audio'][index]

    def to_file(self):
        if self.audio_paths is not None:
            data = {'audio':[],'path':[],'sample_rate':self.sample_rate}
            for index in range(len(self.audio_paths)):
                audio_file = self.audio_paths[index]
                audio, sr = torchaudio.load(audio_file)
                assert sr == self.sample_rate, "The sample rate of audio is not equal to the sample rate expected."

                data['audio'].append(audio)
                data['path'].append(os.path.basename(audio_file)[:-4])
        elif self.audio_files is not None:
            data = self.audio_files
        return data


    def get_data_loader(self, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, reverse = None):
        if reverse is not None and isinstance(reverse, bool):
            self.reverse = reverse
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)


class ComplexSpec(Dataset):
    def __init__(self, config, dataset='voicebank', subset='train', shuffle_spec=None, normalize_audio=None, return_raw = False, return_spec=True, dummy=False):
        self.sample_rate = config.data.get('sample_rate', 16000)
        self.audio_length = config.data.get('max_audio_length', 32000)
        assert subset in ['train', 'valid', 'test']

        assert dataset in ['voicebank','timit+wham'], f'Dataset {dataset} is not supported yet.'
        self.data_dir = config.datasets[dataset]
        self.subset = subset
        self.spatial_channels = config.data.get('spatial_channels', 1)
        self.num_frames = config.data.get('num_frames', 256)
        self.hop_length = config.data.get('hop_length', 128)

        self.clean_files = AudioFolder(audio_path_or_pt_file=os.path.join(self.data_dir, subset, 'clean'), sample_rate=self.sample_rate)
        self.noisy_files = AudioFolder(audio_path_or_pt_file=os.path.join(self.data_dir, subset, 'noisy'), sample_rate=self.sample_rate)

        self.shuffle_spec = shuffle_spec
        self.normalize_audio =  config.data.get('normalize_audio', True) if normalize_audio is None else normalize_audio
        self.return_spec = return_spec
        self.return_raw = return_raw
        self.dummy = dummy

    def stft(self, x):
        return STFTUtil.stft(x)

    def istft(self, x, length=None):
        return STFTUtil.istft(x, length)

    def __getitem__(self, i):
        x = self.clean_files[i]
        y = self.noisy_files[i]

        min_len = min(x.size(-1), y.size(-1))
        x, y = x[..., : min_len], y[..., : min_len]

        if x.ndimension() == 2 and self.spatial_channels == 1:
            x, y = x[0].unsqueeze(0), y[0].unsqueeze(0)  # Select first channel

        # Select channels
        assert self.spatial_channels <= x.size(0), f"You asked too many channels ({self.spatial_channels}) for the given dataset ({x.size(0)})"
        x, y = x[: self.spatial_channels], y[: self.spatial_channels]
        if self.return_raw:
            return x, y

        normfac = y.abs().max()
        target_len = self.audio_length if not self.return_spec else (self.num_frames - 1) * self.hop_length
        current_len = x.size(-1)
        pad = max(target_len - current_len, 0)
        if pad == 0:
            # extract random part of the audio file
            if self.shuffle_spec:
                start = int(np.random.uniform(0, current_len - target_len))
            else:
                start = int((current_len - target_len) / 2)
            x = x[..., start:start + target_len]
            y = y[..., start:start + target_len]
        else:
            # pad audio if the length T is smaller than num_frames
            x = F.pad(x, (pad // 2, pad // 2 + (pad % 2)), mode='constant')
            y = F.pad(y, (pad // 2, pad // 2 + (pad % 2)), mode='constant')

        if self.normalize_audio:
            # normalize both based on noisy speech, to ensure same clean signal power in x and y.
            x = x / normfac
            y = y / normfac

        if self.return_spec:
            X, Y = self.stft(x), self.stft(y)
            return  X, Y

        return x, y

    def __len__(self):
        if self.dummy:
            return 16
        return len(self.noisy_files)

