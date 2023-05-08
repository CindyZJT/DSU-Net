import importlib

import numpy as np
import torch
from scipy.ndimage import rotate, map_coordinates, gaussian_filter
from scipy.ndimage.filters import convolve
from skimage.filters import gaussian
from skimage.segmentation import find_boundaries
from torchvision.transforms import Compose
import scipy.ndimage




class Normalize:
    """
    Normalizes a given input tensor to be 0-mean and 1-std.
    mean and std parameter have to be provided explicitly.
    """

    def __init__(self, mean, std, eps=1e-4, **kwargs):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, m):
        return (m - self.mean) / (self.std + self.eps)


class RangeNormalize:
    def __init__(self, max_value=255, **kwargs):
        self.max_value = max_value

    def __call__(self, m):
        return m / self.max_value


class GaussianNoise:
    def __init__(self, random_state, max_sigma, max_value=255, **kwargs):
        self.random_state = random_state
        self.max_sigma = max_sigma
        self.max_value = max_value

    def __call__(self, m):
        # pick std dev from [0; max_sigma]
        std = self.random_state.randint(self.max_sigma)
        gaussian_noise = self.random_state.normal(0, std, m.shape)
        noisy_m = m + gaussian_noise
        return np.clip(noisy_m, 0, self.max_value).astype(m.dtype)


class ToTensor:
    """
    Converts a given input numpy.ndarray into torch.Tensor. Adds additional 'channel' axis when the input is 3D
    and expand_dims=True (use for raw data of the shape (D, H, W)).
    """

    def __init__(self, expand_dims, dtype=np.float32, **kwargs):
        self.expand_dims = expand_dims
        # self.resize = resize
        self.dtype = dtype

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'
        # add channel dimension
        # if self.expand_dims and m.ndim == 3:
        #     m = np.expand_dims(m, axis=0)
        # res = []

        # if self.resize:
        #     m = np.asarray(m)
        #     m = m[:, 170:370, 130:390]
        #     size = 96/m.shape[0]
        #     res = scipy.ndimage.zoom(m, size, order=3)
        # else:
        #     res = m

        if self.expand_dims and m.ndim == 3:
            m = np.expand_dims(m, axis=0)
        return torch.from_numpy(m.astype(dtype=self.dtype))


class Identity:
    def __call__(self, m):
        return m


def get_transformer(config, mean, std, phase):
    if phase == 'val':
        phase = 'test'

    assert phase in config, f'Cannot find transformer config for phase: {phase}'
    phase_config = config[phase]
    return Transformer(phase_config, mean, std)


class Transformer:
    def __init__(self, phase_config, mean, std):
        self.phase_config = phase_config
        self.config_base = {'mean': mean, 'std': std}
        self.seed = 47

    def raw_transform(self):
        return self._create_transform('raw')

    def label_transform(self):
        return self._create_transform('label')

    def weight_transform(self):
        return self._create_transform('weight')

    @staticmethod
    def _transformer_class(class_name):
        m = importlib.import_module('augment.transforms')
        clazz = getattr(m, class_name)
        return clazz

    def _create_transform(self, name):
        assert name in self.phase_config, f'Could not find {name} transform'
        return Compose([
            self._create_augmentation(c) for c in self.phase_config[name]
        ])

    def _create_augmentation(self, c):
        config = dict(self.config_base)
        config.update(c)
        config['random_state'] = np.random.RandomState(self.seed)
        aug_class = self._transformer_class(config['name'])
        return aug_class(**config)


def _recover_ignore_index(input, orig, ignore_index):
    if ignore_index is not None:
        mask = orig == ignore_index
        input[mask] = ignore_index

    return input
