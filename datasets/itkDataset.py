import collections
import importlib


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import augment.transforms as transforms
from unet3d.utils import get_logger
import os
import SimpleITK as sitk
import random


class SliceBuilder:
    def __init__(self, raw_shape, label_shape, weight_dataset, patch_shape, stride_shape, phase, ignore_bg=True):
        # random ignore_bg for fast training
        if ignore_bg > random.random():
            ignore_bg = True
        else:
            ignore_bg = False
        self._raw_slices = self._build_slices(raw_shape, patch_shape, stride_shape, phase, ignore_bg)
        if label_shape is None:
            self._label_slices = None
        else:
            # take the first element in the label_datasets to build slices
            self._label_slices = self._build_slices(label_shape, patch_shape, stride_shape, phase, ignore_bg)
            assert len(self._raw_slices) == len(self._label_slices)
        if weight_dataset is None:
            self._weight_slices = None
        else:
            self._weight_slices = self._build_slices(weight_dataset[0], patch_shape, stride_shape, phase, ignore_bg)
            assert len(self.raw_slices) == len(self._weight_slices)

    @property
    def raw_slices(self):
        return self._raw_slices

    @property
    def label_slices(self):
        return self._label_slices

    @property
    def weight_slices(self):
        return self._weight_slices

    @staticmethod
    def _build_slices(data_shape, patch_shape, stride_shape, phase, ignore_bg):
        """Iterates over a given n-dim dataset patch-by-patch with a given stride
        and builds an array of slice positions.

        Returns:
            list of slices, i.e.
            [(slice, slice, slice, slice), ...] if len(shape) == 4
            [(slice, slice, slice), ...] if len(shape) == 3
        """
        slices = []
        if len(data_shape) == 4:
            in_channels, i_z, i_y, i_x = [data_shape[i] for i in range(0,len(data_shape))]
        else:
            i_z, i_y, i_x = [data_shape[i] for i in range(0,len(data_shape))]

        k_z, k_y, k_x = patch_shape
        s_z, s_y, s_x = stride_shape
        z_steps = SliceBuilder._gen_indices(i_z, k_z, s_z, phase, ignore_bg)
        for z in z_steps:
            y_steps = SliceBuilder._gen_indices(i_y, k_y, s_y, phase, ignore_bg)
            for y in y_steps:
                x_steps = SliceBuilder._gen_indices(i_x, k_x, s_x, phase, ignore_bg)
                for x in x_steps:
                    slice_idx = (
                        slice(z, z + k_z),
                        slice(y, y + k_y),
                        slice(x, x + k_x)
                    )
                    if len(data_shape) == 4:
                        slice_idx = (slice(0, in_channels),) + slice_idx
                    print(slice_idx)
                    slices.append(slice_idx)
        return slices

    @staticmethod
    def _gen_indices(i, k, s, phase, ignore_bg):
        assert i >= k, 'Sample size has to be bigger than the patch size'
        if i < 256 or phase == 'test' or not ignore_bg:
            for j in range(0, i - k + 1, s):
                yield j
            if j + k < i:
                yield i - k

        else:
            for j in range(100, 412 - k + 1, s):
                yield j
            if j + k < 412:
                yield 412-k
            # for j in range(0, i - k + 1, s):
            #     yield j
            # if j + k < i:
            #     yield i - k

class ITKDataset(Dataset):
    """
    Implementation of torch.utils.data.Dataset backed by the HDF5 files, which iterates over the raw and label datasets
    patch by patch with a given stride.
    """

    def __init__(self, file_path, patch_shape, stride_shape, phase, transformer_config,
                 raw_internal_path='raw', label_internal_path='label',
                 weight_internal_path=None,slice_builder_cls=SliceBuilder, DS_module=None):
        """
        :param file_path: path to H5 file containing raw data as well as labels and per pixel weights (optional)
        :param patch_shape: the shape of the patch DxHxW
        :param stride_shape: the shape of the stride DxHxW
        :param phase: 'train' for training, 'val' for validation, 'test' for testing; data augmentation is performed
            only during the 'train' phase
        :param transformer_config: data augmentation configuration
        :param raw_internal_path (str or list): H5 internal path to the raw dataset
        :param label_internal_path (str or list): H5 internal path to the label dataset
        :param weight_internal_path (str or list): H5 internal path to the per pixel weights
        :param slice_builder_cls: defines how to sample the patches from the volume
        """
        assert phase in ['train', 'val', 'test']

        self.phase = phase
        self.file_path = file_path
        self.DS_module = DS_module
        self.patch_shape = patch_shape
        self.stride_shape = stride_shape
        self.transformer_config = transformer_config


        # convert raw_internal_path, label_internal_path and weight_internal_path to list for ease of computation
        if isinstance(raw_internal_path, str):
            raw_internal_path = [raw_internal_path]
        if isinstance(label_internal_path, str):
            label_internal_path = [label_internal_path]
        if isinstance(weight_internal_path, str):
            weight_internal_path = [weight_internal_path]

        # WARN: we load everything into memory due to hdf5 bug when reading H5 from multiple subprocesses, i.e.
        # File "h5py/_proxy.pyx", line 84, in h5py._proxy.H5PY_H5Dread
        # OSError: Can't read data (inflate() failed)
        """
        get each volume and label
        """
        volume_path = os.path.join(file_path, 'data.nii.gz')
        itk_CT = sitk.ReadImage(volume_path)
        img_arr = sitk.GetArrayFromImage(itk_CT).astype(np.float32)
        # self.raws = [self.window_normalize(img_arr, WW=1500, WL=-500)]
        self.raws = [self.window_normalize(img_arr, WW=transformer_config['WW'], WL=transformer_config['WL'])]
        # calculate global mean and std for Normalization augmentation
        mean, std = self._calculate_mean_std(self.raws[0])

        self.transformer = transforms.get_transformer(transformer_config, mean, std, phase)
        self.raw_transform = self.transformer.raw_transform()

        if phase != 'test':
            # create label/weight transform only in train/val phase
            self.label_transform = self.transformer.label_transform()
            gt_path = os.path.join(file_path, 'label.nii.gz')
            itk_gt = sitk.ReadImage(gt_path)
            self.labels = [sitk.GetArrayFromImage(itk_gt).astype(np.float32)]

            one_hot = transformer_config.get('one_hot', None)
            if one_hot:
                for i in range(len(self.labels)):
                    self.labels[i] = np.where(self.labels[i] == one_hot, 1, 0)
            if weight_internal_path is not None:
                # look for the weight map in the raw file
                self.weight_maps = [input_file[internal_path][...] for internal_path in weight_internal_path]
                self.weight_transform = self.transformer.weight_transform()
            else:
                self.weight_maps = None

            #test the data from multi-class
            # for i in range(len(self.labels[0])):
            #     # self.labels[0][i] = np.where(self.labels[0][i] > 0, 1, 0)
            #     print(i)

            self._check_dimensionality(self.raws, self.labels)
            # DS_module
            if self.DS_module == 'fnd' or self.DS_module == 'dense_fnd':
                DS_path = os.path.join(file_path, 'fnd.nii.gz')
                DS_itk_CT = sitk.ReadImage(DS_path)
                self.fnd = [sitk.GetArrayFromImage(DS_itk_CT).astype(np.float32)]
                self._check_dimensionality(self.raws, self.fnd)
        else:
            # 'test' phase used only for predictions so ignore the label dataset
            self.label_transform = self.transformer.label_transform()
            gt_path = os.path.join(file_path, 'label.nii.gz')
            itk_gt = sitk.ReadImage(gt_path)
            self.labels = [sitk.GetArrayFromImage(itk_gt).astype(np.float32)]
            one_hot = transformer_config.get('one_hot', None)
            if one_hot:
                for i in range(len(self.labels)):
                    self.labels[i] = np.where(self.labels[i] == one_hot, 1, 0)
            self.weight_maps = None
            self._check_dimensionality(self.raws, self.labels)

        if phase != 'test':
            raw_shape = [self.raws[0].shape[0], 200, 260]
            size = 96/raw_shape[0]
            re_raw_shape = [96, int(size*raw_shape[1]),int(size*raw_shape[2])]

            labels_shape = [self.labels[0].shape[0], 200, 260]
            size = 96 / labels_shape[0]
            re_labels_shape = [96, int(size * labels_shape[1]), int(size * labels_shape[2])]

        else:
            re_raw_shape = self.raws.shape
            re_labels_shape = self.labels.shape

        slice_builder = slice_builder_cls(
            re_raw_shape, re_labels_shape, self.weight_maps, patch_shape, stride_shape, phase, transformer_config['ignore_bg'])
        self.raw_slices = slice_builder.raw_slices
        print(len(self.raw_slices))
        self.label_slices = slice_builder.label_slices

        self.weight_slices = slice_builder.weight_slices

        self.patch_count = len(self.raw_slices)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        # get the slice for a given index 'idx'
        raw_idx = self.raw_slices[idx]
        # get the raw data patch for a given slice
        raw_transformed = self._transform_patches(self.raws, raw_idx, self.raw_transform)


        # raw_cut = raw_transformed[:, :, 170:370, 130:390]
        # raw_cut =raw_cut.resize_(1,64,128,128)
        if self.phase == 'test':
            # just return the transformed raw patch and the metadata
            return raw_transformed, raw_idx
        else:
            label_idx = self.label_slices[idx]
            label_transformed = self._transform_patches(self.labels, label_idx, self.label_transform)
            # label_cut = label_transformed[:, :, 170:370, 130:390]
            # label_cut = label_cut.resize_(1, 64, 128, 128)
            if self.DS_module == 'fnd' or self.DS_module == 'dense_fnd':
                fnd_transformed = self._transform_patches(self.fnd, label_idx, self.label_transform)
                # fnd_cut = fnd_transformed[:, :, 170:370, 130:390]
                # fnd_cut = fnd_cut.resize_(1, 64, 128, 128)
                return raw_transformed, label_transformed, fnd_transformed
            else:
                return raw_transformed, label_transformed

    def window_normalize(self, img, WW, WL, dst_range=(0, 1)):
        """
        WW: window width
        WL: window level
        dst_range: normalization range
        """
        src_min = WL - WW / 2
        src_max = WL + WW / 2
        outputs = (img - src_min) / WW * (dst_range[1] - dst_range[0]) + dst_range[0]
        outputs[img >= src_max] = 1
        outputs[img <= src_min] = 0
        return outputs

    @staticmethod
    def _transform_patches(datasets, label_idx, transformer):
        transformed_patches = []
        # transformed_dataset = transformer(datasets)
        transformed_dataset = transformer(datasets[0])
        for dataset in datasets:
            # get the label data and apply the label transformer
            if transformed_dataset.ndim == 4:
                transformed_patch = transformed_dataset[0][label_idx]
                transformed_patch = torch.unsqueeze(transformed_patch, dim=0)
            else:
                transformed_patch = transformed_dataset[label_idx]

            transformed_patches.append(transformed_patch)

        # if transformed_patches is a singleton list return the first element only
        if len(transformed_patches) == 1:
            return transformed_patches[0]
        else:
            return transformed_patches

    def __len__(self):
        return self.patch_count

    @staticmethod
    def _calculate_mean_std(input):
        """
        Compute a mean/std of the raw stack for normalization.
        This is an in-memory implementation, override this method
        with the chunk-based computation if you're working with huge H5 files.
        :return: a tuple of (mean, std) of the raw data
        """
        return input.mean(keepdims=True), input.std(keepdims=True)

    @staticmethod
    def _check_dimensionality(raws, labels):
        for raw in raws:
            assert raw.ndim in [3, 4], 'Raw dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
            if raw.ndim == 3:
                raw_shape = raw.shape
            else:
                raw_shape = raw.shape[1:]

        for label in labels:
            assert label.ndim in [3, 4], 'Label dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
            if label.ndim == 3:
                label_shape = label.shape
            else:
                label_shape = label.shape[1:]
            assert raw_shape == label_shape, 'Raw and labels have to be of the same size'

    @staticmethod
    def _check_patch_shape(patch_shape):
        assert len(patch_shape) == 3, 'patch_shape must be a 3D tuple'
        assert patch_shape[1] >= 64 and patch_shape[2] >= 64, 'Height and Width must be greater or equal 64'
        assert patch_shape[0] >= 16, 'Depth must be greater or equal 16'


def _get_slice_builder_cls(class_name):
    m = importlib.import_module('datasets.itkDataset')
    clazz = getattr(m, class_name)
    return clazz


def get_train_loaders(config):
    """
    Returns dictionary containing the training and validation loaders
    (torch.utils.data.DataLoader) backed by the datasets.hdf5.HDF5Dataset.

    :param config: a top level configuration object containing the 'loaders' key
    :return: dict {
        'train': <train_loader>
        'val': <val_loader>
    }
    """
    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']
    DS_module = loaders_config.get('DS_module', None)

    logger = get_logger('ITKDataset')
    logger.info('Creating training and validation set loaders...')

    # get train and validation files
    train_root_path = loaders_config['train_path']
    val_root_path = loaders_config['val_path']
    train_paths = [os.path.join(train_root_path, volume_name) for volume_name in
                   os.listdir(os.path.join(train_root_path))]
    val_paths = [os.path.join(val_root_path, volume_name) for volume_name in
                   os.listdir(os.path.join(val_root_path))]
    assert isinstance(train_paths, list)
    assert isinstance(val_paths, list)
    # get h5 internal paths for raw and label
    raw_internal_path = loaders_config['raw_internal_path']
    label_internal_path = loaders_config['label_internal_path']
    weight_internal_path = loaders_config.get('weight_internal_path', None)
    # get train/validation patch size and stride

    # get train/validation patch size and stride
    train_patch = tuple(loaders_config['train_patch'])
    train_stride = tuple(loaders_config['train_stride'])
    val_patch = tuple(loaders_config['val_patch'])
    val_stride = tuple(loaders_config['val_stride'])

    # get slice_builder_cls
    slice_builder_str = loaders_config.get('slice_builder', 'SliceBuilder')
    logger.info(f'Slice builder class: {slice_builder_str}')
    slice_builder_cls = _get_slice_builder_cls(slice_builder_str)

    train_datasets = []
    for train_path in train_paths:
        try:
            logger.info(f'Loading training set from: {train_path}...')
            # create H5 backed training and validation dataset with data augmentation
            train_dataset = ITKDataset(train_path, train_patch, train_stride, phase='train',
                                       transformer_config=loaders_config['transformer'],
                                       raw_internal_path=raw_internal_path,
                                       label_internal_path=label_internal_path,
                                       weight_internal_path=weight_internal_path,
                                       slice_builder_cls=slice_builder_cls,
                                       DS_module=DS_module)
            train_datasets.append(train_dataset)
        except Exception:
            logger.info(f'Skipping training set: {train_path}', exc_info=True)

    val_datasets = []
    for val_path in val_paths:
        try:
            logger.info(f'Loading validation set from: {val_path}...')
            val_dataset = ITKDataset(val_path, val_patch, val_stride, phase='val',
                                     transformer_config=loaders_config['transformer'],
                                     raw_internal_path=raw_internal_path,
                                     label_internal_path=label_internal_path,
                                     weight_internal_path=weight_internal_path,
                                     DS_module=DS_module)
            val_datasets.append(val_dataset)
        except Exception:
            logger.info(f'Skipping validation set: {val_path}', exc_info=True)

    num_workers = loaders_config.get('num_workers', 1)
    logger.info(f'Number of workers for train/val datasets: {num_workers}')
    # when training with volumetric data use batch_size of 1 due to GPU memory constraints
    return {
        'train': DataLoader(ConcatDataset(train_datasets), batch_size=1, shuffle=True, num_workers=num_workers),
        'val': DataLoader(ConcatDataset(val_datasets), batch_size=1, shuffle=True, num_workers=num_workers)
    }


def get_test_loaders(config):
    """
    Returns a list of DataLoader, one per each test file.

    :param config: a top level configuration object containing the 'datasets' key
    :return: generator of DataLoader objects
    """

    def my_collate(batch):
        error_msg = "batch must contain tensors or slice; found {}"
        if isinstance(batch[0], torch.Tensor):
            return torch.stack(batch, 0)
        elif isinstance(batch[0], slice):
            return batch[0]
        elif isinstance(batch[0], collections.Sequence):
            transposed = zip(*batch)
            return [my_collate(samples) for samples in transposed]

        raise TypeError((error_msg.format(type(batch[0]))))

    logger = get_logger('ITKDataset')

    assert 'datasets' in config, 'Could not find data sets configuration'
    datasets_config = config['datasets']

    # get train and validation files
    test_root_path = datasets_config['test_path']
    test_paths = [os.path.join(test_root_path, volume_name) for volume_name in
                  os.listdir(os.path.join(test_root_path))]
    assert isinstance(test_paths, list)
    # get h5 internal path
    raw_internal_path = datasets_config['raw_internal_path']
    # get train/validation patch size and stride
    patch = tuple(datasets_config['patch'])
    stride = tuple(datasets_config['stride'])
    num_workers = datasets_config.get('num_workers', 1)

    # construct datasets lazily
    datasets = (ITKDataset(test_path, patch, stride, phase='test', raw_internal_path=raw_internal_path,
                           transformer_config=datasets_config['transformer']) for test_path in test_paths)

    # use generator in order to create data loaders lazily one by one
    for dataset in datasets:
        logger.info(f'Loading test set from: {dataset.file_path}...')
        yield DataLoader(dataset, batch_size=1, num_workers=num_workers, collate_fn=my_collate)