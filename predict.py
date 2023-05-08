import os

# import h5py
import numpy as np
import torch
from torch import nn as nn

from datasets.hdf5 import get_test_loaders
from unet3d import utils
from unet3d.config import load_config
from unet3d.model import get_model
# from unet3d.crf import dence_crf as dcrf
from utils.evaluator import Evaluator_dice, Evaluator_hd95, Evaluator_hd95V2

# from utils.calFPFN import cal_FN,cal_FP

import SimpleITK as sitk


logger = utils.get_logger('UNet3DPredictor')

def predict(model, data_loader, output_file, config):
    """
    Return prediction masks by applying the model on the given dataset.
    The predictions are saved in the output H5 file on a patch by patch basis.
    If your dataset fits into memory use predict_in_memory() which is much faster.

    Args:
        model (Unet3D): trained 3D UNet model used for prediction
        data_loader (torch.utils.data.DataLoader): input data loader
        output_file (str): path to the output H5 file
        config (dict): global config dict
    """
    def _volume_shape(dataset):
        # TODO: support multiple internal datazsets
        raw = dataset.raws[0]
        if raw.ndim == 3:
            return raw.shape
        else:
            return raw.shape[1:]

    out_channels = config['model'].get('out_channels')
    if out_channels is None:
        out_channels = config['model']['dt_out_channels']

    prediction_channel = config.get('prediction_channel', None)
    if prediction_channel is not None:
        logger.info(f"Using only channel '{prediction_channel}' from the network output")

    device = config['device']
    output_heads = config['model'].get('output_heads', 1)

    logger.info(f'Running prediction on {len(data_loader)} patches...')

    # dimensionality of the the output (CxDxHxW)
    volume_shape = _volume_shape(data_loader.dataset)

    if prediction_channel is None:
        prediction_maps_shape = (out_channels,) + volume_shape
        # attention_maps_shape = (out_channels,) + volume_shape
    else:
        # single channel prediction map
        prediction_maps_shape = (1,) + volume_shape
        # attention_maps_shape = (1,) + volume_shape

    logger.info(f'The shape of the output prediction maps (CDHW): {prediction_maps_shape}')

    # allocate datasets for probability maps
    prediction_map = np.zeros(prediction_maps_shape, dtype='float32')
    # attention_map = np.zeros(attention_maps_shape, dtype='float32')

    # allocate datasets for normalization masks
    normalization_mask = np.zeros(prediction_maps_shape, dtype='uint8')
    # attnormalize_mask = np.zeros(attention_maps_shape, dtype='uint8')

    # Sets the module in evaluation mode explicitly, otherwise the final Softmax/Sigmoid won't be applied!
    model.eval()
    # Run predictions on the entire input dataset
    with torch.no_grad():
        for patch, index in data_loader:
            logger.info(f'Predicting slice:{index}')

            # save patch index: (C,D,H,W)
            if prediction_channel is None:
                channel_slice = slice(0, out_channels)
            else:
                channel_slice = slice(0, 1)

            index = (channel_slice,) + tuple(index)

            # send patch to device
            patch = patch.to(device)
            # forward pass
            # prediction = model(patch)
            # prediction, attention = model(patch)
            prediction = model(patch)

            # attention = torch.tensor(attention)

            # wrap predictions into a list if there is only one output head from the network
            # if output_heads == 1:
            #     predictions = [predictions]

            # squeeze batch dimension and convert back to numpy array
            prediction = prediction.squeeze(dim=0).cpu().numpy()

            #
            # attention = attention.squeeze(dim=0).cpu().numpy()

            if prediction_channel is not None:
                # use only the 'prediction_channel'
                logger.info(f"Using channel '{prediction_channel}'...")
                prediction = np.expand_dims(prediction[prediction_channel], axis=0)
                # attention = np.expand_dims(attention[prediction_channel], axis=0)

            # unpad in order to avoid block artifacts in the output probability maps
            u_prediction, u_index = utils.unpad(prediction, index, volume_shape)
            # accumulate probabilities into the output prediction array
            prediction_map[u_index] += u_prediction
            # count voxel visits for normalization
            normalization_mask[u_index] += 1

            # unpad in order to avoid block artifacts in the output probability maps
            # u_attention, u_attidx = utils.unpad(attention, index, volume_shape)
            # # accumulate probabilities into the output prediction array
            # attention_map[u_attidx] += u_attention
            # # count voxel visits for normalization
            # attnormalize_mask[u_attidx] += 1

    # normalize the prediction_maps inside the H5
    # TODO: iterate block by block
    # split the volume into 4 parts and load each into the memory separately
    z, y, x = volume_shape
    mid_x = x // 2
    mid_y = y // 2
    prediction_map[:, :, 0:mid_y, 0:mid_x] /= normalization_mask[:, :, 0:mid_y, 0:mid_x]
    prediction_map[:, :, mid_y:, 0:mid_x] /= normalization_mask[:, :, mid_y:, 0:mid_x]
    prediction_map[:, :, 0:mid_y, mid_x:] /= normalization_mask[:, :, 0:mid_y, mid_x:]
    prediction_map[:, :, mid_y:, mid_x:] /= normalization_mask[:, :, mid_y:, mid_x:]

    # attention_map[:, :, 0:mid_y, 0:mid_x] /= attnormalize_mask[:, :, 0:mid_y, 0:mid_x]
    # attention_map[:, :, mid_y:, 0:mid_x] /= attnormalize_mask[:, :, mid_y:, 0:mid_x]
    # attention_map[:, :, 0:mid_y, mid_x:] /= attnormalize_mask[:, :, 0:mid_y, mid_x:]
    # attention_map[:, :, mid_y:, mid_x:] /= attnormalize_mask[:, :, mid_y:, mid_x:]

    # return torch.from_numpy(prediction_map), torch.from_numpy(attention_map)
    return torch.from_numpy(prediction_map)



def _get_output_file(dataset,path):
    # return f'{os.path.splitext(dataset.file_path)[0]}{suffix}.h5'
    number = dataset.file_path[-1:-3:-1][-1:-3:-1]
    # b = a[-1:-3:-1]
    # b = b[-1:-3:-1]

    return f'{path}{number}'


def _get_dataset_names(config, number_of_datasets, prefix='predictions'):
    dataset_names = config.get('dest_dataset_name')
    if dataset_names is not None:
        if isinstance(dataset_names, str):
            return [dataset_names]
        else:
            return dataset_names
    else:
        if number_of_datasets == 1:
            return [prefix]
        else:
            return [f'{prefix}{i}' for i in range(number_of_datasets)]


def main():
    # Load configuration
    config = load_config()

    # Create the model
    model = get_model(config)

    # Load model state
    model_path = config['model_path']
    logger.info(f'Loading model from {model_path}...')
    utils.load_checkpoint(model_path, model)
    logger.info(f"Sending the model to '{config['device']}'")
    model = model.to(config['device'])

    logger.info(f"Saving the model to '{config['predict_path']}'")
    predict_path = config['predict_path']
    #
    # fpd_path = config['fpd_path']
    # fnd_path = config['fnd_path']
    # attention_path = config['attention_path']

    # predict_crf_path = config.get('predict_crf_path', None)
    check_mkdir(predict_path)
    # check_mkdir(fpd_path)
    # check_mkdir(fnd_path)
    # check_mkdir(attention_path)

    logger.info('Loading HDF5 datasets...')
    store_predictions_in_memory = config.get('store_predictions_in_memory', True)
    if store_predictions_in_memory:
        logger.info('Predictions will be stored in memory. Make sure you have enough RAM for you dataset.')

    eval_dice = Evaluator_dice()
    # eval = Evaluator_hd95()
    eval_hd95 = Evaluator_hd95V2()
    total_dice_eval = 0
    total_hd95_eval = 0

    for test_loader in get_test_loaders(config):
        logger.info(f"Processing '{test_loader.dataset.file_path}'...")
        output_file = _get_output_file(test_loader.dataset, predict_path)
        # fp_file = _get_output_file(test_loader.dataset, fpd_path)
        # fn_file = _get_output_file(test_loader.dataset, fnd_path)
        # attention_file = _get_output_file(test_loader.dataset, attention_path, suffix='_attention')

        # run the model prediction on the entire dataset and save to the 'output_file' H5
        # predict_in_memory(model, test_loader, output_file, config)
        # prediction, attention_maps = predict(model, test_loader, output_file, config) # shape = (channel, d, h, w)
        prediction = predict(model, test_loader, output_file, config) # shape = (channel, d, h, w)

        # attention_maps = attention_maps.numpy()
        # attention_map = np.zeros(attention_maps.shape[1:4])

        # for i in range(attention_maps.shape[0]):
        #     attention_map += attention_maps[i]


        # prediction = dence_crf(prediction)

        if config['model']['out_channels'] == 1: #onehot training
            gt = test_loader.dataset.labels[0] == 1
            prediction = prediction.squeeze(dim=0).numpy()
            pred_map = prediction > 0.1
            eval_hd95.add_volume(gt, pred_map)
            total_eval = eval_hd95.get_eval()
            logger.info(f"eval '{total_eval}'...")

        else:
            # image = test_loader.dataset.labels[0]
            # image = window_normalize(image,1500,-500)
            # res = dcrf(prediction, image)
            # pred_map = res.numpy()
            # pred_map = prediction.argmax(dim=0).numpy()
            prediction = prediction.squeeze(dim=0).numpy()
            pred_map = prediction > 0.5


            fp = np.zeros(pred_map.shape)
            fn = np.zeros(pred_map.shape)

            eval_diceper_class = np.ones(config['model']['out_channels'])
            eval_hd95per_class = np.ones(config['model']['out_channels'])
            for idx in range(config['model']['out_channels']):
                # No need to calculate background eval
                if idx == 0:
                    continue
                # add each volume at each label
                gt = test_loader.dataset.labels[0] == idx
                pred_map_idx = pred_map == idx

                preddice_pixel, targetdice_pixel = eval_dice.add_volume(pred_map_idx, gt, idx)
                # pred95_pixel, target95_pixel = eval_hd95.add_volume(pred_map_idx, gt, idx)

                eval_diceper_class[idx] = eval_dice.get_eval(channel=idx)
                # eval_hd95per_class[idx] = eval_hd95.get_eval(channel=idx)
                logger.info(f"eval_dice_per_class '{idx}--{eval_diceper_class[idx]}'...")
                # logger.info(f"eval_hd95_per_class '{idx}--{eval_hd95per_class[idx]}'...")

                # fp += cal_FP(gt, pred_map_idx, idx)
                # fn += cal_FN(gt, pred_map_idx, idx)

            total_dice_eval = eval_dice.get_eval()
            # total_hd95_eval = eval_hd95.get_eval()
            logger.info(f"eval '{total_dice_eval}'...")
            # logger.info(f"eval '{total_hd95_eval}'...")
        if store_predictions_in_memory:
            logger.info(f'Saving predictions to: {output_file}')
            out = sitk.GetImageFromArray(pred_map.astype(np.float32))
            sitk.WriteImage(out, '{}.nii.gz'.format(output_file))
            #
            # fp_out = sitk.GetImageFromArray(fp.astype(np.float32))
            # fn_out = sitk.GetImageFromArray(fn.astype(np.float32))
            # sitk.WriteImage(fp_out, '{}.nii.gz'.format(fp_file))
            # sitk.WriteImage(fn_out, '{}.nii.gz'.format(fn_file))

            # att_out = sitk.GetImageFromArray(attention_map.astype(np.float32))
            # sitk.WriteImage(att_out, '{}.nii.gz'.format(attention_file))

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

def window_normalize(img, WW, WL, dst_range=(0, 1)):

    src_min = WL - WW / 2
    src_max = WL + WW / 2
    outputs = (img - src_min) / WW * (dst_range[1] - dst_range[0]) + dst_range[0]
    outputs[img >= src_max] = 1
    outputs[img <= src_min] = 0
    return outputs


if __name__ == '__main__':
    main()
