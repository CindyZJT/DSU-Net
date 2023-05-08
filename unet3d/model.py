import importlib

import torch
import torch.nn as nn

from unet3d.buildingblocks import Encoder, Decoder,DecoderWithAspp, FinalConv, DoubleConv, ExtResNetBlock, SingleConv, DSBlock,DSBlock_SE,Decoder_Dense,DSBlock_Dense
from unet3d.utils import create_feature_maps
from .aspp import ASPP
from .aspp import ASPP_Full
from .EMA import EMAU
import numpy as np

import torch.nn.functional as F


class UNet3D(nn.Module):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='crg', num_groups=8,
                 **kwargs):
        super(UNet3D, self).__init__()

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=4)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num, basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        # self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, kernel_size=1, padding=0)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)
        # a = x.cpu().detach().numpy()


        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if not self.training:
            x = self.final_activation(x)

        # b = x.cpu().detach().numpy()


        return x

class UNet3D_DS(nn.Module):

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='crg', num_groups=8,
                 **kwargs):
        super(UNet3D_DS, self).__init__()

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=4)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        reversed_f_maps = list(reversed(f_maps))

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        dsBlocks = []
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num, basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
            decoders.append(decoder)
            dsBlock = DSBlock(out_feature_num, f_maps[0])
            dsBlocks.append(dsBlock)

        self.decoders = nn.ModuleList(decoders)
        self.DSBlocks = nn.ModuleList(dsBlocks)

        self.final_conv = nn.Conv3d(f_maps[0], out_channels, kernel_size=1, padding=0)

        self.fnd_final = nn.Conv3d(f_maps[0], out_channels=1, kernel_size=1)
        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # x = self.aspp(x)

        # decoder part
        fn_outputs = []

        for idx, decoder, encoder_features, dsBlock in zip(range(len(encoders_features)), self.decoders, encoders_features, self.DSBlocks):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)
            # DS module
            x, fn_output = dsBlock(x)
            fn_output = F.interpolate(fn_output, scale_factor=2 ** (len(encoders_features) - idx - 1), mode='trilinear', align_corners=True)
            fn_output = self.fnd_final(fn_output)
            fn_outputs.append(fn_output)
        x = self.final_conv(x)
        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if fn_outputs[0].shape != fn_outputs[1].shape or fn_outputs[1].shape != fn_outputs[2].shape:
            _, _, C, H, W = fn_outputs[2].shape
            fn_outputs[0] = F.upsample(fn_outputs[0], size=(C, H, W), mode='trilinear')
            fn_outputs[1] = F.upsample(fn_outputs[1], size=(C, H, W), mode='trilinear')
        if not self.training:
            x = self.final_activation(x)
            return x
        else:
            return x, fn_outputs

class UNet3D_aspp_DS(nn.Module):

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='crg', num_groups=8,
                 **kwargs):
        super(UNet3D_aspp_DS, self).__init__()

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=4)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        reversed_f_maps = list(reversed(f_maps))
        # ASPP module
        self.aspp = ASPP(in_channels=reversed_f_maps[0], out_channels=reversed_f_maps[0])

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        dsBlocks = []
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num, basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
            decoders.append(decoder)
            dsBlock = DSBlock(out_feature_num, f_maps[0])
            dsBlocks.append(dsBlock)

        self.decoders = nn.ModuleList(decoders)
        self.DSBlocks = nn.ModuleList(dsBlocks)

        self.final_conv = nn.Conv3d(f_maps[0], out_channels, kernel_size=1, padding=0)

        self.fnd_final = nn.Conv3d(f_maps[0], out_channels=1, kernel_size=1)
        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # x = self.aspp(x)

        # decoder part
        fn_outputs = []

        for idx, decoder, encoder_features, dsBlock in zip(range(len(encoders_features)), self.decoders, encoders_features, self.DSBlocks):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)
            # DS module
            x, fn_output = dsBlock(x)
            fn_output = F.interpolate(fn_output, scale_factor=2 ** (len(encoders_features) - idx - 1), mode='trilinear', align_corners=True)
            fn_output = self.fnd_final(fn_output)
            fn_outputs.append(fn_output)
        x = self.final_conv(x)
        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if fn_outputs[0].shape != fn_outputs[1].shape or fn_outputs[1].shape != fn_outputs[2].shape:
            _, _, C, H, W = fn_outputs[2].shape
            fn_outputs[0] = F.upsample(fn_outputs[0], size=(C, H, W), mode='trilinear')
            fn_outputs[1] = F.upsample(fn_outputs[1], size=(C, H, W), mode='trilinear')
        if not self.training:
            x = self.final_activation(x)
            return x
        else:
            return x, fn_outputs

class UNet3D_Dense_DS(nn.Module):

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='crg', num_groups=8,
                 **kwargs):
        super(UNet3D_Dense_DS, self).__init__()

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=4)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)

            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)

            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

        reversed_f_maps = list(reversed(f_maps))

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder

        self.decoder0 = Decoder_Dense(512+256, 256, basic_module=DoubleConv, conv_layer_order=layer_order, num_groups=num_groups)
        self.decoder1 = Decoder_Dense(512+256+128, 128, basic_module=DoubleConv, conv_layer_order=layer_order, num_groups=num_groups)
        self.decoder2 = Decoder_Dense(512+256+128+64, 64, basic_module=DoubleConv, conv_layer_order=layer_order, num_groups=num_groups)

        self.conv = nn.Conv3d(64, 64, kernel_size=1)
        a = f_maps[0]
        self.dsBlock0 = DSBlock_Dense(512, f_maps[0], 0)
        self.dsBlock1 = DSBlock_Dense(256, f_maps[0], 1)
        self.dsBlock2 = DSBlock_Dense(128, f_maps[0], 2)
        self.dsBlock3 = DSBlock_Dense(64, f_maps[0], 3)

        self.de_conv0 = nn.Conv3d(512,out_channels, kernel_size=1, padding=0)
        self.de_conv1 = nn.Conv3d(256,out_channels, kernel_size=1, padding=0)
        self.de_conv2 = nn.Conv3d(128,out_channels, kernel_size=1, padding=0)
        self.de_conv3 = nn.Conv3d(64,out_channels, kernel_size=1, padding=0)

        self.final_conv = nn.Conv3d(f_maps[0], out_channels, kernel_size=1, padding=0)

        self.fnd_final = nn.Conv3d(f_maps[0], out_channels=1, kernel_size=1)
        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        fn_outputs_finals = []
        fn_extractors = []
        decoder_features = []
        pre_decoders = []

        x, fn_extractor = self.dsBlock0(x, fn_extractors, 0)
        fn_extractors.append(fn_extractor)
        decoder_features.append(x)
        # channel 256+512 -> 256
        x = self.decoder0(encoders_features, decoder_features, 0)

        x, fn_extractor = self.dsBlock1(x, fn_extractors, 1)
        fn_extractors.append(fn_extractor)
        decoder_features.append(x)
        # channel 128+256+512->128
        x = self.decoder1(encoders_features, decoder_features, 1)

        x, fn_extractor = self.dsBlock2(x, fn_extractors, 2)
        fn_extractors.append(fn_extractor)
        decoder_features.append(x)
        # channel 64+128+256+512->64
        x = self.decoder2(encoders_features, decoder_features, 2)

        x, fn_extractor = self.dsBlock3(x, fn_extractors, 3)
        decoder_features.append(x)
        fn_extractors.append(fn_extractor)
        # channel 64->32
        x = self.conv(x)

        x = self.final_conv(x)

        for i in range(len(fn_extractors)):
            fn_extractors[i] = self.fnd_final(fn_extractors[i])

        pre_decoder0 = self.de_conv0(decoder_features[0])
        pre_decoder1 = self.de_conv1(decoder_features[1])
        pre_decoder2 = self.de_conv2(decoder_features[2])
        pre_decoder3 = self.de_conv3(decoder_features[3])

        pre_decoders.append(pre_decoder0)
        pre_decoders.append(pre_decoder1)
        pre_decoders.append(pre_decoder2)
        pre_decoders.append(pre_decoder3)


        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if fn_extractors[0].shape != fn_extractors[1].shape or fn_extractors[1].shape != fn_extractors[2].shape \
            or fn_extractors[2].shape != fn_extractors[3].shape:
            _, _, C, H, W = fn_extractors[3].shape
            fn_extractors[0] = F.upsample(fn_extractors[0], size=(C, H, W), mode='trilinear')
            fn_extractors[1] = F.upsample(fn_extractors[1], size=(C, H, W), mode='trilinear')
            fn_extractors[2] = F.upsample(fn_extractors[2], size=(C, H, W), mode='trilinear')

        if pre_decoders[0].shape != pre_decoders[1].shape or pre_decoders[1].shape != pre_decoders[2].shape or pre_decoders[2].shape != pre_decoders[3].shape:
            _, _, C, H, W = pre_decoders[3].shape
            pre_decoders[0] = F.upsample(pre_decoders[0], size=(C, H, W), mode='trilinear')
            pre_decoders[1] = F.upsample(pre_decoders[1], size=(C, H, W), mode='trilinear')
            pre_decoders[2] = F.upsample(pre_decoders[2], size=(C, H, W), mode='trilinear')

        if not self.training:
            x = self.final_activation(x)
            return x
        else:
            return x, fn_extractors, pre_decoders

class UNet3D_DS_SE(nn.Module):

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='crg', num_groups=8,
                 **kwargs):
        super(UNet3D_DS_SE, self).__init__()

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=4)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        reversed_f_maps = list(reversed(f_maps))
        # ASPP module
        # self.aspp = ASPP(in_channels=reversed_f_maps[0], out_channels=reversed_f_maps[0])

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        dsBlocks = []
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num, basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
            decoders.append(decoder)
            dsBlock = DSBlock_SE(out_feature_num, f_maps[0])
            dsBlocks.append(dsBlock)

        self.decoders = nn.ModuleList(decoders)
        self.DSBlocks = nn.ModuleList(dsBlocks)

        self.final_conv = nn.Conv3d(f_maps[0], out_channels, kernel_size=1, padding=0)

        self.fnd_final = nn.Conv3d(f_maps[0], out_channels=1, kernel_size=1)
        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # x = self.aspp(x)

        # decoder part
        fn_outputs = []
        for idx, decoder, encoder_features, dsBlock in zip(range(len(encoders_features)), self.decoders, encoders_features, self.DSBlocks):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)
            # DS module
            x, fn_output = dsBlock(x)
            fn_output = F.interpolate(fn_output, scale_factor=2 ** (len(encoders_features) - idx - 1), mode='trilinear', align_corners=True)
            fn_output = self.fnd_final(fn_output)
            fn_outputs.append(fn_output)
        x = self.final_conv(x)
        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if not self.training:
            x = self.final_activation(x)
            return x
        else:
            return x, fn_outputs



class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes, stride=1,num_groups = 8):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups,planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups, planes)
        self.conv3 = nn.Conv3d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.gn3 = nn.GroupNorm(num_groups, self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                # nn.GroupNorm(num_groups)
            )

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = F.relu(self.gn2(self.conv2(out)))
        out = self.gn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FPN_3D(nn.Module):
    def __init__(self,block,num_blocks,num_groups=8, final_sigmoid=True, training=True,**kwargs):
        super(FPN_3D, self).__init__()
        self.training = training
        self.in_planes = 8

        self.conv1 = nn.Conv3d(1, 8, kernel_size=7, stride=2, padding=3, bias=False)
        self.gn1 = nn.GroupNorm(num_groups,8)

        # Bottom-up layers

        self.layer1 = self._make_layer(block,  8, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 16, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 32, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 64, num_blocks[3], stride=2)

        # Top layer
        self.toplayer = nn.Conv3d(128, 8, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Lateral layers
        self.latlayer1 = nn.Conv3d(64, 8, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv3d(32, 8, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv3d(16, 8, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Conv3d(8, 8, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv3d(8, 8, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv3d(8, 8, kernel_size=3, stride=1, padding=1)
        self.finconv = nn.Conv3d(8, 1, kernel_size=3, stride=1, padding=1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)



    def _make_layer(self,block, planes, num_blocks,  stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,C,H,W = y.size()
        return F.upsample(x, size=(C,H,W), mode='trilinear') + y

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.gn1(self.conv1(x)))
        c1 = F.max_pool3d(c1, kernel_size=3, stride=1, padding=1)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # Top-down
        # p5 = self.toplayer(c5)
        _, _, C, H, W = x.size()
        p5 = F.upsample(self.toplayer(c5), size=(C, H, W), mode='trilinear')
        p5 = self.finconv(p5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        # p4 = self.smooth1(p4)
        p4 = F.upsample(self.smooth1(p4), size=(C, H, W), mode='trilinear')
        p4 = self.finconv(p4)
        # p3 = self.smooth2(p3)
        p3 = F.upsample(self.smooth2(p3), size=(C, H, W), mode='trilinear')
        p3 = self.finconv(p3)
        p2 = F.upsample(self.smooth3(p2), size=(C, H, W), mode='trilinear')
        p2 = self.finconv(p2)

        s = p5 + p4 + p3 + p2

        if not self.training:
            s = self.final_activation(p2)
        return s


def outS(i):
    i = int(i)
    i = (i + 1) / 2
    i = int(np.ceil((i + 1) / 2.0))
    i = (i + 1) / 2
    i = int(np.ceil((i + 1) / 2.0))
    return i


def conv3x3(in_planes, out_planes, stride=1):
    "3x3x3 convolution with padding"
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)




class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation_=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        # self.bn1 = nn.BatchNorm3d(planes, affine=affine_par)
        self.bn1 =nn.GroupNorm(num_groups=8, num_channels=planes)

        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = dilation_
        # if dilation_ == 2:
        #    padding = 2
        # elif dilation_ == 4:
        #    padding = 4
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=padding, bias=False, dilation=dilation_)
        # self.bn2 = nn.BatchNorm3d(planes, affine=affine_par)
        self.bn2 = nn.GroupNorm(num_groups=8, num_channels=planes)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm3d(planes * 4, affine=affine_par)
        self.bn3 = nn.GroupNorm(num_groups=8, num_channels=planes * 4)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, NoLabels):
        super(Classifier_Module, self).__init__()
        self.conv3d_list = nn.ModuleList()
        self.bn3d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv3d_list.append(
                nn.Conv3d(2048, 256, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))
            # self.bn3d_list.append(nn.BatchNorm3d(256, affine=affine_par))
            self.bn3d_list.append(nn.GroupNorm(num_groups=8,num_channels=256))
        self.num_concats = len(self.conv3d_list) + 2
        # add global pooling, add batchnorm
        self.conv1x1_1 = nn.Conv3d(2048, 256, kernel_size=1, stride=1)
        self.conv1x1_2 = nn.Conv3d(2048, 256, kernel_size=1, stride=1)
        self.conv1x1_3 = nn.Conv3d(256 * self.num_concats, 256, kernel_size=1, stride=1)
        self.conv1x1_4 = nn.Conv3d(256, NoLabels, kernel_size=1, stride=1)

        # self.bn1 = nn.BatchNorm3d(256, affine=affine_par)
        self.bn1 = nn.GroupNorm(num_groups=8,num_channels=256)

        # self.bn2 = nn.BatchNorm3d(256 * self.num_concats, affine=affine_par)
        self.bn2 = nn.GroupNorm(num_groups=8,num_channels=256 * self.num_concats)
        # self.bn3 = nn.BatchNorm3d(256, affine=affine_par)
        self.bn3 = nn.GroupNorm(num_groups=8,num_channels=256)
        # global avg pool
        # input = 1x512xdim1xdim2xdim3
        # output = 1x512x1x1x1
        # XXX check

        for m in self.conv3d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.bn3d_list[0](self.conv3d_list[0](x))
        # concatenate multiple atrous rates
        for i in range(len(self.conv3d_list) - 1):
            # XXX add batch norm?
            out = torch.cat([out, self.bn3d_list[i + 1](self.conv3d_list[i + 1](x))], 1)

        # concatenate global avg pooling (avg global pool -> 1x1 conv (256 filter) -> batchnorm -> interpolate -> concat)
        self.glob_avg_pool = nn.AvgPool3d(kernel_size=(x.size()[2], x.size()[3], x.size()[4]))
        self.iterp_orig = nn.Upsample(size=(out.size()[2], out.size()[3], out.size()[4]), mode='trilinear')

        out = torch.cat([out, self.iterp_orig(self.bn1(self.conv1x1_1(self.glob_avg_pool(x))))], 1)

        # concatenate 1x1 convolution
        out = torch.cat([out, self.conv1x1_2(x)], 1)

        # apply batch norm on concatenated output
        out = self.bn2(out)

        # apply 1x1 convolution to get back to 256 filters
        out = self.conv1x1_3(out)

        # apply last batch norm
        out = self.bn3(out)

        # apply 1x1 convolution to get last labels
        out = self.conv1x1_4(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, NoLabels):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)  # / 2
        # self.bn1 = nn.BatchNorm3d(64, affine=affine_par)
        self.bn1 = nn.GroupNorm(num_groups=8,num_channels=64)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=0, ceil_mode=True)  # / 4
        self.layer1 = self._make_layer(block, 64, layers[0])

        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)  # / 8
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)  # / 16

        self.layer4_0 = self._make_layer(block, 512, layers[3], stride=1, dilation__=2)
        self.layer4_1 = self._make_layer(block, 512, layers[3], stride=1, dilation__=4)
        self.layer4_2 = self._make_layer(block, 512, layers[3], stride=1, dilation__=8)
        self.layer4_3 = self._make_layer(block, 512, layers[3], stride=1, dilation__=16)

        self.layer5 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], NoLabels)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                for i in m.parameters():
                    i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation__=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm3d(planes * block.expansion, affine=affine_par),
                nn.GroupNorm(num_groups=8, num_channels=planes * block.expansion),
            )
            for i in downsample._modules['1'].parameters():
                i.requires_grad = False

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation_=dilation__, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation_=dilation__))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels):
        return block(dilation_series, padding_series, NoLabels)

    def forward(self, x):
        # print('A - x', x.size())
        x = self.conv1(x)
        # print('B - conv1', x.size())
        x = self.bn1(x)
        # print('C - bn1', x.size())
        x = self.relu(x)
        # print('D - relu', x.size())
        x = self.maxpool(x)
        # print('E - maxpool', x.size())
        x = self.layer1(x)
        # print('F - layer1', x.size())
        x = self.layer2(x)
        # print('G - layer2', x.size())
        x = self.layer3(x)
        # print('H - layer3', x.size())
        x = self.layer4_0(x)
        # print('I - layer4_0', x.size())
        x = self.layer4_1(x)
        # print('J - layer4_1', x.size())
        x = self.layer4_2(x)
        # print('K - layer4_2', x.size())
        x = self.layer4_3(x)
        # print('L - layer4_3', x.size())
        x = self.layer5(x)
        # print('M - layer5 (classification)', x.size())
        return x


#
class MS_Deeplab(nn.Module):
    def __init__(self, Bottleneck, NoLabels=2,**kwargs):
        super(MS_Deeplab, self).__init__()
        block = Bottleneck
        self.Scale = ResNet(block, [3, 4, 23, 3], NoLabels)

    def forward(self, x):
        s0 = x.size()[2]
        s1 = x.size()[3]
        s2 = x.size()[4]
        # self.interp3 = nn.Upsample(size = ( outS(s0), outS(s1), outS(s2) ), mode= 'nearest')
        self.interp = nn.Upsample(size=(s0, s1, s2), mode='trilinear')
        out = self.interp(self.Scale(x))
        return out


def get_model(config):
    def _model_class(class_name):
        m = importlib.import_module('unet3d.model')
        clazz = getattr(m, class_name)
        return clazz

    assert 'model' in config, 'Could not find model configuration'
    model_config = config['model']
    model_class = _model_class(model_config['name'])
    if model_class == MS_Deeplab:
        return model_class(Bottleneck,1, **model_config)
    else:
        return model_class(**model_config)