# camera-ready

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()

        dilations = [1, 2, 3] # maybe dilations should be set larger
        # dilations = [12, 24, 36] when output_stride==8

        self.conv_1x1_1 = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.bn_conv_1x1_1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)

        self.conv_3x3_1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=dilations[0], dilation=dilations[0])
        self.bn_conv_3x3_1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)

        self.conv_3x3_2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=dilations[1], dilation=dilations[1])
        self.bn_conv_3x3_2 = nn.GroupNorm(num_groups=8, num_channels=in_channels)

        self.conv_3x3_3 = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=dilations[2], dilation=dilations[2])
        self.bn_conv_3x3_3 = nn.GroupNorm(num_groups=8, num_channels=in_channels)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.conv_1x1_2 = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.bn_conv_1x1_2 = nn.GroupNorm(num_groups=8, num_channels=in_channels)

        self.conv_1x1_3 = nn.Conv3d(in_channels*5, in_channels, kernel_size=1) # (1280 = 5*256)
        self.bn_conv_1x1_3 = nn.GroupNorm(num_groups=8, num_channels=in_channels)

        # self.conv_1x1_4 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, feature_map):
        # (feature_map has shape (batch_size, 512, h, w)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet instead is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8))
        feature_map_z = feature_map.size()[2] # (== z)
        feature_map_h = feature_map.size()[3] # (== h)
        feature_map_w = feature_map.size()[4] # (== w)

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map))) # (shape: (batch_size, in_channels, z, h, w))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map))) # (shape: (batch_size, in_channels, z, h, w))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map))) # (shape: (batch_size, in_channels, z, h, w))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map))) # (shape: (batch_size, in_channels, z, h, w))

        out_img = self.avg_pool(feature_map) # (shape: (batch_size, in_channels, 1, 1))
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) # (shape: (batch_size, 256, 1, 1))
        out_img = F.upsample(out_img, size=(feature_map_z, feature_map_h, feature_map_w), mode="trilinear") # (shape: (batch_size, 256, z, h, w))

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) # (shape: (batch_size, in_channels*5, z, h, w))
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) # (shape: (batch_size, 256, z, h, w))
        # out = self.conv_1x1_4(out) # (shape: (batch_size, num_classes, z, h, w))

        return out




class ASPP_Full(nn.Module):
    def __init__(self, in_channels, out_channels, dilations):
        super(ASPP_Full, self).__init__()

        # dilations = [1, 2, 3] # maybe dilations should be set larger
        # dilations = [12, 24, 36] when output_stride==8


        self.conv_1x1_1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        self.conv_3x3_1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilations[0], dilation=dilations[0])
        self.bn_conv_3x3_1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        self.conv_3x3_2 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilations[1], dilation=dilations[1])
        self.bn_conv_3x3_2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        self.conv_3x3_3 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilations[2], dilation=dilations[2])
        self.bn_conv_3x3_3 = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.conv_1x1_2 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        self.conv_1x1_3 = nn.Conv3d(in_channels*5, out_channels, kernel_size=1) # (1280 = 5*256)
        self.bn_conv_1x1_3 = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        # self.conv_1x1_4 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, feature_map):
        # (feature_map has shape (batch_size, 512, h, w)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet instead is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8))
        feature_map_z = feature_map.size()[2] # (== z)
        feature_map_h = feature_map.size()[3] # (== h)
        feature_map_w = feature_map.size()[4] # (== w )

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map))) # (shape: (batch_size, in_channels, z, h, w))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map))) # (shape: (batch_size, in_channels, z, h, w))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map))) # (shape: (batch_size, in_channels, z, h, w))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map))) # (shape: (batch_size, in_channels, z, h, w))

        out_img = self.avg_pool(feature_map) # (shape: (batch_size, in_channels, 1, 1))
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) # (shape: (batch_size, 256, 1, 1))
        out_img = F.upsample(out_img, size=(feature_map_z, feature_map_h, feature_map_w), mode="trilinear") # (shape: (batch_size, 256, z, h, w))

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) # (shape: (batch_size, in_channels*5, z, h, w))
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) # (shape: (batch_size, 256, z, h, w))
        # out = self.conv_1x1_4(out) # (shape: (batch_size, num_classes, z, h, w))

        return out

