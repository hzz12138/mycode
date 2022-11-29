# -*- coding: utf-8 -*-
# @Time    : 2022/11/29 14:20
# @Author  : Zph
# @Email   : hhs_zph@mail.imu.edu.cn
# @File    : ResNet.py
# @Software: PyCharm

"""
    构建ResNet网络
"""

import torch
from torch import nn
from torchsummary import summary


class BottleNeck(nn.Module):
    """
    in_channel：输入通道
    out_channel：输出通道
    stride：卷积步长
    downsample：控制shortcut图片的下采样
    """
    expansion = 4  # 残差块的第三个卷积层，膨胀倍率

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.downsample = downsample  # 定义旁路的下采样
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel * self.expansion, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel * self.expansion),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, in_lyr):
        residual = in_lyr  #
        if self.downsample is not None:
            residual = self.downsample(in_lyr)
        out_lyr = self.stage(in_lyr)
        out_lyr = out_lyr + residual
        return self.relu(out_lyr)


class ResNet(nn.Module):
    def __init__(self, block, block_num, num_classes=100):
        """
        对ResNet进行初始化
        :param block: 选用的残差块类型
        :param block_num: 残差块数量
        :param num_classes: 分类数量
        """
        super(ResNet, self).__init__()
        self.in_channel = 64  # 第一个残差块的输入通道数
        # 第一个残差块前的卷积池化操作
        self.stage_1 = nn.Sequential(
            nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 第一个残差块，尺寸不改变，因此stride=1
        self.stage_2 = self._make_layer(block, first_channel=64, block_num=block_num[0], stride=1)
        # 第二、三、四个残差块，尺寸减少为原来的一半，因此stride=2
        self.stage_3 = self._make_layer(block, first_channel=128, block_num=block_num[1], stride=2)
        self.stage_4 = self._make_layer(block, first_channel=256, block_num=block_num[2], stride=2)
        self.stage_5 = self._make_layer(block, first_channel=512, block_num=block_num[3], stride=2)
        self.stage_6 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=512 * block.expansion, out_features=num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 针对于relu，使用kaiming初始化方法
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, first_channel, block_num, stride=1):  # 保护函数，只允许其子类或本身可访问
        """
        循环创建残差块
        :param block：残差块的类型(BottleNeck)
        :param first_channel: 第一个残差块的通道
        :param block_num: 残差块的数量
        :param stride: 步长
        :return:
        """
        downsample = None
        if stride != 1 or self.in_channel != first_channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=first_channel * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(first_channel * block.expansion)
            )
        # 在第1个残差块中，输入通道、第一个通道、输出通道不相同，输入通道下采样与输出通道相同，才可相加。
        layers = [block(self.in_channel, first_channel, stride=stride, downsample=downsample)]
        self.in_channel = first_channel * block.expansion
        for _ in range(1, block_num):
            # 在第2、3、4个残差块中，输入通道和输出通道已完全相同，因此不需要再进行下采样调整通道。
            layers.append(block(self.in_channel, first_channel))

        return nn.Sequential(*layers)

    def forward(self, in_lyr):
        out_lyr = self.stage_1(in_lyr)
        out_lyr = self.stage_2(out_lyr)
        out_lyr = self.stage_3(out_lyr)
        out_lyr = self.stage_4(out_lyr)
        out_lyr = self.stage_5(out_lyr)
        out_lyr = self.stage_6(out_lyr)
        out_lyr = torch.flatten(out_lyr, 1)
        out_lyr = self.fc(out_lyr)
        return out_lyr


if __name__ == '__main__':
    resnet50 = ResNet(block=BottleNeck, block_num=[3, 4, 6, 3], num_classes=1000).cuda()
    # print(resnet50)
    summary(resnet50, input_size=(3, 256, 256))
