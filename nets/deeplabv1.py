# -*- coding: utf-8 -*-
# @Time : 2022/11/28 22:00
# @Author : Zph
# @FileName: deeplabv1.py
# @Software: PyCharm
# @Github ：https://github.com/hzz12138
import torch
from torch import nn
from torchsummary import summary


class VGG13(nn.Module):
    def __init__(self):
        super(VGG13, self).__init__()
        self.stage_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.stage_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.stage_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.stage_4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        )
        self.stage_5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )

    def forward(self, input_lyr):
        stage_1 = self.stage_1(input_lyr)
        stage_2 = self.stage_2(stage_1)
        stage_3 = self.stage_3(stage_2)
        stage_4 = self.stage_4(stage_3)
        stage_5 = self.stage_5(stage_4)
        return [stage_1, stage_2, stage_3, stage_4, stage_5]


class DeepLabV1(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV1, self).__init__()
        self.backbone = VGG13()
        self.num_classes = num_classes
        self.stage_1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.stage_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=self.num_classes, kernel_size=3, stride=1, padding=1, bias=True)
        )

    def forward(self, input_lyr):
        backbone = self.backbone(input_lyr)[-1]
        stage_1 = self.stage_1(backbone)
        stage_1a = nn.functional.interpolate(input=stage_1, scale_factor=8, mode='bilinear')
        stage_2 = self.stage_2(stage_1a)
        return stage_2




if __name__ == '__main__':
    net = DeepLabV1(2)
    summary(net, input_size=(3, 224, 224), batch_size=1)
