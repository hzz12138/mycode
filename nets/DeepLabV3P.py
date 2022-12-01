# -*- coding: utf-8 -*-
# @Time    : 2022/11/29 14:20
# @Author  : Zph
# @Email   : hhs_zph@mail.imu.edu.cn
# @File    : DeepLabV3P.py
# @Software: PyCharm

"""
    构建DeepLabV3+网络
"""

import torch
from torch import nn
from torchsummary import summary
from nets.ResNet import ResNet
from nets.ResNet import BottleNeck


class ResNet_V1(ResNet):
    def __init__(self, block, block_num):
        super(ResNet_V1, self).__init__(block, block_num)

    def forward(self, in_lyr):
        out_lyr = self.stage_1(in_lyr)
        low_level_lyr = self.stage_2(out_lyr)
        out_lyr = self.stage_3(low_level_lyr)
        out_lyr = self.stage_4(out_lyr)
        out_lyr = self.stage_5(out_lyr)
        # out_lyr = self.stage_6(out_lyr)
        return [low_level_lyr, out_lyr]


class ASPPConv(nn.Sequential):
    def __init__(self, in_channel, out_channel, dilation):
        modules = [
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channel, out_channel):
        # super(ASPPPooling, self).__init__()
        # modules = [
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(out_channel),
        #     nn.ReLU(inplace=True)
        # ]
        # self.stage = nn.Sequential(*modules)
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, in_lyr):
        size = in_lyr.shape[-2:] # 获取原始的特征图大小

        out_lyr = super(ASPPPooling, self).forward(in_lyr)
        # out_lyr = in_lyr
        # for module in self:
        #     out_lyr = module(out_lyr)
        out_lyr = nn.functional.interpolate(out_lyr, size=size, mode='bilinear', align_corners=False)
        return out_lyr


class ASPP(nn.Module):
    def __init__(self, in_channel, out_channel, atrous_rates):
        super(ASPP, self).__init__()
        modules = []
        # 1*1卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=1,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        ))
        # 多尺度空洞卷积
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channel,out_channel,rate))
        # 全局平均池化
        modules.append(ASPPPooling(in_channel, out_channel))
        # 所有的并行处理
        self.stages = nn.ModuleList(modules)

        # 所有层的后处理
        self.final_stage = nn.Sequential(
            nn.Conv2d(len(self.stages)*out_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, in_lyr):
        res = []
        res.append(self.stages[0](in_lyr))
        res.append(self.stages[1](in_lyr))
        res.append(self.stages[2](in_lyr))
        res.append(self.stages[3](in_lyr))
        res.append(self.stages[4](in_lyr))

        # for stage in self.stages:
        #     res.append(stage(in_lyr))
        res = torch.cat(res, dim=1)
        return self.final_stage(res)


class DeepLabV3P(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(DeepLabV3P, self).__init__()
        self.in_channel = in_channel
        # # 调整通道数
        # self.first_conv = nn.Conv2d(in_channels=in_channel, out_channels=3, kernel_size=1, padding=0)
        # 骨干网络
        self.backbone = ResNet_V1(block=BottleNeck, block_num=[3, 4, 6, 3])
        low_level_channel = 256

        # 低维信息处理
        self.low_process = nn.Sequential(
            nn.Conv2d(in_channels=low_level_channel, out_channels=48, kernel_size=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # 高维信息处理
        # 空洞空间金字塔池化
        self.aspp = ASPP(in_channel=2048, out_channel=256, atrous_rates=[6, 12, 18])
        self.upsample = nn.functional.interpolate
        self.final_stage = nn.Sequential(
            nn.Conv2d(in_channels= 304, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
        )



    def forward(self,in_lyr):
        # if self.in_channel == 4:
        #     in_lyr = self.first_conv(in_lyr)
        out_lyr = self.backbone(in_lyr)
        # 低维信息处理
        low_level_lyr = self.low_process(out_lyr[0])
        # 高维信息处理
        out_lyr = self.aspp(out_lyr[1])
        out_lyr = self.upsample(out_lyr, size=(low_level_lyr.size(2), low_level_lyr.size(3)), mode='bilinear', align_corners=True)
        out_lyr = torch.cat((low_level_lyr, out_lyr), dim=1)
        out_lyr = self.final_stage(out_lyr)
        out_lyr = self.upsample(out_lyr, size=(in_lyr.size(2), in_lyr.size(3)), mode='bilinear', align_corners=True)

        return out_lyr



if __name__ == '__main__':

    # asppp = ASPP(2048,256,[6,12,18]).cuda()
    # summary(asppp, input_size=(2048, 7, 7))

    # resnet50 = ResNet_V1(block=BottleNeck, block_num=[3,4,6,3]).cuda()
    # test = torch.randn(size=(1, 3, 256, 256)).cuda()
    # result = resnet50(test)
    # print(result[0].shape,result[1].shape)
    #
    # summary(resnet50, input_size=(3, 256, 256))

    deep = DeepLabV3P(3,2)
    test = torch.randn(size=(2, 3, 224, 224))
    result = deep(test)
    print(result.shape)
