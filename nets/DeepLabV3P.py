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
    def __init__(self, block, block_num, num_classes=1000):
        super(ResNet_V1, self).__init__(block, block_num, num_classes)

    def forward(self, in_lyr):
        out_lyr = self.stage_1(in_lyr)
        low_level_lyr = self.stage_2(out_lyr)
        out_lyr = self.stage_3(low_level_lyr)
        out_lyr = self.stage_4(out_lyr)
        out_lyr = self.stage_5(out_lyr)
        out_lyr = self.stage_6(out_lyr)
        out_lyr = torch.flatten(out_lyr, 1)
        out_lyr = self.fc(out_lyr)
        return [low_level_lyr, out_lyr]


if __name__ == '__main__':
    resnet50 = ResNet_V1(block=BottleNeck, block_num=[3, 4, 6, 3], num_classes=1000).cuda()
    summary(resnet50, input_size=(3, 256, 256))

    test = torch.randn(size=(1,3,224,224)).cuda()
    result = resnet50(test)
    print(result[0].shape,result[1].shape)
