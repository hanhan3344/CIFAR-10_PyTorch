import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel,
                      kernel_size=3, stride=stride, padding=1),  # 只下采样一次
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel)
        )

        self.shortcut = nn.Sequential()
        if in_channel != out_channel or stride > 1:  # 如果输入通道数和输出通道数不一致，或者stride大于1，需要进行跳连
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel,
                          kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out1 = self.layer(x)  # 先进行主干分支的计算
        out2 = self.shortcut(x)  # 再进行跳连分支的计算
        out = out1 + out2  # 最后将两个分支的结果相加
        out = F.relu(out)  # 最后再进行激活函数的计算

        return out


class ResNet(nn.Module):

    def make_layer(self, block, out_channel, stride, num_block):
        layers_list = []
        for i in range(num_block):
            if i == 0:
                in_stride = stride
            else:
                in_stride = 1
            layers_list.append(
                block(self.in_channel, out_channel, in_stride)
            )
            self.in_channel = out_channel

        return nn.Sequential(*layers_list)

    def __init__(self, ResBlock):
        super(ResNet, self).__init__()
        self.in_channel = 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # self.layer1 = ResBlock(in_channel=32, # 输入channel到输出channel的变化
        #                        out_channel=64,
        # stride=2) # 一般stride=2，输出channel数会变为输入channel数的2倍

        # self.layer2 = ResBlock(in_channel=64,
        #                        out_channel=64,
        #                        stride=1)

        # self.layer3 = ResBlock(in_channel=64,
        #                        out_channel=128,
        #                        stride=2)

        # self.layer4 = ResBlock(in_channel=128,
        #                        out_channel=128,
        #                        stride=2)

        self.layer1 = self.make_layer(ResBlock, 64, 2, 2)
        self.layer2 = self.make_layer(ResBlock, 128, 2, 2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, 2)
        self.layer4 = self.make_layer(ResBlock, 512, 2, 2)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, kernel_size=2)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # 没有softmax，因为交叉熵损失函数中已经包含了softmax
        return out


def resnet():
    return ResNet(ResBlock)
