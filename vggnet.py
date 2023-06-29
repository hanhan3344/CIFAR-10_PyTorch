import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGbase(nn.Module):
    def __init__(self):
        super(VGGbase, self).__init__()

        # 3 * 28 * 28（crop后被resize）
        self.conv1 = nn.Sequential(
            # 输入通道数，输出通道数，卷积核大小，步长，padding
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),  # 输出通道数
            nn.ReLU()  # 使用relu激活函数来完成非线性变换，增加网络的表达能力
        )

        self.max_pooling1 = nn.MaxPool2d(
            kernel_size=2, stride=2)  # 进行过一次pooling后，图像channel数会翻倍

        # 14 * 14
        self.conv2_1 = nn.Sequential(
            # 进行过一次pooling后，图像channel数会翻倍64 -> 128
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=1,
                padding=1),
            # 输入通道数，输出通道数，卷积核大小，步长，padding
            nn.BatchNorm2d(128),  # 输出通道数
            nn.ReLU()  # 使用relu激活函数来完成非线性变换，增加网络的表达能力
        )

        self.conv2_2 = nn.Sequential(
            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                stride=1,
                padding=1),
            # 输入通道数，输出通道数，卷积核大小，步长，padding
            nn.BatchNorm2d(128),  # 输出通道数
            nn.ReLU()  # 使用relu激活函数来完成非线性变换，增加网络的表达能力
        )

        self.max_pooling2 = nn.MaxPool2d(
            kernel_size=2, stride=2)  # 进行过一次pooling后，图像channel数会翻倍

        # 7 * 7
        self.conv3_1 = nn.Sequential(
            # 进行过一次pooling后，图像channel数会翻倍128 -> 256
            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                stride=1,
                padding=1),
            # 输入通道数，输出通道数，卷积核大小，步长，padding
            nn.BatchNorm2d(256),  # 输出通道数
            nn.ReLU()  # 使用relu激活函数来完成非线性变换，增加网络的表达能力
        )

        self.conv3_2 = nn.Sequential(
            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                stride=1,
                padding=1),
            # 输入通道数，输出通道数，卷积核大小，步长，padding
            nn.BatchNorm2d(256),  # 输出通道数
            nn.ReLU()  # 使用relu激活函数来完成非线性变换，增加网络的表达能力
        )

        self.max_pooling3 = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=1)  # 进行过一次pooling后，图像channel数会翻倍

        # 4 * 4
        self.conv4_1 = nn.Sequential(
            # 进行过一次pooling后，图像channel数会翻倍256 -> 512
            nn.Conv2d(
                256,
                512,
                kernel_size=3,
                stride=1,
                padding=1),
            # 输入通道数，输出通道数，卷积核大小，步长，padding
            nn.BatchNorm2d(512),  # 输出通道数
            nn.ReLU()  # 使用relu激活函数来完成非线性变换，增加网络的表达能力
        )

        self.conv4_2 = nn.Sequential(
            nn.Conv2d(
                512,
                512,
                kernel_size=3,
                stride=1,
                padding=1),
            # 输入通道数，输出通道数，卷积核大小，步长，padding
            nn.BatchNorm2d(512),  # 输出通道数
            nn.ReLU()  # 使用relu激活函数来完成非线性变换，增加网络的表达能力
        )

        self.maxpooling4 = nn.MaxPool2d(
            kernel_size=2, stride=2)  # 进行过一次pooling后，图像channel数会翻倍

        # Linear的参数：
        #   输入维度：maxpooling4输出的特征图大小与通道数的乘积
        #   输出维度：10
        # batchsize * 512 * 2 * 2 -> batchsize * (512 * 2 * 2)
        self.fc = nn.Linear(512 * 4, 10)

    def forward(self, x):
        batchsize = x.size(0)

        out = self.conv1(x)
        out = self.max_pooling1(out)

        out = self.conv2_1(out)
        out = self.conv2_2(out)
        out = self.max_pooling2(out)

        out = self.conv3_1(out)
        out = self.conv3_2(out)
        out = self.max_pooling3(out)

        out = self.conv4_1(out)
        out = self.conv4_2(out)
        out = self.maxpooling4(out)

        # 在定义FC层前需要将tensor展平，即batchsize * (512 * 2 * 2)
        out = out.view(batchsize, -1)  # -1表示自动计算

        out = self.fc(out)
        out = F.log_softmax(out, dim=1)

        return out


def VGGNet():
    return VGGbase()
