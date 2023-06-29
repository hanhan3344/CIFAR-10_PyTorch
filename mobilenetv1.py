import torch
import torch.nn as nn
import torch.nn.functional as F


class mobilenet(nn.Module):

    def conv_dw(self, in_channel, out_channel, stride):
        return nn.Sequential(
            nn.Conv2d(in_channel, in_channel,  # 第一个分组卷积
                      kernel_size=3, stride=stride, padding=1,
                      groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),

            nn.Conv2d(in_channel, out_channel,  # 点卷积 1 * 1
                      kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def __init__(self):
        super(mobilenet, self).__init__()
        self.conv1 = nn.Sequential(  # 第一个卷积层一般都使用标准卷积
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv_dw2 = self.conv_dw(32, 32, 1)
        self.conv_dw3 = self.conv_dw(32, 64, 2)

        self.conv_dw4 = self.conv_dw(64, 64, 1)
        self.conv_dw5 = self.conv_dw(64, 128, 2)

        self.conv_dw6 = self.conv_dw(128, 128, 1)
        self.conv_dw7 = self.conv_dw(128, 256, 2)

        self.conv_dw8 = self.conv_dw(256, 256, 1)
        self.conv_dw9 = self.conv_dw(256, 512, 2)  # 一共进行了16倍的下采样

        self.fc = nn.Linear(512, 10)

    def forward(self, x):

        out = self.conv1(x)
        out = self.conv_dw2(out)
        out = self.conv_dw3(out)
        out = self.conv_dw4(out)
        out = self.conv_dw5(out)
        out = self.conv_dw6(out)
        out = self.conv_dw7(out)
        out = self.conv_dw8(out)
        out = self.conv_dw9(out)
        out = F.avg_pool2d(out, 2)  # 所以这里采用 2 * 2 的平均池化
        out = out.view(-1, 512)  # 转为二维数据
        out = self.fc(out)  # 512维的特征向量转为10维的特征向量

        return out


def mobilenetv1_small():
    return mobilenet()
