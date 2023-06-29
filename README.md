# CIFAR-10 with PyTorch

## 介绍

以www.bilibili.com/video/BV1334y1H7dX/为参考，采用Pytorch完成Cifar-10图像分类

## 安装教程

1. 安装python环境，

   ```shell
   pip install -r requirements.txt
   ```

## 使用说明

1. 下载并解压[CIFAR-10 python version](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)数据集

   ```shell
   wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
   tar -xvf cifar-10-python.tar.gz
   ```

2. 运行 [readcifar10.py](readcifar10.py) ，该操作会在当前目录下生成TRAIN和TEST两个文件夹，其中保存的是提取出的训练集和测试集图片

3. 训练脚本 `train.py`

4. 测试脚本`test.py`

## 参考

1.  https://www.bilibili.com/video/BV1334y1H7dX/ 

2.  https://www.cs.toronto.edu/~kriz/cifar.html
