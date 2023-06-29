import torch
import torch.nn as nn
import torchvision
import os
import tensorboardX
from vggnet import VGGNet
from resnet import resnet
from mobilenetv1 import mobilenetv1_small
from inceptionModule import inceptionnetsmall
from pre_resnet import pytorch_resnet18
from load_cifar10 import train_loader, test_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device is ", device)

batch_size = 128
epoch_num = 200
lr = 0.01

# net = VGGNet().to(device)
net = resnet().to(device)

# loss
# 多分类问题，使用交叉熵损失函数
loss_func = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=5,
                                            gamma=0.9)  # 每经过5个epoch，学习率乘以0.9

model_path = "models/resnet"
log_path = "logs/resnet"
if not os.path.exists(log_path):
    os.makedirs(log_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)
writer = tensorboardX.SummaryWriter(log_path)

step_n = 0

for epoch in range(epoch_num):

    # Batch Normalization：在训练模式下，Batch Normalization
    # 会使用当前小批量数据的均值和方差进行归一化，并在反向传播过程中更新其统计信息（均值和方差）。这是为了确保每个批次数据的归一化效果和统计一致性。训练模式下的 Batch Normalization 与测试模式下的 Batch
    # Normalization 是不同的。

    # Dropout：在训练模式下，Dropout
    # 会按照指定的概率随机失活一部分神经元。这有助于防止过拟合，促使网络学习到更鲁棒的特征表示。在测试模式下，Dropout
    # 不起作用，所有神经元的输出都被保留。

    for i, data in enumerate(train_loader):
        net.train()  # train BN dropout

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print("step: ", i, "loss: ", loss.item()) # mini-batch
        # loss，不是整个epoch的loss，每个epoch的loss对应所有的样本

        _, pred = torch.max(outputs.data, dim=1)

        correct = pred.eq(labels.data).cpu().sum()

        # print("epoch: ", epoch)
        # print("train lr is ", optimizer.state_dict()['param_groups'][0]['lr'])
        # print("train step: ", i, "loss: ", loss.item(),
        #       "mini-batch correct is: ", 100.0 * correct / batch_size, "%")
        writer.add_scalar("train loss",
                          loss.item(),
                          global_step=step_n)
        writer.add_scalar("train correct",
                          100.0 * correct.item() / batch_size,
                          global_step=step_n)

        im = torchvision.utils.make_grid(inputs)
        writer.add_image("train image", im, global_step=step_n)

        step_n += 1

    torch.save(net.state_dict(), "{}/{}.pth".format(model_path, epoch + 1))
    scheduler.step()

    sum_loss = 0.0
    sum_correct = 0

    for i, data in enumerate(test_loader):
        net.eval()  # eval BN dropout

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        _, pred = torch.max(outputs.data, dim=1)
        correct = pred.eq(labels.data).cpu().sum()

        sum_loss += loss.item()
        sum_correct += correct

        im = torchvision.utils.make_grid(inputs)
        writer.add_image("test image", im, global_step=step_n)

    test_loss = sum_loss / len(test_loader)
    test_correct = sum_correct * 100.0 / len(test_loader) / batch_size

    writer.add_scalar("test loss",
                      test_loss,
                      global_step=epoch + 1)
    writer.add_scalar("test correct",
                      test_correct.item(),
                      global_step=epoch + 1)

    print("epoch is ", epoch + 1, "test loss is ", test_loss,
          "test correct is ", test_correct, "%")

writer.close()
