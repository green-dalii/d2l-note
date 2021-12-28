### 训练集、验证集、测试集

我们将数据划分训练集、验证集和测试集。在训练集上训练模型，在验证集上评估模型，一旦找到的最佳的参数，就在测试集上最后测试一次，测试集上的误差作为泛化误差的近似。

**引入包&下载数据**
```
import collections
import math
import os
import shutil
# 倒腾文件
import pandas as pd
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

d2l.DATA_HUB['cifar10_tiny'] = (d2l.DATA_URL + 'kaggle_cifar10_tiny.zip',
                                '2068874e4b9a9f0fb07ebe0ad2b29754449ccacd')
#此数据集是d2l course存在Amazon的demo而非完整的来自于kaggle的cifar-10

demo = True
#判断是下载完整数据集还是小样

if demo:
    data_dir = d2l.download_extract('cifar10_tiny')
else:
    data_dir = '../code/data/cifar-10/'
```
**划分数据**

```
def read_csv_labels(fname):
    #labels存在csv文件里
    with open(fname, 'r') as f:
        lines = f.readlines()[1:]
        #第一行是抬头
        #readlines()读取整个文件，以行读取。
    tokens = [l.rstrip().split(',') for l in lines]
    #strip()删除字符串头尾特定的字符，默认是空格和换行符
    #rstrip()在此删去右侧换行符
    #split()中间按照逗号分隔
    return dict(((name, label) for name, label in tokens))
    #把[]换成()的列表生成式，就是一个generator
    #可以用list/tuple(generator)方式转换
    #如果是成对元素，就可以dict(generator)转换成字典，相当于dict(list)的方式定义字典
    #以上的方式都会遍历generator
    #当同一个generator被遍历过一遍时，再次调用会返回None
    #比如生成器g,先list(g)，会得到所有元素
    #再tuple(g),得到空元组，所以需要重新定义g

labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))

def copyfile(filename, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    #exist_ok(default)=False，如果目标文件夹已存在会报错
    #对于已存在的文件夹不会覆盖
    shutil.copy(filename, target_dir)
    #copy文件到目录

def reorg_train_valid(data_dir, labels, valid_ratio):
    n = collections.Counter(labels.values()).most_common()[-1][1]
    #collections是高性能容量数据类型
    #返回的是一个类的实例
    #元素以字典形式保存
    #most_common(num)返回最频繁的类型和频率，如果不指定个数，default所有类从多到少排列
    #取最少的类的元素数
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    print(n_valid_per_label)
    #math.floor向下取整
    #n * valid_ratio决定了一个上限
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        #os.listdir()返回指定目录下的文件和文件夹列表
        #train文件夹下是解压后的一张张图片，即train_file
        label = labels[train_file.split('.')[0]]
        #分离文件（图片）名与后缀，只提取文件名，也就是图片的序号
        #再在lables这个字典里拿出对应标号的类型名
        #print(labels[train_file.split('.')[0]])='ship'
        fname = os.path.join(data_dir, 'train', train_file)
        #把文件路径赋值给fname
        copyfile(
            fname,
            os.path.join(data_dir, 'train_valid_test', 'train_valid', label))
        #把文件复制到训练验证文件夹
        if label not in label_count or label_count[label] < n_valid_per_label:
             #创建验证集
            #如果label不在dict_label_count里或者值小于n_valid_per_label
            #也就是说如果该类型的数目没有达到上限
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'valid', label))
            #调用前面定义的copyfile()函数
            #先创建../data/kaggle_cifar10_tiny/train_valid_test/valid/labels[train_file.split('.')[0]]文件夹
            #第一个就是../train_valid_test/valid/ship文件夹
            #再把图片放进去
            label_count[label] = label_count.get(label, 0) + 1
            #用字典计数
            #dict.get(key, default=None),查找类型，如果没有则返回默认值
            #借此赋值dict[key]
            #通过+1计数，并重新赋值
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train', label))
            #创建训练集
    return n_valid_per_label
    #返回测试集的大小
#整体是把文件全部复制到train_valid并且分为train和valid，每个下面再以类名建立文件夹细分

def reorg_test(data_dir):
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(
            os.path.join(data_dir, 'test', test_file),
            os.path.join(data_dir, 'train_valid_test','test','unkonwn'))
        #创建测试集，并且只有一个unknown文件夹，表示所有类型均为不可知

def reorg_cifar10_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)
#调用上方三个函数

batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_cifar10_data(data_dir, valid_ratio)
```

**数据加载&预处理**

```
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(40),
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                                ratio=(1.0, 1.)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])
#注意Compose([])传入的是一个列表，因为没有[]造成传入多个参数会报错

train_ds, train_valid_ds = [
    torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=transform_train) for folder in ['train', 'train_valid']]
#ImageFolder是一个通用的数据集加载器，也就是对自定义的图片文件夹
#传入的是根目录地址，也就是各类别目录的上一级地址
#把data_dir/rain_valid_test/train作为训练集
#把data_dir/rain_valid_test/train_valid作为训练验证集
valid_ds, test_ds = [
    torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=transform_test) for folder in ['valid', 'test']]
#把data_dir/rain_valid_test/valid作为验证集
#把data_dir/rain_valid_test/test/unknown作为测试集
```

```
train_ds, train_valid_ds = [
    torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=transform_train) for folder in ['train', 'train_valid']]
#ImageFolder是一个通用的数据集加载器，也就是对自定义的图片文件夹
#传入的是根目录地址，也就是各类别目录的上一级地址
#把data_dir/rain_valid_test/train作为训练集
#把data_dir/rain_valid_test/train_valid作为训练验证集
valid_ds, test_ds = [
    torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=transform_test) for folder in ['valid', 'test']]
#把data_dir/rain_valid_test/valid作为验证集
#把data_dir/rain_valid_test/test/unknown作为测试集
```
```
train_iter, train_valid_iter = [
    torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]
#drop_last=True表示裁去最后不满batch_size的一批数据，default=False
#为实现随机SGD，必须shuffle=True，否则数据集顺序也会被误学
valid_iter = torch.utils.data.DataLoader(
    valid_ds, batch_size, shuffle=False, drop_last=True)

test_iter =  torch.utils.data.DataLoader(test_ds, batch_size, shuffle=True,
                                         drop_last=False)
```
**定义模型**
```
def get_net():
    num_classes = 10
    net = d2l.resnet18(num_classes, 3)
    return net

loss = nn.CrossEntropyLoss(reduction='none')
```
```
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices,
          lr_period, lr_decay):
    #随机梯度优化收敛的前提是lr的逐渐减小，减少抖动
    #每隔(lr_period)个epoch,lr减少(lr_decay)
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                              weight_decay=wd)
    #momentum使用指数加权平均之后的梯度代替原梯度进行参数更新。
    #因为每个指数加权平均后的梯度含有之前梯度的信息，动量梯度下降法因此得名。
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    #学习率下降率每lr_period*epoch, lr = lr * lr_decay
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3) # loss, accuracy, numel
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0])
            #train_iter返回features和labels,labels是一维向量，shape[0]就是一个标量
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
            #if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (metric[0] / metric[2], metric[1] / metric[2], None))
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
        scheduler.step()
        #每个epoch后scheduler需要更新一下
    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    #两句f''之间不用逗号，就是同一个字符串
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```