# 23 - 使用块的网络VGG

---

### 🎦 本节课程视频地址 👉
[![Bilibil](https://i1.hdslb.com/bfs/archive/4fbdc632ed7cbf51e5097fc1c10c196887376775.jpg@640w_400h_100Q_1c.webp)](https://www.bilibili.com/video/BV1Ao4y117Pd)
## 使用块的网络——VGG(Visual Geometry Group)

A比L更深更大以期获得更好的精度，能不能更深更大？

——更多的全量接层（占内存，成本高）
——更多的卷积层（没有复制3x3的必要）
——将卷积层组合成快√

**VGG块**

3x3卷积层，填充1（n层，m通道，输入通道等于输出通道）
2x2最大池化层（步幅2）

实践证明，更多的3x3好于少儿大的5x5

多个VGG块后接全连接层

不同次数的重复快得到不同的架构，VGG-16，VGG-19。

![](\Images/Overall-architecture-of-the-Visual-Geometry-Group-16-VGG-16-model-VGG-16-comprises.png)

有点儿像是**更大更深**的AlexNet。

![](\Images/Comparison-of-popular-CNN-architectures-The-vertical-axis-shows-top-1-accuracy-on.png)

**总结**

VGG使用可重复使用的卷积块来构建深度卷积神经网络；
不同的卷积块和超参数可以得到不同复杂度的变种；

## 代码实现

```
import torch
from torch import nn
from d2l import torch as d2l
#定义块
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
#分别是卷积层数输出通道数,VGG11
#经典设计，高宽减半，通道数翻一倍

#定义模型
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    #在函数的顺序结构里被改变了
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(
            num_convs, in_channels, out_channels))
        in_channels = out_channels
    
    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(),
        nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(),
        nn.Dropout(0.5), nn.Linear(4096, 10))

net = vgg(conv_arch)

#测试

X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__, 'output shape:\t', X.shape)

#训练
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
#通道数减半
net = vgg(small_conv_arch)
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```