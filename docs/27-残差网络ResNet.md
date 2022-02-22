# 27 - 残差网络 ResNet

---

### 🎦 本节课程视频地址 👇

[![Bilibil](https://i2.hdslb.com/bfs/archive/300fb344d7e0f1fb18e169c9ed3ecb7af8841143.jpg@640w_400h_100Q_1c.webp)](https://www.bilibili.com/video/BV1bV41177ap)

## 问题引出：加更多的层总是改善精度吗？

以下图示例来说，对于非嵌套函数（non-nested function）类，较复杂的函数类并不总是向“真”函数 $f^∗$ 靠拢（区域大小代表模型复杂度，复杂度由 $\mathcal{F1}$ 向 $\mathcal{F6}$ 递增）。 在下图左边，虽然 $\mathcal{F3}$ 比 $\mathcal{F1}$ 更接近 $f^∗$ ，但 $\mathcal{F6}$ 却离的更远了。 相反对于下图右侧的嵌套函数（nested function）类 $\mathcal{F1}\subseteq…\subseteq \mathcal{F6}$ ，我们可以避免上述问题。

![functionclasses](https://zh.d2l.ai/_images/functionclasses.svg)

因此，只有**当较复杂的函数类包含较小的函数类时**，我们才能确保提高它们的性能（相当于在原有区域逐渐增大覆盖面积来逼近最优解）。 对于深度神经网络，如果我们能将新添加的层训练成**恒等映射**（identity function）$f(x)=x$，新模型和原模型将同样有效。 同时，由于新模型可能得出更优的解来拟合训练数据集，因此添加层似乎更容易降低训练误差。

## 残差块（Residual blocks）

假设我们的原始输入为 $x$ ，而希望学出的理想映射为 $f(x)$ （作为下图上方激活函数的输入）。 下图虚线框中的部分需要直接拟合出该映射 $f(x)$ ，而右图虚线框中的部分则需要拟合出残差映射 $f(x)−x$ 。 残差映射在现实中往往更容易优化。 以本节开头提到的恒等映射作为我们希望学出的理想映射 $f(x)$ ，我们只需将下图右侧虚线框内上方的加权运算（如仿射）的权重和偏置参数设成$0$，那么 $f(x)$ 即为恒等映射。 实际中，当理想映射 $f(x)$ 极接近于恒等映射时，残差映射也易于**捕捉恒等映射的细微波动**。 右图是ResNet的基础架构–残差块（residual block）。 在残差块中，输入可通过跨层数据线路更快地向前传播。

![residual-block](https://zh.d2l.ai/_images/residual-block.svg)

- 串联一个层改变函数类，我们希望能扩大函数类。

- 残差块加入快速通道（右边）来得到$f(x)=x+g(x)$

- 相当于在后面复杂网络嵌入了前面的简单网络。

### 残差块细节

残差块有两种实现方式， 一种是当`use_1x1conv=False`时，应用ReLU非线性函数之前，将输入添加到输出。 另一种是当`use_1x1conv=True`时，添加通过 $1×1$ 卷积调整通道和分辨率。

![2-resnet-blocks](https://zh.d2l.ai/_images/resnet-block.svg)

## ResNet网络结构

ResNet的前两层跟之前介绍的GoogLeNet中的一样： 在输出通道数为64、步幅为2的 $7×7$ 卷积层后，接步幅为2的 $3×3$ 的最大汇聚层。 不同之处在于ResNet每个卷积层后增加了批量规范化层。

![resnet_arch](Images/resnet_arch.png)

每个模块有4个卷积层（不包括恒等映射的 $1×1$ 卷积层）。 加上第一个 $7×7$ 卷积层和最后一个全连接层，共有18层。 因此，这种模型通常被称为ResNet-18。 通过配置不同的通道数和模块里的残差块数可以得到不同的ResNet模型，例如更深的含152层的ResNet-152。

![resnet18](https://zh.d2l.ai/_images/resnet18.svg)

- 高宽减半 ResNet 块(stride=2)
- 后接多个高宽不变的 ResNet
  - 用 1x1Conv skip 可以改变输出通道匹配 ResNet
- 类似于 VGG 和 GooleNet 的总体架构
  - 一般是 5 个 Stage
  - $7×7$ Conv + BN + $3×3$ MaxPooling
  - 每一个 Stage 的具体框架很灵活
- 但替换成了 ResNet 块

## 总结

- 残差块使得很深的网络更加容易训练
  - 甚至可以训练一千层的网络
- 残差网络对随后的深层神经网络设计产生了深远影响，无论是卷积累网络还是全连接类网络

## 代码实现

- 定义Residual class

```python
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        #kernel_size=3, padding=1，输出高宽不变
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        #default_stride=1，同上输出高宽不变
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        # 如果指定了旁路卷积
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        # self.relu = nn.ReLU(inplace=True)
    
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
```

- 测试

```python
blk = Residual(3, 3)
X = torch.rand(4, 3, 6, 6)
Y = blk(X)
Y.shape
# Out:
# torch.Size([4, 3, 6, 6])

blk = Residual(3, 6, use_1x1conv=True, strides=2)
blk(X).shape
# Out:
# torch.Size([4, 6, 3, 3])
```

- 定义Residual block

```python
#照搬GoogleNet b1
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            #第一块高宽减半
            blk.append(
                Residual(input_channels, num_channels, use_1x1conv=True,
                         strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
            #高宽不变
    return blk
```

- 构建Stage

```python
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))
```

- 测试网络输出

```python
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
# Out:
# Sequential output shape:     torch.Size([1, 64, 56, 56])
# Sequential output shape:     torch.Size([1, 64, 56, 56])
# Sequential output shape:     torch.Size([1, 128, 28, 28])
# Sequential output shape:     torch.Size([1, 256, 14, 14])
# Sequential output shape:     torch.Size([1, 512, 7, 7])
# AdaptiveAvgPool2d output shape:      torch.Size([1, 512, 1, 1])
# Flatten output shape:        torch.Size([1, 512])
# Linear output shape:         torch.Size([1, 10])
```

- 模型训练

```python
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
# Out:
# loss 0.009, train acc 0.998, test acc 0.922
# 4702.7 examples/sec on cuda:0
```

![output_resnet](https://zh.d2l.ai/_images/output_resnet_46beba_102_1.svg)

> 通过与之前模型结果的对比，可以看出ResNet得益于残差设计，使得梯度传播更快，模型收敛更快、训练精度更高，也就是模型特征提取能力更强，速度也更快。（比Alexnet稍快、比VGG快将近100%、比NiN快将近50%、比GoogLeNet快将近35%）

## Q&A🤓

**Q：为什么$f(x)=x+g(x)$就能保证结果至少不会变差？假如$g(x)$变得更差呢？**

**🙋‍♂️**：在神经网络训练中，如果反向传播时算法发现$g(x)$对模型`loss`损失函数没有贡献（或者有负贡献），就会逐渐将$g(x)$的梯度置零（或者反方向降低$g(x)$的影响直到权重为零），最后$g(x)$就得不到梯度更新，网络最终结果$f(x)$也会忽略$g(x)$而向$x$靠近。

**Q：是不是训练精度总是会比测试精度高？**

**🙋‍♂️**：也不一定，在许多有Data Argument（数据增强）的任务中，比如图片识别，测试精度是可能高于训练精度（因为训练图片有添加噪声等干扰，而测试图片没有）。
