## ResNet

**Problem**：加更多的层总是改善精度吗？

随着模型复杂度的增加，并不总能够距离最优解更近，因此，如果新模型的作用域能够包含当前模型，就能保证这一点。

### 残差块(Residual blocks)

- 串联一个层改变函数类，我们希望能扩大函数类。

- 残差块加入快速通道（右边）来得到$f(x)=x+g(x)$

![](\Images/Res%20Block.png)

**程序框架**

![](\Images/resnet_arch.png)

- 高宽减半ResNet块(stride=2)
- 后接多个高宽不变的ResNet
  - 用1x1Conv skip可以改变输出通道匹配ResNet
- 类似于VGG和GooleNet的总体架构
  - 一般是5个stage
  - 7x7Conv,BN, 3x3MaxPool
  - 每一个stage的具体框架很灵活
  - 几成当前标配
- 但替换成了ResNet块
  
###  总结

- 残差块使得很深的网络更加容易训练
  - 甚至可以训练一千层的网络
- 残差网络对随后的深层神经网络设计产生了深远影响，无论是卷积累网络还是全连接类网络

### 代码实现

```import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
#定义Residual class
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        #高宽不变
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        #default_stride=1
        #高宽不变
     
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        #self.relu = nn.ReLU(inplace=True)
    def forward(self, X):
        #这里写错了forward，报了一个NonImplementedError
        #所以在定义函数时的笔误，不会追溯具体位置。
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
```
```
#测试
blk = Residual(3, 3)
X = torch.rand(4, 3, 6, 6)
Y = blk(X)
Y.shape

blk = Residual(3, 6, use_1x1conv=True, strides=2)
blk(X).shape
```
```
#定义Residual block
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
#照搬GoogleNet b1
#出现stride=2就可以视为高宽减半
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

### ResNet补充

为什么能训练出上千层的模型（解决了梯度消失）。

$$y=f(x)\quad {\partial y\over\partial w}\quad w=w-lr{\partial y\over\partial w}$$

$$y\prime=g(f(x))\quad{\partial y\prime\over\partial w}={\partial y\prime\over\partial y}{\partial y\over\partial w}$$

因为梯度传递，小数相乘会导致梯度消失，但对于一个残差网络：

$$y\prime\prime=f(x)+g(f(x))\quad {\partial y\prime\prime\over\partial w}={\partial y\over\partial w}+{\partial y\prime\over\partial w}$$

保证了下面的层也可以拿到较大的梯度。



