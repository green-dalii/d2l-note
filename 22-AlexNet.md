# 22 - 深度卷积神经网络AlexNet

---

### 🎦 本节课程视频地址 👉
[![Bilibil](https://i2.hdslb.com/bfs/archive/d3ac6a33084e673003dfd4f16685419e891d1bc9.jpg@640w_400h_100Q_1c.webp)](https://www.bilibili.com/video/BV1h54y1L7oe)
### deep learning 发展历程

2000~ 核方法：有一套完整的数学模型，SVM

2000~ 几何学：把计算机视觉的问题描述成几何问题

2010~ 特征工程：如何抽取图片的特征，SIFT

**核心是数据**

Eg: ImageNet(2010)

自然物体的彩色图: 469X387;
样本数：1.2M
类数：1000

### AlexNet

赢了2012年的ImageNet竞赛；
更深更大的LeNet；
主要改进：丢弃法、ReLU、MaxPooling
计算机视觉方法论的改变：从人工提取特征（SVM）到自动获得特征（CNN），分类器和特征提取同时训练；并且构造CNN简单高效——从原始数据（字符串、像素）到最终学习结果。

**基本架构**
![](\Images/1_3B8iO-se13vA2QfZ4OBRSw.png)

![](\Images/1_bD_DMBtKwveuzIkQTwjKQQ.png)

$$
\begin{array}{l}
Input:X=(3,224,224)\\
Conv1:kernel=11\times11,stride=4\rightarrow X=(96,55,55)\\
MaxPool1:kernel=3\times3,stride=2\rightarrow X=(256,27,27)\\
Conv2:kernel=5\times5,stride=2\rightarrow X=(384,13,13)\\
MaxPool2:kernel=3\times3,stride=2\rightarrow X=(384,13,13)\\
Conv3:3\times3\\
Conv4:3\times3\\
Conv5:3\times3\\
MaxPool3:3\times3\\
Dense1:4096\\
Dense2:4096\\
Dense3:1000
\end{array}
$$

**Details**

- 从Sigmoid(LeNet)变到了ReLU（减缓梯度消失）；
- 在全连接层后加入了丢弃层(dropout)；
- 数据增强

**总结**

AlexNet是更大更深的LeNet，10x参数个数，260x计算复杂度；

新进入了丢弃法、LeRU、最大池化层和数据增强；

AlexNet当赢下了2012ImageNet竞赛后，标志着新的一轮神经网络热潮的开始。

### 代码实现

```
import torch
from torch import nn
from d2l import torch as d2l
#网络
net = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
    nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    #Dropout(p=0.5)==>50%输出为零，剩下的乘2
    nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(4096,10)
)
```
```
#测试
X = torch.randn(1, 1, 224, 224)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'Output shape:\t', X.shape)
```
```
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
#我们将它们从原来的28x28增加到224x224
#事实上不可取的，因为没有增大信息量
lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```