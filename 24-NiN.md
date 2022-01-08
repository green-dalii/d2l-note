# 24 - 网络中的网络NiN

### 🎦 本节课程视频地址 👉[![Bilibil](https://i2.hdslb.com/bfs/archive/d20ed4243d80ed7d2c047bc421254b5ff2797c8e.jpg@640w_400h_100Q_1c.webp)](https://www.bilibili.com/video/BV1Uv411G71b)
## Net in Network

**全连接层的问题**

卷积层需要较少的参数
$$c_i\times c_o\times k^2$$

但卷积层后的第一个全连接层的参数

LeNet $16\times5\times5\times120=48k$
AlexNet $256\times5\times5\times4096=26M$
VGG $512\times7\times7\times4096=102M$

！！！！会引起过拟合的问题

### NiN

其的主旨就是取代去连接层

**NIN**

一个卷积层后跟两个全连接层（混合通道数）
- 步幅1，无填充，输出形状跟卷积层输出一样；
- 起到全连接层的作用

![](\Images/1_Oa-HQ4r0TJ7eMb0SLj8YvQ.png)

**NiN架构**

- 无全连接层
- 交替使用NiN块和步幅为2的最大池化层
  - 逐步减小高宽和增大通道数
  - 对每个像素增加了非线性性
- 最后使用全局平均池化层得到输出
  - 其输入通道数是类别数
  - 从每个通道拿出一个值，作为对其类比的预测，再求softmax。
- AlexNet中的全连接层
  - 参数少了，不容易发生过拟合

### 代码实现

```
#定义块
def nin_block(in_channels, out_channels, kernel_size,
             strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding), 
        nn.ReLU(), nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(), nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU())
#输入通道数，输出通道数，以及第一个卷积层的核参数
#ReLU增加了非线性性
```
```
#定义网咯
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2), nn.Dropout(0.5),
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    #根据(1, 1)的需求做平均池化，步长和填充都自动计算出来
    #这里相当于把每个通道全局做平均
    nn.Flatten()
    )
```
```
#测试
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape\t', X.shape)
```

