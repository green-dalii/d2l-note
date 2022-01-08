# 21 - 经典卷积神经网络LeNet

---

### 🎦 本节课程视频地址 👉
[![Bilibil](https://i2.hdslb.com/bfs/archive/31bbec5db68b26ae5c86a5c7a6ac5b52f5751de1.jpg@640w_400h_100Q_1c.webp)](https://www.bilibili.com/video/BV1t44y1r7ct)
### MNIST

50000训练数据集；10000测试数据集，图像大小28x28，10类(0-9)。

![](\Images/LeNet.gif)
![](Images/Screenshot-from-2021-03-18-12-52-17.png)

**总结**

是早期成功的神经网络；先使用卷积层来学习图片的空间信息，然后使用全连接层来转换到类别空间。

**逻辑**
通道增加，高宽减少，最后做全连接输出，预测分类。

## 代码实现

**定义网络**

```
import torch
from torch import nn
from d2l import torch as d2l

class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)
        #Returns a new tensor with the same data as the self tensor 
        #but of a different shape
        #批量不变，通道唯一，
    
net = torch.nn.Sequential(
    Reshape(), nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))
```

**测试**

```
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)
    #调用layer的名字
    
```

**训练**

```
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, torch.nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
            #查看存储设备
        metric = d2l.Accumulator(2)
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
        return metric[0] / metric[1]

```
```
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    #默认取平均损失
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                           legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        # metric.data=[0.,0.,0.]
        net.train()
        for i, (X, y) in enumerate(train_iter):
            # X.shape=(256,1,28,28)
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            #返回三个标量，l*256,accuracy,256
            
            timer.stop()
            train_l = metric[0] / metric[2]
            #平均损失
            train_acc = metric[1] / metric[2]
            #正确率
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
                
```

```
lr, num_epochs = 0.9,a 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```
![](Images/屏幕截图%202021-12-24%20170823.png)