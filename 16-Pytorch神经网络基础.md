# 16 - Pytorch神经网络基础

### 🎦 本节课程视频地址 👉[![Bilibil](https://i0.hdslb.com/bfs/archive/9a09827a8220e688f6866c928f58f5a256788aab.jpg@640w_400h_100Q_1c.webp)](https://www.bilibili.com/video/BV1AK4y1P7vs)

### 层和块

**块**（block）可以描述单个层、由多个层组成的组件或整个模型本身。使用块进行抽象的一个好处是可以将一些块组合成更大的组件， 这一过程通常是递归的。 通过定义代码来按需生成任意复杂度的块， 我们可以**通过简洁的代码实现复杂的神经网络**。

从编程的角度来看，块由**类**（class）表示。 它的任何子类都必须定义一个将其输入转换为输出的前向传播函数， 并且必须存储任何必需的参数。 

从多层感知机入手：
```
import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256,10))

X = torch.rand(2, 20)
net(X)

## 利用Sequential定义了一个特殊的Module
```
任何一个层、神经网络都可以看作Module的一个子类。

我们通过实例化nn.Sequential来构建我们的模型， 层的执行顺序是作为参数传递的。

也可以自定义块：
```
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)
        
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
    #此处必须是forward,相当于对nn.Module里的__call__()下的forward重定义；
```
还可以自己实现Sequential:
```
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            #把args的参数传入作为self.modules这个有序字典的键-值对
            self._modules[block] = block
    
    def forward(self, X):
        for block in self._modules.values():
            #依次调用有序字典里的模块
            X = block(X)
        return X
    
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)
```
可以自定义块来使算法更灵活：
```
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        #随机生成20*20不参与训练的权重
        self.linear = nn.Linear(20, 20)
        
    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        ##torch.mm()是矩阵乘法，假设有一个偏移
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /= 2
            return X.sum()
    
net = FixedHiddenMLP()
net(X)
## 通过继承nn.Module可以更灵活地定义模型
```
块之间可以嵌套：
```
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)
        
    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(X)
```

综上，块可以理解为能够实现一个或多个层的类，通过定义类的实例化来完成神经网络的运算。

### 初始化参数
> state_dict() #查看字典形式的数值

```
print(net[2].state_dict())
#可以把Sequential看作一个list，可以切片拿出每一层的参数。
#是一个字典。
#module.state_dict().keys()=['weight','bias']
```

>nn.bias/.weight(.data/.grad)#直接查看参数
```
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
# .data/.grad
```

> .named_parameters()#返回iterator，用于循环，返回(参数名, 参数数值)。
```
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
# *代表把list/tuple里的元素分开，而非整个输出
```
>add_module(name, module)#在当前模块添加子模块，以（命名，模块）的方式
```
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}', block1())
        #嵌套四个block1
        #add_module(name, module)
        #The module can be accessed as an attribute using the given name.
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
```
> nn.init.normal_/zeros_/constant_/uniform_ #初始化参数
> nn.apply() #对模块应用方法

```
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        # _表示直接替换掉m.weight，而非返回值，或者说直接赋值
        nn.init.zeros_(m.bias)
    
net.apply(init_normal)
#相当于一个for loop
net[0].weight.data[0], net[0].bias.data[0]
--------------------------------------
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)

net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
--------------------------------------
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        #uniform distribution
-----------------------------------   
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)
        #nn.init函数设置模块初始值

net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```
>简单粗暴的直接赋值方式
```
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0] #默认取行
```
>参数联动
```
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), shared, nn.ReLU(), shared,
                   nn.ReLU(), nn.Linear(8, 1))
net(X)
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] == 100
# 会同时修改两个shared, 相当于同一个实例的赋值
print(net[2].weight.data[0] == net[4].weight.data[0])
```

### 自定义层
```
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, X):
        return X - X.mean()

layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))

net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())

Y = net(torch.rand(4, 8))
Y.mean()
```
> nn.Parameter(tensor, required_grad=True) #把传入张量当作模块参数，可以对其求导的
```
#自定义带参数的层
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
        #理论上torch.randn(units,)与torch.randn(units)没有区别
        #逗号后省略表示维度只有1
        #如果是randn(2, 1)，就是一个二维张量了。
        
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data） + self.bias.data
        return F.relu(linear)
    
dense = MyLinear(5, 3)
dense.weight
```
### 读写文件

>torch.save(parameter, 'filename')
>torch.load('filename')

```
#存储一个tensor
X = torch.arange(4)
torch.save(X, 'x-file')

X2 = torch.load('x-file')
X2
```
```
#存储高维度
y = torch.zeros(4)
torch.save([X, y], 'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
```
```
#存储字典
mydict = {'x': X, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```
```
#存储模型参数
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)
    
    def forward(self, X):
        return self.output(F.relu(self.hidden(X)))
    
net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

torch.save(net.state_dict(), 'mlp.params') #存储的实际是模型参数而非模型本身

clone = MLP()
#先克隆原模型本身
clone.load_state_dict(torch.load('mlp.params'))
#再载入并重写克隆的模型
clone.eval()
#eval()设置一个模型参数为不可导，并返回模型本身
Y_clone = clone(X)
Y_clone == Y  ===>  True
```