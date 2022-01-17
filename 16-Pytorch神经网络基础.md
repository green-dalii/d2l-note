# 16 - Pytorch神经网络基础

---

### 🎦 本节课程视频地址 👉
[![Bilibil](https://i0.hdslb.com/bfs/archive/9a09827a8220e688f6866c928f58f5a256788aab.jpg@640w_400h_100Q_1c.webp)](https://www.bilibili.com/video/BV1AK4y1P7vs)

## 层和块

**块**（block）可以描述单个层、由多个层组成的组件或整个模型本身。使用块进行抽象的一个好处是可以将一些块组合成更大的组件， 这一过程通常是递归的，如下图所示。 通过定义代码来按需生成任意复杂度的块， 我们可以**通过简洁的代码实现复杂的神经网络**。

![block](https://zh.d2l.ai/_images/blocks.svg)

从编程的角度来看，块由**类**（class）表示。 它的任何子类都必须定义一个将其输入转换为输出的前向传播函数， 并且必须存储任何必需的参数。 

### 使用Sequential实现层

`nn.Sequential`定义了一种特殊的Module，通过实例化nn.Sequential来构建我们的模型， 层的执行顺序是作为参数传递的。

```python
import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256,10))

X = torch.rand(2, 20)
net(X)
```

### 使用Block实现块

任何一个层、神经网络都可以看作Module的一个子类。

```python
class MLP(nn.Module):
    # 必须先使用父类的init初始化，接下来可以定义各层
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)
    # 必须重新定义前馈过程   
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
    
net = MLP()
net(X)
```

### 自定义Sequential实现

```python
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，`module`是`Module`子类的一个实例。我们把它保存在'Module'类的成员
            # 变量`_modules` 中。`module`的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X
    
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)
```

### 自定义Block实现

```python
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        #随机生成20*20不参与训练的权重
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)
        
    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1)   ##torch.mm()矩阵乘法，假设有一个偏移1
        X = self.linear(X)
        # 可以在前向计算中使用Python控制流来实现更复杂的过程
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
    
net = FixedHiddenMLP()
net(X)
```

> 在这个`FixedHiddenMLP`模型中，我们实现了一个隐藏层，
其权重（`self.rand_weight`）在实例化时被随机初始化，之后为常量。
这个权重不是一个模型参数，因此它永远不会被反向传播更新。
然后，神经网络将这个固定层的输出通过一个全连接层。
>
> 注意，在返回输出之前，模型做了一些不寻常的事情：
它运行了一个while循环，在$L_1$范数大于$1$的条件下，
将输出向量除以$2$，直到它满足条件为止。
最后，模型返回了`X`中所有项的和。
注意，此操作可能不会常用于在任何实际任务中，
我们只是向你展示如何将任意代码集成到神经网络计算的流程中。


### 混合Sequential和Block使用

```python
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

## 参数管理

### 初始化参数

> state_dict() #查看字典形式的数值

```python
print(net[2].state_dict())
#可以把Sequential看作一个list，可以切片拿出每一层的参数。
#是一个字典。
#module.state_dict().keys()=['weight','bias']
```

>nn.bias/.weight(.data/.grad)#直接查看参数

```python
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
# .data/.grad
```

> .named_parameters()#返回iterator，用于循环，返回(参数名, 参数数值)。

```python
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
# *代表把list/tuple里的元素分开，而非整个输出
```

>add_module(name, module)#在当前模块添加子模块，以（命名，模块）的方式

```python
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

```python
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

```python
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0] #默认取行
```

>参数联动

```python
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

```python
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

```python
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

```python
#存储一个tensor
X = torch.arange(4)
torch.save(X, 'x-file')

X2 = torch.load('x-file')
X2
```

```python
#存储高维度
y = torch.zeros(4)
torch.save([X, y], 'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
```

```python
#存储字典
mydict = {'x': X, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```

```python
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
