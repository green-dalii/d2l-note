# 09 - Softmax回归
### 🎦 本节课程视频地址 👉[Bilibil](https://www.bilibili.com/video/BV1K64y1Q7wu)

**是以回归之名的分类**

回归估计一个连续值 VS 分类预测一个离散类别

**回归**

单连续值输出
自然区间R
跟真实值的区别作为损失

**分类**

通常多个输出  
输出$i$表示预测为第$i$类的置信度

## 从回归到多类 —— 均方损失

对类别进行有效编码  
$${\bf{y}}=[y_1, y_2,...,y_n]^T$$
$$y_i=
\left \{
\begin{array}{l}
1\ if\ i=y \\
0\ otherwise
\end{array}
\right.
$$
使用均方损失训练  
最大值最为预测（最大化$o_i$的置信度的值）
$$\hat y = arg\,\max_{i}o_i$$
无校验比例
## 从回归到多类分类——无校验比例
**需要更置信的识别正确类（大余量）**

正确类的置信度要远大于其他非正确类的置信度，数学表示为一个阈值。
$$o_y-o_i\ge\Delta(y,i)$$

## 从回归到多类分类——校验比例

输出匹配概率（非负，和为1）
$$\hat{\bf y}=softmax({\bf o})$$
$$\hat y_i={\exp{o_i}\over\sum_k\exp{o_k}}$$
预测概率$\hat{\bf{y}}$与真实概率$\bf{y}$的比较。
## Softmax和交叉熵损失
交叉熵通常用来衡量两个概率的区别：
$${H(\bf{p},\bf{q})}=\sum_{i}-p_i\log(q_i)$$
将他作为损失：
$$L(\bf{y}-\hat{\bf{y}})=-\sum_{i}y_i\log\hat{y}=-\log\hat{y}_y$$
其梯度是真实概率与预测概率的区别：
$$\partial_{o_i}L(\bf{y}-\hat{\bf{y}})=softmax({\bf o})_i-y_i$$
# 损失函数
**Huber's Robust Loss**
$$
L(y-y\prime)=
\begin{cases}
|y-y\prime|-{1\over2}&if\ |y-y\prime|\gt1\\
{1\over2}(y-y\prime)^2&otherwise
\end{cases}
$$
# 图像分类数据集读取
图像分类中使用最为广泛的数据集**MNIST**，创造与1986，用于识别手写数字，过于简单，此处用较为复杂的**Fashion MNIST**。
- 导入各库
```
%matplotlib inline
import torch
import torchvision
from torch.utils import data #读取数据小批量的函数集
from torchvision import transforms #数据操作
from d2l import torch as d2l
d2l.use_svg_display()
#svg可放缩矢量图形，有利于图片的高清显示
```
- 下载/导入数据
```
trans = transforms.ToTensor()
#ToTensor()把IPL图片转化为Tensor
#并除以255使所有像素均值在0-1之间
mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=False)
#从FashionMNIST拿训练数据，没有则下载，transform代表改变图像为张量。
mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=False)
#从FashionMNIST拿测试数据
len(mnist_train), len(mnist_test)
#结果分别为60000和10000张图片。
mnist_train[0][0].shape #数据示例
#输出torch.Size([1, 28, 28])，1代表RGB通道，为黑白图片，长×宽=28×28
```
- 两个可视化数据集的函数
```
def get_fashion_mnist_labels(labels): 
## 返回FashionMNIST的文本标签。 
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #titles=None,是默认，想改变手动键入。
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    #函数表示为fig , ax = plt.subplots(nrows, ncols)，fig代表名称，ax对应图例
    #figsize指定画布的大小，(宽度,高度)，单位为英寸。
    axes = axes.flatten()
    #flatten()是numpy中用于降低维度的函数，把n×m矩阵变成1×n*m的行向量。
    for i, (ax, img) in enumerate(zip(axes, imgs)):
    #enumerate(sequence, [start=0])返回元组列表（序号，元素）。
    #zip([iterable, ...])函数将多个迭代器对应元素打包以元组返回。
        if torch.is_tensor(img):
        #判断对象是否是torch的张量
            ax.imshow(img.numpy())
            # 先把张量转化为numpy的数组，并通过热图显示。
            # imshow()传入的变量是存储图像的数组，可以是浮点型数组、unit8数组以及PIL图像
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        #设置坐标轴为不可见
        if titles:
            ax.set_title(titles[i])
    return axes
```
- 几个样本的图像和标签
```
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));
```
- 读取一小批量图片
```
batch_size = 256 #传入批量大小为256

def get_dataloader_workers(): 
    """使用4个进程来读取数据"""
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True，num_workers=get_dataloader_workers())
# 从m_t读取数据，批量大小，打乱，同时工作的进程，输出一个生成器。

timer = d2l.Timer()
#计时开始
for X, y in train_iter:
    continue
f'{timer.stop():.2f} sec'
#⏲结束
```
- 定义数据读取的函数
```
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    # 指定给一个方法集
    if resize:
        trans.insert(0, transforms.Resize(resize))
    # 如果制定了大小，则插入一个图片格式改变方法，再转换为张量
    trans = transforms.Compose(trans)
    # 打包成torch可理解的函数集。

    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))
```



