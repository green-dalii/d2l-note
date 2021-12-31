<font face='微软雅黑'>一点感想：写程序在时分块，层层嵌套，优先定义最基本的功能，再逐层调用到高级功能上，最终实现整套程序。一则程序的独立性高，可以多次调用；二则方便debug；三则每部分代码不必过于冗长，方便理解。</font>

**导入包和生成锚框**

```
%matplotlib inline
import torch
from d2l import torch as d2l

img = d2l.plt.imread('./Image/Lions.png')
print(img.shape)

h, w = img.shape[:2]
#一般图片都是(H,W,C)的格式

#在特征图(fmap)上生成锚框(anchors)，每个单位像素作为锚框的中心
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))
    #batch_size=1,channels=10
    #只需要图片尺寸生成锚框，所以内容全零
    #定义了一个图幅，并不一定是原图尺寸
    #以这个图幅的像素为中心设立锚框
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    #放缩到一个像素上的尺寸
    bbox_scale = torch.tensor((w, h, w, h))
    #放大比例
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)
```

**块的实现**

```
%matplotlib inline
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


def cls_predictor(num_inputs, num_anchors, num_classes):
    #输入通道、锚框数/pixel、类别数
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), #1代表背景类
                     kernel_size=3, padding=1)
#对每个锚框做类别预测
#输入输出宽高不变，也就对应h*w*num_anchors的锚框数
#就是说不变的高宽代表每一个像素点
#num_anchors * classes的输出通道代表每一个像素的锚框对每一种类别的预测结果

def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)
#同理，每个锚框给你预测的四维

def forward(x, block):
    return block(x)
#在块里传入输入传出输出

Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
Y1.shape, Y2.shape
#很明显，除了批量以外，输入图的尺寸都不一致，所以需要统一

def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)
    #permute()传入维度索引，改变维度的顺序
    #Flatten()也是按照维度从内到外的顺序
    #可以把CHW想象成一个空间坐标下的长方体，不同维度顺序就是观测的面变化
    #把通道维放在最后，拉直后对每个像素的预测值就是连续的

def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)
#再把两部分首位相接，变成连续矩阵，
#也就是说在仅有的相同维度batch_size上进行连接

```

**网络的实现**

```
#高宽减半的CNN，通常是pre-trained
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)
#两层神经网络，依靠最大池化层高宽减半，并改变通道数

#从原始图片抽取特征到fmap
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
        #循环常用写法，如果出现i+1，就要在循环处有range(len())-1
    return nn.Sequential(*blk)
    #三次高宽减半，通道数加倍

forward(torch.zeros((2, 3, 256, 256)), base_net()).shape

#整个网络，5个stage，在5个尺度上做目标检测
def get_blk(i):
    if i == 0:
        blk = base_net()
    #得到fmp
    elif i == 1:
        blk = down_sample_blk(64, 128)
    #第二层fmp通道加倍
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    #最后全局最大池化
    else:
        blk = down_sample_blk(128, 128)
    #二、三层fmp通道不变，数据集不够大，所以没必要提取过多特征
    return blk

def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    #fmp
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    #只用提取Y的后二维(h, w)生成锚框
    cls_preds = cls_predictor(Y)
    #这里的cls_predictor是直接调用已经定义好的块(返回值是类的实例，即cnn)
    #也就是提前传入了num_inputs,num_anchors,num_classes...
    #该CNN实例的__call__()作用于Y
    #前向预测并不关心锚框的形状，看到的是整个feature_map
    #backward运算时才会调整锚框的偏差
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)

sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
#越往上走，锚框看的越大，所以s持续增长
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1

class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        #类别数
        idx_to_in_channels = [64, 128, 128, 128, 128]
        #每个stage的输出通道数，也就是fmp的输入通道数
        for i in range(5):
            setattr(self, f'blk_{i}', get_blk(i))
            #setattr(self, name, value)
            #object, 属性名，属性值
            #把stages的块传入作为属性
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            #把cls_predictor作为属性传入，return一个CNN，对应上方只传入Y的代码。
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))
    # 定义属性的时候，绑定的是模块，模块传入的参数定义好了cnn的实例，可以被直接调用
    def forward(self, X):
    # forward的功能就是调用定义好的块（调用定义好的net实例）
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        #经典循环空列表
        for i in range(5):
            # getattr(self,'blk_%d'%i)即访问self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
            #getarttr(self, name, default=None)，返回属性
            #传参，定义每一个块
            #一个定义好的块块嵌套其他一个定义好的块
            #返回的X是blk[i](X)，实现了迭代
            #size和ratio是全局变量，可以直接拿来用
            
        anchors = torch.cat(anchors, dim=1)
        #连成一行
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds

net = TinySSD(num_classes=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)
```

<font size=5 color=red>后面的内容需要参考course-33的函数，以及一些其他函数。问题主要是各个输入输出的维度，最好结合GPU运算后debug，总之，逻辑可以理解，但是要写极难，正如李沐建议在识别上一般运用别人的成熟算法</font>

**训练**

```
batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)

device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')
#用L1损失，真实值与预测值差的绝对值
#很可能实值过远，导致平方项过大，所以不用L2平方项损失

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    # 参见上方，(batch_size, num_boxes, num_classes)
    cls = cls_loss(cls_preds.reshape(-1, num_classes), #批量大小维和锚框维合并
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    cls_labels.shape()=(batch_size, 4*num_anchors)
    bbox = bbox_loss(bbox_preds * bbox_masks, #masks真实框对应1，背景框对应0。
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox
#前向计算不看锚框位置
#在计算loss的时候会告诉该位置对应的类，然后往上拟合

def cls_eval(cls_preds, cls_labels):
    
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())
    #计算预测准确类型的个数
def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
```
**训练过程**

```
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
net = net.to(device)
for epoch in range(num_epochs):
    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)
        anchors, cls_preds, bbox_preds = net(X)
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        # 根据锚框与对应边界框的映射求出该边界框的类型和四维
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')

X = torchvision.io.read_image('../img/banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()
```

**要做NMS**

```
def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    #做nms使锚框与边界框一一对应，其余负类
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]
    

output = predict(X)
```
```
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output.cpu(), threshold=0.9)
```