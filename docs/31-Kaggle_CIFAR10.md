# 31 - Kaggle å®æ - CIFAR-10

---

### ð¦ æ¬èè¯¾ç¨è§é¢å°å ð

[![Bilibil](https://i2.hdslb.com/bfs/archive/1060d9d14c8d840fefaf6972f8b539d05655aa5d.jpg@640w_400h_100Q_1c.webp)](https://www.bilibili.com/video/BV1Gy4y1M7Cu)

## CIFAR-10æ°æ®é

ä¹åå èä¸­ï¼æä»¬ä¸ç´å¨ä½¿ç¨æ·±åº¦å­¦ä¹ æ¡æ¶çé«çº§ API ç´æ¥è·åå¼ éæ ¼å¼çå¾åæ°æ®éã ä½æ¯å¨å®è·µä¸­ï¼å¾åæ°æ®ééå¸¸ä»¥å¾åæä»¶çå½¢å¼åºç°ã å¨æ¬èä¸­ï¼æä»¬å°ä»åå§å¾åæä»¶å¼å§ï¼ç¶åéæ­¥ç»ç»ãè¯»åå¹¶å°å®ä»¬è½¬æ¢ä¸ºå¼ éæ ¼å¼ã

æä»¬å°æ°æ®ååè®­ç»éãéªè¯éåæµè¯éãå¨è®­ç»éä¸è®­ç»æ¨¡åï¼å¨éªè¯éä¸è¯ä¼°æ¨¡åï¼ä¸æ¦æ¾å°çæä½³çåæ°ï¼å°±å¨æµè¯éä¸æåæµè¯ä¸æ¬¡ï¼æµè¯éä¸çè¯¯å·®ä½ä¸ºæ³åè¯¯å·®çè¿ä¼¼ã

æ¯èµæ°æ®éåä¸ºè®­ç»éåæµè¯éï¼å¶ä¸­è®­ç»éåå« 50000 å¼ ãæµè¯éåå« 300000 å¼ å¾åã å¨æµè¯éä¸­ï¼10000 å¼ å¾åå°è¢«ç¨äºè¯ä¼°ï¼èå©ä¸ç 290000 å¼ å¾åå°ä¸ä¼è¢«è¿è¡è¯ä¼°ï¼åå«å®ä»¬åªæ¯ä¸ºäºé²æ­¢æå¨æ è®°æµè¯éå¹¶æäº¤æ è®°ç»æã ä¸¤ä¸ªæ°æ®éä¸­çå¾åé½æ¯ png æ ¼å¼ï¼é«åº¦åå®½åº¦åä¸º 32 åç´ å¹¶æä¸ä¸ªé¢è²ééï¼RGBï¼ã è¿äºå¾çå±æ¶µç 10 ä¸ªç±»å«ï¼é£æºãæ±½è½¦ãé¸ç±»ãç«ãé¹¿ãçãéèãé©¬ãè¹åå¡è½¦

## ä»£ç å®ç°

- å¼å¥å&ä¸è½½æ°æ®

```python
import collections
import math
import os
# shutil æ¨¡åæä¾äºä¸ç³»åå¯¹æä»¶åæä»¶éåçé«é¶æä½ã
import shutil
import pandas as pd
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

d2l.DATA_HUB['cifar10_tiny'] = (d2l.DATA_URL + 'kaggle_cifar10_tiny.zip','2068874e4b9a9f0fb07ebe0ad2b29754449ccacd')
#æ­¤æ°æ®éæ¯ä¿å­å¨AWSçdemoèéå®æ´çcifar-10

demo = True

if demo:
    data_dir = d2l.download_extract('cifar10_tiny')
else:
    data_dir = '../code/data/cifar-10/'
```

æ°æ®éç»æï¼

- ../data/cifar-10/train/[1-50000].png
- ../data/cifar-10/test/[1-300000].png
- ../data/cifar-10/trainLabels.csv
- ../data/cifar-10/sampleSubmission.csv

> train å test æä»¶å¤¹åå«åå«è®­ç»åæµè¯å¾åï¼trainLabels.csv å«æè®­ç»å¾åçæ ç­¾ï¼ sample_submission.csv æ¯æäº¤æä»¶çèä¾ã

- æ´çæ°æ®é

```python
def read_csv_labels(fname):
    #labelså­å¨csvæä»¶é
    with open(fname, 'r') as f:
        lines = f.readlines()[1:]
        #readlines() æè¡è¯»åæ´ä¸ªæä»¶ï¼ã
    tokens = [l.rstrip().split(',') for l in lines]
    #strip()å é¤å­ç¬¦ä¸²å¤´å°¾ç¹å®çå­ç¬¦ï¼é»è®¤æ¯ç©ºæ ¼åæ¢è¡ç¬¦
    #rstrip()å¨æ­¤å å»å³ä¾§æ¢è¡ç¬¦
    #split()ä¸­é´æç§éå·åé
    return dict(((name, label) for name, label in tokens))
    #æ[]æ¢æ()çåè¡¨çæå¼ï¼å°±æ¯ä¸ä¸ªgenerator
    #å¯ä»¥ç¨list/tuple(generator)æ¹å¼è½¬æ¢
    #å¦ææ¯æå¯¹åç´ ï¼å°±å¯ä»¥dict(generator)è½¬æ¢æå­å¸ï¼ç¸å½äºdict(list)çæ¹å¼å®ä¹å­å¸
    #ä»¥ä¸çæ¹å¼é½ä¼éågenerator
    #å½åä¸ä¸ªgeneratorè¢«éåè¿ä¸éæ¶ï¼åæ¬¡è°ç¨ä¼è¿åNone
    #æ¯å¦çæå¨g,ålist(g)ï¼ä¼å¾å°ææåç´ 
    #åtuple(g),å¾å°ç©ºåç»ï¼æä»¥éè¦éæ°å®ä¹g

labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
```

- å°éªè¯éä»åå§è®­ç»éä¸­æå

```python
def copyfile(filename, target_dir):
    # æ°å»ºæä»¶å¤¹
    # exist_ok(default)=Falseï¼å¦æç®æ æä»¶å¤¹å·²å­å¨ä¼æ¥é
    #å¯¹äºå·²å­å¨çæä»¶å¤¹ä¸ä¼è¦ç
    os.makedirs(target_dir, exist_ok=True)
    #copyæä»¶å°ç®å½
    shutil.copy(filename, target_dir)

def reorg_train_valid(data_dir, labels, valid_ratio):
    # æ´ä½æ¯ææä»¶å¨é¨å¤å¶å°train_validå¹¶ä¸åä¸ºtrainåvalidï¼æ¯ä¸ªä¸é¢åä»¥ç±»åå»ºç«æä»¶å¤¹ç»å
    # collections.Counteræ¯è®¡æ°å¨
    # åç´ ä»¥å­å¸å½¢å¼ä¿å­
    # most_common(num)è¿åæé¢ç¹çç±»ååé¢çï¼å¦æä¸æå®ä¸ªæ°ï¼defaultææç±»
    # è®­ç»æ°æ®éä¸­æ ·æ¬æå°çç±»å«ä¸­çæ ·æ¬æ°
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # math.flooråä¸åæ´
    # valid_ratioå³å®äºä¸ä¸ªä¸é
    # éªè¯éä¸­æ¯ä¸ªç±»å«çæ ·æ¬æ°
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    print(n_valid_per_label)
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        #åç¦»æä»¶ï¼å¾çï¼åä¸åç¼ï¼åªæåæä»¶åï¼ä¹å°±æ¯å¾ççåºå·
        #åå¨lablesè¿ä¸ªå­å¸éæ¿åºå¯¹åºæ å·çç±»åå
        label = labels[train_file.split('.')[0]]
        # æä»¶å
        fname = os.path.join(data_dir, 'train', train_file)
        # å¤å¶å°è®­ç»æä»¶å¤¹
        copyfile(fname,os.path.join(data_dir, 'train_valid_test', 'train_valid', label))
        # å¦ælabelä¸å¨label_countéæè¯¥labelæ°éæªè¾¾å°æ¯ç±»éªè¯éè¦æ±
        if label not in label_count or label_count[label] < n_valid_per_label:
            # å¤å¶å°éªè¯éæä»¶å¤¹
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'valid', label))
            #ç¨å­å¸è®¡æ°
            #dict.get(key, default=None),æ¥æ¾ç±»åï¼å¦ææ²¡æåè¿åé»è®¤å¼
            #åæ­¤èµå¼dict[key]
            #éè¿+1è®¡æ°ï¼å¹¶éæ°èµå¼
            label_count[label] = label_count.get(label, 0) + 1
        # å¦ælabelåå«å¨label_count
        else:
            # å¤å¶å°è®­ç»éæä»¶å¤¹
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train', label))
    return n_valid_per_label
```

- å¨é¢æµæé´æ´çæµè¯éï¼æ¹ä¾¿è¯»å

```python
# åå»ºæµè¯éï¼å¹¶ä¸åªæä¸ä¸ªunknownæä»¶å¤¹ï¼è¡¨ç¤ºææç±»ååä¸ºä¸å¯ç¥
def reorg_test(data_dir):
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(
            os.path.join(data_dir, 'test', test_file),
            os.path.join(data_dir, 'train_valid_test','test','unkonwn'))
```

- è°ç¨åé¢å®ä¹çå½æ°

```python
def reorg_cifar10_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)

batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_cifar10_data(data_dir, valid_ratio)
```

- å¾åå¢å¹¿

```python
# å¯¹è®­ç»éåå¢å¹¿ï¼resize -> éæºå¤§å°ç¼©æ¾åè£åª -> æ°´å¹³ç¿»è½¬ -> å¼ éå -> å½ä¸å
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(40),
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                                ratio=(1.0, 1.)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])

# å¯¹æµè¯éåå¢å¹¿ï¼å¼ éå -> å½ä¸å
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])
```

- è¯»åç±åå§å¾åç»æçæ°æ®é

```python
# ImageFolderæ¯ä¸ä¸ªéç¨çæ°æ®éå è½½å¨ï¼ä¹å°±æ¯å¯¹èªå®ä¹çå¾çæä»¶å¤¹
# ædata_dir/rain_valid_test/trainä½ä¸ºè®­ç»é
# ædata_dir/rain_valid_test/train_validä½ä¸ºè®­ç»éªè¯é
train_ds, train_valid_ds = [
    torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=transform_train) for folder in ['train', 'train_valid']]

# ædata_dir/rain_valid_test/validä½ä¸ºéªè¯é
# ædata_dir/rain_valid_test/test/unknownä½ä¸ºæµè¯é
valid_ds, test_ds = [
    torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=transform_test) for folder in ['valid', 'test']]
```

- å®ä¹æ°æ®éè¿­ä»£å¨

```python
# :drop_last=Trueè¡¨ç¤ºè£å»æåä¸æ»¡batch_sizeçä¸æ¹æ°æ®ï¼default=False
# ä¸ºå®ç°éæºSGDï¼å¿é¡»shuffle=Trueï¼å¦åæ°æ®éé¡ºåºä¹ä¼è¢«è¯¯å­¦
train_iter, train_valid_iter = [
    torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(
    valid_ds, batch_size, shuffle=False, drop_last=True)

test_iter =  torch.utils.data.DataLoader(test_ds, batch_size, shuffle=True,
                                         drop_last=False)
```

- å®ä¹æ¨¡å&æå¤±å½æ°

```python
def get_net():
    num_classes = 10
    # :num_classes ç±»å«æ°ï¼3ä¸ºRGBééæ°
    net = d2l.resnet18(num_classes, 3)
    return net

loss = nn.CrossEntropyLoss(reduction='none')
```

- å®ä¹è®­ç»è¿ç¨

```python
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices,
          lr_period, lr_decay):
    # momentumä½¿ç¨ææ°å æå¹³åä¹åçæ¢¯åº¦ä»£æ¿åæ¢¯åº¦è¿è¡åæ°æ´æ°ã
    # å ä¸ºæ¯ä¸ªææ°å æå¹³ååçæ¢¯åº¦å«æä¹åæ¢¯åº¦çä¿¡æ¯ï¼å¨éæ¢¯åº¦ä¸éæ³å æ­¤å¾åã
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                              weight_decay=wd)
    # ä½¿ç¨è°åº¦å¨æ¥å¯ç¨èªéåºå­¦ä¹ ç
    # æ¯élr_periodä¸ªepoch, å­¦ä¹ çä¸élr * lr_decay
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    # ä½å¾ç¸å³
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    # æ°æ®å¹¶è¡
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3) # loss, accuracy, numel
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (metric[0] / metric[2], metric[1] / metric[2], None))
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
        # ä½¿ç¨èªéåºå­¦ä¹ çåï¼ååç`optimizer.step()`åæ³
        # éæ¿æ¢ä¸º`scheduler.step()`
        scheduler.step()

    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

- è®­ç»åéªè¯æ¨¡å

```python
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 20, 2e-4, 5e-4
lr_period, lr_decay, net = 4, 0.9, get_net()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
# Out:
# train loss 0.685, train acc 0.751, valid acc 0.359
# 804.0 examples/sec on [device(type='cuda', index=0), device(type='cuda', index=1)]
```

![output_kaggle-cifar10](https://zh.d2l.ai/_images/output_kaggle-cifar10_42a34e_129_1.svg)

- å¨ Kaggle ä¸å¯¹æµè¯éè¿è¡åç±»å¹¶æäº¤ç»æ

```python
net, preds = get_net(), []
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

for X, _ in test_iter:
    y_hat = net(X.to(devices[0]))
    preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())
sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key=lambda x: str(x))
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])
df.to_csv('submission.csv', index=False)
# Out:
# train loss 0.684, train acc 0.759
# 1052.1 examples/sec on [device(type='cuda', index=0), device(type='cuda', index=1)]
```

![output_kaggle-cifar10](https://zh.d2l.ai/_images/output_kaggle-cifar10_42a34e_138_1.svg)

## Python æ¨¡ååèææ¡£

- `shutil` Python é«é¶æä»¶æä½å ð§[å®æ¹ä¸­æ](https://docs.python.org/zh-cn/3/library/shutil.html)
- `torchvision.datasets.ImageFolder` torchvison ä»æä»¶å¤¹åæé æ°æ®éæ¹æ³ ð§[ä¸­æ](https://pytorch-cn.readthedocs.io/zh/latest/torchvision/torchvision-datasets/#imagefolder) | [å®æ¹è±æ](https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.DatasetFolder)
- `torch.optim.lr_scheduler` Pytorch èªéåºå­¦ä¹ çç¸å³ææ¡£ ð§[å®æ¹è±æ](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)

---

## Q&Að¤

**Qï¼æ·±åº¦å­¦ä¹ çæå¤±å½æ°ä¸è¬æ¯éå¸çåï¼**

**ðââï¸**ï¼ä¸è¬æå¤±å½æ°çæ°å­¦å½¢å¼ï¼å¦äº¤åçµæå¤±å½æ°ãçº¿æ§åå½çæå°äºä¹æ³ç­ï¼æ¯å¸å½æ°ï¼ä½æéèå±åæ¿æ´»å½æ°çç¥ç»ç½ç»çæ°å­¦å½¢å¼é½æ¯éå¸çï¼å¸¦æ¥å¶æå¤±å½æ°çæ±è§£å°±æ¯éå¸ä¼åé®é¢ãä½åªè¿½æ±å¸å½æ°æ¯æ²¡ææä¹çï¼å¸å½æ°çè¡¨ç¤ºè½åæéï¼ä¸è½æåå¤æé®é¢ã
