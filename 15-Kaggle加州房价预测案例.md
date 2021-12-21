**准备工作——导入包/下载数据**
```
import hashlib 
#把任意长度的输入，通过某种hash算法，变换成固定长度的输出，
#主要用来加密和解密。
import os
import tarfile #TarFile类对于就是tar压缩包实例
import zipfile
import requests

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

def download(name, cache_dir=os.path.join('..', 'data')):  
    #os.path.join()路径拼接，相当于../data文件夹。
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    #os.makedirs()y用于创建目录
    #exist_ok=True，默认为False,目标文件夹存在会报错
    fname = os.path.join(cache_dir, url.split('/')[-1])
    #把url按照/拆分，提取最后一个字符串连接目录，作为文件名
    if os.path.exists(fname):
        #os.path.exists()检测目录/文件是否存在
        sha1 = hashlib.sha1()
        #sha1算法加密
        with open(fname, 'rb') as f:
            #读取二进制文件，一般用于非文本文件如图片等。
            while True:
                data = f.read(1048576)
                #file.read([size])指定一次最多可读取的字符（字节）个数
                #默认读取所有内容。
                #size=1048576=1024*1024,每次的读取1Mb。
                if not data:
                #如过返回空
                    break
                sha1.update(data)
                #update()合并与覆盖当前内容
        if sha1.hexdigest() == sha1_hash:
            #把所有内容加密后与文件对比，
            #如果密码相同，则返回文件名
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    #从链接获取内容
    with open(fname, 'wb') as f:
        #因为fname文件不存在，创建新文件。
        #写二进制文件
        f.write(r.content)
        #写入链接内容
    return fname
```
```
def download_extract(name, folder=None):
    fname = download(name)
    base_dir = os.path.dirname(fname)
    #去掉文件名，返回目录
    data_dir, ext = os.path.splitext(fname)
    #分离文件名与扩展名
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
        #读取压缩文件
    elif ext in ('tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False
    fp.extractall(base_dir)
    #extractall()即解压缩。
    return os.path.join(base_dir, folder) if folder else data_dir
    #要么返回join的路径，要么返回base的路径

def download_all():
    for name in DATA_HUB:
        download(name)
```
```
%matplotlib inline
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

DATA_HUB['kaggle_house_train'] = (  
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))
```
```
'''
现在来看程序执行过程：
DATA_HUB={'kaggle_house_train': ('http://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_test.csv','fa19780a7b011d9b009e8bff8e99922a8ee2eb90')}
download(name = 'kaggle_house_train')
url, sha1_hash = ('http://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_test.csv','fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
os.makedirs(cache_dir, exist_ok=True):创建目录"../data"
fname = os.path.join(cache_dir, url.split('/')[-1])
fname = "../data/kaggle_house_pred_test.csv"
起初并不存在，所以要先下载：
print(f'正在从{url}下载{fname}...')
r = requests.get(url, stream=True, verify=True)
with open(fname, 'wb') as f:
        f.write(r.content)
创建并写入数据。
如果已经下载过了：通过While循环，把文件的所有数据加密，返回Hash
if sha1.hexdigest() == sha1_hash=='fa19780a7b011d9b009e8bff8e99922a8ee2eb90':
表示已存在文件与目标文件内容一致。
有一点需要注意：加密返回值的长短并不是与字符串成正比，而是一个固定值。
'''
```
**数据预处理**

```
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:-1]))
#默认按列连接
#刨去第一列ID,和最后一列labels

#对数值列进行normalization
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
#在pandas里，用object表示string。
all_features[numeric_features] = all_features[numeric_features].apply(
lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)

all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape
#get_dummies()使one-hot编码，把字符串数据转换为0-1；
#每列有n个不同的字符串，那么独热向量的长度就为n；
#因此df的列数大增
```
**网络算法**

```
#定义输入输出
n_train = train_data.shape[0] #1460
train_features = torch.tensor(all_features[:n_train].values,
                                         dtype=torch.float32)
#没有iloc的切片，默认对行；
#取每列去标题行前n_train=1460行。1460*331
test_features = torch.tensor(all_features[n_train:].values,
                                         dtype=torch.float32)
#1459*331
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1),
                           dtype=torch.float32)
#1460*1

#损失函数和神经网络
loss = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net

#normalize损失
def log_rmse(net, features, labels):
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    #clamp(x, min, max)，限制的值在[min, max]之间，超出取最外值。
    #把小于1的值变为1
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    #Loss求均方损失，输出一个标量
    return rmse.item()
#tensor.item()=number
#Returns the value of this tensor as a standard Python number.
#This only works for tensors with one element. 

#训练函数
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(), 
                                 lr=learning_rate,
                                weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        #每个epoch计算每次的losss并添加到列表
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
        #每个epoch计算每次的losss并添加到列表
    return train_ls, test_ls
    #train_ls，test_ls出入的就是 1*n=epochs 的列表

#K-折交叉验证数据集的定义
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    # z整除
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        #slice()返回切片
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
            #valid是验证集
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
            #torch.cat(tensors, dim=0) 
    return X_train, y_train, X_valid, y_valid
#返回的是第i个折的训练、测试集

#定义训练过程
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                  weight_decay, batch_size)
        #*data表示把其返回值元组(X_train, y_train, X_valid, y_valid)传参
        #i在epoch之前，表示对于每一折，进行epochs次训练
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        #把最后一个epoch的损失提出来，也就是最精确的一组
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                    xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                    legend=['train', 'valid'], yscale='log')
            #只画第一折的图
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f},'
             f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum /k, valid_l_sum / k
    #求所有折的平均损失

k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                         weight_decay, batch_size)
#X_train: train_features.shape=(1459, 331), y_train: train_labels.shape=(1459, 1), k=5, batch_size=64
#data = get_k_fold_data(k, i, X_train, y_train)
#fold_size=1459//5=291
#i=0, X_part, y_part = X[(0:291), :], y[0:291]=X_valid, y_valid; X_train, y_train=X[(291:), :], y[291:]

print(f'{k}-折验证：平均训练log rmse:{float(train_l):f},'
     f'平均验证log rmse: {float(valid_l):f}')#表示小数定义精度，默认六位浮点数
#f'{}'是{}.format()的简易写法
```
![](\Images/微信截图_20211221142847.png)

实际上在前方的K-折交叉验证里并没有用到测试集。
**预测测试集并保存输出**
```
def train_and_pred(train_features, test_feature, train_labels, test_data,
                  num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                       num_epochs, lr, weight_decay, batch_size)
    #传出的是每个epoch的loss，而test_ls=None
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel = 'epoch',
            ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'train log rmse {float(train_ls[-1]):f}')
    #输出最后一个epoch的值。
    preds = net(test_features).detach().numpy()
    #net返回预测的test_labels
    #detach()使梯度计算就此打住
    #numpy()返回数组
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    #(1, -1)排列为一行，但是规定是二维张量，尽管行为1.
    #[0]取第一行，一个一维行向量
    #pd.Series()转化为一个dataframe
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    #把['Id']列和['labels']以行相连。
    submission.to_csv('submission.csv', index=False)
    #保存路径为./submission.csv
    #index=False，就不保存索引列
train_and_pred(train_features, test_features, train_labels, test_data,
              num_epochs, lr, weight_decay, batch_size)
```
![](/Images/微信截图_20211221174519.png)