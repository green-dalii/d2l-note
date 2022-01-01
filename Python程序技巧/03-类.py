## Tip 1
'''
关于类的内建属性方法
'''
## __getitem__(self, idx)
## 使类的实例可以被索引

class A():
    def __getitem__(self, idx):
        L = list(range(6))
        print(L[idx])

B = A()
B[4]
## output: 4