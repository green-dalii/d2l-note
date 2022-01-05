## Tip 1
'''
关于类的魔术方法
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

## __iter__() & __next__():
'''
__iter__()使实例成为可迭代的对象，如果返回的是一个迭代器，就可以直接调用；
否则，就需要定义__next__()，使被调用时直到返回的下一个是什么，从而组成一个完整的迭代器
'''

# Eg:

class A(object):

    def __init__(self, n):
        self.n = n
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.n += 1
        if self.n > 5:
            raise StopIteration
        return self.n - 1
        #这里return语句必须放在最后，相当于每次循环只会运行到return就重新开始
    
a = A(2)
for _ in a:
    print(_)
## output(2, 3, 4)