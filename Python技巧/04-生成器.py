## Tip1: 生成器可以被列表调用
#Eg1:
g = (x for x in range(5))
L = list(g)
print(L)
# output = [0, 1, 2, 3, 4]

# Eg2:
g1 =(x for x in ['a', 'b', 'c', 'd'])
dict1 = {i: x for i, x in enumerate(g1)}
print(dict1)
# output = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}

## Tip2: 同一个生成器里的元素，只能被调用一次，如果被提取完了，会返回空

#Eg:

g3 = (x for x in range(2))
l1 = list(g3)
print(l1)
# output = [0, 1]
l2 = list(g3)
print(l2)
# output = []