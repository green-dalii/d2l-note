## Tip 1
'''
列表可以进行 += 赋值
赋值的对象必须是可迭代的(iterable)，如其他的列表、字符串等
并且是把可迭代对象的元素（最外层维度）拆开，分别放入列表中
'''
# Eg1
L = []
str1 = 'list'
L += str1
print(L)
## output: ['l', 'i', 's', 't']
# Eg2
L = []
list1 = list(range(5))
L += list1
print(L)
## output: [0, 1, 2, 3, 4]

