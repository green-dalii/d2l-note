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

'''
列表作用于字符串，把字符串拆开传入
'''

list('happy')
## output: ['h', 'a', 'p', 'p', 'y']

##　Tip 2:

'''
列表生成式可以双重嵌套
'''
list1 = [[1, 2], [3, 4]]
[x for list2 in list1 for x in list2]

## output: [1, 2, 3, 4]
'''
也可以实现条件判断
'''
list2 = [3, 4, 6, 7]
[x for x in list2 if x > 5]
## output: [6, 7]

'''
补全条件需要改变顺序
'''
list3 = [3, 4, 6, 7]
[x if x < 5 else x - 1 for x in list2 ]
## output: [3, 4, 5, 6]