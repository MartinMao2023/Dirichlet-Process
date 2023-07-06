
a = {1, 2}
b = a
dict = {1: a, 2: "b"}

del dict[1]
del a
print(dict)
print(b)


