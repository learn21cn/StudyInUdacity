import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    expL = np.exp(L)
    return np.divide(expL, expL.sum())

# 与下面的这个函数是一样的
def softmaxa(L):
    expL = np.exp(L)
    sumExpL = sum(expL)
    result = []
    for i in expL:
        result.append(i*1.0/sumExpL)
    return result

L=[5,6,7]
result = softmaxa(L)
print(result)


