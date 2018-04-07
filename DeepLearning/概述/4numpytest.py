#%%
import numpy as np
s = np.array(5)
print(s)
print(s.shape)

x = s + 1
print(x)

#%%
# 要创建一个向量，你可以将 Python 列表传递给 array 函数，像这样：
v = np.array([1,2,3])
print(v)
print(v.shape)
print(v[1:])

#%%
# 所以要创建一个包含数字 1 到 9 的 3x3 矩阵，你可以这样做：
m = np.array([[1,2,3], [4,5,6], [7,8,9]])
print(m)
# 检查它的 shape 属性将返回元组 (3, 3)，表示它有两个维度，每个维度的长度为 3。
print(m.shape)
# 以像向量一样访问矩阵的元素
print(m[1][1])


#%%
# 张量
t = np.array([[[[1],[2]],[[3],[4]],[[5],[6]]],[[[7],[8]],\
    [[9],[10]],[[11],[12]]],[[[13],[14]],[[15],[16]],[[17],[18]]]])

print(t)
print(t.shape)
print(t[1][0][1][0])

#%%
# 更改形状
v = np.array([1,2,3,4])
print(v)
print(v.shape)

x = v.reshape(1,4)
print(x)
print(x.shape)

y = v.reshape(4,1)
print(v)
print(y)
print(y.shape)

# 下面这些代码创建一个切片，查看 v 的所有项目，要求 NumPy 为相关轴添加大小为 1 的新维度
# 注意逗号
m = v[None,:]
print(m)

n = v[:, None]
print(n)

#%%
values = [1,2,3,4,5]
values = np.array(values) + 5
print(values)

x = np.multiply(values, 5)
print(x)
# multiply等价于
x = x * 5
print(x)

#%% 
m = np.array([[1,2,3],[4,5,6]])
n = m * 0.25
print(n)
print(m * n)
print(np.multiply(m, n))

#%%
a = np.array([[1,2,3,4],[5,6,7,8]])
b = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
c = np.matmul(a, b)
print(c)
# 有时候，在你以为要用 matmul 函数的地方，你可能会看到 NumPy 的 dot 函数
# 事实证明，如果矩阵是二维的，那么 dot 和 matmul 函数的结果是相同的
d = np.dot(a, b)
print(d)

#%%
# NumPy 在进行转置时不会实际移动内存中的任何数据 - 只是改变对原始矩阵的索引方式 - 所以是非常高效的

m = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(m)
print(m.T)

# 注意的地方
m_t = m.T
m_t[3][1] = 200
print(m_t)
print(m)
