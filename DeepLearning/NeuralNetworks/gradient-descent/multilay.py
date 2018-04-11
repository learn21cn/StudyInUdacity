#%%

# 权重需要有两个索引ij​	 ，其中 i 表示输入单元，j 表示隐藏单元。



import numpy as np

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1/(1+np.exp(-x))

# Network size
N_input = 4
N_hidden = 3
N_output = 2

np.random.seed(42)
# Make some fake data
X = np.random.randn(4)

weights_input_to_hidden = np.random.normal(0, scale=0.1, size=(N_input, N_hidden))
weights_hidden_to_output = np.random.normal(0, scale=0.1, size=(N_hidden, N_output))


# TODO: Make a forward pass through the network

hidden_layer_in = np.dot(X, weights_input_to_hidden)
hidden_layer_out = sigmoid(hidden_layer_in)

print('Hidden-layer Output:')
print(hidden_layer_out)

output_layer_in = np.dot(hidden_layer_out, weights_hidden_to_output)
output_layer_out = sigmoid(output_layer_in)

print('Output-layer Output:')
print(output_layer_out)


#%%
# 关于列向量
# 对一维数组来说，转置还是行向量。所以你可以用 arr[:,None] 来创建一个列向量：
features = np.array([ 0.49671415, -0.1382643 ,  0.64768854])
print(features)
print(features.T)
print(features[:, None])

# 当然，你可以创建一个二维数组，然后用 arr.T 得到列向量。
newarray = np.array(features, ndmin=2)
print(newarray)
print(newarray.T)


