
# coding: utf-8

# # Implementing the Gradient Descent Algorithm
# 
# In this lab, we'll implement the basic functions of the Gradient Descent algorithm to find the boundary in a small dataset. First, we'll start with some functions that will help us plot and visualize the data.

# In[5]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Some helper functions for plotting and drawing lines

def plot_points(X, y):
    # argwhereFind the indices of array elements that are non-zero, grouped by element.
    # print(np.argwhere(y==1))
    # 注意argwhere返回的是索引
    admitted = X[np.argwhere(y==1)]
    # print(X)
    # print(admitted)
    rejected = X[np.argwhere(y==0)]
    # 注意上面结果的形式，下面使用了二维数组
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'blue', edgecolor = 'k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'red', edgecolor = 'k')

# range返回的是range object，而np.nrange返回的是numpy.ndarray
# range仅可用于迭代，而np.nrange作用远不止于此，它是一个序列，可被当做向量使用
# range不支持步长为小数，np.arange支持步长为小数
def display(m, b, color='g--'):
    # 设置x轴与y轴刻度的取值范围
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    # 最后一个参数规定了步长
    x = np.arange(-10, 10, 0.1)
    plt.plot(x, m*x+b, color)


# ## Reading and plotting the data

# In[6]:


data = pd.read_csv('data.csv', header=None)
# 注意多列时两个中括号
X = np.array(data[[0,1]])
# 这里的类型numpy.ndarray，注意这种转化方式
# print(type(X))
# print(X)
# 单列不需要两个中括号
y = np.array(data[2])
# print(type(y))
# print(y)
plot_points(X,y)
plt.show()


# ## TODO: Implementing the basic functions
# Here is your turn to shine. Implement the following formulas, as explained in the text.
# - Sigmoid activation function
# 
# $$\sigma(x) = \frac{1}{1+e^{-x}}$$
# 
# - Output (prediction) formula
# 
# $$\hat{y} = \sigma(w_1 x_1 + w_2 x_2 + b)$$
# 
# - Error function
# 
# $$Error(y, \hat{y}) = - y \log(\hat{y}) - (1-y) \log(1-\hat{y})$$
# 
# - The function that updates the weights
# 
# $$ w_i \longrightarrow w_i + \alpha (y - \hat{y}) x_i$$
# 
# $$ b \longrightarrow b + \alpha (y - \hat{y})$$

# In[ ]:


# Implement the following functions

# Activation (sigmoid) function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Output (prediction) formula
def output_formula(features, weights, bias):
    return sigmoid(np.dot(features, weights) + bias )

# Error (log-loss) formula
def error_formula(y, output):
    return -y * np.log(output) - (1-y) * np.log(1 - output)

# Gradient descent step
def update_weights(x, y, weights, bias, learnrate):
    output = output_formula(x, weights, bias)    
    
    d_error = -(y - output) 
    # 导数为 -(y - output) * x ,即 d_error * x，反方向用减法
    weights -= learnrate * d_error * x
    bias -= learnrate * d_error
    
    return weights, bias


# ## Training function
# This function will help us iterate the gradient descent algorithm through all the data, for a number of epochs. It will also plot the data, and some of the boundary lines obtained as we run the algorithm.

# In[ ]:


np.random.seed(44)

epochs = 100
learnrate = 0.01

def train(features, targets, epochs, learnrate, graph_lines=False):
    
    errors = []
    n_records, n_features = features.shape
    print("n_records, n_features:",n_records, n_features)
    last_loss = None
    weights = np.random.normal(scale=1 / n_features**.5, size=n_features)
    print("initialweights:",weights)
    bias = 0
    for e in range(epochs):
        # del_w = np.zeros(weights.shape)
        for x, y in zip(features, targets):
            
            output = output_formula(x, weights, bias)
            error = error_formula(y, output)                                
            weights, bias = update_weights(x, y, weights, bias, learnrate)
            
        
        # Printing out the log-loss error on the training set
        out = output_formula(features, weights, bias)
        loss = np.mean(error_formula(targets, out))
        errors.append(loss)
        # print(features.shape)
        # print("features")
        # print(features)
        # print("weights")
        # print(weights.shape)
        # print(weights)
        # print("out")
        # print(out)
        
        if e % (epochs / 10) == 0:
            print("\n========== Epoch", e,"==========")
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            predictions = out > 0.5
            
            accuracy = np.mean(predictions == targets)
            print("Accuracy: ", accuracy)
        if graph_lines and e % (epochs / 100) == 0:
            display(-weights[0]/weights[1], -bias/weights[1])
            

    # Plotting the solution boundary
    plt.title("Solution boundary")
    display(-weights[0]/weights[1], -bias/weights[1], 'black')

    # Plotting the data
    plot_points(features, targets)
    plt.show()

    # Plotting the error
    plt.title("Error Plot")
    plt.xlabel('Number of epochs')
    plt.ylabel('Error')
    plt.plot(errors)
    plt.show()


# ## Time to train the algorithm!
# When we run the function, we'll obtain the following:
# - 10 updates with the current training loss and accuracy
# - A plot of the data and some of the boundary lines obtained. The final one is in black. Notice how the lines get closer and closer to the best fit, as we go through more epochs.
# - A plot of the error function. Notice how it decreases as we go through more epochs.

# In[ ]:


train(X, y, epochs, learnrate, True)

