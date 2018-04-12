# 反向传播

#%%
import numpy as np
import pandas as pd
import os
# 注意这里的工作目录

np.random.seed(21)
cwd = os.getcwd()
print(cwd)
# 如果读不出来，添加路径
# admissions = pd.read_csv(cwd+'\\gradient-descent\\test.csv')
admissions = pd.read_csv('./gradient-descent/test.csv')
print(admissions)

#%%
# 数据预处理
# Make dummy variables for rank
data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)
data = data.drop('rank', axis=1)

# Standarize features
for field in ['gre', 'gpa']:
    mean, std = data[field].mean(), data[field].std()
    # 比如loc[0:2,[“a”,”b”]]。取0到第2行（左闭右开，非整型值时左闭右闭），”a”列与”b”列。
    data.loc[:,field] = (data[field]-mean)/std
    
# Split off random 10% of the data for testing

sample = np.random.choice(data.index, size=int(len(data)*0.9), replace=False)
data, test_data = data.iloc[sample], data.drop(sample)

# Split into features and targets
features, targets = data.drop('admit', axis=1), data['admit']
features_test, targets_test = test_data.drop('admit', axis=1), test_data['admit']

print(features[:10])

#%%
# 激活函数
def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


#%%
# 测试数据
# 用于理解程序的过程
weights_input_hidden_test = np.random.normal(scale=1 / 6 ** .5,
                                        size=(6, 2))
weights_hidden_output_test = np.random.normal(scale=1 / 6 ** .5,
                                         size=2)
print('weights_input_hidden_test:')
print(weights_input_hidden_test)
print('weights_hidden_output_test:')
print(weights_hidden_output_test)

del_w_input_hidden_test = np.zeros(weights_input_hidden_test.shape)
del_w_hidden_output_test = np.zeros(weights_hidden_output_test.shape)

print('del_w_input_hidden_test:')
print(del_w_input_hidden_test)
print('del_w_hidden_output_test:')
print(del_w_hidden_output_test)


a = np.zeros(weights_hidden_output_test.shape)
print('测试加法')
print(a)
print(a + 5)
n = np.array([ 2, -1])
b = np.array([ 0.5, -0.5 ,  0.5, 1, 1, 1])
print(b)
print(b[:, None])
print('测试乘法')
print(n * b[:, None])


features_miny = features[:5]
targets_miny = targets[:5]

count = 0
for x, y in zip(features_miny.values, targets_miny):    
    
    count = count + 1
    print('第'+ str(count) + '次计算开始')
    # TODO: Calculate the output
    # x在此处是个6维向量
    print('x:')
    print(x)
    # 此处为2维向量，由 (1，6)点乘(6,2)得到
    hidden_input = np.dot(x, weights_input_hidden_test)
    print('hidden_input:')
    print(hidden_input)

    # 此处为2维向量
    hidden_output = sigmoid(hidden_input)
    print('hidden_output:')
    print(hidden_output)

    # 两个2维向量点乘得到一个具体的值，之后是使用激活函数处理
    output = sigmoid(np.dot(hidden_output, weights_hidden_output_test))
    # 得到一个具体的值
    print('output:')
    print(output)

    ## Backward pass ##
    # TODO: Calculate the network's prediction error
    # 得到一个具体的值
    error = y - output
    print('error:')
    print(error)

    # TODO: Calculate error term for the output unit
    # 计算输出层δ，误差乘以激活函数导数，​此处得到一个具体的值
    output_error_term = error * output * (1 - output)
    print('output_error_term:')
    print(output_error_term)   

    ## propagate errors to hidden layer

    # TODO: Calculate the hidden layer's contribution to the error
    # (1,) 点乘2维向量，此处为2维向量，注意不是矩阵
    hidden_error = np.dot(output_error_term, del_w_hidden_output_test)
    print('hidden_error:')
    print(hidden_error)

    # TODO: Calculate the error term for the hidden layer
    # 计算隐藏层δ，误差乘以激活函数导数，​此处得到一个2维向量，分别对应两个隐藏层
    hidden_error_term = hidden_error * hidden_output * (1 - hidden_output)
    print('hidden_error_term:')
    print(hidden_error_term)

    # TODO: Update the change in weights
    # 2维向量加一个具体的数 ΔW=δa，每个维度分别加上
    del_w_hidden_output_test += output_error_term * hidden_output
    print('del_w_hidden_output_test:')
    print(del_w_hidden_output_test)
    # (6,2)+(6,2)
    del_w_input_hidden_test += hidden_error_term * x[:, None]
    print('x[:, None]:')
    print(x[:, None])
    # 注意这里是乘法，不是点乘，这里2维向量乘以6维向量，得到一个(6，2)
    print(hidden_error_term * x[:, None])
    print('del_w_input_hidden_test:')
    print(del_w_input_hidden_test)

    print('第'+ str(count) + '次计算结束')
    print('------------------------------------')


print('new del_w_input_hidden_test:')
print(del_w_input_hidden_test)
print('new del_w_hidden_output_test:')
print(del_w_hidden_output_test)

#%%
# 以下是正式部分
# Hyperparameters
n_hidden = 2  # number of hidden units
epochs = 900
# 通常由η表示
learnrate = 0.005

n_records, n_features = features.shape
# print(n_records, n_features)
last_loss = None
# Initialize weights
weights_input_hidden = np.random.normal(scale=1 / n_features ** .5,
                                        size=(n_features, n_hidden))
weights_hidden_output = np.random.normal(scale=1 / n_features ** .5,
                                         size=n_hidden)


for e in range(epochs):
    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    del_w_hidden_output = np.zeros(weights_hidden_output.shape)
    for x, y in zip(features.values, targets):
        ## Forward pass ##
        # TODO: Calculate the output
        hidden_input = np.dot(x, weights_input_hidden)
        hidden_output = sigmoid(hidden_input)

        output = sigmoid(np.dot(hidden_output,
                                weights_hidden_output))

        ## Backward pass ##
        # TODO: Calculate the network's prediction error
        error = y - output

        # TODO: Calculate error term for the output unit
        output_error_term = error * output * (1 - output)
        

        ## propagate errors to hidden layer

        # TODO: Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(output_error_term, weights_hidden_output)

        # TODO: Calculate the error term for the hidden layer
        hidden_error_term = hidden_error * hidden_output * (1 - hidden_output)

        # TODO: Update the change in weights
        del_w_hidden_output += output_error_term * hidden_output
        del_w_input_hidden += hidden_error_term * x[:, None]

    # TODO: Update weights
    weights_input_hidden += learnrate * del_w_input_hidden / n_records
    weights_hidden_output += learnrate * del_w_hidden_output / n_records

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        hidden_output = sigmoid(np.dot(x, weights_input_hidden))
        out = sigmoid(np.dot(hidden_output,
                             weights_hidden_output))
        loss = np.mean((out - targets) ** 2)

        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
hidden = sigmoid(np.dot(features_test, weights_input_hidden))
out = sigmoid(np.dot(hidden, weights_hidden_output))
predictions = out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))