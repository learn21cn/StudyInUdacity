
#%%
import pandas as pd

# TODO: Set weight1, weight2, and bias

# And感知器的权重与偏差
# weight1 = 1.0
# weight2 = 1.0
# bias = -2

# OR 感知器和 AND 感知器很相似，可以通过 “增大权重”、“减小偏差大小”实现

# Not感知器的权重与偏差
weight1 = 0.0
weight2 = -1.0
bias = 0.5


# DON'T CHANGE ANYTHING BELOW
# Inputs and outputs
test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
correct_outputs = [True, False, True, False]
outputs = []

# Generate and check output
for test_input, correct_output in zip(test_inputs, correct_outputs):
    linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
    output = int(linear_combination >= 0)
    is_correct_string = 'Yes' if output == correct_output else 'No'
    outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])

# Print output
num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
output_frame = pd.DataFrame(outputs, columns=['Input 1', '  Input 2', '  Linear Combination', '  Activation Output', '  Is Correct'])
if not num_wrong:
    print('Nice!  You got it all correct.\n')
else:
    print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
print(output_frame.to_string(index=False))


# XOR感知器
# 神经网络的输入来自第一个节点。输出来自最后一个节点。
# 假定将input1 与 input2 作为输入输入
# A节点 And运算 即input1 AND input2
# B节点 OR运算 即input1 OR input2
# C节点 对A节点的输出结果Not运算 即NOT(input1 AND input2)
# D 节点 对B与C节点的输出进行AND运算
#  output = ((NOT(input1 And input2)) AND (input1 OR input2))
# 以上为XOR感知器的计算


#%%
import numpy as np  

filename = './data.csv'

# 使用numpy加载文件，得到的是ndarray类型
my_matrix = np.loadtxt(open(filename,"rb"),delimiter=",",skiprows=0,usecols=[0,1,2])
# print(my_matrix)

# 可以使用切片获取特定的列
X_matrix = my_matrix[:, 0:2]
# print(X_matrix)
y_matrix = my_matrix[:, 2:3]
# print(y_matrix)

# 使用numpy写入文件
# np.savetxt('new.csv', my_matrix, delimiter = ',') 

# 如果使用pandas读取数据，则采用以下方式，usecols指定了要加载的列，header表明数据中是否包含标题行，names形成新的列名
data_df = pd.read_csv(filename ,header=None, names=['x1', 'x2', 'h'],usecols=[0,1,2])
# print(data_df)

# 形成数组
row_array = data_df.values.tolist()
# print(row_array)
column_array = data_df.T.values.tolist()
# print(column_array)

# 转成矩阵
new_matrix = np.mat(row_array)
# print(new_matrix)
# 也是可以利用切片的，但类型是matrix，不是ndarray
X_nm = new_matrix[:, 0:2]
y_nm = new_matrix[:, 2:3]
# print(X_nm)
# print(y_nm)

# 这里注意的的是，可以使用getA()方法（例如，X_nm.getA(), y_nm.getA()）将上面的矩阵类型转化为ndarray类型
# 通常会做这样的处理

# 使用pandas写文件
# data_df.to_csv('pd_data.csv')

# 上面是数据的处理，分了两种方式，下面是一些函数
# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)

def stepFunction(t):
    '''在直线上方，则标注1，否则标注0 '''
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):   
    
    return stepFunction((np.matmul(X,W)+b)[0])


def perceptronStep(X, y, W, b, learn_rate = 0.01):
    ''' 
    y_hat 是预测的值，只能是为0或1，参见上面的stepFunction函数
    y_hat=1，表明点在直线上方
    y_hat=0，表明点在直线下方
    y[i] 是实际标签值，值为0或1，
    当 y[i]-y_hat == 0 时，表明区分正确，不做任何处理
    当 y[i]-y_hat == 1 时，只能说明y[i]为1，y_hat为0，
    即实际应该在直线上方的点被分到了直线的下方，需要直线向下移动，使用加法
    例如，原直线为ax + cy + b = 0，调整后(a + 0.01x)x + (c + 0.01y)y + b + 0.01 显然会大于0 
    当 y[i]-y_hat == -1 时，只能说明y[i]为0，y_hat为1，
    即实际应该在直线下方的点被分到了直线的上方，需要直线向上移动，使用减法
    '''
    for i in range(len(X)):
        y_hat = prediction(X[i],W,b)
        if y[i]-y_hat == 1:
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i]-y_hat == -1:
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
    return W, b    
    
# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines

# 测试
# print(X_matrix[0])
# print(X_matrix.T[0])
# print(np.random.rand(1))
# print(np.random.rand(2,1))

print(type(X_matrix))
print(type(X_nm))
print(type(X_nm.getA()))

# trainPerceptronAlgorithm(X_matrix, y_matrix)
# 这一语句同上面的作用是一样的，但是注意这里需要转成ndarray
trainPerceptronAlgorithm(X_nm.getA(), y_nm.getA())
# 原因：X_nm，y_nm的类型是矩阵，下面的测试中x_mina是求不出最小值的
# x_mina = min(X_nm.T[0])
# x_minb = min(X_nm.getA().T[0]) 
# print(x_mina)
# print(x_minb)



