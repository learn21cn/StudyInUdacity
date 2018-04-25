#%%
import numpy as np
from keras.utils import np_utils
import tensorflow as tf
#%% 
def run():
    output = None
    x = tf.placeholder(tf.int32)

    with tf.Session() as sess:
        output = sess.run(x, feed_dict={x: 123})

    return output

run()
#%%
x = tf.constant(20)
y = tf.constant(2)
z = tf.subtract(tf.divide(x,y),tf.cast(tf.constant(1), tf.float64))

# TODO: Print z from a session
with tf.Session() as sess:
    output = sess.run(z)
    print(output)


#%%

# tf.Variable 类创建一个 tensor，其初始值可以被改变，就像普通的 Python 变量一样。
# 该 tensor 把它的状态存在 session 里，所以你必须手动初始化它的状态。
# 使用 tf.global_variables_initializer() 函数来初始化所有可变 tensor
def get_weights(n_features, n_labels):
    """
    Return TensorFlow weights
    :param n_features: Number of features
    :param n_labels: Number of labels
    :return: TensorFlow weights
    在tf.truncated_normal中如果x的取值在区间（μ-2σ，μ+2σ）之外则重新进行选择。这样保证了生成的值都在均值附近。
    """    
    return tf.Variable(tf.truncated_normal((n_features, n_labels)))

def get_biases(n_labels):
    """
    Return TensorFlow bias
    :param n_labels: Number of labels
    :return: TensorFlow bias
    """
    return tf.Variable(tf.zeros(n_labels))

def linear(input, w, b):
    """
    Return linear function in TensorFlow
    :param input: TensorFlow input
    :param w: TensorFlow weights
    :param b: TensorFlow biases
    :return: TensorFlow linear function
    """   
    return tf.add(tf.matmul(input, w), b)
#%%
import os
# 注意这里的工作目录
cwd = os.getcwd()
print(cwd)

#%%
from tensorflow.examples.tutorials.mnist import input_data

def mnist_features_labels(n_labels):
    """
    Gets the first <n> labels from the MNIST dataset
    :param n_labels: Number of labels to use
    :return: Tuple of feature list and label list
    """
    mnist_features = []
    mnist_labels = []
    
    # input_data这个脚本会检查路径中数据集是否已经存在，如果不存在则进行下载
    # 如果数据集下载不了，请到下面的地址手动下载    
    # http://yann.lecun.com/exdb/mnist/
    mnist = input_data.read_data_sets('../../../source/datasets/mnist', one_hot=True)

    # In order to make quizzes run faster, we're only looking at 10000 images
    for mnist_feature, mnist_label in zip(*mnist.train.next_batch(10000)):

        # Add features and labels if it's for the first <n>th labels
        if mnist_label[:n_labels].any():
            mnist_features.append(mnist_feature)
            mnist_labels.append(mnist_label[:n_labels])   

    return mnist_features, mnist_labels

#%%
# Number of features (28*28 image is 784 features)
n_features = 784
# Number of labels
n_labels = 3

# Features and Labels
features = tf.placeholder(tf.float32)
labels = tf.placeholder(tf.float32)

# Weights and Biases
w = get_weights(n_features, n_labels)
b = get_biases(n_labels)

# Linear Function xW + b
logits = linear(features, w, b)

# Training data
train_features, train_labels = mnist_features_labels(n_labels)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    # Softmax
    prediction = tf.nn.softmax(logits)

    # Cross entropy
    # This quantifies how far off the predictions were.
    # You'll learn more about this in future lessons.
    cross_entropy = -tf.reduce_sum(labels * tf.log(prediction), reduction_indices=1)

    # Training loss
    # You'll learn more about this in future lessons.
    loss = tf.reduce_mean(cross_entropy)

    # Rate at which the weights are changed
    # You'll learn more about this in future lessons.
    learning_rate = 0.08

    # Gradient Descent
    # This is the method used to train the model
    # You'll learn more about this in future lessons.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Run optimizer and get loss
    _, l = session.run(
        [optimizer, loss],
        feed_dict={features: train_features, labels: train_labels})

# Print loss
print('Loss: {}'.format(l))

#%%
# Softmax 函数可以把它的输入，通常被称为 logits 或者 logit scores，处理成 0 到 1 之间，并且能够把输出归一化到和为 1。
# 这意味着 softmax 函数与分类的概率分布等价。
# 它是一个网络预测多分类问题的最佳输出激活函数。
# tf.nn.softmax() 直接实现了 softmax 函数

output = None
logit_data = [2.0, 1.0, 0.1]
logits = tf.placeholder(tf.float32)
    
# TODO: Calculate the softmax of the logits
softmax = tf.nn.softmax(logits)     

with tf.Session() as sess:
    # TODO: Feed in the logit data
    output = sess.run(softmax, feed_dict={logits: logit_data})
print(output)

#%%

# tf.reduce_sum() 函数输入一个序列，返回它们的和
# x = tf.reduce_sum([1, 2, 3, 4, 5])  # 15
# tf.log() 它返回所输入值的自然对数。
softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)

cross_entropy = -tf.reduce_sum(tf.multiply(one_hot, tf.log(softmax)))
with tf.Session() as sess:
    output = sess.run(cross_entropy, feed_dict={softmax: softmax_data, one_hot: one_hot_data})
    print(output)

#%%
# from tensorflow.examples.tutorials.mnist import input_data
# import tensorflow as tf

n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# Import MNIST data
mnist = input_data.read_data_sets('../../../source/datasets/mnist', one_hot=True)

# The features are already scaled and the data is shuffled
train_features = mnist.train.images
print(train_features.size)
# 以上打印的值乘以4即为占用的内存字节数 （每个float32占用4个字节内存）
test_features = mnist.test.images

train_labels = mnist.train.labels.astype(np.float32)
test_labels = mnist.test.labels.astype(np.float32)

# Weights & bias
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

#%%
def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    # TODO: Implement batching
    output_batches = []
    
    sample_size = len(features)
    # 第三个参数为步长
    for start_i in range(0, sample_size, batch_size):
        
        print('start_i:')
        print(start_i)
        end_i = start_i + batch_size
        print('end_i:')
        print(end_i)
        batch = [features[start_i:end_i], labels[start_i:end_i]]
        output_batches.append(batch)
        
    return output_batches
example_features = [
    ['F11','F12','F13','F14'],
    ['F21','F22','F23','F24'],
    ['F31','F32','F33','F34'],
    ['F41','F42','F43','F44']]
# 4 Samples of labels
example_labels = [
    ['L11','L12'],
    ['L21','L22'],
    ['L31','L32'],
    ['L41','L42']]

# PPrint prints data structures like 2d arrays, so they are easier to read
from pprint import pprint
pprint(batches(2, example_features, example_labels))
# 注意这里不用考虑溢出
pprint(example_features[2:9])


#%%
# ReLU是一个非线性函数，又称修正线性单元。
# ReLU 函数对所有负的输入，返回 0；所有 x >0 的输入，返回 x
hidden_layer_weights = [
    [0.1, 0.2, 0.4],
    [0.4, 0.6, 0.6],
    [0.5, 0.9, 0.1],
    [0.8, 0.2, 0.8]]
out_weights = [
    [0.1, 0.6],
    [0.2, 0.1],
    [0.7, 0.9]]

# Weights and biases
weights = [
    tf.Variable(hidden_layer_weights),
    tf.Variable(out_weights)]
biases = [
    tf.Variable(tf.zeros(3)),
    tf.Variable(tf.zeros(2))]

# Input
features = tf.Variable([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0], [11.0, 12.0, 13.0, 14.0]])

# TODO: Create Model with Dropout
keep_prob = tf.placeholder(tf.float32)
hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
hidden_layer = tf.nn.relu(hidden_layer)
hidden_layer = tf.nn.dropout(hidden_layer, keep_prob)

logits = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])

# TODO: Print logits from a session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(logits, feed_dict={keep_prob: 0.5}))


