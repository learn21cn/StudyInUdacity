#%%
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
import os

cwd = os.getcwd()
print(cwd)



# 定义函数来加载train，test和validation数据集
def load_dataset(path):
    data = load_files(path)
    print(np.array(data['target']))    
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 6)
    return dog_files, dog_targets

# 加载train，test和validation数据集
train_files, train_targets = load_dataset('./TestData/train')
print('There are %d training dog images.' % len(train_files))
print(train_targets)

#%%
from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # 用PIL加载RGB图像为PIL.Image.Image类型
    img = image.load_img(img_path, target_size=(224, 224))
    # 将PIL.Image.Image类型转化为格式为(224, 224, 3)的3维张量
    x = image.img_to_array(img)
    # 将3维张量转化为格式为(1, 224, 224, 3)的4维张量并返回
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

s = path_to_tensor('./TestData/train/001.Affenpinscher/Affenpinscher_00001.jpg')
# print(s)
print(s.shape)