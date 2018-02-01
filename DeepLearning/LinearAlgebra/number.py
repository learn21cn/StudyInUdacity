import numpy as np

from pylab import *  

from helper import *
from matplotlib import pyplot as plt

# print(np.random.rand(4,4))

# x_values = arange(0.0, math.pi * 4, 0.01)  
# y_values = sin(x_values)  
# plot(x_values, y_values, linewidth=1.0)  
# xlabel('x')  
# ylabel('sin(x)')  
# title('Simple plot')  
# grid(True)  
# savefig("sin.png")  
# show()  



# %matplotlib inline

X,Y = generatePoints(seed=15,num=100)
# print(X)

# print([[x, 1] for x in X])
# print([ [x,y] for x,y in zip(X, Y) ])


# print(len(list(zip(X,Y))))
# print(len(X))
# print(len(Y))

## 可视化
# plt.xlim((-5,5))
# plt.xlabel('x',fontsize=18)
# plt.ylabel('y',fontsize=18)
# plt.scatter(X,Y,c='b')
# plt.show()

def calculateMSE(X,Y,m,b): 
    n = len(list(zip(X,Y)))    
    # return  sum([math.pow((y - m * x -b),2)  for x,y in zip(X,Y)]) /n
    return  sum([(y - m * x -b) * (y - m * x -b) for x,y in zip(X,Y)]) /n

# m1, b1 = 3.5, 7
# print('test',calculateMSE(X,Y,m1,b1))

def linearRegression(X,Y):
    
    MX = [[x, 1] for x in X]
    MY = [[y] for y in Y]    
   
    MXT = transpose(MX)
    A = matxMultiply(MXT, MX)
    b = matxMultiply(MXT, MY)
    
    M = augmentMatrix(A, b)    
   
    result = gj_Solve(A, b, decPts=4, epsilon = 1.0e-16)
    print(result)

    print(result[0][0],result[1][0])
    
    return result[0][0],result[1][0]


def gj_Solve(A, b, decPts=4, epsilon = 1.0e-16):
    # 检查行数是否相等
    if len(A) != len(b):
        return None
    
    else:
        # 构造增广矩阵Ab
        M = augmentMatrix(A, b)
        # print(M)
        
        # 对于每一列j
        for j in range(len(M)):
            # 获取特定列含对角线以及以下元素的最大值以及所在行
            col_list = [ abs(M[i][j]) for i in range(j,len(M))]
            max_value = max(col_list)
            i = j + col_list.index(max_value)
            
            # 是否为奇异矩阵
            if max_value <= epsilon:
                return None
            
            else:
                # 交换 将绝对值最大值所在行i交换到对角线元素所在行j
                swapRows(M, i, j)
                # 将列j的对角线元素缩放为1
                scaleRow(M, j, 1/M[j][j])
                
                # 将列j的其他元素消为0
                for k in range(0,len(M)):
                    if k != j:
                        addScaledRow(M, k, j, -M[k][j])

        result_list = list(map(list, zip(*M)))[len(M)] 
        result_b = [[round(c, decPts)] for c in result_list]               
        # return M
        return result_b
        
        
# 构造增广矩阵    
def augmentMatrix(A, b):
    return [i+j for i,j in zip(A,b)] 

# 交换两行
def swapRows(M, r1, r2):
    M[r1], M[r2] = M[r2], M[r1]

# 扩大每行
def scaleRow(M, r, scale):
    if scale ==0:
        raise ValueError('scale的值不能为零')
    else:
        M[r] = [scale * i for i in M[r]]

# 将r2扩大scale后加到r1上        
def addScaledRow(M, r1, r2, scale):
    M[r1] = [scale * j + i for i,j in zip(M[r1],M[r2])] 


def matxMultiply(A, B):
    countlist_A = [len(c) for c in A]
    countset_A = set(countlist_A)
    
    countlist_B = [len(c) for c in B]
    countset_B = set(countlist_B)
    
    try:
        assert len(countset_A) ==1 and len(countset_B) ==1 
        
        if len(B) == len(A[0]):   
            return [[sum(a * b for a, b in zip(a, b)) for b in zip(*B)] for a in A]
        else:
            raise ValueError('矩阵A的列数与B的行数不相等，不能相乘')
    
    except AssertionError:
        raise Exception('A和必须是是有效的矩阵')    


def transpose(M):    
    
    return list(map(list, zip(*M)))

# TODO 每个元素四舍五入到特定小数数位
# 直接修改参数矩阵，无返回值
def matxRound(M, decPts=4):    
    for row in M:
        for i,c in enumerate(row):
#             row[i] = Decimal(c).quantize(Decimal('0.' + '0'*decPts )) 
            row[i] = round(c, decPts) 


linearRegression(X,Y)


