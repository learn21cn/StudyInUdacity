# TODO 实现 Gaussain Jordan 方法求解 Ax = b

"""
步骤1 检查A，b是否行数相同
步骤2 构造增广矩阵Ab
步骤3 逐列转换Ab为化简行阶梯形矩阵 中文维基链接
对于Ab的每一列（最后一列除外）
    当前列为列c
    寻找列c中 对角线以及对角线以下所有元素（行 c~N）的绝对值的最大值
    如果绝对值最大值为0
        那么A为奇异矩阵，返回None (你可以在选做问题2.4中证明为什么这里A一定是奇异矩阵)
    否则
        使用第一个行变换，将绝对值最大值所在行交换到对角线元素所在行（行c） 
        使用第二个行变换，将列c的对角线元素缩放为1
        多次使用第三个行变换，将列c的其他元素消为0
步骤4 返回Ab的最后一列
"""

""" Gaussian Jordan 方法求解 Ax = b.
    参数
        A: 方阵 
        b: 列向量
        decPts: 四舍五入位数，默认为4
        epsilon: 判读是否为0的阈值，默认 1.0e-16
        
    返回列向量 x 使得 Ax = b 
    返回None，如果 A，b 高度不同
    返回None，如果 A 为奇异矩阵
"""

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



A =  [[-7,5,-1],
     [1,-3,-8],
     [-10,-2,9]]

b = [[1],[1],[1]]

m = gj_Solve(A, b, decPts=4, epsilon = 1.0e-16)

print(m)
print(b)
