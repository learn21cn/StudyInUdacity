

def transpose(M):
    # new_col_count = len(M)
    print(M)
    for i in zip(*M):
        print(i)
    print(list(zip(*M)))
    return list(map(list,zip(*M)))

# 这个zip在不同版本的不同反应了python的一个演变：python3中大部分返回list的函数不在返回list，而是返回一个支持遍历的对象，比如map、fiter之类

def transpose2(M):
    for i in range(len(M)):
        for j in range(i):
            M[i][j], M[j][i] = M[j][i], M[i][j]
    return M


def multi(M1, M2):
    if isinstance(M1, (float, int)) and isinstance(M2, (tuple, list)):
        return [[M1*i for i in j] for j in M2]
    if isinstance(M1, (tuple, list)) and isinstance(M2, (tuple, list)):
        return [[sum(map(lambda x: x[0]*x[1], zip(i,j))) for j in zip(*M2)] for i in M1]



def matrixMul(A, B):
    res = [[0] * len(B[0]) for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                res[i][j] += A[i][k] * B[k][j]
    return res

def matrixMul2(A, B):
    return [[sum(a * b for a, b in zip(a, b)) for b in zip(*B)] for a in A]


def matxMultiply(A, B):
    countlist_A = [len(c) for c in A]
    countset_A = set(countlist_A)
    
    countlist_B = [len(c) for c in B]
    countset_B = set(countlist_B)
    
    try:
        assert len(countset_A) ==1 and len(countset_B) ==1 
        
        if len(B) == len(A[0]): 
            # print(list(zip(*B)))
            print([[ list(zip(a,b)) for b in zip(*B)]  for a in A])  
            
            # 这个是错误的结果
            print([[sum(i*j for i, j in zip(a, b)) for a in A] for b in zip(*B)])
            # 这个是正确的结果，注意顺序
            return [[sum(i*j for i, j in zip(a, b)) for b in zip(*B)] for a in A]
            
        else:
            raise ValueError('矩阵A的列数与B的行数不相等，不能相乘')
    
    except AssertionError:
        raise Exception('A和必须是是有效的矩阵')


a = [[1,2], [3,4], [5,6], [7,8]]
b = [[1,2,3,4], [5,6,7,8]]
# print(matrixMul(a,b))
# print(matrixMul(b,a))
# print("-"*90)
# print(matrixMul2(a,b))
# print(matrixMul2(b,a))

# print("-"*90)
# print(multi(a, b))

# print(matxMultiply(a,b))



I = [[1,0,2,0],
     [0,1,0,0],
     [0,0,1,0],
     [0,0,0,1]]

# print(transpose(I))
# print(transpose2(I))


# 用于测试
m1 = [1,1]
m2 = [2,5]
l = list(zip(m1,m2))
# print(l)
# print([sum(i*j for i, j in l)])


A = [[1,1]]
B = [[2],[5]]
# 虽然这个计算的结果和下面是一样的，但过程是错误的，多维的情况下会不适应
# print([[sum(i*j for i, j in zip(a, b)) for a in A] for b in zip(*B)])
# 正确的方式
# print([[sum(i*j for i, j in zip(a, b)) for b in zip(*B)] for a in A])

# print(matxMultiply(A,B))


# 增广矩阵
b =[[1],[2],[3],[4]]

l = list( map(list,zip(I,b)) )

# print(l)
# 正确结果
print([i+j for i,j in zip(I,b)] )


def scaleRow(M, r, scale):
    if scale ==0:
        raise ValueError('scale的值不能为零')
    else:
        M[r] = [scale * i for i in M[r]]

scaleRow(I,2,5)
print(I)

def addScaledRow(M, r1, r2, scale):
    M[r1] = [scale * j + i for i,j in zip(M[r1],M[r2])]



M = [[1,0,2,10],
     [9,1,0,0],
     [0,15,1,0],
     [9,0,0,5]]

result = [x for xx in M for x in xx if 2<x<10]

result = [abs(x) for xx in M for x in xx if 2<x<10]

result =[ [i,j, abs (M[i][j])] for i in range(len(M)) for j in range(i+1) ]


m =[  abs (M[i][j]) for i in range(len(M)) for j in range(i+1) ]

# print(max(M))
# print(M.index(max(M)))

print(max(m))

print(result)
# print(max(result))
# print(list(zip(*result)))
# 相当于上面的列表表达式
for i in range(len(M)):
    for j in range(i+1):
        pass
        # print(M[i][j])



for j in range(len(M)):
    col_list = [ abs(M[i][j]) for i in range(j,len(M))]
    m = max(col_list)
    i = j + col_list.index(m)
    print(m,i)




    
 
        
