# import math
# from decimal import Decimal, getcontext
# from vector import Vector

class Test(object):
    
    # BASEPT_AND_DIR_VECTORS_MUST_BE_IN_SAME_DIM_MSG = ('The basepoint and direction vectors all live in the same dimension')

    def __init__(self, basepoint, direction_vectors):
   
        
        self.basepoint = basepoint
        self.direction_vectors = direction_vectors

        # self.dimension = self.basepoint.dimension 

        # print('=======================')  
        # print(self.basepoint)
        print(self.direction_vectors)  

        # print('=======================')     

        # try:
        #     print(direction_vectors)
        #     for item in direction_vectors:
        #         v = Vector(item)
        #         print(v)                
        #         assert v.dimension == self.dimension
        
        # except AssertionError:
        #     raise Exception(BASEPT_AND_DIR_VECTORS_MUST_BE_IN_SAME_DIM_MSG)


# Vector: (Decimal('-0.3010471533'), Decimal('-0.4919135404'), Decimal('0'))
# [[Decimal('-0.09084497715'), Decimal('0.5086234347'), 1]]


print(12345)
if __name__ == "__main__":
    # p = Test(basepoint = Vector([Decimal('-0.3010471533'), Decimal('-0.4919135404'), Decimal('0')]), direction_vectors=[[Decimal('-0.09084497715'), Decimal('0.5086234347'), 1]])

    # p = Test(basepoint = Vector([Decimal('-0.3010471533'), Decimal('-0.4919135404'), Decimal('0')]), direction_vectors=[Decimal('-0.09084497715'), Decimal('0.5086234347'), 1])

    # print(p)

    t = Test(basepoint = 1, direction_vectors =2)

    print(t)




