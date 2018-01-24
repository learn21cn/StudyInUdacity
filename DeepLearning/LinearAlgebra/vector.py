import math
from decimal import Decimal, getcontext

# getcontext().prec = 15
class Vector(object):
    def __init__(self, coordinates):
        self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG = 'Can not normalize the zero vector'
        self.No_UNIQUE_PARALLEL_COMPONENT_MSG = 'There is no unique parallel component'
        self.No_UNIQUE_ORTHOGONAL_COMPONENT_MSG = 'There is no unique orthogonal component'
        self.ONLY_DEFINED_INTWO_RHREE_DIMS_MSG = 'Only support two or three dimention to compute cross product'

        try:
            if not coordinates:
                raise ValueError
            # self.coordinates = tuple(coordinates)
            # 这样的目的时确保所有坐标都是小数，而不是浮点数或者整数
            self.coordinates = tuple([Decimal(x) for x in coordinates])
            self.dimension = len(coordinates)
        
        except ValueError:
            raise ValueError('The coordinates must be nonempty')
        
        except TypeError:
            raise TypeError('The coordinates must be an iterable')
        
    def __str__(self):
        return 'Vector: {}'.format(self.coordinates)

    def __eq__(self, v):
        return self.coordinates == v.coordinates

    def plus(self, v):
        # 使用列表推导式
        # new_coordinates = [x+y for x, y in zip(self.coordinates, v.coordinates)]
        
        # 使用for循环
        new_coordinates = []
        n = len(self.coordinates)
        for i in range(n):
            new_coordinates.append(self.coordinates[i] + v.coordinates[i])            
        
        return Vector(new_coordinates)

    def minus(self, v):
        new_coordinates = [x-y for x, y in zip(self.coordinates, v.coordinates)]
        return Vector(new_coordinates)

    def times_scalar(self, c):
        new_coordinates = [c*x for x in self.coordinates]
        return Vector(new_coordinates)

    def magnitude(self):
        coordinates_squared = [x**2 for x in self.coordinates]
        return Decimal( math.sqrt(sum(coordinates_squared)))
        # return math.sqrt(sum(coordinates_squared))

    def normalized(self):
        try:
            magnitude = self.magnitude()
            # 确保为小数而不是浮点数或者整数
            return self.times_scalar(Decimal('1.0') / magnitude)
            # return self.times_scalar(1.0 / magnitude)
        except ZeroDivisionError:
            raise Exception('Can not normalize the zero vector')

    def dot(self, v):
        return sum([Decimal(x*y) for x, y in zip(self.coordinates, v.coordinates)])
       
    def angle_with(self, v, in_degrees=False):
        try:
            u1 = self.normalized()
            u2 = v.normalized()
            angle_in_radians = math.acos(u1.dot(u2))

            if in_degrees:
                degrees_per_radian = 180./ math.pi
                return angle_in_radians * degrees_per_radian
            else:
                return angle_in_radians
        
        except Exception as e:
            if str(e) == self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG:
                raise Exception('Can not compute an angle with zero vector')
            else:
                raise e
                
    def is_zero(self, tolerance = 1e-10):
        return self.magnitude() < tolerance

    def is_parallel_to(self, v):
        # print(self.angle_with(v))
        # print(math.pi)
        return(self.is_zero() or self.angle_with(v) == 0 or self.angle_with(v) == math.pi)

    def is_orthogonal_to(self, v, tolerance=1e-10):
        return abs(self.dot(v)) < tolerance

    # 投影 
    def component_parallel_to(self, basis):
        try:
            u = basis.normalized()
            weight = self.dot(u)
            return u.times_scalar(weight)    
        
        except Exception as e:
            if str(e) == self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG:
                raise Exception(self.No_UNIQUE_PARALLEL_COMPONENT_MSG)
            else:
                raise e


    def component_orthogonal_to(self, basis):
        try:
            projection = self.component_parallel_to(basis)
            return self.minus(projection)

        except Exception as e:
            if str(e) == self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG:
                raise Exception(self.No_UNIQUE_ORTHOGONAL_COMPONENT_MSG)
            else:
                raise e

    def cross(self, v):
        try:
            x_1, y_1, z_1 = self.coordinates
            x_2, y_2, z_2 = v.coordinates
            new_coordinates = [y_1*z_2 - y_2*z_1, -(x_1*z_2 - x_2*z_1), x_1*y_2 - x_2*y_1]
            return Vector(new_coordinates)

        except ValueError as e:
            msg = str(e)
            if msg =='need more than 2 values to unpack':
                self_embedded_in_R3 = Vector(self.coordinates + ('0',))
                v_embedded_in_R3 = Vector(v.coordinates + ('0',))
                return self_embedded_in_R3.cross(v_embedded_in_R3)
            elif msg == 'too many values to unpack' or msg == 'need more than 1 value to unpack':
                raise Exception(self.ONLY_DEFINED_INTWO_RHREE_DIMS_MSG)
            else:
                raise e

    def area_of_triangle_with(self, v):
        return self.area_of_parallelogram_with(v) / Decimal('2.0')

    def area_of_parallelogram_with(self, v):
        cross_product = self.cross(v)
        return cross_product.magnitude()




# a = Vector([1, 2, 3])
# print(a)

# v11 = Vector([8.218, -9.341])
# v12 = Vector([-1.129, 2.111])
# print(v11.plus(v12))


# v21 = Vector([7.119, 8.215])
# v22 = Vector([-8.223, 0.878])
# print(v21.minus(v22))

# c = 7.41
# v = Vector([1.671, -1.012, -0.318] )
# print (v.times_scalar(c))

# v = Vector([-0.221, 7.437])
# print(v.magnitude())

# v = Vector([8.813, -1.331, -6.247])
# print(v.magnitude())

# v = Vector([5.581, -2.136])
# print(v.normalized())

# v = Vector([1.996, 3.108, -4.554])
# print(v.normalized()) 

# print(Decimal('1.0') / Decimal('1.5'))

# v = Vector([7.887, 4.138])
# w = Vector([-8.802, 6.776])
# print(v.dot(w))

# v = Vector([-5.955, -4.904, -1.874])
# w = Vector([-4.496, -8.755, 7.103])
# print(v.dot(w))

# v = Vector([3.183, -7.627])
# w = Vector([-2.668, 5.319])
# print(v.angle_with(w, False))


# v = Vector([7.35, 0.221, 5.188])
# w = Vector([2.751, 8.259, 3.985])
# print(v.angle_with(w, True))

# v = Vector([-7.579, -7.88])
# w = Vector([22.737, 23.64])

# v = v.normalized()
# w = w.normalized()
# getcontext().prec = 10  
            
# print(v.dot(w))
# print(getcontext())
# result = math.acos(Decimal(v.dot(w)))
# print(result)
# print(v.is_parallel_to(w))
# print(v.is_orthogonal_to(w))


# v = Vector([-2.029, 9.97, 4.172])
# w = Vector([-9.231, -6.639, -7.245])
# print(v.is_parallel_to(w))
# print(v.is_orthogonal_to(w))

# v = Vector([-2.328, -7.284, -1.214])
# w = Vector([-1.821, 1.072, -2.94])
# print(v.is_parallel_to(w))
# print(v.is_orthogonal_to(w))

# v = Vector([3.039, 1.879])
# b = Vector([0.825, 2.036])
# print(v.component_parallel_to(b))

# v = Vector([3.009, -6.172, 3.692, -2.51])
# b = Vector([6.404, -9.144, 2.759, 8.718])
# print(v.component_parallel_to(b))
# print(v.component_orthogonal_to(b))

# v = Vector([-9.88, -3.264, -8.159])
# b = Vector([-2.155, -9.353, -9.473])
# print(v.component_parallel_to(b))
# print(v.component_orthogonal_to(b))

# a = [1, 2, 3]
# # b = tuple([x for x in a])
# b = tuple([Decimal(x) for x in a])
# print(b + ('0',))

# v = Vector([8.462, 7.893,-8.187])
# w = Vector([6.984, -5.975, 4.778])
# print(v.cross(w))

# v = Vector([-8.987, -9.838, 5.031])
# w = Vector([-4.268, -1.861, -8.866])
# print(v.area_of_parallelogram_with(w))

# v = Vector([1.5, 9.547, 3.691])
# w = Vector([-6.007, 0.124, 5.772])
# print(v.area_of_triangle_with(w))