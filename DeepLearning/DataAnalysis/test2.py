#%%
import numpy as np
# PROBLEM 2
# 
# Figure out what value of the boost velocity will allow the spaceship to 
# return safely to earth. In order to do this, you will 
# have to fill in the moon_position, acceleration, 
# and apply_boost functions. Further details are below.
# 

import math
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns


earth_mass = 5.97e24 # kg
earth_radius = 6.378e6 # m (at equator)
gravitational_constant = 6.67e-11 # m3 / kg s2
moon_mass = 7.35e22 # kg
moon_radius = 1.74e6 # m
moon_distance = 400.5e6 # m (actually, not at all a constant)
moon_period = 27.3 * 24.0 * 3600. # s
moon_initial_angle = math.pi / 180. * -61. # radian

total_duration = 12. * 24. * 3600. # s
marker_time = 0.5 * 3600. # s
tolerance = 100000. # m

def moon_position(time):
# Task 1: Compute the moon's position (a vector) at time t. Let it start at moon_initial_angle, not on the horizontal axis.   
    ###Your code here.
    moon_angle = moon_initial_angle + 2.0 * math.pi * time / moon_period
    position = np.zeros(2)
    position[0] = moon_distance * np.array(math.cos(moon_angle))
    position[1] = moon_distance * np.array(math.sin(moon_angle))

    return position

def acceleration(time, position):
    # Task 2: Compute the spacecraft's acceleration due to gravity
	###Your code here.
    moon_pos = moon_position(time)
    vector_from_moon = position - moon_pos
    vector_from_earth = position    
    acc = -gravitational_constant * ((earth_mass / np.linalg.norm(vector_from_earth)** 3) * vector_from_earth + (moon_mass / np.linalg.norm(vector_from_moon)** 3) * vector_from_moon)
    return acc

axes = plt.gca()
axes.set_xlabel('Longitudinal position in m')
axes.set_ylabel('Lateral position in m')

# Task 5: (First see the other tasks below.) What is the appropriate boost to apply?
# Try -10 m/s, 0 m/s, 10 m/s, 50 m/s and 100 m/s and leave the correct amount in as you submit the solution.

def apply_boost():

    # Do not worry about the arrays position_list, velocity_list, and times_list.  
    # They are simply used for plotting and evaluating your code, so none of the 
    # code that you add should involve them.
    
    boost = 10. # m/s Change this to the correct value from the list above after everything else is done.
    position_list = [np.array([-6.701e6, 0.])] # m
    velocity_list = [np.array([0., -10.818e3])] # m / s
    times_list = [0]
    position = position_list[0]
    velocity = velocity_list[0]
    current_time = 0.
    h = 0.1 # s, set as initial step size right now but will store current step size
    h_new = h # s, will store the adaptive step size of the next step
    mcc2_burn_done = False
    dps1_burn_done = False

    while current_time < total_duration:
        #Task 3: Include a retrograde rocket burn at 101104 seconds that reduces the velocity by 7.04 m/s
        # and include a rocket burn that increases the velocity at 212100 seconds by the amount given in the variable called boost.
        # Both velocity changes should happen in the direction of the rocket's motion at the time they occur.
        
        ###Your code here.
        if not mcc2_burn_done and current_time >= 101104:
            velocity -= 7.04 / np.linalg.norm(velocity) * velocity
            mcc2_burn_done = True
            
        if not dps1_burn_done and current_time >= 212100:
            velocity += boost / np.linalg.norm(velocity) * velocity
            dps1_burn_done = True

        #Task 4: Implement Heun's method with adaptive step size. Note that the time is advanced at the end of this while loop.
        ###Your code here.

        ###Your code here.
        acceleration0 = acceleration(current_time, position)
        velocityE = velocity + h * acceleration0
        positionE = position + h * velocity
        
        velocityH = velocity + h * 0.5 * (acceleration0 + acceleration(current_time + h, positionE))
        positionH = position + h * 0.5 * (velocity + velocityE)
        
        velocity = velocityH
        position = positionH
        
        error =  np.linalg.norm(positionH - positionE) + total_duration * np.linalg.norm(velocityH - velocityE)
        h_new = h * math.sqrt(tolerance / error)

        h_new = min(0.5 * marker_time, max(0.1, h_new)) # restrict step size to reasonable range
            
        current_time += h
        h = h_new
        position_list.append(position.copy())
        velocity_list.append(velocity.copy())
        times_list.append(current_time)

    return position_list, velocity_list, times_list, boost

position, velocity, current_time, boost = apply_boost()


def plot_path(position_list, times_list):
    axes = plt.gca()
    axes.set_xlabel('Longitudinal position in m')
    axes.set_ylabel('Lateral position in m')
    previous_marker_number = -1;
    for position, current_time in zip(position_list, times_list):
         if current_time >= marker_time * previous_marker_number:
            previous_marker_number += 1
            plt.scatter(position[0], position[1], s = 2., facecolor = 'r', edgecolor = 'none')
            moon_pos = moon_position(current_time)
            if np.linalg.norm(position - moon_pos) < 30. * moon_radius: 
                axes.add_line(matplotlib.lines.Line2D([position[0], moon_pos[0]], [position[1], moon_pos[1]], alpha = 0.3, c = 'g')) 
    axes.add_patch(matplotlib.patches.CirclePolygon((0., 0.), earth_radius, facecolor = 'none', edgecolor = 'b'))
    for i in range(int(total_duration / marker_time)):
        moon_pos = moon_position(i * marker_time)
        axes.add_patch(matplotlib.patches.CirclePolygon(moon_pos, moon_radius, facecolor = 'none', edgecolor = 'g', alpha = 0.7))

    plt.axis('equal')

plot_path(position, current_time)

	
#%%
h = 1
num_steps = 5
x = np.zeros([11,2])
x[0, 0] = 1
x[0, 1] = 2

for step in range(num_steps):
    x[step + 1] = x[step] + h * x[step] 
print(x)
#%%

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
print(a + b)

#%%
a = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
    
print(a.sum())
# 若数轴为0，则计算每一列
print(a.sum(axis=0))
# 若数轴为1，则计算每一行
print(a.sum(axis=1))

#%%
# Subway ridership for 5 stations on 10 different days
ridership = np.array([
    [   0,    0,    2,    5,    0],
    [1478, 3877, 3674, 2328, 2539],
    [1613, 4088, 3991, 6461, 2691],
    [1560, 3392, 3826, 4787, 2613],
    [1608, 4802, 3932, 4477, 2705],
    [1576, 3933, 3909, 4979, 2685],
    [  95,  229,  255,  496,  201],
    [   2,    0,    1,   27,    0],
    [1438, 3785, 3589, 4174, 2215],
    [1342, 4043, 4009, 4665, 3033]
])

# 起始行列都是从0开始
# print(ridership[1, 3])
# print(ridership[1:3, 3:5])
# print(ridership[1, :])

# 前两行相加
# print(ridership[0, :] + ridership[1, :])
# 前两列相加
# print(ridership[:, 0] + ridership[:, 1])


def mean_riders_for_max_station(ridership):
    '''
    Fill in this function to find the station with the maximum riders on the
    first day, then return the mean riders per day for that station. Also
    return the mean ridership overall for comparsion.
    
    Hint: NumPy's argmax() function might be useful:
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html
    '''
    max_station = ridership[0, :].argmax()

    mean_for_max = ridership[:, max_station].mean()

    overall_mean = ridership.mean()
        
    return (overall_mean, mean_for_max)

result = mean_riders_for_max_station(ridership)
print(result)


def min_and_max_riders_per_day(ridership):
    '''
    Fill in this function. First, for each subway station, calculate the
    mean ridership per day. Then, out of all the subway stations, return the
    maximum and minimum of these values. That is, find the maximum
    mean-ridership-per-day and the minimum mean-ridership-per-day for any
    subway station.
    '''
    # 按列计算 
    station_riders = ridership.mean(axis=0)

    max_daily_ridership = station_riders.max()    
    min_daily_ridership = station_riders.min()
    return (max_daily_ridership, min_daily_ridership)


result = min_and_max_riders_per_day(ridership)
print(result)

#%%
import pandas as pd
# You can create a DataFrame out of a dictionary mapping column names to values
df_1 = pd.DataFrame({'A': [1, 1, 2], 'B': [3, 4, 5]})
print(df_1)

# You can also use a list of lists or a 2D NumPy array
df_2 = pd.DataFrame([[0, 1, 2], [3, 4, 5]], columns=['A', 'B', 'C'])
print(df_2)
# 注意以上两种方式的差别

#%%
df = pd.DataFrame({'A': [0, 1, 2], 'B': [3, 4, 5]})
print(df.sum())
print(df.sum(axis=1))
print(df.values.sum())

#%%
# Subway ridership for 5 stations on 10 different days
ridership_df = pd.DataFrame(
    data=[[   0,    0,    2,    5,    0],
          [1478, 3877, 3674, 2328, 2539],
          [1613, 4088, 3991, 6461, 2691],
          [1560, 3392, 3826, 4787, 2613],
          [1608, 4802, 3932, 4477, 2705],
          [1576, 3933, 3909, 4979, 2685],
          [  95,  229,  255,  496,  201],
          [   2,    0,    1,   27,    0],
          [1438, 3785, 3589, 4174, 2215],
          [1342, 4043, 4009, 4665, 3033]],
    index=['05-01-11', '05-02-11', '05-03-11', '05-04-11', '05-05-11',
           '05-06-11', '05-07-11', '05-08-11', '05-09-11', '05-10-11'],
    columns=['R003', 'R004', 'R005', 'R006', 'R007']
)

# 第一行
# print(ridership_df.iloc[0])
# 按行索引打印
# print(ridership_df.loc['05-05-11'])
# 按列名打印
# print(ridership_df['R003'])
# 按坐标打印（第二行第四列）
# print(ridership_df.iloc[1, 3])

# 第二三四行
# print(ridership_df.iloc[1:4])

print(ridership_df[['R003', 'R005']])

#%%
def mean_riders_for_max_station2(ridership):
    # 这里返回的是列名，不同于numpy(numpty返回的是位置)
    max_station = ridership.iloc[0].idxmax()
    print(max_station)
    mean_for_max = ridership[max_station].mean()

    # 无法计算数据框的总平均值
    # 因此要使用.values来获取numpy数组的平均值
    overall_mean = ridership.values.mean()
        
    return (overall_mean, mean_for_max)

result = mean_riders_for_max_station2(ridership_df)
print(result)

#%%
filename = '../DataMaterials/nyc-subway-weather.csv'
subway_df = pd.read_csv(filename)

print(subway_df.head())
# subway_df.describe()


def correlation(x, y):
    '''
    Fill in this function to compute the correlation between the two
    input variables. Each input is either a NumPy array or a Pandas
    Series.
    
    correlation = average of (x in standard units) times (y in standard units)
    
    Remember to pass the argument "ddof=0" to the Pandas std() function!
    '''

    std_x = (x - x.mean()) / x.std(ddof=0)
    std_y = (y - y.mean()) / y.std(ddof=0)
    return (std_x * std_y).mean()

entries = subway_df['ENTRIESn_hourly']
cum_entries = subway_df['ENTRIESn']
rain = subway_df['meanprecipi']
temp = subway_df['meantempi']

print(correlation(entries, rain))
print(correlation(entries, temp))
print(correlation(rain, temp))

print(correlation(entries, cum_entries))

#%%
df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
df2 = pd.DataFrame({'a': [10, 20, 30], 'b': [40, 50, 60], 'c': [70, 80, 90]})
print(df1 + df2)

#%%
df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
df2 = pd.DataFrame({'d': [10, 20, 30], 'c': [40, 50, 60], 'b': [70, 80, 90]})
print(df1 + df2)

#%%
# Adding DataFrames with overlapping row indexes
# 有行索引的话按行相加
df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]},
                    index=['row1', 'row2', 'row3'])
df2 = pd.DataFrame({'a': [10, 20, 30], 'b': [40, 50, 60], 'c': [70, 80, 90]},
                    index=['row4', 'row3', 'row2'])
print(df1 + df2)

# row1    1   4   7
# row2    2   5   8
# row3    3   6   9

# row4    10  40  70
# row3    20  50  80
# row2    30  60  90



# Cumulative entries and exits for one station for a few hours.
entries_and_exits = pd.DataFrame({
    'ENTRIESn': [3144312, 3144335, 3144353, 3144424, 3144594,
                 3144808, 3144895, 3144905, 3144941, 3145094],
    'EXITSn': [1088151, 1088159, 1088177, 1088231, 1088275,
               1088317, 1088328, 1088331, 1088420, 1088753]
})

# print(entries_and_exits.shift(1))

def get_hourly_entries_and_exits(entries_and_exits):
    '''
    Fill in this function to take a DataFrame with cumulative entries
    and exits (entries in the first column, exits in the second) and
    return a DataFrame with hourly entries and exits (entries in the
    first column, exits in the second).
    '''
    return entries_and_exits - entries_and_exits.shift(1)

get_hourly_entries_and_exits(entries_and_exits)


#%%
# DataFrame applymap()

df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [10, 20, 30],
    'c': [5, 10, 15]
})

def add_one(x):
    return x + 1
    
print(df.applymap(add_one))

grades_df = pd.DataFrame(
    data={'exam1': [43, 81, 78, 75, 89, 70, 91, 65, 98, 87],
          'exam2': [24, 63, 56, 56, 67, 51, 79, 46, 72, 60]},
    index=['Andre', 'Barry', 'Chris', 'Dan', 'Emilio', 
           'Fred', 'Greta', 'Humbert', 'Ivan', 'James']
)

def convert_grade(grade):
    if grade >= 90:
        return 'A'
    elif grade >= 80:
        return 'B'
    elif grade >= 70:
        return 'C'
    elif grade >= 60:
        return 'D'
    else:
        return 'F'     
    
def convert_grades(grades):
    '''
    Fill in this function to convert the given DataFrame of numerical
    grades to letter grades. Return a new DataFrame with the converted
    grade.
    
    The conversion rule is:
        90-100 -> A
        80-89  -> B
        70-79  -> C
        60-69  -> D
        0-59   -> F
    '''
    return grades.applymap(convert_grade)

print(convert_grades(grades_df))


#%%
# DataFrame apply()

grades_df = pd.DataFrame(
    data={'exam1': [43, 81, 78, 75, 89, 70, 91, 65, 98, 87],
          'exam2': [95, 63, 56, 56, 67, 51, 79, 46, 72, 60]},
    index=['Andre', 'Barry', 'Chris', 'Dan', 'Emilio', 
           'Fred', 'Greta', 'Humbert', 'Ivan', 'James']
)

def convert_grades_curve(exam_grades):
    # Pandas has a bult-in function that will perform this calculation
    # This will give the bottom 0% to 10% of students the grade 'F',
    # 10% to 20% the grade 'D', and so on. You can read more about
    # the qcut() function here:
    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html
    return pd.qcut(exam_grades,
                    [0, 0.1, 0.2, 0.5, 0.8, 1],
                    labels=['F', 'D', 'C', 'B', 'A'])
    
# qcut() operates on a list, array, or Series. This is the
# result of running the function on a single column of the
# DataFrame.
print(convert_grades_curve(grades_df['exam1']))
    
# qcut() does not work on DataFrames, but we can use apply()
# to call the function on each column separately
print(grades_df.apply(convert_grades_curve))

#%%

# 注意，计算得出的默认标准偏差类型在 numpy 的 .std() 和 pandas 的 .std() 函数之间是不同的。
# 默认情况下，numpy 计算的是总体标准偏差，ddof = 0。另一方面，pandas 计算的是样本标准偏差，ddof = 1。
# 如果我们知道所有的分数，那么我们就有了总体——因此，要使用 pandas 进行归一化处理，我们需要将“ddof”设置为 0。

def standardize_column(column):
    return (column - column.mean()) / column.std(ddof=0)

def standardize(df):
    '''
    Fill in this function to standardize each column of the given
    DataFrame. To standardize a variable, convert each value to the
    number of standard deviations it is above or below the mean.
    '''
    return df.apply(standardize_column)

standardize_df = standardize(grades_df)
print(standardize_df)

#%%
import numpy as np
import pandas as pd

df = pd.DataFrame({
    'a': [4, 5, 3, 1, 2],
    'b': [20, 10, 40, 50, 30],
    'c': [25, 20, 5, 15, 10]
})

# DataFrame apply() - use case 2
# 将列转化为单个值
print(df.apply(np.mean))
print(df.apply(np.max))
# 相当于
print(df.mean())
print(df.max())
    
def second_largest_in_column(column):
    sorted_column = column.sort_values(ascending=False)
    return sorted_column.iloc[1]


def second_largest(df):
    '''
    Fill in this function to return the second-largest value of each 
    column of the input DataFrame.
    '''
    return df.apply(second_largest_in_column)

print(second_largest_in_column(df['a']))
print(second_largest(df))


#%%
# Adding a Series to a square DataFrame

s = pd.Series([1, 2, 3, 4])
df = pd.DataFrame({
    0: [10, 20, 30, 40],
    1: [50, 60, 70, 80],
    2: [90, 100, 110, 120],
    3: [130, 140, 150, 160]
})

# print(df)
# print('') # Create a blank line between outputs
# print(df + s)

# Adding a Series to a one-row DataFrame 
s = pd.Series([1, 2, 3, 4])
df = pd.DataFrame({0: [10], 1: [20], 2: [30], 3: [40]})

# print(df)
# print('') # Create a blank line between outputs
# print(df + s)

# Adding a Series to a one-column DataFrame

s = pd.Series([1, 2, 3, 4])
df = pd.DataFrame({0: [10, 20, 30, 40]})

print(df)
print('') # Create a blank line between outputs
print(df + s)
# 注意区别
print(df.add(s, axis='columns'))
print(df.add(s, axis='index'))

# Adding when DataFrame column names match Series index

s = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
df = pd.DataFrame({
    'a': [10, 20, 30, 40],
    'b': [50, 60, 70, 80],
    'c': [90, 100, 110, 120],
    'd': [130, 140, 150, 160]
})

# print(df)
# print('') # Create a blank line between outputs
# print(df + s)

# Adding when DataFrame column names don't match Series index
s = pd.Series([1, 2, 3, 4])
df = pd.DataFrame({
    'a': [10, 20, 30, 40],
    'b': [50, 60, 70, 80],
    'c': [90, 100, 110, 120],
    'd': [130, 140, 150, 160]
})

# print(df)
# print('') # Create a blank line between outputs
# print(df + s)

#%%

s = pd.Series([1, 2, 3, 4])
df = pd.DataFrame({
    0: [10, 20, 30, 40],
    1: [50, 60, 70, 80],
    2: [90, 100, 110, 120],
    3: [130, 140, 150, 160]
})

print(df)
print('') # Create a blank line between outputs
print(df + s)
    
# Adding with axis='index'

s = pd.Series([1, 2, 3, 4])
df = pd.DataFrame({
    0: [10, 20, 30, 40],
    1: [50, 60, 70, 80],
    2: [90, 100, 110, 120],
    3: [130, 140, 150, 160]
})

print(df)
print('') # Create a blank line between outputs
print(df.add(s, axis='index'))
    # The functions sub(), mul(), and div() work similarly to add()
    
# Adding with axis='columns'

s = pd.Series([1, 2, 3, 4])
df = pd.DataFrame({
    0: [10, 20, 30, 40],
    1: [50, 60, 70, 80],
    2: [90, 100, 110, 120],
    3: [130, 140, 150, 160]
})

print(df)
print('') # Create a blank line between outputs
print(df.add(s, axis='columns'))
# The functions sub(), mul(), and div() work similarly to add()


#%%
grades_df = pd.DataFrame(
    data={'exam1': [43, 81, 78, 75, 89, 70, 91, 65, 98, 87],
          'exam2': [24, 63, 56, 56, 67, 51, 79, 46, 72, 60]},
    index=['Andre', 'Barry', 'Chris', 'Dan', 'Emilio', 
           'Fred', 'Greta', 'Humbert', 'Ivan', 'James']
)

def standardize(df):
    '''
    Fill in this function to standardize each column of the given
    DataFrame. To standardize a variable, convert each value to the
    number of standard deviations it is above or below the mean.
    
    This time, try to use vectorized operations instead of apply().
    You should get the same results as you did before.
    '''
    return (df - df.mean()) / df.std(ddof = 0)

def standardize_rows(df):
    '''
    Optional: Fill in this function to standardize each row of the given
    DataFrame. Again, try not to use apply().
    
    This one is more challenging than standardizing each column!
    '''
    mean_diffs = df.sub(df.mean(axis='columns'), axis='index')
    result = mean_diffs.div(df.std(axis='columns', ddof = 0), axis='index')
    return result

# print(standardize(grades_df))

# 用于测试
# 默认 
grades_df.mean(axis='index')

# 相当于求每一行的平均值
grades_df.mean(axis='columns')

# 索引要对应起来
mean_diffs = grades_df.sub(grades_df.mean(axis='columns'), axis='index')
# 以上用于测试

print(standardize_rows(grades_df))


#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

values = np.array([1, 3, 2, 4, 1, 6, 4])
example_df = pd.DataFrame({
    'value': values,
    'even': values % 2 == 0,
    'above_three': values > 3 
}, index=['a', 'b', 'c', 'd', 'e', 'f', 'g'])

# Examine DataFrame
print(example_df)

# Examine groups
grouped_data = example_df.groupby('even')
# The groups attribute is a dictionary mapping keys to lists of row indexes
print(grouped_data.groups)

grouped_data = example_df.groupby(['even', 'above_three'])
print(grouped_data.groups)

grouped_data = example_df.groupby('even')
print(grouped_data.sum())

 
grouped_data = example_df.groupby('even')    
# You can take one or more columns from the result DataFrame
print(grouped_data.sum()['value'])
print('\n') # Blank line to separate results
# You can also take a subset of columns from the grouped data before 
# collapsing to a DataFrame. In this case, the result is the same.
print(grouped_data['value'].sum())
    

#%%
subway_df.head()
subway_df.groupby('day_week').mean()
# 某一列
ridership_by_day = subway_df.groupby('day_week').mean()['ENTRIESn_hourly']

#%pylab inline
ridership_by_day.plot()


#%%
import numpy as np
import pandas as pd

values = np.array([1, 3, 2, 4, 1, 6, 4])
example_df = pd.DataFrame({
    'value': values,
    'even': values % 2 == 0,
    'above_three': values > 3 
}, index=['a', 'b', 'c', 'd', 'e', 'f', 'g'])

# Standardize each group

def standardize(xs):
    return (xs - xs.mean()) / xs.std()
grouped_data = example_df.groupby('even')

print(grouped_data['value'].apply(standardize))
    
# Find second largest value in each group

def second_largest(xs):
    sorted_xs = xs.sort_values(inplace=False, ascending=False)
    return sorted_xs.iloc[1]

# grouped_data = example_df.groupby('even')
print(grouped_data['value'].apply(second_largest))




#%%
# 每小时入站和出站数
# DataFrame with cumulative entries and exits for multiple stations
ridership_df = pd.DataFrame({
    'UNIT': ['R051', 'R079', 'R051', 'R079', 'R051', 'R079', 'R051', 'R079', 'R051'],
    'TIMEn': ['00:00:00', '02:00:00', '04:00:00', '06:00:00', '08:00:00', '10:00:00', '12:00:00', '14:00:00', '16:00:00'],
    'ENTRIESn': [3144312, 8936644, 3144335, 8936658, 3144353, 8936687, 3144424, 8936819, 3144594],
    'EXITSn': [1088151, 13755385,  1088159, 13755393,  1088177, 13755598, 1088231, 13756191,  1088275]
})

def hourly_for_group(entries_and_exits):    
    return entries_and_exits - entries_and_exits.shift(1)

def get_hourly_entries_and_exits_new(entries_and_exits):
    '''
    Fill in this function to take a DataFrame with cumulative entries
    and exits and return a DataFrame with hourly entries and exits.
    The hourly entries and exits should be calculated separately for
    each station (the 'UNIT' column).
    
    Hint: Take a look at the `get_hourly_entries_and_exits()` function
    you wrote in a previous quiz, DataFrame Vectorized Operations. If
    you copy it here and rename it, you can use it and the `.apply()`
    function to help solve this problem.
    '''
    # 注意选出具体的列'ENTRIESn', 'EXITSn'，因为其它列可能不能应用hourly_for_group函数
    return entries_and_exits.groupby('UNIT')[['ENTRIESn', 'EXITSn']].apply(hourly_for_group)


print(get_hourly_entries_and_exits_new(ridership_df))

#%%
subway_df = pd.DataFrame({
    'UNIT': ['R003', 'R003', 'R003', 'R003', 'R003', 'R004', 'R004', 'R004',
             'R004', 'R004'],
    'DATEn': ['05-01-11', '05-02-11', '05-03-11', '05-04-11', '05-05-11',
              '05-01-11', '05-02-11', '05-03-11', '05-04-11', '05-05-11'],
    'hour': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ENTRIESn': [ 4388333,  4388348,  4389885,  4391507,  4393043, 14656120,
                 14656174, 14660126, 14664247, 14668301],
    'EXITSn': [ 2911002,  2911036,  2912127,  2913223,  2914284, 14451774,
               14451851, 14454734, 14457780, 14460818],
    'latitude': [ 40.689945,  40.689945,  40.689945,  40.689945,  40.689945,
                  40.69132 ,  40.69132 ,  40.69132 ,  40.69132 ,  40.69132 ],
    'longitude': [-73.872564, -73.872564, -73.872564, -73.872564, -73.872564,
                  -73.867135, -73.867135, -73.867135, -73.867135, -73.867135]
})

weather_df = pd.DataFrame({
    'DATEn': ['05-01-11', '05-01-11', '05-02-11', '05-02-11', '05-03-11',
              '05-03-11', '05-04-11', '05-04-11', '05-05-11', '05-05-11'],
    'hour': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'latitude': [ 40.689945,  40.69132 ,  40.689945,  40.69132 ,  40.689945,
                  40.69132 ,  40.689945,  40.69132 ,  40.689945,  40.69132 ],
    'longitude': [-73.872564, -73.867135, -73.872564, -73.867135, -73.872564,
                  -73.867135, -73.872564, -73.867135, -73.872564, -73.867135],
    'pressurei': [ 30.24,  30.24,  30.32,  30.32,  30.14,  30.14,  29.98,  29.98,
                   30.01,  30.01],
    'fog': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'rain': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'tempi': [ 52. ,  52. ,  48.9,  48.9,  54. ,  54. ,  57.2,  57.2,  48.9,  48.9],
    'wspdi': [  8.1,   8.1,   6.9,   6.9,   3.5,   3.5,  15. ,  15. ,  15. ,  15. ]
})

def combine_dfs(subway_df, weather_df):
    '''
    Fill in this function to take 2 DataFrames, one with subway data and one with weather data,
    and return a single dataframe with one row for each date, hour, and location. Only include
    times and locations that have both subway data and weather data available.
    '''
    # 如果两个表中关联的列名称不同，使用left_on 与right_on分别对应表中的列名
    # return subway_df.merge(weather_df, left_on=['DATEn', 'hour', 'latitude', 'longitude'], right_on=['DATEn', 'hour', 'latitude', 'longitude'], how='inner' )

    return subway_df.merge(weather_df, on=['DATEn', 'hour', 'latitude', 'longitude'], how='inner' )

combine_dfs(subway_df, weather_df)


#%%
values = np.array([1, 3, 2, 4, 1, 6, 4])
example_df = pd.DataFrame({
    'value': values,
    'even': values % 2 == 0,
    'above_three': values > 3 
}, index=['a', 'b', 'c', 'd', 'e', 'f', 'g'])

# groupby() without as_index

# first_even = example_df.groupby('even').first()
# print(first_even)
# print(first_even['even']) # Causes an error. 'even' is no longer a column in the DataFrame
# 出错的原因是'even'已经不再是DataFrame中的列，而是变成了行索引值

# groupby() with as_index=False

first_even = example_df.groupby('even', as_index=False).first()
print(first_even)
print(first_even['even']) # Now 'even' is still a column in the DataFrame

#%%
# 绘制散点图
filename = '../DataMaterials/nyc-subway-weather.csv'
subway_df = pd.read_csv(filename)

data_by_location = subway_df.groupby(['latitude', 'longitude'],as_index=False).mean()
data_by_location.head()
data_by_location.head()['latitude']
# 气泡大小，可调整
scaled_entries = 2 * (data_by_location['ENTRIESn_hourly'] / data_by_location['ENTRIESn_hourly'].std())
plt.scatter(data_by_location['latitude'], data_by_location['longitude'], s=scaled_entries)



