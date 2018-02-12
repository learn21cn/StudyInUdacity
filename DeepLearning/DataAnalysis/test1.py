import unicodecsv
def read_csv(file_name):
    with open(file_name, 'rb') as f:
        reader = unicodecsv.DictReader(f)
        return list(reader)
# G:/StudyInUdacity/DeepLearning/DataMaterials
engagement_filename = '../DataMaterials/daily-engagement.csv'
submissions_filename = '../DataMaterials/project-submissions.csv'

daily_engagement = read_csv(engagement_filename)     # Replace this with your code
project_submissions = read_csv(submissions_filename)  # Replace this with your code


#%%
import pandas as pd
engagement_filename = '../DataMaterials/daily-engagement.csv'
daily_engagement = pd.read_csv(engagement_filename)

len(daily_engagement['acct'].unique())

#%%
def max_employment(countries, employment):
    max_country = None
    max_employment = 0

    for i in range(len(countries)):
        country = countries[i]
        country_employment = employment[i]

        if country_employment > max_employment:
            max_country = country
            max_employment = country_employment
    
    return (max_country, max_employment)

def max_employment2(countries, employment):
    i = employment.argmax()
    return (countries[i], employment[i])




import numpy as np

def standardize_data(values):
    standardize_values = (values - values.mean()) / values.std()
    
    return standardize_values
    

countries = np.array([
    'Afghanistan', 'Albania', 'Algeria', 'Angola', 'Argentina',
    'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas',
    'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium',
    'Belize', 'Benin', 'Bhutan', 'Bolivia',
    'Bosnia and Herzegovina'
])

# Employment data in 2007 for those 20 countries
employment = np.array([
    55.70000076,  51.40000153,  50.5       ,  75.69999695,
    58.40000153,  40.09999847,  61.5       ,  57.09999847,
    60.90000153,  66.59999847,  60.40000153,  68.09999847,
    66.90000153,  53.40000153,  48.59999847,  56.79999924,
    71.59999847,  58.40000153,  70.40000153,  41.20000076
])

print(max_employment2(countries, employment))

print(standardize_data(employment))

#%%
a = np.array([1, 2, 3, 4])
b = np.array([1, 2, 1, 2])

print('==========')  
print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a ** b)

#%%
a = np.array([1, 2, 3, 4])
b = 2
print('==========')
print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a ** b)

#%%
a = np.array([True, True, False, False])
b = np.array([True, False, True, False])

print('==========')   
print(a & b)
print(a | b)
print(~a)
    
print(a & True)
print(a & False)
    
print(a | True)
print(a | False)

#%%
print('===========================')   
a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 4, 3, 2, 1])
    
print(a > b)
print(a >= b)
print(a < b)
print(a <= b)
print(a == b)
print(a != b)

#%%
a = np.array([1, 2, 3, 4])
b = 2

print(a > b)
print(a >= b)
print(a < b)
print(a <= b)
print(a == b)
print(a != b)

countries = np.array([
       'Algeria', 'Argentina', 'Armenia', 'Aruba', 'Austria','Azerbaijan',
       'Bahamas', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Bolivia',
       'Botswana', 'Brunei', 'Bulgaria', 'Burkina Faso', 'Burundi',
       'Cambodia', 'Cameroon', 'Cape Verde'
])

# Female school completion rate in 2007 for those 20 countries
female_completion = np.array([
    97.35583,  104.62379,  103.02998,   95.14321,  103.69019,
    98.49185,  100.88828,   95.43974,   92.11484,   91.54804,
    95.98029,   98.22902,   96.12179,  119.28105,   97.84627,
    29.07386,   38.41644,   90.70509,   51.7478 ,   95.45072
])

# Male school completion rate in 2007 for those 20 countries
male_completion = np.array([
     95.47622,  100.66476,   99.7926 ,   91.48936,  103.22096,
     97.80458,  103.81398,   88.11736,   93.55611,   87.76347,
    102.45714,   98.73953,   92.22388,  115.3892 ,   98.70502,
     37.00692,   45.39401,   91.22084,   62.42028,   90.66958
])

def overall_completion_rate(female_completion, male_completion):
    '''
    Fill in this function to return a NumPy array containing the overall
    school completion rate for each country. The arguments are NumPy
    arrays giving the female and male completion of each country in
    the same order.
    '''
    return (female_completion + male_completion) / 2

overall_completion_rate(female_completion, male_completion)

#%%
a = np.array([1, 2, 3, 4])
b = np.array([True, True, False, False])
    
print(a[b])
print(a[np.array([True, False, True, False])])

#%%
a = np.array([1, 2, 3, 2, 1])
b = (a >= 2)
    
print(a[b])
print(a[a >= 2])

#%%
a = np.array([1, 2, 3, 4, 5])
b = np.array([1, 2, 3, 2, 1])
    
print(b == 2)
print(a[b == 2])

#%%

def mean_time_for_paid_students(time_spent, days_to_cancel):
    '''
    Fill in this function to calculate the mean time spent in the classroom
    for students who stayed enrolled at least (greater than or equal to) 7 days.
    Unlike in Lesson 1, you can assume that days_to_cancel will contain only
    integers (there are no students who have not canceled yet).
    
    The arguments are NumPy arrays. time_spent contains the amount of time spent
    in the classroom for each student, and days_to_cancel contains the number
    of days until each student cancel. The data is given in the same order
    in both arrays.
    '''
    return time_spent[days_to_cancel >=7].mean()

# Time spent in the classroom in the first week for 20 students
time_spent = np.array([
       12.89697233,    0.        ,   64.55043217,    0.        ,
       24.2315615 ,   39.991625  ,    0.        ,    0.        ,
      147.20683783,    0.        ,    0.        ,    0.        ,
       45.18261617,  157.60454283,  133.2434615 ,   52.85000767,
        0.        ,   54.9204785 ,   26.78142417,    0.
])

# Days to cancel for 20 students
days_to_cancel = np.array([
      4,   5,  37,   3,  12,   4,  35,  38,   5,  37,   3,   3,  68,
     38,  98,   2, 249,   2, 127,  35
])

mean_time_for_paid_students(time_spent, days_to_cancel)

#%%
# 注意区别
a = np.array([1, 2, 3, 4])
b = a
a += np.array([1, 1, 1, 1])
# 这里变了
print(b)


#%%
a = np.array([1, 2, 3, 4])
b = a
a = a + np.array([1, 1, 1, 1])
# 这里没变
print(b)

#%%
a = np.array([1, 2, 3, 4, 5])
slice = a[:3] 
slice[0] = 6
# 这里变了
print(a)

#%%
a = [1, 2, 3, 4, 5]
slice = a[:3] 
slice[0] = 6
# 这里没变
print(a)


#%%
import pandas as pd

def variable_correlation(variable1, variable2):
    both_above = (variable1 > variable1.mean()) & (variable2 > variable2.mean())
    both_below = (variable1 < variable1.mean()) & (variable2 < variable2.mean())

    is_same_direction = both_above | both_below
    num_same_direction = is_same_direction.sum()
    num_different_direction = len(variable1) - num_same_direction

    return (num_same_direction, num_different_direction)

countries = ['Albania', 'Algeria', 'Andorra', 'Angola', 'Antigua and Barbuda',
             'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan',
             'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus',
             'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia']

life_expectancy_values = [74.7,  75. ,  83.4,  57.6,  74.6,  75.4,  72.3,  81.5,  80.2,
                          70.3,  72.1,  76.4,  68.1,  75.2,  69.8,  79.4,  70.8,  62.7,
                          67.3,  70.6]

gdp_values = [ 1681.61390973,   2155.48523109,  21495.80508273,    562.98768478,
              13495.1274663 ,   9388.68852258,   1424.19056199,  24765.54890176,
              27036.48733192,   1945.63754911,  21721.61840978,  13373.21993972,
                483.97086804,   9783.98417323,   2253.46411147,  25034.66692293,
               3680.91642923,    366.04496652,   1175.92638695,   1132.21387981]

# Life expectancy and gdp data in 2007 for 20 countries
life_expectancy = pd.Series(life_expectancy_values)
gdp = pd.Series(gdp_values)



# print(life_expectancy[0])
# print(gdp[3:6])

# for country_life_expectancy in life_expectancy:
#     print('Examining life expectancy {}'.format(country_life_expectancy))

print(life_expectancy.mean())
print(life_expectancy.std())
print(gdp.max())
print(gdp.sum())

print(variable_correlation(life_expectancy, gdp))

#%%
a = pd.Series([1, 2, 3, 4])
b = pd.Series([1, 2, 1, 2])
  
print(a + b)
print(a * 2)
print(a >= 3)
print(a[a >= 3])



#%%
a = np.array([1, 2, 3, 4])
s = pd.Series([1, 2, 3, 4])

s.describe()

# pandas 带索引，像列表和字典的集合
number = pd.Series([1, 15, 2, 21, 5], index=['A', 'B', 'C', 'D', 'E'])
print(number) 

# 使用索引
number.loc['E']
# 使用元素的位置
number.iloc[1]

#%%
def max_employment(employment):
    # max_country = employment.argmax()
    max_country = employment.idxmax()
    max_value = employment.loc[max_country]
    return (max_country, max_value)

countries = [
    'Afghanistan', 'Albania', 'Algeria', 'Angola',
    'Argentina', 'Armenia', 'Australia', 'Austria',
    'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh',
    'Barbados', 'Belarus', 'Belgium', 'Belize',
    'Benin', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina',
]
employment_values = [
    55.70000076,  51.40000153,  50.5       ,  75.69999695,
    58.40000153,  40.09999847,  61.5       ,  57.09999847,
    60.90000153,  66.59999847,  60.40000153,  68.09999847,
    66.90000153,  53.40000153,  48.59999847,  56.79999924,
    71.59999847,  58.40000153,  70.40000153,  41.20000076,
]

employment = pd.Series(employment_values, index=countries)
max_employment(employment)

#%%
# Addition when indexes are the same
if True:
    s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
    s2 = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
    print(s1 + s2)

# Indexes have same elements in a different order
if True:
    s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
    s2 = pd.Series([10, 20, 30, 40], index=['b', 'd', 'a', 'c'])
    print(s1 + s2)

# Indexes overlap, but do not have exactly the same elements
if True:
    s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
    s2 = pd.Series([10, 20, 30, 40], index=['c', 'd', 'e', 'f'])
    print(s1 + s2)

# Indexes do not overlap
if True:
    s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
    s2 = pd.Series([10, 20, 30, 40], index=['e', 'f', 'g', 'h'])
    print(s1 + s2)
    print((s1 + s2).dropna())

    s = s1.add(s2, fill_value=0)
    print(s)

#%%
if False:
    s = pd.Series([1, 2, 3, 4, 5])
    def add_one(x):
        return x + 1
    print(s.apply(add_one))

names = pd.Series([
    'Andre Agassi',
    'Barry Bonds',
    'Christopher Columbus',
    'Daniel Defoe',
    'Emilio Estevez',
    'Fred Flintstone',
    'Greta Garbo',
    'Humbert Humbert',
    'Ivan Ilych',
    'James Joyce',
    'Keira Knightley',
    'Lois Lane',
    'Mike Myers',
    'Nick Nolte',
    'Ozzy Osbourne',
    'Pablo Picasso',
    'Quirinus Quirrell',
    'Rachael Ray',
    'Susan Sarandon',
    'Tina Turner',
    'Ugueth Urbina',
    'Vince Vaughn',
    'Woodrow Wilson',
    'Yoji Yamada',
    'Zinedine Zidane'
])

def reverse_name(name):
    split_name = name.split(" ")
    first_name = split_name[0]
    last_name = split_name[1]
    return last_name + ', ' + first_name
    
def reverse_names(names):
    '''
    Fill in this function to return a new series where each name
    in the input series has been transformed from the format
    "Firstname Lastname" to "Lastname, FirstName".
    
    Try to use the Pandas apply() function rather than a loop.
    '''
    return names.apply(reverse_name)

reverse_names(names)

#%%

import seaborn as sns
path = '../DataMaterials/'
employment = pd.read_csv(path + 'employment-above-15.csv', index_col='Country')
female_completion = pd.read_csv(path + 'female-completion-rate.csv', index_col='Country')
male_completion = pd.read_csv(path + 'male-completion-rate.csv', index_col='Country')
life_expectancy = pd.read_csv(path + 'life-expectancy.csv', index_col='Country')
gdp = pd.read_csv(path + 'gdp-per-capita.csv', index_col='Country')


employment_us = employment.loc['United States']
female_completion_us = female_completion.loc['United States']
male_completion_us = male_completion.loc['United States']
life_expectancy_us = life_expectancy.loc['United States']
gdp_us = gdp.loc['United States']

# %pylab inline
gdp_us.plot()
# female_completion_us.plot()
# male_completion_us.plot()
# life_expectancy_us.plot()
# employment_us.plot()