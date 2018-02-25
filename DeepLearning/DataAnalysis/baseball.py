

#%%
# 读取文件
import pandas as pd

filename = '../../../Data/baseballdatabank-2017.1/core/Salaries.csv'
salaries_df = pd.read_csv(filename)

filename = '../../../Data/baseballdatabank-2017.1/core/Teams.csv'
teams_df = pd.read_csv(filename)

battinf_filename = '../../../Data/baseballdatabank-2017.1/core/Batting.csv'
batting_df = pd.read_csv(battinf_filename)

print(salaries_df.head())
salaries_df.describe()

#%%
grouped_data = salaries_df.groupby(['yearID', 'teamID', 'lgID'], as_index=False)
# print(grouped_data.groups)
print(grouped_data.mean())

mean_salaries_df = grouped_data.mean()
print(len(mean_salaries_df['yearID']))

years = set(salaries_df['yearID'])
print(len(years))

teams = set(salaries_df['teamID'])
print(len(teams))

#%%
max_salaries_df = grouped_data.max()
print(max_salaries_df)

#%%
new_grouped_data = salaries_df.groupby(['yearID', 'teamID']).describe().reset_index()
# print(grouped_data.groups)
print(new_grouped_data)

#%%
import matplotlib.pyplot as plt
import seaborn as sns

# 气泡大小，可调整 s=scaled_entries
scaled_entries = 5 * (mean_salaries_df['salary'] / mean_salaries_df['salary'].std())
plt.scatter(mean_salaries_df['teamID'], mean_salaries_df['yearID'], s=scaled_entries)


#%%
# 平均工资增长趋势
# grouped_data_by_year = salaries_df.groupby(['yearID'], as_index=False)
grouped_data_by_year = salaries_df.groupby(['yearID'])
mean_salaries_df_by_year = grouped_data_by_year.mean()
# print(mean_salaries_df_by_year)
mean_salaries_df_by_year.describe()
mean_salaries_df_by_year.plot()

#%%
def combine_dfs(teams_df, mean_salaries_df):   
   
    return teams_df.merge(mean_salaries_df, on=['yearID', 'teamID', 'lgID'], how='inner' )

def new_combine_dfs(df1, df2, on_list):   
   
    return df1.merge(df2, on=on_list, how='inner' )



cf = teams_df[teams_df['yearID'] >= 1985]
print(len(cf))

onlist=['yearID', 'teamID', 'lgID']
# 其实这里合并意义不大，因为未考虑时间因素，此处仅做测试
new_teams_df = new_combine_dfs(cf, mean_salaries_df, onlist)
new_teams_df.head()
print(len(new_teams_df))
print(len(mean_salaries_df))
print(len(teams_df))

teams1 = set(mean_salaries_df['teamID'])
print(len(teams1))

teams2 = set(cf['teamID'])
print(len(teams2))

print(teams1 - teams2)
# 可使用&（并）与| （或）实现多条件筛选
# cf1 = teams_df[teams_df['yearID'] >= 1985 & teams_df['yearID'] < 2015]


#%%
def correlation(x, y):
    '''
    用于相关系数的计算
    '''
    std_x = (x - x.mean()) / x.std(ddof=0)
    std_y = (y - y.mean()) / y.std(ddof=0)
    return (std_x * std_y).mean()

salary = new_teams_df['salary']
rank = new_teams_df['Rank']
w = new_teams_df['W']
bl = new_teams_df['W'] / new_teams_df['G']
r = new_teams_df['R']
# 防御率（ERA）
era = new_teams_df['ERA']
# 自责分（ER）
er = new_teams_df['ER']
# 救援点（SV）
sv = new_teams_df['SV']
# 完投（CG）
cg = new_teams_df['CG']
# 完封（SHO）
sho = new_teams_df['SHO']
# 守备失误（E）
e = new_teams_df['E']


# HA             Hits allowed
ha = new_teams_df['HA']
# HRA            Homeruns allowed
hra = new_teams_df['HRA']
# BBA            Walks allowed
bba = new_teams_df['BBA']
# SOA            Strikeouts by pitchers
soa = new_teams_df['SOA']
# FP             Fielding  percentage
fp = new_teams_df['FP']


# print(correlation(rank, salary))
# print(correlation(rank, bl))
# print(correlation(rank, r))
# print(correlation(rank, era))
# print(correlation(rank, e))
# print('----------------------')
# print(correlation(salary, bl))
# print(correlation(salary, r))
# print(correlation(salary, era))
# print(correlation(salary, e))

# 打数（AB）
ab = new_teams_df['AB']
# 得分（R）
r = new_teams_df['R']
# 安打（H）
h = new_teams_df['H']
# 全垒打（HR）
hr = new_teams_df['HR']

# 保送（BB）
bb = new_teams_df['BB']
# 触身球（HBP）
hbp = new_teams_df['HBP']
# 牺牲打（SF）
sf= new_teams_df['SF']

# 打击率（BA）
ba = h / ab
# 上垒率（OBP）
obp = (h + bb + hbp) / (ab + bb + hbp + sf)

# 与工资的相关性未考虑时间因素
# print(correlation(salary, ab))
# print(correlation(salary, r))
# print(correlation(salary, h))
# print(correlation(salary, hr))
# print(correlation(salary, rbi))
# print(correlation(salary, bb))
# print(correlation(salary, hbp))
# print(correlation(salary, sf))
# print(correlation(salary, ba))
# print(correlation(salary, obp))

print('--------------------------------')
# print(correlation(rank, ab))
# print(correlation(rank, r))
# print(correlation(rank, h))
# print(correlation(rank, hr))
# print(correlation(rank, rbi))
# print(correlation(rank, bb))
# print(correlation(rank, hbp))
# print(correlation(rank, sf))
# print(correlation(rank, ba))
# print(correlation(rank, obp))
print(correlation(rank, cg))
print(correlation(rank, sho))
print(correlation(rank, fp))
print(correlation(rank, ha))
print(correlation(hra, ha))

print('--------------------------------')
print(correlation(bl, ab))
print(correlation(bl, r))
print(correlation(bl, h))
print(correlation(bl, hr))
print(correlation(bl, rbi))
print(correlation(bl, bb))
print(correlation(bl, hbp))
print(correlation(bl, sf))
print(correlation(bl, ba))
print(correlation(bl, obp))
print(correlation(bl, salary))
print('--------------------------------')
print(correlation(w, ab))
print(correlation(w, r))
print(correlation(w, h))
print(correlation(w, hr))
print(correlation(w, rbi))
print(correlation(w, bb))
print(correlation(w, hbp))
print(correlation(w, sf))
print(correlation(w, ba))
print(correlation(w, obp))
print(correlation(w, salary))


#%%

on_list = ['playerID', 'yearID', 'teamID', 'lgID']
new_batting_df = new_combine_dfs(salaries_df, batting_df, on_list)
# print(new_batting_df.head())
print(len(new_batting_df))

salary = new_batting_df['salary']
# 打数（AB）
ab = new_batting_df['AB']
# 得分（R）
r = new_batting_df['R']
# 安打（H）
h = new_batting_df['H']
# 全垒打（HR）
hr = new_batting_df['HR']
# 打点（RBI）
rbi = new_batting_df['RBI']
# 保送（BB）
bb = new_batting_df['BB']
# 触身球（HBP）
hbp = new_batting_df['HBP']
# 牺牲打（SF）
sf= new_batting_df['SF']

# 打击率（BA）
ba = h / ab
# 上垒率（OBP）
obp = (h + bb + hbp) / (ab + bb + hbp + sf)

print(correlation(salary, ab))
print(correlation(salary, r))
print(correlation(salary, h))
print(correlation(salary, hr))
print(correlation(salary, rbi))
print(correlation(salary, bb))
print(correlation(salary, hbp))
print(correlation(salary, sf))
print(correlation(salary, ba))
print(correlation(salary, obp))

#%%
on_list = ['playerID', 'yearID', 'teamID', 'lgID']
max_batting_df = new_combine_dfs(max_salaries_df, batting_df, on_list)
print(len(max_batting_df))

salary_m = max_batting_df['salary']
# 打数（AB）
ab_m = max_batting_df['AB']
# 得分（R）
r_m = max_batting_df['R']
# 安打（H）
h_m = max_batting_df['H']
# 全垒打（HR）
hr_m = max_batting_df['HR']
# 打点（RBI）
rbi = max_batting_df['RBI']
# 保送（BB）
bb_m = max_batting_df['BB']
# 触身球（HBP）
hbp_m = max_batting_df['HBP']
# 牺牲打（SF）
sf_m= max_batting_df['SF']

# 打击率（BA）
ba_m = h / ab
# 上垒率（OBP）
obp_m = (h + bb + hbp) / (ab + bb + hbp + sf)

nf_m = max_batting_df[['AB','R','HR','H','BB','HBP','SF']].copy()
nf_m.loc[:,'BA'] = ba_m
nf_m['OBP'] = obp_m

nf_m.describe()
#%%
new_batting_df.describe()

#%%
nf = new_batting_df[['AB','R','HR','H','BB','HBP','SF']]

# nf[:,'OBP'] = obp
nf.head()
#%%
nf1 = pd.DataFrame({'AB': ab, 'R': r})
nf1.loc[:,'BA'] = ba
nf1.loc['BA'] = ba
nf1.head()

nf2 = nf.copy()
nf2.loc[:,'BA'] = ba
nf2['OBP'] = obp

nf2.describe()


#%%
# 用来划分工资水平
def pct_rank_qcut(series, n):
    edges = pd.Series([float(i) / n for i in range(n + 1)])
    f = lambda x: (edges >= x).values.argmax()
    return series.rank(pct=1).apply(f)

# q = pct_rank_qcut(df.loss_percent, 10)

def new_pct_rank_qcut(df, n):
    edges = pd.Series([float(i) / n for i in range(n + 1)])
    f = lambda x: (edges >= x).values.argmax()
    df['levelname'] = df['salary'].rank(pct=1).apply(f)
    return df

def convert_salaries_curve(salary_grades):
    
    return pd.qcut(salary_grades,
                    [0, 0.333, 0.666,  1],
                    labels=[ 'C', 'B', 'A'])



# ngrouped_data = salaries_df.groupby(['yearID']).apply(lambda df:print(convert_salaries_curve(df['salary']))) 
# ngrouped_data = salaries_df.groupby(['yearID'], as_index=False).apply(lambda df:convert_salaries_curve(df['salary'])) 
# print(ngrouped_data.groups)

# ngrouped_data = salaries_df.copy().groupby(['yearID']).apply(lambda df: convert_salaries_curve(df['salary'])).reset_index() 

filename = '../../../Data/baseballdatabank-2017.1/core/Salaries.csv'
salaries_df = pd.read_csv(filename)

# ngrouped_data = salaries_df.groupby(['yearID']).apply(lambda df: pct_rank_qcut(df['salary'], 3)).reset_index() 
# mgrouped_data = salaries_df.groupby(['yearID','teamID']).apply(lambda df: pct_rank_qcut(df['salary'], 3)).reset_index() 

# print(ngrouped_data['salary'])
ngrouped_data = salaries_df.groupby(['yearID']).apply(lambda df: new_pct_rank_qcut(df, 3)).reset_index() 
print(ngrouped_data)
# print(ngrouped_data[ngrouped_data['yearID'] == 1985])

#%%


high_salary_group = ngrouped_data[ngrouped_data['levelname'] == 3]
# print(high_salary_group)

middle_salary_group = ngrouped_data[ngrouped_data['levelname'] == 2]
# print(middle_salary_group)

low_salary_group = ngrouped_data[ngrouped_data['levelname'] == 1]
# print(low_salary_group)

on_list = ['playerID', 'yearID', 'teamID', 'lgID']
high_batting_df = new_combine_dfs(high_salary_group, batting_df, on_list)
print(len(high_batting_df))
middle_batting_df = new_combine_dfs(middle_salary_group, batting_df, on_list)
print(len(middle_batting_df))
low_batting_df = new_combine_dfs(low_salary_group, batting_df, on_list)
print(len(low_batting_df))


def key_batting_indexes(df):
    
    yearID = df['yearID']
    salary = df['salary']
    # 打数（AB）
    ab = df['AB']
    # 得分（R）
    r = df['R']
    # 安打（H）
    h = df['H']
    # 全垒打（HR）
    hr = df['HR']
    # 打点（RBI）
    rbi = df['RBI']
    # 保送（BB）
    bb = df['BB']
    # 触身球（HBP）
    hbp = df['HBP']
    # 牺牲打（SF）
    sf= df['SF']
    # 打击率（BA）
    ba = h / ab
    # 上垒率（OBP）
    obp = (h + bb + hbp) / (ab + bb + hbp + sf)

    ndf = df[['yearID','AB','R','HR','H','BB','HBP','SF','salary']].copy()
    ndf.loc[:,'BA'] = ba
    ndf['OBP'] = obp

    return ndf

high_ndf = key_batting_indexes(high_batting_df)

middle_ndf = key_batting_indexes(middle_batting_df)

low_ndf = key_batting_indexes(low_batting_df)


print(low_ndf.mean())


low_ndf = low_ndf[ ~(pd.isnull(low_ndf['BA']))  & ~(pd.isnull(low_ndf['OBP']))]
middle_ndf = middle_ndf[ ~(pd.isnull(middle_ndf['BA']))  & ~(pd.isnull(middle_ndf['OBP']))]
high_ndf = high_ndf[ ~(pd.isnull(high_ndf['BA']))  & ~(pd.isnull(high_ndf['OBP']))]
print(low_ndf.mean())
# print('---------------------------------------')

# print(middle_ndf.mean())

# print('---------------------------------------')
# print(high_ndf.mean())

print(low_ndf)

# print(low_ndf.groupby(['yearID']).mean()['BA'])
print('---------------------------------------')
low_data = low_ndf.groupby(['yearID']).mean().loc[:,['BA','OBP']].reset_index().sort_values(['yearID']) 
# print(low_data)
middle_data = middle_ndf.groupby(['yearID']).mean().loc[:,['BA','OBP']].reset_index().sort_values(['yearID'])

high_data = high_ndf.groupby(['yearID']).mean().loc[:,['BA','OBP']].reset_index().sort_values(['yearID'])

print(low_data['yearID'])

test_data = np.array(low_data['yearID']).tolist()
# train_x_list=train_data.tolist()#list

print(test_data)
print(correlation(high_ndf['OBP'], high_ndf['BA']))

ba_group = pd.DataFrame({
    
    'low_group': np.array(low_data['BA']).tolist(),
    'middle_group': np.array(middle_data['BA']).tolist(),
    'high_group': np.array(high_data['BA']).tolist()
}, index=np.array(low_data['yearID']).tolist())

# print(ba_group)
ba_group.plot()
# print(middle_ndf.mean())

# print('---------------------------------------')
# print(high_ndf.mean())

obp_group = pd.DataFrame({
    
    'low_group': np.array(low_data['OBP']).tolist(),
    'middle_group': np.array(middle_data['OBP']).tolist(),
    'high_group': np.array(high_data['OBP']).tolist()
}, index=np.array(low_data['yearID']).tolist())

obp_group.plot()


#%%
# print(correlation(low_ndf['salary'], low_ndf['AB']))
# print(correlation(low_ndf['salary'], low_ndf['R']))
# print(correlation(low_ndf['R'], low_ndf['AB']))
# print('--------------------------------')
# print(correlation(middle_ndf['salary'], middle_ndf['AB']))
# print(correlation(middle_ndf['salary'], middle_ndf['R']))
# print(correlation(middle_ndf['R'], middle_ndf['AB']))
# print('--------------------------------')
# print(correlation(high_ndf['salary'], high_ndf['AB']))
# print(correlation(high_ndf['salary'], high_ndf['R']))
# print(correlation(high_ndf['R'], high_ndf['AB']))


#%%
import numpy as np
values = np.array([1, 3, 2, 4, 1, 6, 4])
example_df = pd.DataFrame({
    'value': values,
    'even': values % 2 == 0,
    'above_three': values > 3 
})

print(example_df['value'].mean())

example_df.sort_values(['value'])
#%%

import pandas as pd
import numpy as np
from pandas import *

L = [4, None ,6]
df = Series(L)

print(df)


if(pd.isnull(df[1])):
    print("Found")



if(np.isnan(df[1])):
    print("Found")

