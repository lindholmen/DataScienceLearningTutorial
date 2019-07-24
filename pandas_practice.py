import pandas as pd
import numpy as np

print("pandas.version:",pd.__version__)

#%% series object
data = pd.Series([0.25,0.5,0.75,1])
print("data.value:", data.values)
print("data.index:",data.index)
data = pd.Series([0.25,0.5,0.75,1],
                 index= ["甲","乙","丙","丁"])
print(data)
print("乙之data为：",data["乙"])
population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict)
print(population)
print(population["California":"New York"])

#%% dataframe object
area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
             'Florida': 170312, 'Illinois': 149995}
area = pd.Series(area_dict)
# from multiple series
states = pd.DataFrame({"population":population,"area":area})
print(states, "\nstates index:\n", states.index)
print("states column:\n", states.columns)
area_data = states["area"]
print("area data:\n",area_data)

# cast to np array data!!!!
np_area_data = np.array(area_data)
print("np_area_data：", np_area_data)

# Create a DataFrame object from a list of tuple or dictionary
students_list = [('jack', 34, 'Sydeny'),
            ('Riti', 30, 'Delhi'),
            ('Aadi', 16, 'New York')]
students_df = pd.DataFrame(students_list, columns=['Name', 'Age', 'City'])
print(students_df)
print("choose specific row:\n",students_df.loc[0:1,])
print("choose specific columns:\n",students_df.loc[:, ["Age","Name"]])

dictionalry_data = [{"age":i,"expirence":i*2} for i in range(10,15)]
experience_df = pd.DataFrame(dictionalry_data)
print(experience_df)

# Special: Create a Dataframe directly from a dictionary with array as values!!!

df = pd.DataFrame({'a': np.random.randn(10) + 1, 'b': np.random.randn(10),
                   'c': np.random.randn(10) - 1})
print(df) # 10 row

df2= pd.DataFrame([{'a': np.random.randn(10) + 1, 'b': np.random.randn(10),
                   'c': np.random.randn(10) - 1}])

print(df2)# 1 row

# Create a DataFrame object from a 2D numpy array
rain_possibility = pd.DataFrame(np.random.rand(3,5),
                                columns= ["Monday","Tuesday","Wednesdsay","Thursday","Friday"],
                                index=["Chongqing","Shanghai","Uppsala"])
print(rain_possibility)

# Create a DataFrame object from a structured numpy array

contacts_info = np.zeros(3,dtype = {"names":['name',"telephonenr"],
                                   "formats":["U10","i4"]})
print(contacts_info)
print(pd.DataFrame(contacts_info))


#%% panda index object
indA = pd.Index([2,3,4,5])
indB = pd.Index([3,4,5,7,8])
print("indexA are immutable array:",indA, "size:",indA.size," shape:", indA.shape)
print("indexB are immutable array:",indB)
print("index are also ordered set than can have intersection:", indA & indB)
print("index are also ordered set than can have union:", indA.union(indB))


#%% indexing of series, which is both 1-D np array and dictionary alike
data = pd.Series([1,2,3,4.0],
                 index=["a","b","c","d"])
print(data)
print(data["b"])
print(data.keys())
print(list(data.items()))
data["something_new"] = 999
print("extend the series like a dict:", data)
print("slicing:\n",data["a":"c"])

confusedData = pd.Series(["a","b","c"], index=[1,3,5])
print("explicit indexing:",confusedData[1])
print("implicit indexing:",confusedData[1:5])

#using loc attribute always refer to explicit indexing
print("loc with explict:",confusedData.loc[1])
print("loc with explict:",confusedData.loc[1:5])

#using iloc attribute always refer to implicit indexing
print("loc with implicit:",confusedData.iloc[1])
print("loc with implicit:",confusedData.iloc[1:5])


#%% indexing of dataframe, which is both 1-D np array and dictionary alike
area = pd.Series({'California': 423967, 'Texas': 695662,
                  'New York': 141297, 'Florida': 170312,
                  'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193,
                 'New York': 19651127, 'Florida': 19552860,
                 'Illinois': 12882135})
data = pd.DataFrame({'area':area, 'pop':pop})
print(data["area"] is data.area)
# add a new column
data['density'] = data["pop"] / data["area"]
print(data)
# update a value
data.iloc[0,2] = 150
# masking and fancy indexing (explicit indexing)
print(data.loc[data["density"] > 100, ["pop","area"]])

#slicing to produce rows and indexing to produce columns
print(data["area"])
print(data["California":"Florida"])
print(data[0:1])
print(data[data["density"]>100])

#%% index preserved - ufuncs

r = np.random.RandomState(42)
new_series = pd.Series(r.randint(0,10,4))
print(new_series)

df = pd.DataFrame(r.randint(0,10,(3,4)),
                  columns=['A',"B","C","D"])
print(df)

print("indices preserved:")
print(np.exp(new_series))

print("columns preserved:")
print(np.sin(df*np.pi/4))

#%% index alignment
A = pd.Series([1,3,5], index=[0,1,2])
B = pd.Series([2,4,6], index=[1,2,3])
print("A+B:")
print(A+B) #same as A.add(B)
print("A+B version 2:")
A.add(B, fill_value=0)

df_A = pd.DataFrame(np.random.RandomState(42).randint(0,20,(2,2)),
                    columns=list("AB"))
print("df_A:")
print(df_A)


df_B = pd.DataFrame(np.random.RandomState(42).randint(0,10,(3,3)),
                    columns=list("ABC"))
print("df_B:")
print(df_B)

print("df_A+df_B:")
print(df_A+df_B)

fill_avg= df_A.stack().mean()
print("filled average of each row of df_a:",fill_avg)
print(df_A.add(df_B,fill_value = fill_avg))

print(df_A)
print(df_A.subtract(df_A["A"],axis = 0))

#%% none
vals1 = np.array([1,None,2,3])
print(vals1)
vals2 = np.array([1,np.nan,2,3])
print(vals2)
print(np.nansum(vals2))
print(pd.Series([1,np.nan,2,None]))
x = pd.Series(range(3),dtype=int)
print(x)
x[0] = None
print("x:")
print(x)
print("check if none or nan")
print(x.isnull())
print("boolean as index")
print(x[x.notnull()])
df = pd.DataFrame([[1,2,3],
                   [2,3,5],
                   [2,np.nan,10]],columns=["A","B","C"])
print(df)

print(df.dropna())

df2 = pd.DataFrame([[1,2,3],
                   [2,3,5],
                   [2,np.nan,10]],columns=["A","B","C"])
print(df2)
print(df2.dropna(axis=1))
df2["X"] = 1
df2["y"] = np.nan
print(df2)
J =df2.dropna(axis=1,how="all")

print("J:")
print(J)
print("fill all NA entries with single value and return a copy:")
print(J.fillna(999))
print("specify a direction to fill and method to fill:")
print(J.fillna(method="ffill",axis=1))

data = pd.Series([1,np.nan,2,None,3])
print("data：")
print(data)
print(data.fillna(method="ffill"))
print(data.fillna(method="bfill"))


#%% MultiIndex
# 构造multi-index 的series 或者dataframe
# 1. 纵向构造：
df = pd.DataFrame(np.random.rand(4,2), columns = ["Class1","Class2"], index= [["Morning","Morning","Afternoon","Afternoon"],["Math","Nature","Math","PE"]])
print(df)

index = pd.MultiIndex.from_arrays([["Morning","Morning","Afternoon","Afternoon"],["Math","PE","Math","PE"]])
s1 = pd.Series([1,20,55,99],index=[('Morning', 'Math'), ('Morning', 'PE'), ('Afternoon', 'Math'), ('Afternoon', 'PE')])
s1 = s1.reindex(index)
print("index from arrays:")
print(s1)

# 2. 横向构造:
array = [["Morning","Morning","Afternoon","Afternoon"],["Math","PE","Math","PE"]]
#The * operator can be used in conjuncton with zip() to unzip the list.
list_of_tuples = list(zip(*array)) # [('Morning', 'Math'), ('Morning', 'PE'), ('Afternoon', 'Math'), ('Afternoon', 'PE')]
print(list_of_tuples)
new_index=pd.MultiIndex.from_tuples(list_of_tuples)

s1 = pd.Series([1,20,55,99],index=new_index)
print("create series with new_index:")
print(s1)

print("or use reindex:")
s1 = s1.reindex(new_index)
print(s1)

#3. 横向 series :using dict with tuple as key!
s2 = pd.Series({('Morning', 'Math'):100, ('Morning', 'PE'):200, ('Afternoon', 'Math'):300,('Afternoon', 'PE'):400})
print(s2)

#4. using stack to convert data frame to multiindex series
s1_df = s1. unstack()
print(s1_df)

s1 = s1_df.stack()
print(s1)
print(s1["Afternoon","Math"])#55
print(s2[s2>200])
print(s1[:,"PE"])



#%% 笛卡尔集和index names
pop_index = pd.MultiIndex.from_product([["Chongqing","Shanghai","Gothenburg"],
                                       [2000,2010]],names=["City","Year"])
pop = pd.Series([30000000,32000000,21000000,28000000,950000,1000000],index=pop_index)
print(pop)

# multiple levels for columns (DataFrame)
print("")
print("# multiple levels for columns (DataFrame)")
np.random.seed(1234)
C = np.random.randint(0,1000,size=(6,4))
row_index = pd.MultiIndex.from_product([["Chongqing","Shanghai","Gothenburg"],
                                       [2000,2010]],names=["City","Year"])

column_index = pd.MultiIndex.from_product([["Urban","Rural"],
                                       ["Woman","Man"]],names=["Geo","Gender"])

health_data = pd.DataFrame(C,index=row_index,columns=column_index)
print(health_data)
print("Urban data:")
print(health_data["Urban"])
print("")


#%% Slicing and indexing a multiindex
# 核心：
# dataSeries[level1_slicing,level2_slicing],
# dataSeries.iloc/loc[level1_slicing_start:level1_slicing_end]

# fancy index for series
print(s1[["Morning"]])

# for dataframe
# 核心：
# applies to only column: dataframe[leve1,level2]
# applies to row: dataframe.iloc/loc[leve1,level2]
# applies to an area(specified by column and row): dataframe.iloc[rowrange,columnrange]
# 或者推荐用dataframe.loc[idx[rowlevel1range,rowlevel2range],idx[columnlevel1range,columnlevel2range]]


print(health_data["Urban","Man"]) # applies to only column
# applies to row
print(health_data.loc["Chongqing",2000])
print(health_data.iloc[0])

#applies to column and row：
print(health_data.iloc[:2,:2])
#如果指定行列，比如行选择2010年的数据，column选择女性
idx = pd.IndexSlice
print(health_data.loc[idx[:,2010],idx[:,"Woman"]])

#%% sort must be done if want to slice
myindex = pd.MultiIndex.from_product([["a","c","b"],[1,2]])
data = pd.Series(np.random.randn(6),index = myindex)
data = data.sort_index()
print(data["a":"b"])


#%% stacking and unstacking
myindex = pd.MultiIndex.from_product([["Chongqing","Shanghai","Gothenburg"],
                                       [2000,2010]],names=["City","Year"])
data = pd.Series(np.random.rand(6),index=myindex)
print(data)
print(data.unstack(level=0))
print(data.unstack(level=1))
print(data.unstack().stack())

#%% index resetting to result in a DataFrame, with column name specified optionally
myindex = pd.MultiIndex.from_product([["Chongqing","Shanghai","Gothenburg"],
                                       [2000,2010]],names=["City","Year"])
data = pd.Series(np.random.rand(6),index=myindex)
data_raw = data.reset_index(name = "precipitation")
print(data_raw)# raw data usually looks like this so we can create a multiindex from the column value
processed_data_frame = data_raw.set_index(["City","Year"])
print(processed_data_frame)

#%% data aggregation
# specify level to suggest level you want to explore (row wise)
# and specify axis to suggest on the column
C = np.random.rand(6,4)
row_index = pd.MultiIndex.from_product([["Chongqing","Shanghai","Gothenburg"],
                                       [2000,2010]],names=["City","Year"])

column_index = pd.MultiIndex.from_product([["Urban","Rural"],
                                       ["Woman","Man"]],names=["Geo","Gender"])

health_data = pd.DataFrame(C,index=row_index,columns=column_index)
print(health_data)
city_avg = health_data.mean(level= "City")
print("Each city's average data:\n", city_avg)
print(city_avg.mean(axis= 1, level = "Gender"))


#%% concat and append

def make_a_quick_df(cols, ind):
    data = { c :[str(c) + str(i) for i in ind] for c in cols}
    return pd.DataFrame(data,index=ind)

make_a_quick_df("ABC",range(3))

x = [1, 2, 3]
y = [4, 5, 6]
z = [7, 8, 9]
print("np.concatenation:")
print(np.concatenate([x,y,z]))

x = [[1,2],
     [3,4]]
print("horizontally \n",np.concatenate([x,x], axis= 1))
print("vertically \n",np.concatenate([x,x], axis= 0))

series1 = pd.Series(["a","b","c"],index = [1,2,3])
series2 = pd.Series(["d","e","f"], index = [4,5,6])
print("concat 2 series:")
print(pd.concat([series1,series2]))


df1 =  make_a_quick_df("AB",[1,2])
df2 = make_a_quick_df("AB",[3,4])
print(df1)
print(df2)
print(pd.concat([df1,df2])) #默认是沿着axis=0方向concat


X =  make_a_quick_df("AB",[0,1])
Y = make_a_quick_df("AB",[0,1])
print("repeat index:")
try:
    pd.concat([X, Y],verify_integrity=True)
except ValueError as e:
    print("ValueError:",e)
print("ignore the index:")
print(pd.concat([X,Y],ignore_index=True))

print("adding multiple keys:")
print(pd.concat([X,Y],keys=["year 2000", "year 2001"]))

print("contatenation with joins")
x2 = make_a_quick_df("ABC",[1,2])
y2 = make_a_quick_df("BCD",[3,4])
print("default is outer:")
print(pd.concat([x2,y2]))
print("change to inner:")
print(pd.concat([x2,y2],join = "inner"))
print("Or use columns from one data source:")
print(pd.concat([x2,y2],join_axes=[x2.columns]))

print("Using append achieves the same goal:")
print(x2.append(y2))
print("X2 did not get changed, unlike append in a list:\n",x2)

#%% database merge (针对相同列的merge,默认是交集，除非指定how = "outer"则为并集)
df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                    'hire_date': [2004, 2008, 2012, 2014]})
df3 = pd.merge(df1,df2)
print("df3:\n",df3)

df4 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
                    'supervisor': ['Carly', 'Guido', 'Steve']})
print("merge df3 and df4:")
print(pd.merge(df3,df4))

df5 = pd.DataFrame({'group': ['Accounting', 'Accounting',
                              'Engineering', 'Engineering', 'HR', 'HR'],
                    'skills': ['math', 'spreadsheets', 'coding', 'linux',
                               'spreadsheets', 'organization']})
print("many to many merge:")
print(pd.merge(df1,df5))


print("using the on keyword:")
print(pd.merge(df1,df2, on = "employee"))


print("merging with different column names:")
df3 = pd.DataFrame({'name': ['Jake', 'Bob', 'Lisa', 'Sue'],
                    'salary': [70000, 80000, 120000, 90000]})
print(pd.merge(df1,df3, left_on = "employee", right_on ="name"))
print("then drop the redundant column")
print(pd.merge(df1,df3, left_on = "employee", right_on ="name").drop("name", axis=1))


#%% database join on indices， (针对相同行的join)
df1a = df1.set_index('employee')
df2a = df2.set_index('employee')
print("df1a:")
print(df1a)
print("df2a:")
print(df2a)
print("join method perform a merge that defaults to join on indices")
print(df1a.join(df2a))

#%% inner and outer join
df6 = pd.DataFrame({'name': ['Peter', 'Paul', 'Mary'],
                    'food': ['fish', 'beans', 'bread']},
                   columns=['name', 'food'])
df7 = pd.DataFrame({'name': ['Mary', 'Joseph'],
                    'drink': ['wine', 'beer']},
                   columns=['name', 'drink'])

print("default is inner join to find intersection:")
print(pd.merge(df6, df7)) #
print('this is outer join:')
print(pd.merge(df6, df7, how='outer'))
print('this is left join:')
print(pd.merge(df6, df7, how='left'))
print('this is right join:')
print(pd.merge(df6, df7, how='right'))

#overlapping column names:

df8 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [1, 2, 3, 4]})
df9 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [3, 1, 4, 2]})

print(pd.merge(df8,df9, on="name"))
pd.merge(df8, df9, on="name", suffixes=["_L", "_R"])

#%% US states data
pop = pd.read_csv("data/state-population.csv")
print(pop.head())

areas = pd.read_csv("data/state-areas.csv")
print(areas.head())

abbrevs = pd.read_csv("data/state-abbrevs.csv")
print(abbrevs.head())

merged = pd.merge(pop, abbrevs, how="outer", left_on= "state/region", right_on="abbreviation")
merged = merged.drop("abbreviation", axis=1)
print(merged)
#验证空项
print(merged.isnull().any())

#找到空项：
print(merged[merged['population'].isnull()])
print(merged.loc[merged["state"].isnull(), "state/region"].unique())

#填充空项
merged.loc[merged["state/region"]=="PR", "state"]= "Puerto Rico"
merged.loc[merged["state/region"] == "USA", "state"] = "United States"
print(merged.isnull().any())

# merged with areas:
final = pd.merge(merged, areas, on = "state",how="left")
print(final.head())
print(final.isnull().any())

#找到area中的空项：
print(final.loc[final["area (sq. mi)"].isnull(), "state"].unique())
# drop all entries where state is "United States"
#final.dropna(inplace=True) # drop所有的null选项
final = final.dropna()
print(final.isnull().any())

# 重点查询！！！！
data2010 = final.query("year == 2010 & ages == 'total'")
print(data2010.head())

# 吧state变成index是关键，这样column之间做运算后，index自动对应相应的state的值。
# inplace 的用法很巧，不用修改之后赋值了。
data2010.set_index("state",inplace = True)
density = data2010["population"]/data2010["area (sq. mi)"]
density.sort_values(ascending=False,inplace = True)
print("density:")
print(density)
print(density.head())
print(density.tail())


