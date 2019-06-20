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
