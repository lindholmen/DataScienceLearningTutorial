#%% IMPORT
import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set()

#%% IMPORT END
print(np.__version__)
multidata= np.array([range(i,i+3) for i in [2,4,6]])
print(multidata)
print(np.random.random((3,3)))
print(np.random.normal(0,1,(3,3)))
print(np.random.randint(0,10,(3,3)))
zeros = np.zeros((2,3),dtype=int)
x = np.arange(1,4)
y = np.arange(7,10)
print("concatenation:",np.concatenate([x,y]))
grid = np.arange(16).reshape((4,4))
upper, lower = np.vsplit(grid,[2])
print("upper:\n", upper)
print("lower:\n", lower)

x = np.arange(5)
y = np.empty(5)
np.multiply(x, 10, out=y)
print("out argument is y:",y)

y = np.zeros(10)
np.power(2,x,out= y[::2])
print(y)

x = np.arange(1,6)
print("reduce to add repeatedly:", np.add.reduce(x))
print("reduce to multiply repeatedly:",np.multiply.reduce(x))
print("accumulate output intermediate results:", np.add.accumulate(x))
print("all pairs for multiplying calculation:\n",np.multiply.outer(x,x))

L = np.random.random(1000)
print("slow sum:",sum(L))
print("fast sum:",np.sum(L))

print("min：", np.min(L), L.min())
print("max：",np.max(L), L.max())

M = np.random.random((3,4))
print("M",M)
print("M sum:", M.sum())
print("M max of each column:", M.max(axis = 0)) # axis =0 指的维度为0的轴collapse,行无意义，只爱看列。
print("M sum of each row:", M.sum(axis = 1))
print("M mean of each row:", M.mean(axis = 1))
print("M std of each row:", M.std(axis = 1))
print("M std of each variance:", M.var(axis = 1))
print("M index of min in each row:", M.argmin(axis = 1))
print("M product of elements in each row:", M.prod(axis = 1))


#%% president of USA
# using !head -4 xxxx.csv to probe 4 rows and column name
import pandas as pd
data = pd.read_csv("data/president_heights.csv")
president_heights_data = np.array(data['height(cm)'])
print(president_heights_data)
print("mean height:",president_heights_data.mean())
print("sd of height:",president_heights_data.std())
print("min height index:",president_heights_data.argmin())
print("president of min height is :", data['name'][president_heights_data.argmin()])
print("25th percentile:", np.percentile(president_heights_data,25))
print("median", np.median(president_heights_data))


plt.hist(president_heights_data)
plt.title("Height distribution of USA presidents")
plt.xlabel("height(cm)")
plt.ylabel("number")
plt.show()

#%% MEAN
# 10 observations, 3 values for each one
a = np.random.randint(1,7,(10,3))
print("mean of each feature:",a.mean(0))
print("mean of each observation:", a.mean(1))


x=np.linspace(0,5,50)
y=np.linspace(0,5,50)[:,np.newaxis]
z = np.sin(x) ** 10 + np.cos(10 + y*x) * np.cos(x)
plt.imshow(z,origin="lower",extent=[0,5,0,5],cmap= "viridis")
plt.colorbar()
plt.title("visulisation of 2d array")
plt.xlabel("")
plt.ylabel("")
#plt.show()

rng = np.random.RandomState(255) # seed is 255
x = rng.randint(0,10,(3,4))
print(x)
y = rng.uniform(0,1,(3,4))
print(y)
print("x>3:",x[x>3])
print("y<=0.5:",np.less_equal(y,0.5))

# count the number of element in each row
# 设axis=i，则Numpy沿着第i个下标变化的方向进行操作。这里0维为行，1维为列，则变化方向是列0，列1，列2方向。
print(np.sum(x>5,axis=1))

print("check if any value is greater than 5 in each row:",np.any(x>5,axis=1))
print(np.sum(x,axis = 1))
print("check if all is greater than 2:",np.all(x>2))
# N = 50
# x = np.random.rand(N)
# print(x)

# %%fancy indexing
X = np.array([[0,1,2,3],
              [4,5,6,7],
              [8,9,10,11]])
row = np.array([0,1,2])
mask = np.array([1,0,1,0], dtype=bool)
print("reshape to column vector:",row[:,np.newaxis])
print(X[row[:,np.newaxis],mask])

X = np.arange(10)
i = np.array([2,1,8])
X[i] -= 10
print(X)

x = np.zeros(10)
i = [2,3,3,4,4,4]
x[i] += 1 # it is the assignment that repeats multiple times
print(x)

x = np.zeros(10)
np.add.at(x,i,1) # the add behaviour repeats multiple times
print(x)

#%% manually plot a histogram
np.random.seed(42)
x= np.random.randn(100)

np.linspace(-5,5,3)
bins = np.linspace(-5,5,20) # divide 20 bins


counts = np.zeros_like(bins) # [0,0,...,0] 代表每个bins里的频率
i = np.searchsorted(bins,x) # 针对每个x放到对应的bin，i表示每个x所对应的bin的index
np.add.at(counts, i, 1) #每一个index位置的count数组都加1，从而得到每个bin的频率
plt.plot(bins,counts,linestyle = "steps")
plt.show()
# 等价: plt.hist(x,bins,histtype="step") which uses np.histogram(x,bins)


#%% sorting
x = np.array([2,1,4,3,5])
print(np.sort(x))
#x.sort()
#print(x)
i = np.argsort(x) # 返回indices
print(x[i]) # fancy indexing


rand = np.random.RandomState(42)
X = rand.randint(0,10,(4,6))
print(X)
print("sort each column:\n",np.sort(X,axis=0))
print("sort each row:\n",np.sort(X,axis=1))

# k nearest neighbors
X = rand.rand(10,2)
plt.scatter(X[:,0],X[:,1], s = 100)
plt.show()

# for each pair of points, compute differences in their coordinates
# 一行代码计算各点到其他点的距离问题:
# dist_sq = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=-1)
differences = X[:, np.newaxis, :] - X[np.newaxis, :, :]
sq_differences = differences ** 2
dist_sq = sq_differences.sum(-1)
nearest = np.argsort(dist_sq, axis=1)
print(nearest) # 第一列不算，是自己和自己的距离最短为0. 从左到右是从小到大离该点最近的点的index
#Each point in the plot has lines drawn to its two nearest neighbors.
K = 2
nearest_partition = np.argpartition(dist_sq, K + 1, axis=1) # 加1 是因为不包括自己这个点
plt.scatter(X[:, 0], X[:, 1], s=100)

for i in range(X.shape[0]):
    for j in nearest_partition[i, :K+1]:
        # plot a line from X[i] to X[j]
        # use some zip magic to make it happen:
        plt.plot(*zip(X[j], X[i]), color='black')

plt.show()

    #%% structured array

name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]

# Use a compound data type for structured arrays
data = np.zeros(4, dtype={'names':('name', 'age', 'weight'),
                          'formats':('U10', 'i4', 'f8')})
data['name'] = name
data['age'] = age
data['weight'] = weight
print(data)
print("names:",data['name'])
print("first row:",data[0])
print("last row :", data[-1]['name'])
print("name with age under 30:", data[data['age']<30]["name"])
#using record arrary
data_rec = data.view(np.recarray)
print("record array:",data_rec.age)

#%% dot of array:
# 情况1： If both a and b are 1-D arrays, it is inner product of vectors (without complex conjugation).
# 本来也就是处理a,b形状一样的情况 也就是shape=(n,)
# 不需要转置直接内积，最后得到一个scalar
# a,b 严格来说既不是column vector也不是row vector, 但是可以act like either one based on the position in a dot product.
a = np.array([1,2,3])
b = np.array([1,2,3])
c = np.dot(a,b)
print("c as inner product of vectors:", c) # c = 14

# 情况2：If both a and b are 2-D arrays, it is matrix multiplication, but using matmul or a @ b is preferred.
a = np.arange(1,5).reshape((2,2))
b = np.arange(5,9).reshape((2,2))
c = np.dot(a,b)
print(c) # [[19 22],[43 50]]


# 情况3：If both a and b are 0-D arrays,  it is equivalent to multiply and using numpy.multiply(a, b) or a * b is preferred.
a = 1
b = 2
c = np.dot(1,2)
print(c) # 2

# 情况4：If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.
a=np.arange(3).reshape((1,3))
b= np.arange(3) # 这里b是1D,shape(3,)
c= np.dot(a,b) # 用a的最后一个axis也就是3 去align b的唯一axis3, 匹配，然后分别乘法相加
print(c) # 5 as shape(1,) 因为上一步a和b的3都align 相当于抵了，剩下一个a的（1，)就作为c的shape
print(c.shape)

# 情况5：If a is an N-D array and b is an M-D array (where M>=2),
# it is a sum product over the last axis of a and the second-to-last axis of b:
# 这种情况就是需要第一个变量的最后axis去匹配第二个变量的倒数第二个axis
d=np.arange(12).reshape((3,4))
c= np.dot(b,d) # b: shape(3,) d: shape(3,4)
print(c) # array([20, 23, 26, 29]) 其中用b的3 去匹配倒数第二个axis也就是3，那么匹配，所以相互乘法后相加
print(c.shape) # 相互约去3之后，只有d剩一个（,4）这种情况放在shape里就是(4,)