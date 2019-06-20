import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import numpy as np
seaborn.set()

#%% seatle precip
rainfall = pd.read_csv("data/Seattle2014.csv")['PRCP'].values
inches = rainfall / 254
plt.hist(inches,40)
print(inches)
print("number of days with more than 0.5 inches but less than 1.0 inches:",np.sum((inches>0.5)& (inches < 1)))
print("number of days without rain:", np.sum(inches == 0))
print("median precip on rainy days:", np.median(inches[inches>0]))
#plt.show()

#%% multivariate_normal visualisation and fancy indexing
mean = [0,0]
cov = [[1,2],
       [2,5]]
X = np.random.multivariate_normal(mean, cov, 100)
print("x shape:", X.shape) # (100,2)
#plt.show()
indices = np.random.choice(X.shape[0],20, replace=False)
#using fancy indexing
selection = X[indices]
#print(selection)
#print(selection.shape)
# need to see what are selected
plt.scatter(X[:,0],X[:,1],c=['r','b'],marker="o", alpha= 0.5,s =80)
plt.scatter(selection[:,0],selection[:,1],c= "k",marker="*")
#plt.show()

#%% multivariate_normal visualisation and fancy indexing