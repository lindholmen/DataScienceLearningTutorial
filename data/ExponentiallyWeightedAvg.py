import numpy as np

np.random.seed(3)
theta = np.random.randint(10,30,100)
print(theta)

def expotentially_weighted_avg(v,idx,beta=0.9):
    v = beta * v + (1-beta) * theta[idx]
    idx = idx+1
    if idx < len(theta):
        return expotentially_weighted_avg(v,idx)
    else:
        return v

print(expotentially_weighted_avg(0,0))
