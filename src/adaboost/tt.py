import numpy as np
T=2
m=3
f=np.zeros((3,5))
Y=np.array([1,2,3])
htx=np.array([1,2,4])
COST=np.zeros((3,5))
print COST
alpha=[]
k=[0,1,2,3,4]
for t in range(T):
            for i in range(m):
                for l in range(len(k)):
                    COST[i,l]=np.exp(f[i,l]-f[i,Y[i]])

            print 't is ',t
            COST[np.array(range(m)), Y] = 0
            d_sum = np.sum(COST, axis = 1)
            COST[np.array(range(m)), Y] = -d_sum
            delta = -np.sum(COST[np.array(range(m)), np.array(htx)])/(np.sum(d_sum))
            print 'delta is ',delta
            print COST
            alpha = 0.5 * np.log(1.0 * (1 + delta) / (1 - delta))
            print 'ALPHA ', alpha

            ind_vec_htx = np.zeros_like(f) 
            ind_vec_htx[np.array(range(m)), np.array(htx)] = alpha
            print f
            f += ind_vec_htx

            print f
            htx=np.array([0,2,4])
