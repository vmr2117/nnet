import numpy as np


def cost2weight(COST,predict):
	return COST[np.array(range(len(predict))), np.array(predict)]


COST=np.array([[1,2,3],[4,5,6],[7,8,9]])
print COST

print cost2weight(COST,[0,1,1])




def C45_read_file(path):
        X=[]
        y=[]
        file1=open(path,'r')
        for example in file1:
            y.append(int(ord(example[0]))-48)
            tuple_exampe=example.split(' | ')
            feature_value=tuple_exampe[1]
            mm=feature_value[:-2].split(' ')
            ss=[float(x) for x in mm] 
            X.append(ss)

        return X,y

if __name__ == '__main__':
	path='/Users/liguifan/Documents/nnet/data/niubi.test'
	X,y=C45_read_file(path)
	print y[1:100]

print type(X[1]



x = np.array(['1.1', '2.2', '3.3'], dtype='|S4')
y = x.astype(np.float)



