import numpy as np
def convert_cost_vw(cost_matrix):
	f=open('liguifan','w')
	_,col=cost_matrix.shape
	print 'the col value is ', col
        for i in range(len(cost_matrix)):
            #m_temp=[]
            for j in range(col):
            	if cost_matrix[i][j]!=0:
            	#m_temp.append(`j+1`+':'+`cost_matrix[i][j]`)
            		f.write(`j+1`+':'+`cost_matrix[i][j]`+' ')
            f.write('\n')
        f.close()


cost_matrix=np.zeros((5,5))
cost_matrix[0,4]=81
cost_matrix[1,1]=58
cost_matrix[2,1]=27
cost_matrix[1,2]=99
cost_matrix[3,1]=5
cost_matrix[4,3]=88
s=convert_cost_vw(cost_matrix)

print cost_matrix
