import numpy as np

def read_MnistFile(file_path):
        examples=open(file_path,"r")
        mnist_after=[]
        for example in examples:
            mnist_after.append(example)
        examples.close()
        
        return mnist_after

def transform(COST_MATRIX, vw_mnist):
        n_samples, n_features = np.shape(COST_MATRIX)
        result = []
        for i in range(n_samples):
            tuple_exampe=vw_mnist[i].split('| ')
            feature_value=tuple_exampe[1]
            vw_csoaa_example=' '.join([' '.join([str(j)+':'+`COST_MATRIX[i,j]` for j in range(n_features) if COST_MATRIX[i,j] != 0]),'|',feature_value])
            print vw_csoaa_example
            result.append(vw_csoaa_example)
        return result



path = '/home/liguifan/Desktop/data/validation.txt'
C=np.zeros((2,10))

C[1,2]=5
C[0,1]=10

choices = [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]
print type(choices)

