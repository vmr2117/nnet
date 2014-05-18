import numpy as np
#import matplotlib.pyplot as plt
import pylab as plt
def plot_x(value):
	fig = plt.figure()
	iters = 5
   	x_axis = range(iters)
  
   	plt.plot(x_axis, value, 'k--', label = 'AdaMM')
   	plt.legend()
   	plt.xlabel('Iterations')
   	plt.ylabel('Accuracy')
   	plt.title('AdaboostMM')
   	plt.grid(True)
   	plt.show()
	#plt.axis([40, 160, 0, 0.03])\
	filename='plot1'
	fig.savefig(filename)
if __name__ == '__main__':
	x=[1,2,7,4,5]
	plot_x(x)