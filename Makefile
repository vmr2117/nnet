# Targets to generate standardized traininig, validation and test set.
gen_std_data:
	python src/data_prep/mnist_dataprep.py \
	    	data/mnist.pkl.gz \
		data/	

# Targets for generating intitial weights for neural network
gen_ur_init_weights:	
	python src/neural_network/weights_generator.py \
		weights/random_init_784_300_10_tanh.weights \
		tanh \
		784 \
		300 \
		10

# Targets for Training and Testing the network.
train_ur_init_network:
	python src/neural_network/nn_runner.py \
	   	train \
		data/numpy_array_multiclass.train \
		data/numpy_array_multiclass.valid \
		weights/random_init_784_300_10_tanh.weights \
		models/ffnn_random_init_784_300_10_tanh.model \
		results/ffnn_random_init_784_300_10_tanh.perf \
		results/ffnn_random_init_784_300_10_tanh.debug \
		50 \
		1563 \
		32 \
		--tanh_actv

train_ll_ur_init_network:
	python src/neural_network/nn_runner.py \
	   	train \
		data/numpy_array_multiclass.train \
		data/numpy_array_multiclass.valid \
		weights/random_init_784_300_10_tanh.weights \
		models/ffnn_random_init_784_300_10_tanh_ll.model \
		results/ffnn_random_init_784_300_10_tanh_ll.perf \
		results/ffnn_random_init_784_300_10_tanh_ll.debug \
		50 \
		1563 \
		32 \
		--tanh_actv \
		--train_layers 1

# Targets for Testing the network.
test_ur_init_network:
	python src/neural_network/nn_runner.py \
	    	test \
		data/numpy_array_multiclass.test \
		models/ffnn_random_init_784_300_10_tanh.model \
	    	--tanh_actv

test_ll_ur_init_network:
	python src/neural_network/nn_runner.py \
	    	test \
		data/numpy_array_multiclass.test \
		models/ffnn_random_init_784_300_10_tanh_ll.model \
	    	--tanh_actv

# Targets for Training performance graphs.
graph_ur_init_network_train_perf:
	python src/neural_network/perf_graph.py \
		results/ffnn_random_init_784_300_10_tanh.perf \
		results/ffnn_random_init_784_300_10_tanh_train_perf.eps \
		'Neural Network Training Error - UR initilized'

graph_ll_ur_init_network_train_perf:
	python src/neural_network/perf_graph.py \
		results/ffnn_random_init_784_300_10_tanh_ll.perf \
		results/ffnn_random_init_784_300_10_tanh_train_perf_ll.eps \
		'LL Neural Network Training Error - UR initialized'

# Targets for Activations and weights debugging.
graph_ur_init_network_debug_params:
	python src/neural_network/distribution_graph.py \
		results/ffnn_random_init_784_300_10_tanh.debug \
		results/ffnn_random_init_784_300_10_tanh_change

graph_ll_ur_init_network_debug_params:
	python src/neural_network/distribution_graph.py \
		results/ffnn_random_init_784_300_10_tanh_ll.debug \
		results/ffnn_random_init_784_300_10_tanh_change_ll

