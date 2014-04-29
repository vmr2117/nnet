# Targets to generate standardized traininig, validation and test set.
generate_standardized_data:
	python src/data_prep/mnist_dataprep.py \
	    	data/mnist.pkl.gz \
		data/	

# Targets for adaboost training and testing.
adaboost_train:
	python src/adaboost/adaboostSAMME.py \
		train \
		data/numpy_array_multiclass.train \
		data/numpy_array_multiclass.valid \
		models/adaboost.model \
		results/adaboost_train.png \
		1 \
		1 \
		--log_loss \
		--l2

adaboost_test:
	python src/adaboost/adaboostSAMME.py \
	    	test \
		data/numpy_array_multiclass.test \
		models/adaboost.model

# Targets for generating intitial weights for neural network
ffnn_weights_from_adaboost:	
	python src/adaboost/model_rewriter.py \
	    	models/adaboost.model \
		weights/ffnn_adaboost_init.weights

ffnn_weights_uniform_random:	
	python src/neural_network/weights_generator.py \
		weights/ffnn_400tanh_random_init.weights \
		tanh \
		784 \
		400 \
		10

# Targets for Training the network.
ffnn_400tanh_random_init_train:
	python src/neural_network/nn_runner.py \
	   	train \
		data/numpy_array_multiclass.train \
		data/numpy_array_multiclass.valid \
		weights/ffnn_400tanh_random_init.weights \
		models/ffnn_400tanh_random_init.model \
		results/ffnn_400tanh_random_init.db \
		500 \
		1563 \
		32 \
		--tanh_actv

ffnn_400tanh_adaboost_init_train:
	python src/neural_network/nn_runner.py \
	    	train \
		data/numpy_array_multiclass.train \
		data/numpy_array_multiclass.valid \
		weights/ffnn_400tanh_adaboost_init.weights \
		models/ffnn_400tanh_adaboost_init.model \
		results/ffnn_400tanh_adaboost_init.db \
		500 \
		1563 \
		32 \
		--tanh_actv

# Targets for Testing the network.
ffnn_400tanh_random_init_test:
	python src/neural_network/nn_runner.py \
	    	test \
		data/numpy_array_multiclass.test \
		models/ffnn_400tanh_random_init.model \
	    	--tanh_actv

ffnn_400tanh_adaboost_init_test:
	python src/neural_network/nn_runner.py \
       		test \
		data/numpy_array_multiclass.test \
		models/ffnn_400tanh_adaboost_init.model \
	    	--tanh_actv 

# Targets for Training performance graphs.
ffnn_400tanh_random_init_train_graph:
	python src/neural_network/perf_graph.py \
		results/ffnn_400tanh_random_init.db \
		results/ffnn_400tanh_random_init.png \
		'Neural Network Training graph - uniformly randomly initialized'

ffnn_400tanh_adaboost_init_train_graph:
	python src/neural_network/perf_graph.py \
		results/ffnn_400tanh_adaboost_init.db \
		results/ffnn_400tanh_adaboost_init.png \
		'Neural Network Training graph - adaboost initialized'

ffnn_400tanh_train_compare_graph:
	python src/neural_network/perf_graph_comp.py \
	    	results/ffnn_400tanh_random_init.db \
		results/ffnn_400tanh_adaboost_init.db \
		results/ffnn_400tanh_train_compare.png \
		'Neural Network Training graph - initializaion comparison'

# Targets for hinton diagrams:
ffnn_random_init_hinton_comparison_fig:
	python src/neural_network/hinton_diagram.py \
	    	weights/ffnn_400tanh_random_init.weights \
		models/ffnn_400tanh_random_init.model \
		'Neural Network - Randomly initialized weights' \
		results/ffnn_random_init_comp_hinton_fig.png

ffnn_adaboost_init_hinton_comparison_fig:
	python src/neural_network/hinton_diagram.py \
	    	weights/ffnn_400tanh_adaboost_init.weights \
		models/ffnn_400tanh_adaboost_init.model \
		'Neural Network - Adaboost initialized weights' \
		results/ffnn_adaboost_init_comp_hinton_fig.png

# Targets for histograms
ffnn_random_init_hist_comparison_fig:
	python src/neural_network/histogram.py \
	    	weights/ffnn_400tanh_random_init.weights \
		models/ffnn_400tanh_random_init.model \
		'Neural Network - Randomly initialized weights' \
		results/ffnn_random_init_comp_hist_fig.png

ffnn_adaboost_init_hist_comparison_fig:
	python src/neural_network/histogram.py \
	    	weights/ffnn_400tanh_adaboost_init.weights \
		models/ffnn_400tanh_adaboost_init.model \
		'Neural Network - Adaboost initialized weights' \
		results/ffnn_adaboost_init_comp_hist_fig.png
