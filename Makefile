neuralnet_random_init_train:
	@python src/neural_network/nn_runner.py train --logistic_actv \
											       data/numpy_array_multiclass.train \
											       data/numpy_array_multiclass.valid \
											       models/neuralnet_random_init.model \
											       results/neuralnet_random_init_train.db \
												   1000 \
												   1536 \
												   32 \
												   500 &

neuralnet_random_init_test:
	@python src/neural_network/nn_runner.py test  --logistic_actv \
											       data/numpy_array_multiclass.test \
											       models/neuralnet_random.model &

neuralnet_adaboost_init_train:
	@python src/neural_network/nn_runner.py train --logistic_actv \
											       data/numpy_array_multiclass.train \
											       data/numpy_array_multiclass.valid \
											       models/neuralnet_adaboost_init.model \
											       results/neuralnet_adaboost_init_train.db \
												   1000 \
												   1536 \
												   32 \
												   500 \
												   weights/adaboost_50wl_10passes_sgd.weights &

neuralnet_adaboost_init_test:
	@python src/neural_network/nn_runner.py test  --logistic_actv \
											       data/numpy_array_multiclass.test \
											       models/neuralnet_adaboost_init.model &

neuralnet_random_init_train_graph:
	@python src/neural_network/perf_graph.py       results/neuralnet_random_init_train.db \
												   results/neuralnet_random_init_train.png 

neuralnet_adaboost_init_train_graph:
	@python src/neural_network/perf_graph.py       results/neuralnet_adaboost_init_train.db \
												   results/neuralnet_adaboost_init_train.png 
											
