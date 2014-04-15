# Targets for Neural Network with random initial weights.
neuralnet_400random_init_train:
	@python src/neural_network/nn_runner.py train --logistic_actv \
											       data/numpy_array_multiclass.train \
											       data/numpy_array_multiclass.valid \
											       models/neuralnet_400random_init.model \
											       results/neuralnet_400random_init_train.db \
												   1000 \
												   1536 \
												   32 \
												   400
	# 7 hours
neuralnet_5000random_init_train:
	@python src/neural_network/nn_runner.py train --logistic_actv \
											       data/numpy_array_multiclass.train \
											       data/numpy_array_multiclass.valid \
											       models/neuralnet_5000random_init.model \
											       results/neuralnet_5000random_init_train.db \
												   1000 \
												   1536 \
												   32 \
												   5000

neuralnet_400random_init_test:
	@python src/neural_network/nn_runner.py test  --logistic_actv \
											       data/numpy_array_multiclass.test \
												   models/neuralnet_400random_init.model

neuralnet_5000random_init_test:
	@python src/neural_network/nn_runner.py test  --logistic_actv \
											       data/numpy_array_multiclass.test \
												   models/neuralnet_5000random_init.model

# Targets for Neural Network with initial weights obtained from adaboost.
neuralnet_40adaboost_init_train:
	@python src/neural_network/nn_runner.py train --logistic_actv \
											       data/numpy_array_multiclass.train \
											       data/numpy_array_multiclass.valid \
											       models/neuralnet_40adaboost_init.model \
											       results/neuralnet_40adaboost_init_train.db \
												   1000 \
												   1536 \
												   32 \
												   400 \
												   weights/adaboost_40wl_60passes.weights

neuralnet_500adaboost_init_train:
	@python src/neural_network/nn_runner.py train --logistic_actv \
											       data/numpy_array_multiclass.train \
											       data/numpy_array_multiclass.valid \
											       models/neuralnet_500adaboost_init.model \
											       results/neuralnet_500adaboost_init_train.db \
												   1000 \
												   1536 \
												   32 \
												   5000 \
												   weights/adaboost_500wl_1passes.weights

neuralnet_40adaboost_init_test:
	@python src/neural_network/nn_runner.py test  --logistic_actv \
											       data/numpy_array_multiclass.test \
											       models/neuralnet_40adaboost_init.model

neuralnet_500adaboost_init_test:
	@python src/neural_network/nn_runner.py test  --logistic_actv \
											       data/numpy_array_multiclass.test \
											       models/neuralnet_500adaboost_init.model

# Targets for generating different graphs

neuralnet_400random_init_train_graph:
	@python src/neural_network/perf_graph.py       results/neuralnet_400random_init_train.db \
												   results/neuralnet_400random_init_train.png \
												   'Neural Network randomly initialized'

neuralnet_5000random_init_train_graph:
	@python src/neural_network/perf_graph.py       results/neuralnet_5000random_init_train.db \
												   results/neuralnet_5000random_init_train.png \
												   'Neural Network randomly initialized'

neuralnet_40adaboost_init_train_graph:
	@python src/neural_network/perf_graph.py       results/neuralnet_40adaboost_init_train.db \
												   results/neuralnet_40adaboost_init_train.png \
												   'Neural Network adaboost initialized'

neuralnet_500adaboost_init_train_graph:
	@python src/neural_network/perf_graph.py       results/neuralnet_500adaboost_init_train.db \
												   results/neuralnet_500adaboost_init_train.png \
												   'Neural Network adaboost initialized'

neuralnet_400init_comparison_graph:
	@python src/neural_network/perf_graph_comp.py  results/neuralnet_400random_init_train.db \
												   results/neuralnet_40adaboost_init_train.db \
												   results/neuralnet_400init_comparison_train.png \
												   'Neural Network Initialization Comparison'

neuralnet_400init_comparison_graph_blown:
	@python src/neural_network/perf_graph_comp.py  results/neuralnet_400random_init_train.db \
												   results/neuralnet_40adaboost_init_train.db \
												   results/blwn_neuralnet_400init_comparison_train.png \
												   'Neural Network Initialization Comparison' \
												   50 \
												   1

neuralnet_5000init_comparison_graph_blown:
	@python src/neural_network/perf_graph_comp.py  results/neuralnet_5000random_init_train.db \
												   results/neuralnet_500adaboost_init_train.db \
												   results/blwn_neuralnet_5000init_comparison_train.png\
												   'Neural Network Initialization Comparison' \
												   130 \
												   1

# Targets for training adaboost SAMME models.
adaboost_train_100wl:
	@python src/adaboost/adaboostSAMME.py train    --log_loss --l2 \
											       data/numpy_array_multiclass.train \
											       data/numpy_array_multiclass.valid \
											       models/adaboost_100wl_60passes.model \
												   results/adaboost_100wl_60passes_train.png \
												   60 \
											       100

adaboost_train_40wl:
	@python src/adaboost/adaboostSAMME.py train    --log_loss --l2 \
											       data/numpy_array_multiclass.train \
											       data/numpy_array_multiclass.valid \
											       models/adaboost_40wl_60passes.model \
												   results/adaboost_40wl_60passes_train.png \
												   60 \
												   40

	# 2.2 hours
adaboost_train_500wl:
	@python src/adaboost/adaboostSAMME.py train    --log_loss --l2 \
											       data/numpy_array_multiclass.train \
											       data/numpy_array_multiclass.valid \
											       models/adaboost_500wl_1passes.model \
												   results/adaboost_500wl_1passes_train.png \
												   1 \
											       500

	# 1.2 hours
adaboost_test_40wl:
	@python src/adaboost/adaboostSAMME.py test     data/numpy_array_multiclass.test \
												   models/adaboost_40wl_60passes.model

40adaboost_to_nnet_weights:	
	@python src/adaboost/model_rewriter.py         models/adaboost_40wl_60passes.model \
												   weights/adaboost_40wl_60passes.weights
500adaboost_to_nnet_weights:	
	@python src/adaboost/model_rewriter.py         models/adaboost_500wl_1passes.model \
												   weights/adaboost_500wl_1passes.weights
