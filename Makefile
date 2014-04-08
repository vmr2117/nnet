neuralnet_random_init_train:
	@python src/neural_network/nn_runner.py train --logistic_actv \
											       data/numpy_array_multiclass.train \
											       data/numpy_array_multiclass.valid \
											       models/neuralnet_random_init.model \
											       results/neuralnet_random_init_train.db \
												   2 \
												   1536 \
												   32 \
												   500

neuralnet_random_init_test:
	@python src/neural_network/nn_runner.py test  --logistic_actv \
											       data/numpy_array_multiclass.test \
											       models/neuralnet_random.model

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
												   weights/adaboost_50wl_10passes_sgd.weights

neuralnet_adaboost_init_test:
	@python src/neural_network/nn_runner.py test  --logistic_actv \
											       data/numpy_array_multiclass.test \
											       models/neuralnet_adaboost_init.model

neuralnet_random_init_train_graph:
	@python src/neural_network/perf_graph.py       results/neuralnet_random_init_train.db \
												   results/neuralnet_random_init_train.png 

neuralnet_adaboost_init_train_graph:
	@python src/neural_network/perf_graph.py       results/neuralnet_adaboost_init_train.db \
												   results/neuralnet_adaboost_init_train.png 

adaboost_train_100wl:
	@python src/adaboost/adaboostSAMME.py train    --log_loss --l2 \
											       data/numpy_array_multiclass.train \
											       data/numpy_array_multiclass.valid \
											       models/adaboost_100wl_60passes.model \
												   results/adaboost_100wl_60passes_train.png \
												   60 \
											       100
	@# 35 minutes

adaboost_train_40wl:
	@python src/adaboost/adaboostSAMME.py train    --log_loss --l2 \
											       data/numpy_array_multiclass.train \
											       data/numpy_array_multiclass.valid \
											       models/adaboost_40wl_60passes.model \
												   results/adaboost_40wl_60passes_train.png \
												   60 \
												   40

adaboost_test_40wl:
	@python src/adaboost/adaboostSAMME.py test     data/numpy_array_multiclass.test \
												   models/adaboost_40wl_60passes.model

adaboost_to_nnet_weights:	
	@python src/adaboost/model_rewriter.py         models/adaboost_50wl_40passes.model \
												   weights/adaboost_50wl_40passes.weights
