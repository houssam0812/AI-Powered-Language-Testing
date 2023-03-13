.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################

run_initialiaze:
#	 python -c 'from plt.interface.main import initialize_model; initialize_model()'
#	python -c 'from plt.interface.main import compile_model; compile_model(model_1)'
	python -m plt.interface.main
# run_root_mean_squared_error:
# 	python -c 'from plt.interface.main import root_mean_squared_error; root_mean_squared_error()'

run_train:
	python -c 'from plt.interface.main import train; train()'

run_compile:
	python -c 'from plt.interface.main import compile_model; compile_model()'

run_load_weights:
	python -c 'from plt.interface.main import load_weights; load_weights()'

run_evaluate:
	python -c 'from plt.interface.main import evaluate; evaluate()'

run_one_text:
	python -c 'from plt.interface.main import one_text; one_text()'

run_prediction:
	python -c 'from plt.interface.main import prediction; prediction(new_str0)'

run_all: run_initialiaze,  run_compile

#run_load_weights, run_prediction



run_api:
	uvicorn plt.api.fast:app --reload

