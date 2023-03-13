.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################

run_initialiaze:
	python -m plt.interface.main

run_api:
	uvicorn plt.api.fast:app --reload
