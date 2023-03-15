FROM python:3.10.6-buster

COPY raw_data/training_outputs/my_best_model.epoch97-loss0.26.hdf5 raw_data/training_outputs/my_best_model.epoch97-loss0.26.hdf5

# We strip the requirements from useless packages like `ipykernel`, `matplotlib` etc...
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY plt plt
COPY setup.py setup.py
RUN pip install .
# RUN pip install -e .

# COPY Makefile Makefile
# RUN make run_api

CMD uvicorn plt.api.fast:app --host 0.0.0.0 --port $PORT
