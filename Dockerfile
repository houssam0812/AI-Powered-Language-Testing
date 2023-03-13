FROM 

# We strip the requirements from useless packages like `ipykernel`, `matplotlib` etc...
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY plt plt
COPY setup.py setup.py
RUN pip install .

#CMD uvicorn plt.api.fast:app --host 0.0.0.0 --port $PORT