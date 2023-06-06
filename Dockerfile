FROM pytorch/pytorch:latest

ENV PYTHON_VERSION 3.9

WORKDIR /ptoie_dep

COPY pyproject.toml /ptoie_dep/pyproject.toml
COPY poetry.lock /ptoie_dep/poetry.lock

RUN python3 -m pip install poetry==1.5.0
RUN poetry config virtualenvs.create false
RUN poetry install

COPY . /ptoie_dep

ENV PYTHONPATH="$PYTHONPATH:/ptoie_dep"

ENTRYPOINT [ "poetry", "run", "python3", "src/main.py" ]
