FROM python:3.12-slim

RUN pip install poetry==1.8.0

WORKDIR /ptoie_dep

RUN poetry config virtualenvs.create false

COPY pyproject.toml /ptoie_dep/pyproject.toml
COPY poetry.lock /ptoie_dep/poetry.lock

RUN poetry install --no-root --no-directory

COPY . /ptoie_dep

RUN poetry install --no-dev

ENV PYTHONPATH="$PYTHONPATH:/ptoie_dep"

CMD [ "poetry", "run", "python3", "src/noie.py" ]
