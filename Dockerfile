FROM python:3.12-slim

WORKDIR /ptoie_dep

COPY pyproject.toml poetry.lock /ptoie_dep/
RUN pip install poetry  \
    && poetry config virtualenvs.create false \
    && poetry install --only main --no-root --no-directory
COPY . /ptoie_dep
RUN poetry install --only main

ENV PYTHONPATH="$PYTHONPATH:/ptoie_dep"

CMD [ "poetry", "run", "python3", "src/noie.py" ]
