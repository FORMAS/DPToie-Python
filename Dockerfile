FROM python:3.12-slim

WORKDIR /ptoie_python

COPY pyproject.toml poetry.lock /ptoie_python/
RUN pip install poetry  \
    && poetry config virtualenvs.create false \
    && poetry install --only main --no-root --no-directory
COPY . /ptoie_python
RUN poetry install --only main

ENV PYTHONPATH="$PYTHONPATH:/ptoie_python"

CMD [ "poetry", "run", "python3", "src/noie.py" ]
