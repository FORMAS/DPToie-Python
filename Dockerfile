FROM python:3.12-slim

WORKDIR /dptoie_python

COPY pyproject.toml poetry.lock /dptoie_python/
RUN pip install poetry  \
    && poetry config virtualenvs.create false \
    && poetry install --only main --no-root --no-directory
COPY . /dptoie_python

RUN pip install torch==2.7.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN poetry install --only main --no-root

ENV PYTHONPATH="$PYTHONPATH:/dptoie_python"

CMD [ "poetry", "run", "python3", "src/noie.py" ]
