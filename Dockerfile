FROM pytorch/pytorch:latest

ENV PYTHON_VERSION 3.9

WORKDIR /ptoie_dep

RUN python3 -m pip install poetry==1.5.0
RUN poetry config virtualenvs.create false

COPY pyproject.toml /ptoie_dep/pyproject.toml
COPY poetry.lock /ptoie_dep/poetry.lock

RUN poetry install

ENV STANZA_RESOURCES_DIR="/ptoie_dep/.stanza_resources"
RUN poetry run python3 -c "import spacy_stanza; spacy_stanza.load_pipeline('pt')"

COPY . /ptoie_dep

ENV PYTHONPATH="$PYTHONPATH:/ptoie_dep"

ENTRYPOINT [ "poetry", "run", "python3", "src/noie.py" ]
