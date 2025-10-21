# DPToie-Python

Open Information Extractor for Portuguese based on dependency analysis (SpaCy + Stanza).

This guide shows all ways to run the project via `src/dptoie/main.py`, with all argument variations, both locally (Poetry) and with Docker / Docker Compose.

- Minimum requirements: Python 3.12+, Poetry, or Docker (optional)
- Models: Stanza downloads models automatically on first run. You can set `STANZA_RESOURCES_DIR` to use a local models directory (e.g., `./models/.stanza_resources`).

## Table of Contents
- [Installation (Poetry)](#installation-poetry)
- [How to run (local, via Poetry)](#how-to-run-local-via-poetry)
  - [Supported arguments](#supported-arguments)
  - [Practical examples](#practical-examples)
- [How to run with Docker (without Compose)](#how-to-run-with-docker-without-compose)
- [How to run with Docker Compose](#how-to-run-with-docker-compose)
- [Quick references](#quick-references)
- [How to cite this project](#how-to-cite)
- [Authors](#authors)

## Installation (Poetry)

```bash
poetry install
```

## How to run (local, via Poetry)

General form:

```bash
poetry run python3 src/dptoie/main.py \
  -i <input_path> \
  -it <txt|conll> \
  -o <output_path> \
  -ot <json|csv|txt> \
  [-cc] [-sc] [-hs] [-a] [-t] [-debug]
```

### Supported arguments

- -i, --input: path to the input file. Default: `./inputs/teste.txt`
- -it, --input-type: input file type. Options: `txt` or `conll`. Default: `txt`
  - For `txt` input: each line in the file is a sentence; the system generates a temporary `.conll`.
  - For `conll` input: the input file is already in CoNLL-U format (one sentence per block, separated by an empty line).
- -o, --output: path to the output file. Default: `./outputs/output.json`
- -ot, --output-type: output format. Options: `json`, `csv`, `txt`. Default: `json`
- -cc, --coordinating_conjunctions: enable extractions using coordinating conjunctions
- -sc, --subordinating_conjunctions: enable extractions using subordinating conjunctions
- -hs, --hidden_subjects: enable extractions with hidden subjects (Not implemented)
- -a, --appositive: enable appositive extractions
- -t, --transitive: enable transitivity for appositives (only has effect when `-a` is active)
- -debug: verbose debug mode

Important:
- Extraction modules are disabled by default. Enable the ones you want using the flags `-cc -sc -a -t`.

### Practical examples

1) TXT input, JSON output (defaults):
```bash
poetry run python3 src/dptoie/main.py -i ./inputs/ceten-200.txt -it txt -o ./outputs/out.json -ot json
```

2) TXT input, CSV output, enabling coordination and hidden subject (flag example):
```bash
poetry run python3 src/dptoie/main.py -i ./inputs/ceten-200.txt -it txt -o ./outputs/out.csv -ot csv -cc
```

3) TXT input, human-readable text output:
```bash
poetry run python3 src/dptoie/main.py -i ./inputs/ceten-200.txt -it txt -o ./outputs/out.txt -ot txt -cc -sc -a -t
```

4) Input already in CoNLL-U, JSON output:
```bash
poetry run python3 src/dptoie/main.py -i ./inputs/teste.conll -it conll -o ./outputs/out.json -ot json -cc -sc -a -t
```

5) Only coordinating conjunctions:
```bash
poetry run python3 src/dptoie/main.py -i ./inputs/ceten-200.txt -it txt -o ./outputs/cc.json -ot json -cc
```

6) Debug mode for detailed inspection:
```bash
poetry run python3 src/dptoie/main.py -i ./inputs/ceten-200.txt -it txt -o ./outputs/out.json -ot json -cc -debug
```

7) Show arguments list:
```bash
poetry run python3 src/dptoie/main.py -h
```

Expected outputs:
- JSON: a list of objects per sentence, with extractions inside `extractions` and possible `sub_extractions`.
- CSV: columns `id`, `sentence`, `arg1`, `rel`, `arg2` (includes sub-extractions with hierarchical ids like `1.1`).
- TXT: the sentence followed by extractions and sub-extractions formatted as lines.

## How to run with Docker (without Compose)

Build the image (from the project root):
```bash
docker build -t dptoie_python .
```

Run a one-off command (mounting the current directory and pointing to files inside the container):
```bash
docker run --rm -it \
  -e STANZA_RESOURCES_DIR=/dptoie_python/models/.stanza_resources \
  -v "$(pwd)":/dptoie_python \
  -w /dptoie_python \
  dptoie_python \
  poetry run python3 src/dptoie/main.py -i /dptoie_python/inputs/teste.conll -it conll -o /dptoie_python/outputs/out.json -ot json -cc -sc -a -t
```

Note: adjust the `-i` and `-o` paths as needed; use `-it txt` when the input is line-by-line text.

## How to run with Docker Compose

The `docker-compose.yml` file already includes the `dptoie_python` service. You can edit the `command:` line for the desired scenario. Example recommended command:

```yaml
command: poetry run python3 src/dptoie/main.py -i /dptoie_python/inputs/teste.conll -it conll -o /dptoie_python/outputs/out.json -ot json -cc -sc -a -t
```

Then run:
```bash
docker compose up --build
```

Use `run` to execute other custom commands:
```bash
docker compose run dptoie_python poetry run python3 src/dptoie/main.py -i /dptoie_python/inputs/ceten-200.txt -it txt -o /dptoie_python/outputs/out.csv -ot csv -cc
```

Tips:
- The volume `.:/dptoie_python` allows using local files inside the container.
- `STANZA_RESOURCES_DIR` (exposed in the compose file) can point to `models/.stanza_resources` to avoid repeated downloads.

## Quick references

- TXT input: each line is a sentence; the system creates a temporary `.conll`.
- CoNLL-U input: use `-it conll` and ensure sentences are separated by an empty line.
- Rule activation: all rules are disabled by default; add the desired flags.
- Relative paths are interpreted from the project root; in Docker, use absolute paths inside the container (e.g., `/dptoie_python/...`).


## How to cite
If you find this repo helpful, please consider citing:

```bibtex
@Article{dptoie2025, author={xxx xxx}, title={xxxx}, journal={dddd}, year={xxx}, month={x}, day={cc}, issn={xxx}, doi={xxxxx}, url={asas} }
```

## Authors
* Andre Walker
* Rafael Glauber
* [Daniela Barreiro Claro](http://formas.ufba.br/dclaro/)