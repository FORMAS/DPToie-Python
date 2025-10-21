
# DPToie-Python

Extrator de Informação Aberta para língua portuguesa baseado em análise de dependências (SpaCy + Stanza).

Este guia mostra todas as formas de rodar o projeto pelo `src/main.py`, com todas as variações de argumentos, tanto localmente (Poetry) quanto com Docker/Docker Compose.

- Requisitos mínimos: Python 3.12+, Poetry, ou Docker (opcional)
- Modelos: o Stanza faz o download automático na primeira execução. Você pode definir `STANZA_RESOURCES_DIR` para usar um diretório local de modelos (ex.: `./models/.stanza_resources`).

## Instalação (Poetry)

```bash
poetry install
```

## Como executar (local, via Poetry)

Forma geral:

```bash
poetry run python3 src/main.py \
  -i <caminho_entrada> \
  -it <txt|conll> \
  -o <caminho_saida> \
  -ot <json|csv|txt> \
  [-cc] [-sc] [-hs] [-a] [-t] [-debug]
```

### Argumentos suportados

- -i, --input: caminho do arquivo de entrada. Padrão: `./inputs/teste.txt`
- -it, --input-type: tipo do arquivo de entrada. Opções: `txt` ou `conll`. Padrão: `txt`
  - Na entrada `txt`: cada linha do arquivo é uma sentença; o sistema gera um `.conll` temporário.
  - Na entrada `conll`: o arquivo de entrada já está no formato CoNLL-U (uma sentença por bloco, separado por linha vazia).
- -o, --output: caminho do arquivo de saída. Padrão: `./outputs/output.json`
- -ot, --output-type: formato de saída. Opções: `json`, `csv`, `txt`. Padrão: `json`
- -cc, --coordinating_conjunctions: ativa extrações com conjunções coordenativas
- -sc, --subordinating_conjunctions: ativa extrações com conjunções subordinativas
- -hs, --hidden_subjects: ativa extrações com sujeito oculto (Não implementado)
- -a, --appositive: ativa extrações apositivas
- -t, --transitive: ativa a transitividade para apositivas (só tem efeito quando `-a` está ativo)
- -debug: modo verbose para depuração

Importante:
- Os módulos de extração são desativados por padrão. Ative os que deseja usando as flags `-cc -sc -a -t`.

### Exemplos práticos

1) TXT de entrada, JSON de saída (padrões):
    ```bash
    poetry run python3 src/main.py -i ./inputs/ceten-200.txt -it txt -o ./outputs/out.json -ot json
    ```

2) TXT de entrada, CSV de saída, ativando coordenação e sujeito oculto:
    ```bash
    poetry run python3 src/main.py -i ./inputs/ceten-200.txt -it txt -o ./outputs/out.csv -ot csv -cc
    ```

3) TXT de entrada, saída em texto legível:
    ```bash
    poetry run python3 src/main.py -i ./inputs/ceten-200.txt -it txt -o ./outputs/out.txt -ot txt -cc -sc -a -t
    ```

4) Entrada já em CoNLL-U, JSON de saída:
    ```bash
    poetry run python3 src/main.py -i ./inputs/teste.conll -it conll -o ./outputs/out.json -ot json -cc -sc -a -t
    ```

5) Somente conjunções coordenativas:
    ```bash
    poetry run python3 src/main.py -i ./inputs/ceten-200.txt -it txt -o ./outputs/cc.json -ot json -cc
    ```

6) Modo debug para inspeção detalhada:
    ```bash
    poetry run python3 src/main.py -i ./inputs/ceten-200.txt -it txt -o ./outputs/out.json -ot json -cc -debug
    ```

7) Ver lista de argumentos:
    ```bash
    poetry run python3 src/main.py -h
    ```

Saídas esperadas:
- JSON: lista de objetos por sentença, com as extrações dentro de `extractions` e possíveis `sub_extractions`.
- CSV: colunas `id`, `sentence`, `arg1`, `rel`, `arg2` (inclui subextrações com ids hierárquicos como `1.1`).
- TXT: sentença seguida das extrações e subextrações formatadas em linhas.

## Como executar com Docker (sem Compose)

Build da imagem (na raiz do projeto):
```bash
docker build -t ptoie_python .
```

Executar um comando pontual (mapeando o diretório atual e apontando para os arquivos dentro do container):
```bash
docker run --rm -it \
  -e STANZA_RESOURCES_DIR=/ptoie_python/models/.stanza_resources \
  -v "$(pwd)":/ptoie_python \
  -w /ptoie_python \
  ptoie_python \
  poetry run python3 src/main.py -i /ptoie_python/inputs/teste.conll -it conll -o /ptoie_python/outputs/out.json -ot json -cc -sc -a -t
```

Observação: ajuste os caminhos de `-i` e `-o` conforme necessário; use `-it txt` quando a entrada for texto linha-a-linha.

## Como executar com Docker Compose

O arquivo `docker-compose.yml` já inclui o serviço `ptoie_python`. Você pode editar a linha `command:` para o cenário desejado. Exemplo de comando recomendado:

```yaml
command: poetry run python3 src/main.py -i /ptoie_python/inputs/teste.conll -it conll -o /ptoie_python/outputs/out.json -ot json -cc -sc -a -t
```

Então rode:
```bash
docker compose up --build
```

Use o comando `run` para executar outros comandos personalizados:
```bash
docker compose run ptoie_python poetry run python3 src/main.py -i /ptoie_python/inputs/ceten-200.txt -it txt -o /ptoie_python/outputs/out.csv -ot csv -cc
```



Dicas:
- O volume `.:/ptoie_python` permite usar arquivos da pasta local dentro do container.
- `STANZA_RESOURCES_DIR` (exposto no compose) pode apontar para `models/.stanza_resources` para evitar downloads repetidos.

## Referências rápidas

- Entrada TXT: cada linha é uma sentença; o sistema cria um `.conll` temporário.
- Entrada CoNLL-U: use `-it conll` e garanta sentenças separadas por linha vazia.
- Ativação das regras: todas desativadas por padrão; adicione as flags desejadas.
- Caminhos relativos são interpretados a partir da raiz do projeto; no Docker, use caminhos absolutos dentro do container (ex.: `/ptoie_python/...`).

## Como citar este projeto
Se você utilizar este projeto em sua pesquisa, por favor, cite-o da seguinte forma:

```bibtex
@Article{dptoie2025, author={xxx xxx}, title={xxxx}, journal={dddd}, year={xxx}, month={x}, day={cc}, issn={xxx}, doi={xxxxx}, url={asas} }
```