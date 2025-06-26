import argparse
import json
import logging
from typing import Any, Generator
from spacy_conll import init_parser
from spacy_conll.parser import ConllParser
from spacy.tokens import Doc
from src.extraction import Extractor, ExtractorConfig

logging.basicConfig(level=logging.INFO)

def main(input_file: str, output_file: str, conll_format: bool = False, coordinating_conjunctions: bool = True, subordinating_conjunctions: bool = True, appositive: bool = True, transitive: bool = True, debug: bool = False):
    extractor = Extractor(ExtractorConfig(
        coordinating_conjunctions=coordinating_conjunctions,
        subordinating_conjunctions=subordinating_conjunctions,
        appositive=appositive,
        transitive=transitive,
        debug=debug,
    ))

    Doc.set_extension("extractions", getter=extractor.get_extractions_from_doc)

    if not conll_format:
        nlp = init_parser("pt",
                          "stanza",
                          parser_opts={"use_gpu": True, "verbose": False},
                          include_headers=True
                          )
        connl_file = './outputs/input.conll'
        # clean the file if it exists
        with open(connl_file, 'w') as f:
            f.write('')

        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    doc = nlp(line.strip())
                    with open(connl_file, 'a', encoding='utf-8') as fout:
                        fout.write(doc._.conll_str)
                        fout.write('\n')  # Adiciona uma linha em branco entre sentenças
        input_file = connl_file

    nlp = ConllParser(init_parser("pt_core_news_sm", "spacy"))

    extractions = {
        'config': dict(extractor.config),
        'sentences': []
    }

    for i, sentence in enumerate(read_conll_sentences(input_file), 1):
        doc = nlp.parse_conll_text_as_spacy(sentence)

        sentence = {
            'sentence': doc.text.strip(),
            'extractions': []
        }
        for extraction in doc._.extractions:
            sentence['extractions'].append(dict(extraction))
            if debug:
                sentence['extractions'].append({
                    'debug': {
                        'subject': {
                            'token': extraction.subject.core.text.strip(),
                            'pieces': [token.text.strip() for token in extraction.subject.pieces],
                        },
                        'relation': {
                            'token': extraction.relation.core.text.strip(),
                            'pieces': [token.text.strip() for token in extraction.relation.pieces],
                        },
                        'complement': {
                            'token': extraction.complement.core.text.strip() if extraction.complement.core else None,
                            'pieces': [token.text.strip() for token in extraction.complement.pieces],
                        }
                    }
                })
        extractions['sentences'].append(sentence)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(extractions, indent=2, ensure_ascii=False))


def read_conll_sentences(file_path: str) -> Generator[str, Any, None]:
    """
    Lê um arquivo CONLL onde sentenças são separadas por linhas vazias
    Gera cada sentença como uma lista de linhas (strings)
    """
    current_sentence = ''

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line:  # Se a linha não está vazia
                current_sentence += line + '\n'  # Acumula a linha na sentença atual
            else:  # Linha vazia indica fim de sentença
                if current_sentence:  # Se temos uma sentença acumulada
                    yield current_sentence
                    current_sentence = ''  # Reseta a sentença atual

        # Retorna a última sentença se o arquivo não terminar com linha vazia
        if current_sentence:
            yield current_sentence


def extract_facts_from_doc(doc: Doc) -> dict:
    output = {
        'facts': []
    }

    sentence = {
        'text': doc.text,
        'facts': []
    }
    for extraction in doc._.extractions:
        sentence['facts'].append({
            'subject': ' '.join([token.text for token in extraction.subject]),
        })
    output['facts'].append(sentence)

    return output

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract clauses from a text file.')

    parser.add_argument('-path', metavar='path', type=str, help='path to the text file', default='./inputs/teste.txt')
    parser.add_argument('-out', metavar='out', type=str, help='path to the output file', default='./outputs/extractions.json')
    parser.add_argument('-conll', action='store_true', help='input file is in CONLL format')
    parser.add_argument('-cc', '--coordinating_conjunctions', dest='coordinating_conjunctions', action='store_true', help='enable coordinating conjunctions extraction')
    parser.add_argument('-sc', '--subordinating_conjunctions', dest='subordinating_conjunctions', action='store_true', help='enable subordinating conjunctions extraction')
    parser.add_argument('-a', '--appositive', dest='appositive', action='store_true', help='enable appositive extraction')
    parser.add_argument('-t', '--transitive', dest='transitive', action='store_true', help='enable transitive extraction(only for appositive)')
    parser.add_argument('-debug', action='store_true', help='enable debug mode')

    args = parser.parse_args()

    main(input_file=args.path, output_file=args.out, conll_format=args.conll, coordinating_conjunctions=args.coordinating_conjunctions, subordinating_conjunctions=args.subordinating_conjunctions, appositive=args.appositive, transitive=args.transitive, debug=args.debug)