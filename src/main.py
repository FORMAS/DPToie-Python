import logging

import spacy
from spacy import Language
from stanza import DownloadMethod

logging.basicConfig(level=logging.WARNING)

import os
import json
import stanza
import argparse
import spacy_stanza

from tqdm import tqdm
from spacy.tokens import Doc
from typing import Any, Generator
from spacy_conll.parser import ConllParser
from src.extraction import Extractor, ExtractorConfig

def generate_conll_file_from_sentences_file(input_file: str) -> str:
    tokenizer = stanza.Pipeline(lang='pt', processors='tokenize, mwt', use_gpu=False,
                                download_method=DownloadMethod.REUSE_RESOURCES)
    nlp = spacy_stanza.load_pipeline("pt", tokenize_pretokenized=True, use_gpu=False, download_method=None)
    nlp.add_pipe("conll_formatter", last=True)
    connl_file = './outputs/input.conll'

    with open(connl_file, 'w') as f:
        f.write('')

    # 2. Pega o tamanho total do arquivo de entrada em bytes
    file_size = os.path.getsize(input_file)

    with open(input_file, 'r', encoding='utf-8') as f:
        with tqdm(total=file_size,
                  desc="Gerando árvores de dependência",
                  unit='B',  # Define a unidade como Bytes
                  unit_scale=True,  # Mostra KB, MB, GB automaticamente
                  unit_divisor=1024) as pbar:
            for line in f:
                if line.strip():
                    sentence = line.strip()
                    # Process the sentence with Stanza tokenizer
                    doc = tokenizer(sentence)
                    # Convert Stanza Doc to SpaCy Doc
                    spacy_doc = nlp(' '.join([word.text for sent in doc.sentences for word in sent.words]))
                    with open(connl_file, 'a', encoding='utf-8') as fout:
                        fout.write(spacy_doc._.conll_str)
                        fout.write('\n')

                # Atualiza a barra com o número de bytes da linha lida
                pbar.update(len(line.encode('utf-8')))

    return connl_file

def extract_to_json(nlp: Language, input_file: str, output_file: str):
    sentence_iterator = read_conll_sentences(input_file)
    print(f"Processando sentenças de '{input_file}' e salvando em '{output_file}'...")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('[\n')

        is_first_item = True
        for conll_sentence_block in tqdm(sentence_iterator, desc="Extraindo informações"):

            conll_parser = ConllParser(nlp)
            doc = conll_parser.parse_conll_text_as_spacy(conll_sentence_block)

            extractions = doc._.extractions

            # Apenas processa e escreve a sentença se ela tiver extrações
            if extractions:
                sentence_data = {
                    'sentence': doc.text.strip(),
                    'extractions': []
                }

                for extraction in extractions:
                    # Converte cada objeto de extração para um dicionário
                    extraction_dict = dict(extraction)
                    sentence_data['extractions'].append(extraction_dict)

                # Adiciona uma vírgula antes de cada item, exceto o primeiro
                if not is_first_item:
                    f.write(',\n')

                # Converte o dicionário para uma string JSON e escreve no ficheiro
                # indent=2 para manter a formatação legível
                json_string = json.dumps(sentence_data, ensure_ascii=False, indent=2)
                f.write(json_string)

                # Atualiza a flag após o primeiro item ser escrito
                is_first_item = False

        # Fecha o array JSON
        f.write('\n]\n')

    print("Processo concluído com sucesso!")

def extract_to_csv(nlp: Language, input_file: str, output_file: str):
    import csv

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['sentence', 'arg1', 'rel', 'arg2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        sentence_iterator = read_conll_sentences(input_file)

        print(f"Processando sentenças de '{input_file}' e salvando em '{output_file}'...")

        for conll_sentence_block in tqdm(sentence_iterator, desc="Extraindo informações"):
            conll_parser = ConllParser(nlp)
            doc = conll_parser.parse_conll_text_as_spacy(conll_sentence_block)

            for extraction in doc._.extractions:
                row = {
                    'sentence': doc.text.strip(),
                }
                row.update(dict(extraction))
                writer.writerow(row)

    print("Processo concluído com sucesso!")

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

def main(input_file: str, output_type: str, conll_format: bool = False, coordinating_conjunctions: bool = True, subordinating_conjunctions: bool = True, hidden_subjects: bool = True, appositive: bool = True, transitive: bool = True, debug: bool = False):
    extractor = Extractor(ExtractorConfig(
        coordinating_conjunctions=coordinating_conjunctions,
        subordinating_conjunctions=subordinating_conjunctions,
        hidden_subjects=hidden_subjects,
        appositive=appositive,
        transitive=transitive,
        debug=debug,
    ))

    Doc.set_extension("extractions", getter=extractor.get_extractions_from_doc)

    if not conll_format:
        conll_file = generate_conll_file_from_sentences_file(input_file=input_file)
    else:
        conll_file = input_file

    output_file = f'./outputs/output.{output_type}'

    nlp = spacy.blank("pt")
    nlp.add_pipe("conll_formatter", last=True)

    if output_type == 'csv':
        extract_to_csv(nlp=nlp, input_file=conll_file, output_file=output_file)
    elif output_type == 'json':
        extract_to_json(nlp=nlp, input_file=conll_file, output_file=output_file)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract clauses from a text file.')

    parser.add_argument('-input', metavar='input', type=str, help='path to the input file', default='./inputs/teste.txt')
    parser.add_argument('-output-type', metavar='output_type', type=str, choices=['json', 'csv'], help='output file type', default='json')
    parser.add_argument('-conll', action='store_true', help='input file is in CONLL format')
    parser.add_argument('-cc', '--coordinating_conjunctions', dest='coordinating_conjunctions', action='store_true', help='enable coordinating conjunctions extraction')
    parser.add_argument('-sc', '--subordinating_conjunctions', dest='subordinating_conjunctions', action='store_true', help='enable subordinating conjunctions extraction')
    parser.add_argument('-hs', '--hidden_subjects', dest='hidden_subjects', action='store_true', help='enable hidden subjects extraction')
    parser.add_argument('-a', '--appositive', dest='appositive', action='store_true', help='enable appositive extraction')
    parser.add_argument('-t', '--transitive', dest='transitive', action='store_true', help='enable transitive extraction(only for appositive)')
    parser.add_argument('-debug', action='store_true', help='enable debug mode')

    args = parser.parse_args()

    main(
        input_file=args.input,
        output_type=args.output_type,
        conll_format=args.conll,
        coordinating_conjunctions=args.coordinating_conjunctions,
        subordinating_conjunctions=args.subordinating_conjunctions,
        hidden_subjects=args.hidden_subjects,
        appositive=args.appositive,
        transitive=args.transitive,
        debug=args.debug
    )