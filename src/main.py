import logging

logging.basicConfig(level=logging.WARNING)

import os
import json
import spacy
import stanza
import argparse
import spacy_stanza

from tqdm import tqdm
from spacy.tokens import Doc
from spacy import Language
from typing import Any, Generator
from spacy_conll.parser import ConllParser
from src.extraction import Extractor, ExtractorConfig, Extraction


def generate_conll_file_from_sentences_file(input_file: str) -> str:
    tokenizer = stanza.Pipeline(lang='pt', processors='tokenize, mwt', use_gpu=False)
    nlp = spacy_stanza.load_pipeline("pt", tokenize_pretokenized=True, use_gpu=False)
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
            # adiciona identação de 2 espaços
            json_string = '  ' + json_string.replace('\n', '\n  ')
            f.write(json_string)

            # Atualiza a flag após o primeiro item ser escrito
            is_first_item = False

        # Fecha o array JSON
        f.write('\n]\n')

    print("Processo concluído com sucesso!")

def extract_to_csv(nlp: Language, input_file: str, output_file: str):
    import csv

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'sentence', 'arg1', 'rel', 'arg2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        sentence_iterator = read_conll_sentences(input_file)

        print(f"Processando sentenças de '{input_file}' e salvando em '{output_file}'...")

        for conll_sentence_block in tqdm(sentence_iterator, desc="Extraindo informações"):
            conll_parser = ConllParser(nlp)
            doc = conll_parser.parse_conll_text_as_spacy(conll_sentence_block)
            extractions: list[Extraction] = doc._.extractions

            for indice, extraction in enumerate(extractions):
                indice_output = indice + 1
                extraction_dict = dict(extraction)
                row = {
                    'id': str(indice_output) + '.0',
                    'sentence': doc.text.strip(),
                    'arg1': extraction_dict['arg1'],
                    'rel': extraction_dict['rel'],
                    'arg2': extraction_dict['arg2'],
                }
                writer.writerow(row)

                for indice_sub, sub_extraction in enumerate(extraction.sub_extractions):
                    sub_extraction_dict = dict(sub_extraction)
                    sub_row = {
                        'id': str(indice_output) + '.' + str(indice_sub + 1),
                        'sentence': doc.text.strip(),
                        'arg1': sub_extraction_dict['arg1'],
                        'rel': sub_extraction_dict['rel'],
                        'arg2': sub_extraction_dict['arg2'],
                    }
                    writer.writerow(sub_row)

    print("Processo concluído com sucesso!")

def extract_to_txt(nlp: Language, input_file: str, output_file: str):
    """
    Extrai informações de um arquivo CONLL e salva em um arquivo de texto.
    A saída será no formato:
    sentence
        id | arg1 | rel | arg2 (extraction)
            id | arg1 | rel | arg2 (sub_extraction)
    """
    sentence_iterator = read_conll_sentences(input_file)
    print(f"Processando sentenças de '{input_file}' e salvando em '{output_file}'...")

    with open(output_file, 'w', encoding='utf-8') as f:

        indice_output = 0
        for conll_sentence_block in tqdm(sentence_iterator, desc="Extraindo informações"):
            conll_parser = ConllParser(nlp)
            doc = conll_parser.parse_conll_text_as_spacy(conll_sentence_block)

            f.write(f"{doc.text.strip()}\n")
            extractions = doc._.extractions
            for index, extraction in enumerate(extractions):
                extraction_dict = dict(extraction)
                indice_output += 1
                f.write(f"    {extraction_dict['arg1']} | {extraction_dict['rel']} | {extraction_dict['arg2']}\n")
                for sub_index, sub_extraction in enumerate(extraction.sub_extractions):
                    sub_extraction_dict = dict(sub_extraction)
                    f.write(f"      {sub_extraction_dict['arg1']} | {sub_extraction_dict['rel']} | {sub_extraction_dict['arg2']}\n")

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

def main(
    input_file: str,
    input_type: str,
    output_file: str,
    output_type: str,
    coordinating_conjunctions: bool = True,
    subordinating_conjunctions: bool = True,
    hidden_subjects: bool = True,
    appositive: bool = True,
    transitive: bool = True,
    debug: bool = False):
    extractor = Extractor(ExtractorConfig(
        coordinating_conjunctions=coordinating_conjunctions,
        subordinating_conjunctions=subordinating_conjunctions,
        hidden_subjects=hidden_subjects,
        appositive=appositive,
        appositive_transitivity=transitive,
        debug=debug,
    ))

    Doc.set_extension("extractions", getter=extractor.get_extractions_from_doc)

    if input_type == 'txt':
        conll_file = generate_conll_file_from_sentences_file(input_file=input_file)
    else:
        conll_file = input_file

    nlp = spacy.blank("pt")
    nlp.add_pipe("conll_formatter", last=True)

    if output_type == 'csv':
        extract_to_csv(nlp=nlp, input_file=conll_file, output_file=output_file)
    elif output_type == 'json':
        extract_to_json(nlp=nlp, input_file=conll_file, output_file=output_file)
    elif output_type == 'txt':
        extract_to_txt(nlp=nlp, input_file=conll_file, output_file=output_file)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract triples from sentences in a file using SpaCy and Stanza.')

    parser.add_argument('-i', '--input', metavar='input', type=str, help='path to the input file', default='./inputs/teste.txt')
    parser.add_argument('-it', '--input-type', metavar='input_type', type=str, choices=['txt', 'conll'], help='input file type', default='txt')
    parser.add_argument('-o', '--output', metavar='output', type=str, help='path to the output file', default='./outputs/output.json')
    parser.add_argument('-ot', '--output-type', metavar='output_type', type=str, choices=['json', 'csv', 'txt'], help='output file type', default='json')
    parser.add_argument('-cc', '--coordinating_conjunctions', dest='coordinating_conjunctions', action='store_true', help='enable coordinating conjunctions extraction')
    parser.add_argument('-sc', '--subordinating_conjunctions', dest='subordinating_conjunctions', action='store_true', help='enable subordinating conjunctions extraction')
    parser.add_argument('-hs', '--hidden_subjects', dest='hidden_subjects', action='store_true', help='enable hidden subjects extraction')
    parser.add_argument('-a', '--appositive', dest='appositive', action='store_true', help='enable appositive extraction')
    parser.add_argument('-t', '--transitive', dest='transitive', action='store_true', help='enable transitive extraction(only for appositive)')
    parser.add_argument('-debug', action='store_true', help='enable debug mode')

    args = parser.parse_args()

    main(
        input_file=args.input,
        output_file=args.output,
        input_type=args.input_type,
        output_type=args.output_type,
        coordinating_conjunctions=args.coordinating_conjunctions,
        subordinating_conjunctions=args.subordinating_conjunctions,
        hidden_subjects=args.hidden_subjects,
        appositive=args.appositive,
        transitive=args.transitive,
        debug=args.debug
    )