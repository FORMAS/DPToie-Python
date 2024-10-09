import argparse
import json
import logging
import spacy_stanza
import stanza

from spacy.tokens import Doc

from src.extraction.extraction import Extraction

logging.basicConfig(level=logging.INFO)

Doc.set_extension("extractions", getter=Extraction.get_extractions_from_doc)

def main(input_path: str, output_file: str):
    with open(input_path, 'r') as f:
        sentences = f.readlines()

    tokenizer = stanza.Pipeline(lang='pt', processors='tokenize, mwt')
    tokenized_docs = tokenizer.bulk_process(sentences)

    sentences = []
    for tokenized_doc in tokenized_docs:
        for sentence in tokenized_doc.sentences:
            sentences.append(' '.join([token.text for token in sentence.tokens]))

    docs = list(nlp.pipe(sentences))

    output = {
        'facts': []
    }
    for doc in docs:
        sentence = {
            'text': doc.text,
            'facts': []
        }
        for extraction in doc._.extractions:
            sentence['facts'].append({
                'subject': ' '.join([token.text for token in extraction.subject]),
            })
        output['facts'].append(sentence)
    with open(output_file, 'w') as f:
        f.write(json.dumps(output, indent=4, ensure_ascii=False))


if __name__ == "__main__":

    nlp = spacy_stanza.load_pipeline("pt", tokenize_pretokenized=True)

    parser = argparse.ArgumentParser(description='Extract clauses from a text file.')

    parser.add_argument('-path', metavar='path', type=str, help='path to the text file', default='../inputs/teste.txt')
    parser.add_argument('-out', metavar='out', type=str, help='path to the output file', default='../out.txt')

    args = parser.parse_args()

    main(input_path=args.path, output_file=args.out)