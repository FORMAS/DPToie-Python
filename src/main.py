import argparse
import spacy_stanza

from noie import do_extract_clauses

if __name__ == '__main__':
    nlp = spacy_stanza.load_pipeline("pt")
    nlp.add_pipe("openie")

    parser = argparse.ArgumentParser(description='Extract clauses from a text file.')
    parser.add_argument('-path', metavar='path', type=str, help='path to the text file')
    args = parser.parse_args()

    path = args.path

    with open(path, 'r') as f:
        sentences = f.read()

    with open('out.txt', 'w') as output:
        for line in sentences.split('\n'):
            doc = nlp(line)
            output.write(line + '\n')
            for prop in doc._.clauses:
                output.write('\t' + str(prop.to_propositions(inflect=None)) + '\n')
