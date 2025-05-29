from collections import deque
from typing import List

from spacy.tokens import Span, Doc, Token

from src.extraction.subject import Subject

class Extraction:

    def __init__(self):

        self.subject: List[Subject] = []

    def add_subject(self, subject: Subject):
        self.subject.append(subject)

    def extract_subject_from_sentence(self, sentence: Span):
        visited_tokens = [False] * len(sentence)
        stack = deque()

        for token in sentence:
            if token.dep_ in ["nsubj", "nsubj:pass"] and token.text not in ["que", "a", "o"]:
                sbj = Subject(token)
                stack.append(token)
                visited_tokens[token.i] = True

                while stack:
                    current_token = stack.pop()
                    for child in current_token.children:
                        if not visited_tokens[child.i]:
                            if child.dep_ in ["nummod", "advmod", "appos", "nmod", "amod", "dep", "obj", "det", "case", "punct", "conj"] and (child.dep_ != "conj" or child.pos_ != "VERB"):
                                if child.dep_ == "punct" and not self.pontuacao_valida_sujeito(child):
                                    continue
                                sbj.add_piece(child)
                                stack.append(child)
                                visited_tokens[child.i] = True

                self.add_subject(sbj)

    def pontuacao_valida_sujeito(self, token: Token) -> bool:
        valid_punctuation = {"(", ")", "{", "}", "\"", "'", "[", "]", ","}
        return token.text in valid_punctuation

    @staticmethod
    def get_extractions_from_doc(doc: Doc) -> List['Extraction']:
        extractions = []

        for sentence in doc.sents:
            extractions.append(Extraction.from_sentence(sentence))

        return extractions

    @staticmethod
    def from_sentence(sentence: Span):
        extraction = Extraction()

        extraction.extract_subject_from_sentence(sentence)

        return extraction