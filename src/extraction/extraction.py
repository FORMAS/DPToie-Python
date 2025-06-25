from collections import deque
from typing import List, Optional
from spacy.tokens import Span, Doc, Token

class TripleElement:
    def __init__(self, token: Token = None):
        self.token: Token = token
        self.pieces: List[Token] = []

    def __str__(self):
        # print the subject as a string ordered by the token index
        return ' '.join([token.text for token in
                         sorted([t for t in [self.token] + self.pieces if t is not None], key=lambda x: x.i)])

    def add_piece(self, piece: Token):
        self.pieces.append(piece)



class Extraction:

    def __init__(self):
        self.subject: Optional[TripleElement] = None
        self.relation: Optional[TripleElement] = None
        self.complement: Optional[TripleElement] = None

    @staticmethod
    def __extract_subject_from_sentence(sentence: Span) -> List[TripleElement]:
        visited_tokens = [False] * len(sentence)
        stack = deque()

        subjects = []

        for token in sentence:
            if token.dep_ in ["nsubj", "nsubj:pass"] and token.text not in ["que", "a", "o"]:
                sbj = TripleElement(token)
                stack.append(token)
                visited_tokens[token.i] = True

                while stack:
                    current_token = stack.pop()
                    for child in current_token.children:
                        if not visited_tokens[child.i]:
                            if child.dep_ in ["nummod", "advmod", "appos", "nmod", "amod", "dep", "obj", "det", "case", "punct", "conj"] and (child.dep_ != "conj" or child.pos_ != "VERB"):
                                if child.dep_ == "punct" and not Extraction.__valid_punct_subject(child):
                                    continue
                                sbj.add_piece(child)
                                stack.append(child)
                                visited_tokens[child.i] = True

                subjects.append(sbj)

        return subjects

    def extract_relation_from_sentence(self, sentence: Span):
        stack = deque()

        deprel_valid = ["aux:pass", "obj", "iobj", "advmod", "cop", "aux", "expl:pv", "mark"]
        deprel_valid_for_after_subject = ["flat", "expl:pv"]
        punct_invalid = [",", "--"]

        visited_tokens = [False] * len(sentence)
        head_subject = self.subject.token.head

        self.relation.token = head_subject
        visited_tokens[head_subject.i] = True

        stack.append(head_subject)
        while stack:
            current_token = stack.pop()
            for child in current_token.children:
                if not visited_tokens[child.i]:
                    between_subject_and_relation = (child.i < self.relation.token.i
                                                    and (
                                                        (child.i > self.subject.token.i and self.subject.token.i < self.relation.token.i)
                                                        or (child.i < self.subject.token.i and self.subject.token.i > self.relation.token.i)))

                    is_deprel_valid = child.dep_ in deprel_valid
                    is_punct_valid = child.dep_ == "punct" and child.text not in punct_invalid
                    is_deprel_valid_for_after_subject = child.dep_ in deprel_valid_for_after_subject
                    is_punct_hyphen = child.dep_ == "punct" and child.text == "-"
                    is_aclpart_valid = child.dep_ == "acl:part" and self.__acl_part_first_child(child)

                    if (between_subject_and_relation and (is_deprel_valid or is_punct_valid)) or (child.i > head_subject.i and (is_deprel_valid_for_after_subject or is_punct_hyphen or is_aclpart_valid)):
                        self.relation.add_piece(child)
                        stack.append(child)
                        visited_tokens[child.i] = True
                        if is_aclpart_valid:
                            self.relation.token = child

    def extract_complement_from_sentence(self, sentence: Span):
        if self.relation.token is None:
            return

        visited_indices = set()
        visited_indices.add(self.subject.token.i)
        visited_indices.add(self.relation.token.i)

        for p in self.subject.pieces:
            visited_indices.add(p.i)
        for r in self.relation.pieces:
            visited_indices.add(r.i)

        stack = deque()

        for child in self.relation.token.children:
            if child.i in visited_indices:
                continue

            is_complement_start = False
            if child.dep_ in ["nmod", "xcomp", "dobj", "obj", "acl:relcl", "iobj", "nummod", "advmod", "appos", "amod", "dep", "acl:part"]:
                is_complement_start = True
            elif child.dep_ == "conj" and child.pos_ != 'VERB':
                is_complement_start = True
            elif child.dep_ in ["ccomp", "advcl"] and not any(c.dep_.startswith("nsubj") for c in child.children):
                is_complement_start = True
            elif child.dep_ == "punct" and self.__valid_punct_subject(child) and child.i > child.head.i:
                is_complement_start = True

            if is_complement_start:
                stack.append(child)
                visited_indices.add(child.i)
                self.complement.token = child

                while stack:
                    token = stack.pop()
                    for t_child in token.children:
                        if t_child.i not in visited_indices:
                            is_complement_part = False
                            if t_child.dep_ in ["nmod", "xcomp", "dobj", "obj", "acl:relcl", "iobj", "nummod", "advmod", "appos", "amod", "dep", "acl:part"]:
                                is_complement_part = True
                            elif t_child.dep_ == "conj" and t_child.pos_ != 'VERB':
                                is_complement_part = True
                            elif t_child.dep_ in ["ccomp", "advcl"] and not any(c.dep_.startswith("nsubj") for c in t_child.children):
                                is_complement_part = True
                            elif t_child.dep_ == "punct" and self.__valid_punct_subject(t_child) and t_child.i > t_child.head.i:
                                is_complement_part = True

                            if is_complement_part:
                                stack.append(t_child)
                                visited_indices.add(t_child.i)
                                self.complement.pieces.append(t_child)

    @staticmethod
    def __acl_part_first_child(token: Token) -> bool:
        token_head = token.head
        for token_child in token_head.children:
            if token_child.i > token_head.i:
                if token_child.dep_ in ["nmod", "xcomp", "dobj", "obj", "iobj", "nummod", "advmod", "appos", "conj", "amod", "dep"]:
                    return False
                elif token_child.dep_ == "acl:part":
                    return True
        return False

    @staticmethod
    def __valid_punct_subject(token: Token) -> bool:
        valid_punctuation = {"(", ")", "{", "}", "\"", "'", "[", "]", ","}
        return token.text in valid_punctuation

    @staticmethod
    def get_extractions_from_doc(doc: Doc) -> List['Extraction']:
        extractions = []

        for sentence in doc.sents:
            extractions.extend(Extraction.from_sentence(sentence))

        return extractions

    @staticmethod
    def from_sentence(sentence: Span) -> list['Extraction']:

        subject_list = Extraction.__extract_subject_from_sentence(sentence)

        extractions = []
        for subject in subject_list:
            extraction = Extraction()
            extraction.subject = subject
            extraction.relation = TripleElement()
            extraction.complement = TripleElement()

            extraction.extract_relation_from_sentence(sentence)
            extraction.extract_complement_from_sentence(sentence)


            extractions.append(extraction)

        return extractions

    def __iter__(self):
        yield 'arg1', str(self.subject)
        yield 'rel', str(self.relation)
        yield 'arg2', str(self.complement) if self.complement.token else None