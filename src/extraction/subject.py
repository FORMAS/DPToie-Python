from collections import deque
from typing import List, Optional

from spacy.tokens import Token, Doc, Token

class Subject:

    def __init__(self, token: Token, pieces: List[Token] = None):
        self.token: Token = token
        self.pieces: List[Token] = pieces if pieces is not None else []
        self.relations: List[Token] = []
        self.complements: List[Token] = []
        self.relation_nucleus: Optional[Token] = None

    def add_piece(self, piece: Token):
        self.pieces.append(piece)

    def add_relation(self, relation: Token):
        self.relations.append(relation)

    def add_complement(self, complement: Token):
        self.complements.append(complement)

    def __str__(self):
        # print the subject as a string ordered by the token index
        return ' '.join([token.text for token in sorted([self.token] + self.pieces, key=lambda x: x.i)])
    def get_relation_text(self):
        return ' '.join([relation.text for relation in sorted(self.relations, key=lambda x: x.i)])

    def get_complement_text(self):
        return ' '.join([complement.text for complement in sorted(self.complements, key=lambda x: x.i)])