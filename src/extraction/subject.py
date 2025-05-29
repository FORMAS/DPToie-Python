from collections import deque
from typing import List

from spacy.tokens import Token, Doc, Token

class Subject:

    def __init__(self, token: Token, pieces: List[Token] = None):
        self.token: Token = token
        self.pieces: List[Token] = pieces if pieces is not None else []

    def add_piece(self, piece: Token):
        self.pieces.append(piece)

    def __str__(self):
        # print the subject as a string ordered by the token index
        return ' '.join([token.text for token in sorted([self.token] + self.pieces, key=lambda x: x.i)])