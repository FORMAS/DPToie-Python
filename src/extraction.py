from collections import deque
from typing import List, Optional
from spacy.tokens import Span, Doc, Token

class TripleElement:
    def __init__(self, token: Token = None):
        self.core: Optional[Token] = token
        self.pieces: List[Token] = []

    def __str__(self):
        # Imprime o elemento como uma string ordenada pelo índice do token
        tokens = [t for t in [self.core] + self.pieces if t is not None]
        return ' '.join([token.text for token in sorted(tokens, key=lambda x: x.i)])

    def add_piece(self, piece: Token):
        if piece not in self.pieces:
            self.pieces.append(piece)


class Extraction:
    def __init__(self):
        self.subject: Optional[TripleElement] = None
        self.relation: Optional[TripleElement] = None
        self.complement: Optional[TripleElement] = None

    @staticmethod
    def __extract_subject_from_sentence(sentence: Span) -> List[TripleElement]:
        visited_tokens = set()
        subjects = []

        for token in sentence:
            if token.dep_ in ["nsubj", "nsubj:pass"] and token.text.lower() not in ["que", "a", "o"]:
                if token.i in visited_tokens:
                    continue

                sbj = TripleElement(token)
                stack = deque([token])
                visited_tokens.add(token.i)

                current_subject_tokens = {token}

                while stack:
                    current_token = stack.pop()
                    for child in current_token.children:
                        if child.i not in visited_tokens:
                            # Lógica para expansão do sujeito
                            if child.dep_ in ["nummod", "advmod", "appos", "nmod", "amod", "dep", "det", "case",
                                              "punct", "conj"] and (child.dep_ != "conj" or child.pos_ != "VERB"):
                                if child.dep_ == "punct" and not Extraction.__valid_punct(child):
                                    continue
                                sbj.add_piece(child)
                                stack.append(child)
                                visited_tokens.add(child.i)
                                current_subject_tokens.add(child)

                subjects.append(sbj)

        return subjects

    def extract_relation_from_sentence(self):
        if not self.subject or not self.subject.core:
            return

        stack = deque()
        deprel_valid = ["aux:pass", "obj", "iobj", "advmod", "cop", "aux", "expl:pv", "mark"]
        deprel_valid_for_after_subject = ["flat", "expl:pv"]
        punct_invalid = [",", "--"]

        visited_tokens = {p.i for p in self.subject.pieces}
        visited_tokens.add(self.subject.core.i)

        head_subject = self.subject.core.head
        if head_subject is None:
            return

        self.relation = TripleElement(head_subject)
        visited_tokens.add(head_subject.i)

        stack.append(head_subject)
        while stack:
            current_token = stack.pop()
            for child in current_token.children:
                if child.i not in visited_tokens:
                    is_between = (min(self.subject.core.i, self.relation.core.i) < child.i < max(self.subject.core.i,
                                                                                                 self.relation.core.i))

                    is_deprel_valid = child.dep_ in deprel_valid
                    is_punct_valid = child.dep_ == "punct" and child.text not in punct_invalid
                    is_deprel_valid_for_after_subject = child.dep_ in deprel_valid_for_after_subject
                    is_punct_hyphen = child.dep_ == "punct" and child.text == "-"
                    is_aclpart_valid = child.dep_ == "acl:part" and self.__acl_part_first_child(child)

                    if (is_between and (is_deprel_valid or is_punct_valid)) or \
                        (child.i > head_subject.i and (
                            is_deprel_valid_for_after_subject or is_punct_hyphen or is_aclpart_valid)):

                        self.relation.add_piece(child)
                        stack.append(child)
                        visited_tokens.add(child.i)

                        if is_aclpart_valid:
                            self.relation.core = child

    def extract_complement_from_sentence(self):
        if self.relation is None or self.relation.core is None:
            return

        # Consolida os índices de tokens já visitados do sujeito e da relação
        visited_indices = {self.subject.core.i, self.relation.core.i}
        visited_indices.update(p.i for p in self.subject.pieces)
        visited_indices.update(r.i for r in self.relation.pieces)

        # Pilha para a busca em profundidade (DFS)
        stack = deque()
        self.complement = TripleElement()

        # Identifica todos os filhos diretos da relação que são partes válidas do complemento
        initial_complement_parts = sorted(
            [child for child in self.relation.core.children if
             child.i not in visited_indices and self.__is_complement_part(child)],
            key=lambda t: t.i
        )

        if not initial_complement_parts:
            return

        # O primeiro token válido é definido como o núcleo do complemento
        self.complement.core = initial_complement_parts[0]
        visited_indices.add(self.complement.core.i)
        stack.append(self.complement.core)

        # Adiciona os outros tokens iniciais às peças e à pilha para a travessia
        for token in initial_complement_parts[1:]:
            self.complement.add_piece(token)
            visited_indices.add(token.i)
            stack.append(token)

        # Realiza a busca em profundidade (DFS) a partir dos tokens iniciais
        while stack:
            current_token = stack.pop()
            # Explora os filhos do token atual em ordem
            for child in sorted(current_token.children, key=lambda t: t.i):
                if child.i not in visited_indices and self.__is_complement_part(child):
                    self.complement.add_piece(child)
                    visited_indices.add(child.i)
                    stack.append(child)

    @staticmethod
    def __is_complement_part(token: Token) -> bool:
        """
        Verifica se um token pode ser parte de um complemento, com base nas regras
        do código Java e ajustes para capturar construções completas.
        """
        # Exclui conjunções subordinativas como 'que'.
        if token.pos_ == "SCONJ":
            return False

        # Exclui pronomes relativos como 'que' de serem parte do complemento.
        if token.pos_ == "PRON" and "Rel" in token.morph.get("PronType", []):
            return False

        # verificar se o token está na lista de dependências válidas
        if token.dep_ in [
            "nmod", "xcomp", "dobj", "obj", "acl:relcl", "iobj", "acl:part",
            "nummod", "advmod", "appos", "amod", "dep", "case", "mark", "det", "flat", "fixed"
        ]:
            return True

        # 'conj' é válido se não for um verbo
        if token.dep_ == "conj" and token.pos_ != 'VERB':
            return True

        # 'ccomp' e 'advcl' são válidos se não tiverem seu próprio sujeito
        if token.dep_ in ["ccomp", "advcl"] and not any(c.dep_.startswith("nsubj") for c in token.children):
            return True

        # 'punct' é válida sob condições específicas (pontuação permitida e posição)
        if token.dep_ == "punct" and Extraction.__valid_punct(token) and token.i > token.head.i:
            return True

        return False

    @staticmethod
    def __acl_part_first_child(token: Token) -> bool:
        token_head = token.head
        for token_child in token_head.children:
            if token_child.i > token_head.i:
                if token_child.dep_ in ["nmod", "xcomp", "dobj", "obj", "iobj", "nummod", "advmod", "appos", "conj",
                                        "amod", "dep"]:
                    return False
                elif token_child.dep_ == "acl:part":
                    return True
        return False

    @staticmethod
    def __valid_punct(token: Token) -> bool:
        """Verifica se a pontuação é válida para compor um elemento."""
        valid_punctuation = {"(", ")", "{", "}", "\"", "'", "[", "]", ","}
        return token.text in valid_punctuation

    @staticmethod
    def get_extractions_from_doc(doc: Doc) -> List['Extraction']:
        extractions = []
        for sentence in doc.sents:
            extractions.extend(Extraction.get_extractions_from_sentence(sentence))
        return extractions

    @staticmethod
    def get_extractions_from_sentence(sentence: Span) -> list['Extraction']:
        subject_list = Extraction.__extract_subject_from_sentence(sentence)
        extractions = []
        for subject in subject_list:
            extraction = Extraction()
            extraction.subject = subject

            extraction.extract_relation_from_sentence()
            extraction.extract_complement_from_sentence()

            extractions.append(extraction)

        return extractions

    def __iter__(self):
        yield 'arg1', str(self.subject) if self.subject else None
        yield 'rel', str(self.relation) if self.relation else None
        yield 'arg2', str(self.complement) if self.complement and self.complement.core else None