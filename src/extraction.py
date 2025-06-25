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
    def __extract_subject_from_sentence(sentence: Span) -> List['Extraction']:
        visited_tokens = set()
        extractions = []

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
                extraction = Extraction()
                extraction.subject = sbj
                extractions.append(extraction)

        return extractions

    @staticmethod
    def extract_relation(extraction: 'Extraction') -> List['Extraction']:

        extractions: list['Extraction'] = []

        if not extraction.subject or not extraction.subject.core:
            return [extraction]

        stack = deque()
        deprel_valid = ["aux:pass", "obj", "iobj", "advmod", "cop", "aux", "expl:pv", "mark"]
        deprel_valid_for_after_subject = ["flat", "expl:pv"]
        punct_invalid = [",", "--"]

        visited_tokens = {p.i for p in extraction.subject.pieces}
        visited_tokens.add(extraction.subject.core.i)

        head_subject = extraction.subject.core.head
        if head_subject is None:
            return [extraction]

        extraction.relation = TripleElement(head_subject)
        visited_tokens.add(head_subject.i)

        stack.append(head_subject)
        while stack:
            current_token = stack.pop()
            for child in current_token.children:
                if child.i not in visited_tokens:
                    is_between = (min(extraction.subject.core.i, extraction.relation.core.i) < child.i < max(extraction.subject.core.i,
                                                                                                 extraction.relation.core.i))

                    is_deprel_valid = child.dep_ in deprel_valid
                    is_punct_valid = child.dep_ == "punct" and child.text not in punct_invalid
                    is_deprel_valid_for_after_subject = child.dep_ in deprel_valid_for_after_subject
                    is_punct_hyphen = child.dep_ == "punct" and child.text == "-"
                    is_aclpart_valid = child.dep_ == "acl:part" and extraction.__acl_part_first_child(child)

                    if (is_between and (is_deprel_valid or is_punct_valid)) or \
                        (child.i > head_subject.i and (
                            is_deprel_valid_for_after_subject or is_punct_hyphen or is_aclpart_valid)):

                        extraction.relation.add_piece(child)
                        stack.append(child)
                        visited_tokens.add(child.i)

                        if is_aclpart_valid:
                            extraction.relation.core = child

        extractions.append(extraction)
        return extractions

    # Substitua os métodos extract_complement e extract_broken_clauses por este único método.
    # A função auxiliar __dfs_for_complement continua sendo necessária.

    @staticmethod
    def extract_complements(extraction: 'Extraction') -> List['Extraction']:
        if extraction.relation is None or extraction.relation.core is None:
            return [extraction]

        # Conjunto de todos os índices já usados no sujeito e na relação.
        base_visited_indices = {extraction.subject.core.i, extraction.relation.core.i}
        base_visited_indices.update(p.i for p in extraction.subject.pieces)
        base_visited_indices.update(r.i for r in extraction.relation.pieces)

        # Passo 1: Encontra todos os filhos do verbo que são candidatos a iniciar um complemento.
        potential_starts = sorted(
            [child for child in extraction.relation.core.children if
             child.i not in base_visited_indices and extraction.__is_complement_part(child)],
            key=lambda t: t.i
        )

        if not potential_starts:
            return [extraction]  # Retorna a extração original sem complemento.

        # Lista final de extrações que será retornada.
        final_extractions = []

        # Clona a extração original para não modificar a referência passada.
        main_extraction = Extraction()
        main_extraction.subject = extraction.subject
        main_extraction.relation = extraction.relation

        # --- Passo 2: Extrai o Complemento Principal ---
        # O primeiro candidato é o início do complemento principal.
        main_complement_start = potential_starts[0]

        # Realiza a DFS para construir o complemento principal completo.
        main_complement = Extraction.__dfs_for_complement(main_complement_start, base_visited_indices, extraction)
        main_extraction.complement = main_complement
        final_extractions.append(main_extraction)

        # Atualiza o conjunto de tokens visitados com os usados no complemento principal.
        all_visited_indices = base_visited_indices.copy()
        if main_complement.core:
            all_visited_indices.add(main_complement.core.i)
            all_visited_indices.update(p.i for p in main_complement.pieces)

        # --- Passo 3: Extrai as Cláusulas Quebradas (Restantes) ---
        # Itera sobre os mesmos candidatos novamente.
        for start_token in potential_starts:
            # Se o candidato já foi incluído no complemento principal (ou outro), pula.
            if start_token.i in all_visited_indices:
                continue

            # Se não foi usado, é o início de uma cláusula quebrada.
            broken_clause = Extraction.__dfs_for_complement(start_token, all_visited_indices, extraction)

            if broken_clause.core:
                # Cria uma NOVA extração para a cláusula quebrada.
                new_extraction = Extraction()
                new_extraction.subject = extraction.subject
                new_extraction.relation = extraction.relation
                new_extraction.complement = broken_clause
                final_extractions.append(new_extraction)

                # Marca os tokens desta cláusula como visitados para as próximas iterações.
                all_visited_indices.add(broken_clause.core.i)
                all_visited_indices.update(p.i for p in broken_clause.pieces)

        return final_extractions

    @staticmethod
    def __dfs_for_complement(start_token: Token, visited_indices: set, extraction: 'Extraction') -> TripleElement:
        """
        Realiza uma busca em profundidade (DFS) a partir de um token inicial
        para construir um elemento de tripla (sujeito, relação ou complemento).
        """
        complement = TripleElement(start_token)
        stack = deque([start_token])

        # Adiciona o token inicial ao conjunto de visitados para esta busca específica
        local_visited = visited_indices.copy()
        local_visited.add(start_token.i)

        while stack:
            current_token = stack.pop()
            for child in sorted(current_token.children, key=lambda t: t.i):
                if child.i not in local_visited and extraction.__is_complement_part(child):
                    complement.add_piece(child)
                    local_visited.add(child.i)
                    stack.append(child)

        return complement

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
        final_extractions = []

        initial_extractions = Extraction.__extract_subject_from_sentence(sentence)

        for e1 in initial_extractions:
            for e2 in Extraction.extract_relation(e1):
                final_extractions.extend(Extraction.extract_complements(e2))

        return final_extractions

    def __iter__(self):
        yield 'arg1', str(self.subject) if self.subject else None
        yield 'rel', str(self.relation) if self.relation else None
        yield 'arg2', str(self.complement) if self.complement and self.complement.core else None