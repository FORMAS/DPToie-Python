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

    def __iter__(self):
        yield 'arg1', str(self.subject) if self.subject else None
        yield 'rel', str(self.relation) if self.relation else None
        yield 'arg2', str(self.complement) if self.complement and self.complement.core else None

class ExtractorConfig:
    def __init__(self, coordinating_conjunctions: bool = True, subordinating_conjunctions: bool = True, appositive: bool = True, transitive: bool = True, debug: bool = False):
        self.coordinating_conjunctions = coordinating_conjunctions
        self.subordinating_conjunctions = subordinating_conjunctions
        self.appositive = appositive
        self.transitive = transitive
        self.debug = debug

    def __str__(self):
        return f"ExtractorConfig(coordinating_conjunctions={self.coordinating_conjunctions}, " \
               f"subordinating_conjunctions={self.subordinating_conjunctions}, " \
               f"appositive={self.appositive}, transitive={self.transitive}, debug={self.debug})"

    def __iter__(self):
        yield 'coordinating_conjunctions', self.coordinating_conjunctions
        yield 'subordinating_conjunctions', self.subordinating_conjunctions
        yield 'appositive', self.appositive
        yield 'transitive', self.transitive

class Extractor:
    def __init__(self, config: ExtractorConfig = None):
        self.config = config if config else ExtractorConfig()

    @staticmethod
    def get_extractions_from_doc(doc: Doc) -> List['Extraction']:
        extractions = []
        for sentence in doc.sents:
            extractions.extend(Extractor.get_extractions_from_sentence(sentence))
        return extractions

    @staticmethod
    def get_extractions_from_sentence(sentence: Span) -> list['Extraction']:
        final_extractions = []

        initial_extractions = Extractor.__extract_subject_from_sentence(sentence)

        for e1 in initial_extractions:
            for e2 in Extractor.extract_relation(e1):
                final_extractions.extend(Extractor.extract_complements(e2))

        return final_extractions

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
                                if child.dep_ == "punct" and not Extractor.__valid_punct(child):
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
                    is_aclpart_valid = child.dep_ == "acl:part" and Extractor.__acl_part_first_child(child)

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

    @staticmethod
    def extract_complements(extraction: 'Extraction') -> List['Extraction']:
        """
        extrai complementos individuais e combinados.
        """
        if extraction.relation is None or extraction.relation.core is None:
            return [extraction]

        base_visited_indices = {extraction.subject.core.i, extraction.relation.core.i}
        base_visited_indices.update(p.i for p in extraction.subject.pieces)
        base_visited_indices.update(r.i for r in extraction.relation.pieces)

        potential_starts = sorted(
            [child for child in extraction.relation.core.children if
             child.i not in base_visited_indices and Extractor.__is_complement_part(child)],
            key=lambda t: t.i
        )

        if not potential_starts:
            return [extraction]

        final_extractions = []
        all_visited_indices = base_visited_indices.copy()

        # --- Lógica de Acumulação e Clonagem ---

        # 1. Primeiro, encontre todos os complementos independentes (cláusulas quebradas)
        independent_complements = []
        for start_token in potential_starts:
            if start_token.i in all_visited_indices:
                continue

            # Realiza a DFS para cada cláusula independente
            clause = Extractor.__dfs_for_complement(start_token, all_visited_indices, extraction)
            independent_complements.append(clause)

            # Marca os tokens desta cláusula como visitados para não pegá-los novamente
            all_visited_indices.add(clause.core.i)
            all_visited_indices.update(p.i for p in clause.pieces)

        if not independent_complements:
            return [extraction]

        # 2. Gere as extrações baseadas na lógica de acumulação

        # O acumulador que irá crescer a cada passo
        accumulator_complement = TripleElement()

        for clause in independent_complements:
            # Adiciona a cláusula atual ao acumulador
            if accumulator_complement.core is None:
                accumulator_complement.core = clause.core
            else:
                accumulator_complement.add_piece(clause.core)

            for piece in clause.pieces:
                accumulator_complement.add_piece(piece)

            # Tira um "snapshot" (clone) do estado atual e cria uma extração
            snapshot_extraction = Extraction()
            snapshot_extraction.subject = extraction.subject
            snapshot_extraction.relation = extraction.relation

            # Clona o acumulador para o snapshot
            snapshot_complement = TripleElement(accumulator_complement.core)
            snapshot_complement.pieces = accumulator_complement.pieces[:]  # Cópia da lista

            snapshot_extraction.complement = snapshot_complement
            final_extractions.append(snapshot_extraction)

        # 3. Adicione também as extrações individuais.
        # O resultado é uma combinação de extrações individuais e acumuladas.
        # Para garantir que tenhamos as extrações separadas, podemos adicioná-las se houver mais de uma.
        if len(independent_complements) > 1:
            for clause in independent_complements:
                individual_extraction = Extraction()
                individual_extraction.subject = extraction.subject
                individual_extraction.relation = extraction.relation
                individual_extraction.complement = clause

                # Evita adicionar duplicatas exatas das acumuladas
                if not any(str(ex.complement) == str(individual_extraction.complement) for ex in final_extractions):
                    final_extractions.append(individual_extraction)

        # Garante que a extração original sem complemento não seja retornada se encontrarmos algum
        if final_extractions:
            return final_extractions
        else:
            return [extraction]

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
                if child.i not in local_visited and Extractor.__is_complement_part(child):
                    complement.add_piece(child)
                    local_visited.add(child.i)
                    stack.append(child)

        return complement

    @staticmethod
    def __is_complement_part(token: Token) -> bool:
        """
        Verifica se um token pode ser parte de um complemento.
        """

        # Exclui pronomes relativos como 'que' de serem parte do complemento.
        if token.pos_ == "PRON" and "Rel" in token.morph.get("PronType", []):
            return False

        # verificar se o token está na lista de dependências válidas
        if token.dep_ in [
            "nmod", "xcomp", "dobj", "obj", "acl:relcl", "iobj", "acl:part",
            "nummod", "advmod", "appos", "amod", "dep", "case", "mark", "det", "flat", "fixed", "obl", "cop", "aux"
        ]:
            return True

        # 'conj' é válido se não for um verbo
        if token.dep_ == "conj" and token.pos_ != 'VERB':
            return True

        # 'ccomp' e 'advcl' são válidos se não tiverem seu próprio sujeito
        if token.dep_ in ["ccomp", "advcl"] and not any(c.dep_.startswith("nsubj") for c in token.children):
            return True

        # 'punct' é válida sob condições específicas (pontuação permitida e posição)
        if token.dep_ == "punct" and Extractor.__valid_punct(token) and token.i > token.head.i:
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