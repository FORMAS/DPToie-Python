from collections import deque
from typing import List, Optional, Set
from spacy.tokens import Span, Doc, Token


class TripleElement:
    def __init__(self, token: Token = None):
        self.core: Optional[Token] = token
        self.pieces: List[Token] = []

    def __str__(self):
        return ' '.join([token.text for token in self.get_all_tokens()])

    def get_all_tokens(self) -> List[Token]:
        tokens = [t for t in [self.core] + self.pieces if t is not None]
        seen_ids = set()
        unique_tokens = []
        for token in sorted(tokens, key=lambda x: x.i):
            if token.i not in seen_ids:
                unique_tokens.append(token)
                seen_ids.add(token.i)
        return unique_tokens

    def add_piece(self, piece: Token):
        if piece and piece not in self.pieces:
            self.pieces.append(piece)

    def merge(self, other_element: 'TripleElement'):
        if other_element.core:
            self.add_piece(other_element.core)
        for piece in other_element.pieces:
            self.add_piece(piece)


class Extraction:
    def __init__(self):
        self.subject: Optional[TripleElement] = None
        self.relation: Optional[TripleElement] = None
        self.complement: Optional[TripleElement] = None

    def __iter__(self):
        yield 'arg1', str(self.subject) if self.subject else None
        yield 'rel', str(self.relation) if self.relation else None
        yield 'arg2', str(self.complement) if self.complement and (
            self.complement.core or self.complement.pieces) else None


class ExtractorConfig:
    def __init__(self, coordinating_conjunctions: bool = True, subordinating_conjunctions: bool = True,
                 appositive: bool = True, transitive: bool = True, debug: bool = False):
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

    def get_extractions_from_doc(self, doc: Doc) -> List['Extraction']:
        extractions = []
        for sentence in doc.sents:
            extractions.extend(self.get_extractions_from_sentence(sentence))
        return extractions

    def get_extractions_from_sentence(self, sentence: Span) -> list['Extraction']:
        final_extractions = []
        initial_extractions = self.__extract_subject_from_sentence(sentence)

        for e1 in initial_extractions:
            for e2 in self.extract_relation(e1):
                final_extractions.extend(self.extract_complements(e2))

        unique_extractions = []
        seen = set()
        for extr in final_extractions:
            representation = (str(extr.subject), str(extr.relation), str(extr.complement))
            if representation not in seen:
                seen.add(representation)
                unique_extractions.append(extr)

        return unique_extractions

    def extract_complements(self, extraction: 'Extraction') -> List['Extraction']:
        """
        Extrai complementos. Para conjunções, gera:
        1. Extrações "mínimas" para cada item, SEM conectores (vírgulas, 'e').
        2. Uma única extração "completa" com todos os itens E conectores.
        """
        if not extraction.relation or not extraction.relation.core:
            return [extraction] if not any(str(e.complement) for e in [extraction]) else []

        base_visited_indices = {tok.i for tok in extraction.subject.get_all_tokens()}
        base_visited_indices.update(tok.i for tok in extraction.relation.get_all_tokens())
        component_clauses: List[TripleElement] = []
        processed_in_chain = set()
        relation_children = sorted(extraction.relation.core.children, key=lambda t: t.i)
        for child in relation_children:
            if child.i in base_visited_indices or child.i in processed_in_chain:
                continue
            is_conjunction_head = any(c.dep_ == 'conj' for c in child.children)
            if self.__is_complement_part(child):
                if is_conjunction_head and self.config.coordinating_conjunctions:
                    components_found = self.__find_conjunction_components(child, base_visited_indices)
                    component_clauses.extend(components_found)
                    for comp in components_found:
                        processed_in_chain.update(tok.i for tok in comp.get_all_tokens())
                else:
                    single_component = self.__dfs_for_complement(child, base_visited_indices)
                    component_clauses.append(single_component)
                    processed_in_chain.update(tok.i for tok in single_component.get_all_tokens())

        if not component_clauses:
            return [extraction] if not any(str(e.complement) for e in [extraction]) else []

        final_extractions = []

        if len(component_clauses) > 1:
            # 1. Gerar extrações mínimas (limpando os conectores)
            for clause in component_clauses:
                # Usa uma versão limpa do complemento
                cleaned_clause = self.__clean_connectors(clause)

                new_extraction = Extraction()
                new_extraction.subject = extraction.subject
                new_extraction.relation = extraction.relation
                new_extraction.complement = cleaned_clause  # Usa a versão limpa
                final_extractions.append(new_extraction)

            # 2. Gerar UMA extração completa com todos os componentes (sem limpar)
            complete_complement = TripleElement()
            for clause in component_clauses:  # Usa as cláusulas originais
                complete_complement.merge(clause)

            complete_extraction = Extraction()
            complete_extraction.subject = extraction.subject
            complete_extraction.relation = extraction.relation
            complete_extraction.complement = complete_complement
            final_extractions.append(complete_extraction)
        else:
            # Se houver apenas um componente, não há conectores para limpar.
            single_extraction = Extraction()
            single_extraction.subject = extraction.subject
            single_extraction.relation = extraction.relation
            single_extraction.complement = component_clauses[0]
            final_extractions.append(single_extraction)

        return final_extractions

    @staticmethod
    def __clean_connectors(element: TripleElement) -> TripleElement:
        """
        Cria uma nova versão de um TripleElement removendo conectores
        como pontuação (punct) e conjunções coordenativas (cc).
        """
        cleaned_element = TripleElement()

        # Mantém o mesmo 'core' se ele não for um conector
        if element.core and element.core.dep_ not in ['punct', 'cc']:
            cleaned_element.core = element.core

        # Adiciona apenas as peças que não são conectores
        for piece in element.pieces:
            if piece.dep_ not in ['punct', 'cc']:
                cleaned_element.add_piece(piece)

        # Se o core original era um conector, mas há outras peças,
        # tenta eleger um novo core para não ficar vazio.
        if not cleaned_element.core and cleaned_element.pieces:
            cleaned_element.core = cleaned_element.pieces.pop(0)

        return cleaned_element

    def __is_complement_part(self, token: Token) -> bool:
        """Verifica se um token pode ser parte de um complemento."""
        if token.pos_ == "PRON" and "Rel" in token.morph.get("PronType", []):
            return False

        valid_deps = [
            "nmod", "xcomp", "dobj", "obj", "acl:relcl", "iobj", "acl:part",
            "nummod", "advmod", "appos", "amod", "dep", "case", "mark", "det",
            "flat", "fixed", "obl", "cop", "aux",
            "cc"  # <-- BUG 1 CORRIGIDO: Adicionada a dependência 'cc'
        ]
        if token.dep_ in valid_deps:
            return True

        if token.dep_ == "conj":
            return True

        if token.dep_ in ["ccomp", "advcl"] and not any(c.dep_.startswith("nsubj") for c in token.children):
            return True

        # BUG 2 CORRIGIDO: Removida a condição "token.i > token.head.i"
        if token.dep_ == "punct" and self.__valid_punct(token):
            return True

        return False

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
                local_visited = {token.i}
                while stack:
                    current_token = stack.pop()
                    for child in current_token.children:
                        if child.i not in local_visited:
                            if child.dep_ in ["nummod", "advmod", "appos", "nmod", "amod", "dep", "det", "case", "flat",
                                              "punct", "conj"] and (child.dep_ != "conj" or child.pos_ != "VERB"):
                                if child.dep_ == "punct" and not Extractor.__valid_punct(child):
                                    continue
                                sbj.add_piece(child)
                                stack.append(child)
                                local_visited.add(child.i)
                extraction = Extraction()
                extraction.subject = sbj
                extractions.append(extraction)
                visited_tokens.update(local_visited)
        return extractions

    @staticmethod
    def extract_relation(extraction: 'Extraction') -> List['Extraction']:
        if not extraction.subject or not extraction.subject.core:
            return [extraction]
        stack = deque()
        deprel_valid = ["aux:pass", "obj", "iobj", "advmod", "cop", "aux", "expl:pv", "mark"]
        deprel_valid_for_after_subject = ["flat", "expl:pv"]
        punct_invalid = [",", "--"]
        visited_tokens = {p.i for p in extraction.subject.get_all_tokens()}
        head_subject = extraction.subject.core.head
        if head_subject is None or head_subject.i in visited_tokens:
            return []
        extraction.relation = TripleElement(head_subject)
        visited_tokens.add(head_subject.i)
        stack.append(head_subject)
        while stack:
            current_token = stack.pop()
            for child in current_token.children:
                if child.i not in visited_tokens:
                    is_between = (min(extraction.subject.core.i, extraction.relation.core.i) < child.i < max(
                        extraction.subject.core.i, extraction.relation.core.i))
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
        has_verb = any(t.pos_ == 'VERB' or t.pos_ == 'AUX' for t in extraction.relation.get_all_tokens())
        if not has_verb:
            return []
        return [extraction]

    def __find_conjunction_components(self, start_token: Token, visited_indices: set) -> List[TripleElement]:
        components: List[TripleElement] = []
        shared_modifiers = [child for child in start_token.children if child.dep_ in ['case', 'det']]

        def create_component_with_mods(token: Token) -> TripleElement:
            component = self.__dfs_for_complement(token, visited_indices)
            for mod in shared_modifiers:
                component.add_piece(mod)
            return component

        components.append(create_component_with_mods(start_token))
        stack = deque([start_token])
        processed_conj = {start_token.i}
        while stack:
            current = stack.pop()
            for child in current.children:
                if child.dep_ == 'conj' and child.i not in processed_conj:
                    components.append(create_component_with_mods(child))
                    processed_conj.add(child.i)
                    stack.append(child)
        return components

    def __dfs_for_complement(self, start_token: Token, visited_indices: set) -> TripleElement:
        complement = TripleElement(start_token)
        stack = deque([start_token])
        local_visited = visited_indices.copy()
        local_visited.add(start_token.i)
        ignore_deps = {'conj', 'case', 'det'}
        while stack:
            current_token = stack.pop()
            for child in sorted(current_token.children, key=lambda t: t.i):
                if child.i not in local_visited and child.dep_ not in ignore_deps and self.__is_complement_part(child):
                    complement.add_piece(child)
                    local_visited.add(child.i)
                    stack.append(child)
        return complement

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
        valid_punctuation = {"(", ")", "{", "}", "\"", "'", "[", "]", ","}
        return token.text in valid_punctuation