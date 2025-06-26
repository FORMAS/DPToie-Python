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
        yield 'arg1', str(self.subject) if self.subject else ''
        yield 'rel', str(self.relation) if self.relation else ''
        yield 'arg2', str(self.complement) if self.complement and (
            self.complement.core or self.complement.pieces) else ''


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

    def __build_relation_element(self, start_token: Token, visited_tokens: Set[int]) -> (Optional[TripleElement], Optional[Token]):
        relation = TripleElement(start_token)
        effective_verb = start_token

        # Pilha para busca na frase verbal
        stack = deque([start_token])
        # Mantém o controle local para não adicionar o mesmo token várias vezes
        local_visited = visited_tokens.copy()
        local_visited.add(start_token.i)

        while stack:
            current = stack.pop()
            relation.add_piece(current)

            # Procura por filhos que compõem a frase verbal
            for child in current.children:
                if child.i in local_visited:
                    continue

                # Lógica para xcomp: anexa apenas se não for o início de um pedaço de complemento
                if child.dep_ == 'xcomp' and child.pos_ == 'VERB':
                    # Verifica se o xcomp tem uma marca (como 'de', 'para'), indicando um complemento
                    has_marker = any(c.dep_ == 'mark' for c in child.children)
                    if not has_marker:
                        effective_verb = child
                        stack.append(child)
                        local_visited.add(child.i)
                # Anexa outros modificadores verbais
                elif child.dep_ in ["aux:pass", "aux", "cop", "expl", "expl:pv"]:
                    relation.add_piece(child)
                    local_visited.add(child.i)

        has_verb = any(t.pos_ in ['VERB', 'AUX'] for t in relation.get_all_tokens())
        return (relation, effective_verb) if has_verb else (None, None)

    def extract_relation(self, extraction: 'Extraction') -> List['Extraction']:
        if not extraction.subject or not extraction.subject.core:
            return []
        subject_tokens = {p.i for p in extraction.subject.get_all_tokens()}
        relation_head = extraction.subject.core.head
        if relation_head is None or relation_head.i in subject_tokens:
            return []

        base_relation, effective_verb = self.__build_relation_element(relation_head, subject_tokens)
        if not base_relation:
            return []

        extraction.relation = base_relation
        extractions_found = [extraction]

        relation_tokens = {p.i for p in base_relation.get_all_tokens()}
        visited_for_conj = subject_tokens.union(relation_tokens)

        for child in effective_verb.children:
            if child.dep_ == 'conj' and child.pos_ in ['VERB', 'AUX']:
                new_relation, _ = self.__build_relation_element(child, visited_for_conj)
                if new_relation:
                    new_extraction = Extraction()
                    new_extraction.subject = extraction.subject
                    new_extraction.relation = new_relation
                    extractions_found.append(new_extraction)
        return extractions_found

    def extract_complements(self, extraction: 'Extraction') -> List['Extraction']:
        if not extraction.relation or not extraction.relation.core:
            return [extraction] if not any(str(e.complement) for e in [extraction]) else []

        base_visited_indices = {tok.i for tok in extraction.subject.get_all_tokens()}
        base_visited_indices.update(tok.i for tok in extraction.relation.get_all_tokens())

        complement_components: List[TripleElement] = []
        processed_in_chain = set()
        relation_children = sorted(extraction.relation.core.children, key=lambda t: t.i)

        for child in relation_children:
            if child.i in base_visited_indices or child.i in processed_in_chain:
                continue

            if self.__is_complement_part(child):
                is_conjunction_head = any(c.dep_ == 'conj' for c in child.children)
                if is_conjunction_head and self.config.coordinating_conjunctions:
                    components_found = self.__find_conjunction_components(child, base_visited_indices)
                    complement_components.extend(components_found)
                    for comp in components_found:
                        processed_in_chain.update(tok.i for tok in comp.get_all_tokens())
                else:
                    single_component = self.__dfs_for_complement(child, base_visited_indices, ignore_conj=False)
                    complement_components.append(single_component)
                    processed_in_chain.update(tok.i for tok in single_component.get_all_tokens())

        if not complement_components:
            return [extraction]

        final_extractions = []
        if len(complement_components) > 1:
            for component in complement_components:
                cleaned_component = self.__clean_connectors(component)
                new_extraction = Extraction()
                new_extraction.subject = extraction.subject
                new_extraction.relation = extraction.relation
                new_extraction.complement = cleaned_component
                final_extractions.append(new_extraction)
            complete_complement = TripleElement()
            for component in complement_components:
                complete_complement.merge(component)
            complete_extraction = Extraction()
            complete_extraction.subject = extraction.subject
            complete_extraction.relation = extraction.relation
            complete_extraction.complement = complete_complement
            final_extractions.append(complete_extraction)
        else:
            single_extraction = Extraction()
            single_extraction.subject = extraction.subject
            single_extraction.relation = extraction.relation
            single_extraction.complement = complement_components[0]
            final_extractions.append(single_extraction)
        return final_extractions

    @staticmethod
    def __is_complement_part(token: Token) -> bool:
        """
        Verifica se um token pode ser parte de um complemento.
        """
        if token.dep_ == 'conj' and token.pos_ in ['VERB', 'AUX']:
            return False

        if token.pos_ == "PRON" and "Rel" in token.morph.get("PronType", []):
            return False

        valid_deps = [
            "nmod", "xcomp", "dobj", "obj", "acl:relcl", "iobj", "acl:part", "acl",
            "nummod", "advmod", "appos", "amod", "dep", "case", "mark", "det",
            "flat", "fixed", "obl", "obl:agent", "cc"
        ]
        if token.dep_ in valid_deps:
            return True

        # Trata conjunções não-verbais
        if token.dep_ == "conj":
            return True

        if token.dep_ in ["ccomp", "advcl"] and not any(c.dep_.startswith("nsubj") for c in token.children):
            return True

        if token.dep_ == "punct" and Extractor.__valid_punct(token):
            return True

        return False

    @staticmethod
    def __clean_connectors(element: TripleElement) -> TripleElement:
        cleaned_element = TripleElement()
        if element.core and element.core.dep_ not in ['punct', 'cc']:
            cleaned_element.core = element.core
        for piece in element.pieces:
            if piece.dep_ not in ['punct', 'cc']:
                cleaned_element.add_piece(piece)
        if not cleaned_element.core and cleaned_element.pieces:
            cleaned_element.core = cleaned_element.pieces.pop(0)
        return cleaned_element

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
    def __find_conjunction_components(start_token: Token, visited_indices: set) -> List[TripleElement]:
        components: List[TripleElement] = []
        shared_modifiers = [child for child in start_token.children if child.dep_ in ['case', 'det']]

        def create_component_with_mods(token: Token) -> TripleElement:
            component = Extractor.__dfs_for_complement(token, visited_indices, ignore_conj=True)
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

    @staticmethod
    def __dfs_for_complement(start_token: Token, visited_indices: set, ignore_conj: bool = True) -> TripleElement:
        """
        Realiza uma busca em profundidade para construir um único elemento de complemento.
        """
        complement = TripleElement(start_token)
        stack = deque([start_token])
        local_visited = visited_indices.copy()
        local_visited.add(start_token.i)

        ignore_deps = {}
        if ignore_conj:
            ignore_deps['conj'] = True

        while stack:
            current_token = stack.pop()
            for child in sorted(current_token.children, key=lambda t: t.i):
                if child.i not in local_visited and child.dep_ not in ignore_deps and Extractor.__is_complement_part(
                        child):
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