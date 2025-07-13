from collections import deque
from typing import List, Optional, Set, Tuple
from spacy.tokens import Span, Doc, Token


class TripleElement:
    """
    Representa um componente de uma extração (sujeito, relação ou complemento).
    É construído em torno de um token principal e pode incluir tokens relacionados.
    """

    def __init__(self, token: Token = None, text: Optional[str] = None):
        """
        Inicializa o elemento.
        Args:
            token (Token, optional): O token principal do elemento.
            text (Optional[str], optional): Um texto pré-definido para o elemento (útil para relações sintéticas como "é").
        """
        self.core: Optional[Token] = token
        self.pieces: List[Token] = []
        self._text: Optional[str] = text

    def __str__(self):
        """Retorna a representação em string do elemento, limpando pontuações e conectores nas bordas."""
        if self._text:
            return self._text
        text = ' '.join([token.text for token in self.get_output_tokens()])
        return text.strip()

    def is_empty(self) -> bool:
        """Verifica se o elemento está vazio."""
        return not self.core and not self.pieces and not self._text

    def get_output_tokens(self) -> List[Token]:
        """Retorna a lista de tokens, limpa de pontuações e conjunções coordenativas no início e no fim."""
        tokens = self.get_all_tokens()
        if not tokens:
            return []
        # Remove conectores e pontuações do início
        while tokens and (tokens[0].pos_ == 'PUNCT' or tokens[0].dep_ == 'cc'):
            tokens.pop(0)
        # Remove pontuações do final
        while tokens and tokens[-1].pos_ == 'PUNCT':
            tokens.pop(-1)
        return tokens

    def get_all_tokens(self) -> List[Token]:
        """Retorna todos os tokens únicos do elemento, ordenados por sua posição na sentença."""
        tokens = [t for t in [self.core] + self.pieces if t is not None]
        seen_ids = set()
        unique_tokens = []
        # Ordena os tokens pela sua posição (índice 'i') na sentença
        for token in sorted(tokens, key=lambda x: x.i):
            if token.i not in seen_ids:
                unique_tokens.append(token)
                seen_ids.add(token.i)
        return unique_tokens

    def add_piece(self, piece: Token):
        """Adiciona um token às peças do elemento, se ainda não estiver presente."""
        if piece and piece not in self.pieces:
            self.pieces.append(piece)

    def merge(self, other_element: 'TripleElement'):
        """
        Mescla outro TripleElement neste, adicionando seu núcleo e peças.
        Args:
            other_element (TripleElement): O outro elemento a ser mesclado.
        """
        if other_element.core:
            self.add_piece(other_element.core)
        for piece in other_element.pieces:
            self.add_piece(piece)


class Extraction:
    """Representa uma única tripla Sujeito-Relação-Complemento."""

    def __init__(self):
        self.subject: Optional[TripleElement] = None
        self.relation: Optional[TripleElement] = None
        self.complement: Optional[TripleElement] = None

    def __iter__(self):
        """Permite a conversão da extração para um dicionário ou iterável."""
        yield 'arg1', str(self.subject) if self.subject else ''
        yield 'rel', str(self.relation) if self.relation else ''
        yield 'arg2', str(self.complement) if self.complement and not self.complement.is_empty() else ''

    def is_valid(self) -> bool:
        """Verifica se a extração é válida (pelo menos um dos componentes não está vazio)."""
        return bool(str(self.subject) or str(self.relation) or str(self.complement))


class ExtractorConfig:
    """Classe de configuração para controlar o comportamento do extrator."""

    def __init__(self, coordinating_conjunctions: bool = True, subordinating_conjunctions: bool = True,
                 appositive: bool = True, transitive: bool = True, hidden_subjects: bool = False, debug: bool = False):
        self.coordinating_conjunctions = coordinating_conjunctions
        self.subordinating_conjunctions = subordinating_conjunctions
        self.appositive = appositive
        self.transitive = transitive
        self.hidden_subjects = hidden_subjects
        self.debug = debug

    def __str__(self):
        return f"ExtractorConfig({self.coordinating_conjunctions}, {self.subordinating_conjunctions}, {self.hidden_subjects}, {self.appositive}, {self.transitive}, {self.debug})"

    def __iter__(self):
        yield 'coordinating_conjunctions', self.coordinating_conjunctions
        yield 'subordinating_conjunctions', self.subordinating_conjunctions
        yield 'hidden_subjects', self.hidden_subjects
        yield 'appositive', self.appositive
        yield 'transitive', self.transitive


class Extractor:
    """
    Classe principal para realizar a extração de informação aberta de um Doc do spaCy.
    """

    def __init__(self, config: ExtractorConfig = None):
        self.config = config if config else ExtractorConfig()

    def get_extractions_from_doc(self, doc: Doc) -> List['Extraction']:
        extractions = []
        for sentence in doc.sents:
            extractions.extend(self.get_extractions_from_sentence(sentence))
        return extractions

    def get_extractions_from_sentence(self, sentence: Span) -> list['Extraction']:
        final_extractions = []
        processed_tokens = set()

        for token in sentence:
            if token.i in processed_tokens:
                continue

            start_node = token
            is_verb_head = token.pos_ in ['VERB', 'AUX']
            is_nominal_predicate_root = token.dep_ == 'ROOT' and token.pos_ in ['ADJ', 'NOUN'] and any(
                c.dep_ == 'cop' for c in token.children)

            if is_verb_head or is_nominal_predicate_root:
                if is_nominal_predicate_root:
                    copula_candidates = [c for c in token.children if c.dep_ == 'cop']
                    if not copula_candidates: continue
                    start_node = copula_candidates[0]

                subject_element = self.__find_subject(start_node)

                if subject_element is None and not self.config.hidden_subjects:
                    continue

                for rel_extr, effective_verb in self.extract_relation(subject_element, start_node=start_node):
                    final_extractions.extend(self.extract_complements(rel_extr, effective_verb))

                    if rel_extr.relation and rel_extr.relation.core:
                        for t in rel_extr.relation.get_all_tokens():
                            processed_tokens.add(t.i)

        if self.config.appositive:
            final_extractions.extend(self.__extract_from_appositive(sentence))

        unique_extractions = []
        seen = set()
        for extr in final_extractions:
            representation = (str(extr.subject), str(extr.relation), str(extr.complement))
            if representation not in seen and extr.is_valid():
                seen.add(representation)
                unique_extractions.append(extr)

        return unique_extractions

    def __find_subject(self, verb_token: Token) -> Optional[TripleElement]:
        """
        Encontra o sujeito de um determinado verbo.
        Aprimorado para tratar sujeitos pospostos em construções passivas.
        """
        search_node = verb_token
        is_passive = any(c.dep_ == 'aux:pass' for c in search_node.children)

        if verb_token.dep_ in ['cop', 'aux', 'aux:pass']:
            search_node = verb_token.head
            is_passive = is_passive or any(c.dep_ == 'aux:pass' for c in search_node.children)

        for child in search_node.children:
            if child.dep_ in ["nsubj", "nsubj:pass"]:
                if child.pos_ == 'PRON' and 'Rel' in child.morph.get("PronType", []):
                    return self.__dfs_for_nominal_phrase(child.head, is_subject=True)
                return self.__dfs_for_nominal_phrase(child, is_subject=True)
            if child.dep_ == "csubj":
                return self.__dfs_for_complement(child, set())

        existential_verbs = {'haver', 'ocorrer', 'acontecer', 'existir', 'surgir'}
        if is_passive or search_node.lemma_ in existential_verbs:
            for child in search_node.children:
                if child.dep_ == 'obj':
                    return self.__dfs_for_nominal_phrase(child, is_subject=False)

        if search_node.dep_ in ['acl', 'acl:relcl']:
            return self.__dfs_for_nominal_phrase(search_node.head, is_subject=True)

        if self.config.hidden_subjects:
            return TripleElement()

        return None

    def __extract_from_appositive(self, sentence: Span) -> List['Extraction']:
        extractions = []
        for token in sentence:
            if token.dep_ == 'appos':
                subject_head = token.head
                if subject_head.dep_ in ['ccomp', 'xcomp']: continue

                subject = self.__dfs_for_nominal_phrase(subject_head, is_subject=True, ignore_conj=True,
                                                        ignore_appos=True)
                complement = self.__dfs_for_nominal_phrase(token, is_subject=False, ignore_conj=False)
                relation = TripleElement(text="é")

                if subject and complement:
                    extraction = Extraction()
                    extraction.subject = subject
                    extraction.relation = relation
                    extraction.complement = complement
                    extractions.append(extraction)
        return extractions

    def extract_relation(self, subject: Optional[TripleElement], start_node: Token) -> List[Tuple['Extraction', Token]]:
        subject_tokens = {p.i for p in subject.get_all_tokens()} if subject else set()

        base_relation, effective_verb = self.__build_relation_element(start_node, subject_tokens)
        if not base_relation:
            return []

        extraction = Extraction()
        extraction.subject = subject
        extraction.relation = base_relation
        extractions_found = [(extraction, effective_verb)]

        if self.config.coordinating_conjunctions:
            for child in effective_verb.children:
                if child.dep_ == 'conj' and child.pos_ in ['VERB', 'AUX']:
                    has_own_subject = any(c.dep_.startswith("nsubj") for c in child.children)
                    if not has_own_subject:
                        new_relation, new_effective_verb = self.__build_relation_element(child, set())
                        if new_relation:
                            new_extraction = Extraction()
                            new_extraction.subject = subject
                            new_extraction.relation = new_relation
                            extractions_found.append((new_extraction, new_effective_verb))
        return extractions_found

    @staticmethod
    def __dfs_for_nominal_phrase(start_token: Token, is_subject: bool = False,
                                 ignore_conj: bool = False, ignore_appos: bool = False) -> TripleElement:
        element = TripleElement(start_token)
        stack = deque([start_token])
        local_visited = {start_token.i}

        valid_deps = {"nummod", "advmod", "nmod", "amod", "dep", "det", "case", "flat", "flat:name", "punct"}
        if not ignore_conj:
            valid_deps.add("conj")
            valid_deps.add("cc")
        if not ignore_appos:
            valid_deps.add("appos")

        while stack:
            current_token = stack.pop()
            for child in sorted(current_token.children, key=lambda t: t.i):
                if child.i in local_visited: continue
                if is_subject and current_token.i == start_token.i and child.dep_ == 'case': continue
                is_valid_dep = child.dep_ in valid_deps
                is_non_verbal_conj = not (child.dep_ == "conj" and child.pos_ == "VERB")
                if is_valid_dep and is_non_verbal_conj:
                    element.add_piece(child)
                    local_visited.add(child.i)
                    stack.append(child)
        return element

    @staticmethod
    def __build_relation_element(start_token: Token, visited_tokens: Set[int]) -> Tuple[
        Optional[TripleElement], Optional[Token]]:
        relation = TripleElement(start_token)
        effective_verb = start_token
        stack = deque([start_token])
        local_visited = visited_tokens.copy()
        local_visited.add(start_token.i)

        if start_token.dep_ == 'cop':
            effective_verb = start_token.head
            relation.add_piece(effective_verb)
            local_visited.add(effective_verb.i)

        while stack:
            current = stack.pop()
            if current not in relation.get_all_tokens():
                relation.add_piece(current)

            relation_adverbs = {"não", "ja", "ainda", "também", "nunca"}
            for child in current.children:
                if child.i in local_visited: continue
                if child.dep_ in ["aux", "aux:pass", "xcomp"] and child.pos_ in ['VERB', 'AUX']:
                    stack.append(child)
                    local_visited.add(child.i)
                    if child.i > effective_verb.i:
                        effective_verb = child
                elif child.dep_ == 'advmod' and child.lemma_.lower() in relation_adverbs:
                    relation.add_piece(child)
                    local_visited.add(child.i)

        if effective_verb.dep_ in ['aux', 'aux:pass', 'cop'] and effective_verb.head not in relation.get_all_tokens():
            relation.add_piece(effective_verb.head)
            effective_verb = effective_verb.head

        has_verb = any(t.pos_ in ['VERB', 'AUX'] for t in relation.get_all_tokens())
        return (relation, effective_verb) if has_verb else (None, None)

    def extract_complements(self, extraction: 'Extraction', complement_root: Token) -> List['Extraction']:
        if not extraction.relation or not extraction.relation.core:
            return [extraction] if extraction.subject and extraction.is_valid() else []

        base_visited_indices = {tok.i for tok in extraction.subject.get_all_tokens()} if extraction.subject else set()
        base_visited_indices.update(tok.i for tok in extraction.relation.get_all_tokens())

        full_complement = TripleElement()
        complement_parts = []

        if complement_root.pos_ not in ['VERB', 'AUX']:
            component = self.__dfs_for_complement(complement_root, base_visited_indices)
            if not component.is_empty():
                full_complement.merge(component)
                complement_parts.append(component)
                base_visited_indices.update(tok.i for tok in component.get_all_tokens())

        for child in sorted(complement_root.children, key=lambda t: t.i):
            if child.i in base_visited_indices:
                continue

            if self.__is_complement_part(child):
                component = self.__dfs_for_complement(child, base_visited_indices)
                if not component.is_empty():
                    full_complement.merge(component)
                    complement_parts.append(component)
                    base_visited_indices.update(tok.i for tok in component.get_all_tokens())

        final_extractions = []
        if not full_complement.is_empty():
            complete_extraction = Extraction()
            complete_extraction.subject = extraction.subject
            complete_extraction.relation = extraction.relation
            complete_extraction.complement = full_complement
            final_extractions.append(complete_extraction)
        elif extraction.subject and extraction.relation:
            final_extractions.append(extraction)

        if len(complement_parts) > 1 and self.config.coordinating_conjunctions:
            for component in complement_parts:
                cleaned_component = self.__clean_connectors(component)
                if cleaned_component.is_empty(): continue
                new_extraction = Extraction()
                new_extraction.subject = extraction.subject
                new_extraction.relation = extraction.relation
                new_extraction.complement = cleaned_component
                final_extractions.append(new_extraction)

        return final_extractions

    def __is_complement_part(self, token: Token) -> bool:
        non_initiator_deps = {'mark', 'case', 'cop', 'punct', 'aux', 'aux:pass', 'expl:pv', 'nsubj', 'nsubj:pass',
                              'csubj'}
        if token.dep_ in non_initiator_deps:
            return False
        if token.dep_ == 'conj' and token.pos_ in ['VERB', 'AUX']: return False
        if token.pos_ == "PRON" and "Rel" in token.morph.get("PronType", []): return False
        return True

    @staticmethod
    def __clean_connectors(element: TripleElement) -> TripleElement:
        cleaned_element = TripleElement()
        tokens = element.get_output_tokens()  # Usa get_output_tokens para já vir limpo nas bordas
        if not tokens: return cleaned_element
        if tokens[0].dep_ == 'cc':
            tokens.pop(0)
        if tokens:
            cleaned_element.core = tokens.pop(0)
            cleaned_element.pieces = tokens
        return cleaned_element

    @staticmethod
    def __find_conjunction_components(start_token: Token, visited_indices: set) -> List[TripleElement]:
        components: List[TripleElement] = []

        def create_component(token: Token) -> TripleElement:
            return Extractor.__dfs_for_complement(token, visited_indices, ignore_conj=True)

        components.append(create_component(start_token))
        for child in start_token.children:
            if child.dep_ == 'conj':
                components.append(create_component(child))
        return components

    @staticmethod
    def __dfs_for_complement(start_token: Token, visited_indices: set, ignore_conj: bool = False) -> TripleElement:
        complement = TripleElement(start_token)
        stack = deque([start_token])
        local_visited = visited_indices.copy()
        local_visited.add(start_token.i)

        # Ignora sujeitos e marcadores de subordinação
        ignore_deps = {'nsubj', 'nsubj:pass', 'csubj', 'mark'}
        if ignore_conj:
            ignore_deps.add('conj')

        while stack:
            current_token = stack.pop()
            for child in sorted(current_token.children, key=lambda t: t.i):
                if child.i not in local_visited and child.dep_ not in ignore_deps:
                    complement.add_piece(child)
                    local_visited.add(child.i)
                    stack.append(child)
        return complement
