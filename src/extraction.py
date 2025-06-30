from collections import deque
from typing import List, Optional, Set, Tuple
from spacy.tokens import Span, Doc, Token


class TripleElement:
    """
    Represents a component of an extraction (subject, relation, or complement).
    It is built around a core token and can include additional related tokens.
    """

    def __init__(self, token: Token = None, text: Optional[str] = None):
        self.core: Optional[Token] = token
        self.pieces: List[Token] = []
        self._text: Optional[str] = text

    def __str__(self):
        if self._text:
            return self._text
        text = ' '.join([token.text for token in self.get_output_tokens()])
        return text.strip()

    def get_output_tokens(self) -> List[Token]:
        """Returns the list of tokens, cleaned of leading/trailing punctuation."""
        tokens = self.get_all_tokens()
        if not tokens:
            return []
        while tokens and tokens[0].text in [',', '.', 'e']:
            tokens.pop(0)
        while tokens and tokens[-1].text in [',', '.']:
            tokens.pop(-1)
        return tokens

    def get_all_tokens(self) -> List[Token]:
        """Returns all unique tokens of the element, sorted by their position in the sentence."""
        tokens = [t for t in [self.core] + self.pieces if t is not None]
        seen_ids = set()
        unique_tokens = []
        for token in sorted(tokens, key=lambda x: x.i):
            if token.i not in seen_ids:
                unique_tokens.append(token)
                seen_ids.add(token.i)
        return unique_tokens

    def add_piece(self, piece: Token):
        """Adds a token to the element's pieces."""
        if piece and piece not in self.pieces:
            self.pieces.append(piece)

    def merge(self, other_element: 'TripleElement'):
        """Merges another TripleElement into this one."""
        if other_element.core:
            self.add_piece(other_element.core)
        for piece in other_element.pieces:
            self.add_piece(piece)


class Extraction:
    """Represents a single Subject-Relation-Complement triple."""

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
    """Configuration class to control the extractor's behavior."""

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
    Main class to perform open information extraction from a spaCy Doc.
    It works by iterating through sentences and applying a set of rules
    to identify subjects, relations, and complements.
    """

    def __init__(self, config: ExtractorConfig = None):
        self.config = config if config else ExtractorConfig()

    def get_extractions_from_doc(self, doc: Doc) -> List['Extraction']:
        """
        Processes a spaCy Doc and returns a list of all found extractions.

        Args:
            doc (Doc): The spaCy Doc to process.

        Returns:
            List[Extraction]: A list of Extraction objects.
        """
        extractions = []
        for sentence in doc.sents:
            extractions.extend(self.get_extractions_from_sentence(sentence))
        return extractions

    def get_extractions_from_sentence(self, sentence: Span) -> list['Extraction']:
        """
        Extracts triples from a single sentence.

        The main strategy is "verb-centric": it iterates through tokens, identifies potential
        main verbs or predicates of a clause, and then searches for their corresponding
        subjects and complements.

        Args:
            sentence (Span): The spaCy Span representing the sentence.

        Returns:
            list[Extraction]: A list of extractions found in the sentence.
        """
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
                # In sentences like "O céu é azul", the root is "azul" (ADJ).
                # The extraction process must start from its verb of connection (copula), "é".
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
            is_valid = representation[0] or representation[1] or representation[2]
            if representation not in seen and is_valid:
                seen.add(representation)
                unique_extractions.append(extr)

        return unique_extractions

    def __find_subject(self, verb_token: Token) -> Optional[TripleElement]:
        """
        Finds the subject of a given verb, handling direct, passive, post-verbal,
        and clausal subjects.

        Args:
            verb_token (Token): The verb token whose subject is to be found.

        Returns:
            Optional[TripleElement]: The found subject, or None.
        """
        search_node = verb_token
        # In a verb phrase, the subject is syntactically linked to the head of the phrase,
        # not necessarily the auxiliary verb itself.
        if verb_token.dep_ in ['cop', 'aux', 'aux:pass']:
            search_node = verb_token.head

        for child in search_node.children:
            # Clausal subject (csubj): "Comemorar é difícil" -> "Comemorar" is the subject.
            if child.dep_ == "csubj":
                return self.__dfs_for_complement(child, set())

            # Nominal subject (nsubj) or passive nominal subject (nsubj:pass).
            if child.dep_ in ["nsubj", "nsubj:pass"]:
                # Handles relative clauses, finding the antecedent of the pronoun "que".
                # Ex: "... a cidade QUE foi centro..." -> subject of "foi" is "cidade".
                if child.pos_ == 'PRON' and 'Rel' in child.morph.get("PronType", []):
                    return self.__dfs_for_nominal_phrase(child.head, is_subject=True)
                return self.__dfs_for_nominal_phrase(child, is_subject=True)

        # Subject of an adjectival clause (acl).
        if search_node.dep_ in ['acl', 'acl:relcl']:
            return self.__dfs_for_nominal_phrase(search_node.head, is_subject=True)

        # Post-verbal subject, common with verbs like "haver", "ocorrer".
        # Ex: "Houve uma homenagem" -> "uma homenagem" is the subject.
        for child in verb_token.children:
            if child.dep_ == 'obj' and verb_token.lemma_ in ['haver', 'ocorrer', 'acontecer', 'existir', 'surgir']:
                return self.__dfs_for_nominal_phrase(child, is_subject=False)

        return None

    def __extract_from_appositive(self, sentence: Span) -> List['Extraction']:
        """
        Extracts synthetic triples from appositions, creating an "is" relation.
        Ex: "João, o carpinteiro, ..." -> (João, é, o carpinteiro)

        Args:
            sentence (Span): The sentence to process.

        Returns:
            List[Extraction]: A list of extractions from appositions.
        """
        extractions = []
        for token in sentence:
            if token.dep_ == 'appos':
                subject_head = token.head
                if subject_head.dep_ in ['ccomp', 'xcomp']: continue

                subject = self.__dfs_for_nominal_phrase(subject_head, is_subject=True, ignore_conj=True)
                relation = TripleElement(text="é")
                complement = self.__dfs_for_nominal_phrase(token, is_subject=False, ignore_conj=False)

                extraction = Extraction()
                extraction.subject = subject
                extraction.relation = relation
                extraction.complement = complement
                extractions.append(extraction)
        return extractions

    def extract_relation(self, subject: Optional[TripleElement], start_node: Token) -> List[Tuple['Extraction', Token]]:
        """
        Extracts the relation starting from a verb token. It also handles
        coordinated verbs (e.g., "comeu e bebeu").

        Args:
            subject (Optional[TripleElement]): The subject of the relation.
            start_node (Token): The initial verb token.

        Returns:
            List[Tuple[Extraction, Token]]: A list of tuples, where each contains
                                             an extraction and its effective verb.
        """
        subject_tokens = {p.i for p in subject.get_all_tokens()} if subject else set()

        base_relation, effective_verb = self.__build_relation_element(start_node, subject_tokens)
        if not base_relation:
            return []

        extraction = Extraction()
        extraction.subject = subject
        extraction.relation = base_relation
        extractions_found = [(extraction, effective_verb)]

        # Handles verb coordination (e.g., "ele comeu E bebeu").
        if self.config.coordinating_conjunctions:
            for child in effective_verb.children:
                if child.dep_ == 'conj' and child.pos_ in ['VERB', 'AUX']:
                    # Ensures the coordinated verb doesn't have its own subject.
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
                                 ignore_conj: bool = False) -> TripleElement:
        """
        Performs a Depth-First Search starting from a token to build a complete nominal phrase.

        Args:
            start_token (Token): The token to start the search from (usually a NOUN or PROPN).
            is_subject (bool): Flag indicating if it's a subject, to apply specific rules.
            ignore_conj (bool): Flag to ignore conjunctions.

        Returns:
            TripleElement: The constructed nominal phrase.
        """
        element = TripleElement(start_token)
        stack = deque([start_token])
        local_visited = {start_token.i}
        # A specific rule to prevent relative clauses ('acl', 'acl:relcl') from being part of the nominal phrase itself.
        # This avoids capturing verbs from sub-clauses in the subject. Ex: "O homem QUE CORREU..."
        valid_deps = {"nummod", "advmod", "nmod", "amod", "dep", "det", "case", "flat", "flat:name", "punct", "conj",
                      "cc"}

        while stack:
            current_token = stack.pop()
            for child in sorted(current_token.children, key=lambda t: t.i):
                if child.i in local_visited: continue
                if is_subject and current_token.i == start_token.i and child.dep_ == 'case': continue
                if ignore_conj and child.dep_ == 'conj': continue

                is_valid_dep = child.dep_ in valid_deps
                is_non_verbal_conj = not (child.dep_ == "conj" and child.pos_ == "VERB")

                if is_valid_dep and is_non_verbal_conj:
                    element.add_piece(child)
                    local_visited.add(child.i)
                    stack.append(child)
        return element

    @staticmethod
    def __build_relation_element(start_token: Token, visited_tokens: Set[int]) -> (Optional[TripleElement],
                                                                                   Optional[Token]):
        """
        Constructs the relation element by gathering auxiliary verbs, negation, and other modifiers.

        Args:
            start_token (Token): The verb token to start from.
            visited_tokens (Set[int]): A set of token indices already processed.

        Returns:
            A tuple containing the relation (TripleElement) and the effective verb (Token)
            which is the main action verb in a verb phrase.
        """
        relation = TripleElement(start_token)
        effective_verb = start_token
        stack = deque([start_token])
        local_visited = visited_tokens.copy()
        local_visited.add(start_token.i)

        # In copular constructions ("ser", "estar"), the effective "verb" for finding
        # complements is the predicative (adjective or noun).
        if start_token.dep_ == 'cop':
            effective_verb = start_token.head

        while stack:
            current = stack.pop()
            if current not in relation.get_all_tokens():
                relation.add_piece(current)

            for child in current.children:
                if child.i in local_visited: continue
                if child.dep_ in ["aux", "aux:pass", "xcomp"]:
                    if child.pos_ in ['VERB', 'AUX']:
                        stack.append(child)
                        local_visited.add(child.i)
                        if child.i > effective_verb.i:
                            effective_verb = child
                # Generalization to capture adverbial modifiers of the verb, like "não", "já", "também".
                elif child.dep_ == 'advmod':
                    relation.add_piece(child)
                    local_visited.add(child.i)

        has_verb = any(t.pos_ in ['VERB', 'AUX'] for t in relation.get_all_tokens())
        return (relation, effective_verb) if has_verb else (None, None)

    def extract_complements(self, extraction: 'Extraction', complement_root: Token) -> List['Extraction']:
        """
        Extracts all complements associated with a given verb/relation.

        Args:
            extraction (Extraction): The extraction object being built.
            complement_root (Token): The token from which to start searching for complements.
                                     This is the "effective verb" of the relation.
        Returns:
            List[Extraction]: A list of complete extractions (with complements).
        """
        if not extraction.relation or not extraction.relation.core:
            return [extraction] if extraction.subject else []

        base_visited_indices: Set[int] = set()
        if extraction.subject is not None:
            base_visited_indices.update(tok.i for tok in extraction.subject.get_all_tokens())
        base_visited_indices.update(tok.i for tok in extraction.relation.get_all_tokens())

        complement_components: List[TripleElement] = []
        processed_in_chain = set()

        # If the root is not a verb (it's a predicative), it is the complement itself.
        # Ex: In "é brasileiro", "brasileiro" is the root and the complement.
        if complement_root.pos_ not in ['VERB', 'AUX']:
            component = self.__dfs_for_complement(complement_root, base_visited_indices, ignore_conj=False)
            if str(component):
                complement_components.append(component)
                processed_in_chain.update(tok.i for tok in component.get_all_tokens())

        # Also checks the children of the complement root for more complements (e.g., objects, adverbial clauses).
        relation_children = sorted(complement_root.children, key=lambda t: t.i)

        # Defines dependencies that do not start a new complement, like subject or markers.
        non_initiator_deps = {'mark', 'case', 'cop', 'punct', 'aux', 'aux:pass', 'expl:pv'}
        if extraction.subject is not None:
            non_initiator_deps.update({'nsubj', 'nsubj:pass', 'csubj'})

        for child in relation_children:
            if child.i in base_visited_indices or child.i in processed_in_chain: continue

            if self.__is_complement_part(child, non_initiator_deps):
                if any(c.dep_ == 'conj' for c in child.children) and self.config.coordinating_conjunctions:
                    components_found = self.__find_conjunction_components(child, base_visited_indices)
                    complement_components.extend(components_found)
                    for comp in components_found:
                        processed_in_chain.update(tok.i for tok in comp.get_all_tokens())
                else:
                    single_component = self.__dfs_for_complement(child, base_visited_indices, ignore_conj=False)
                    if str(single_component):
                        complement_components.append(single_component)
                        processed_in_chain.update(tok.i for tok in single_component.get_all_tokens())

        if not complement_components:
            return [extraction] if extraction.subject else []

        final_extractions = []
        # Creates one extraction for the full complement.
        full_complement = TripleElement()
        for component in complement_components:
            full_complement.merge(component)

        if str(full_complement):
            complete_extraction = Extraction()
            complete_extraction.subject = extraction.subject
            complete_extraction.relation = extraction.relation
            complete_extraction.complement = full_complement
            final_extractions.append(complete_extraction)

        # Creates fragmented extractions if enabled by the config.
        # This is controlled by 'coordinating_conjunctions' because multiple complements
        # are usually a result of coordination.
        if len(complement_components) > 1 and self.config.coordinating_conjunctions:
            for component in complement_components:
                cleaned_component = self.__clean_connectors(component)
                if not str(cleaned_component): continue
                new_extraction = Extraction()
                new_extraction.subject = extraction.subject
                new_extraction.relation = extraction.relation
                new_extraction.complement = cleaned_component
                final_extractions.append(new_extraction)

        return final_extractions or ([extraction] if extraction.subject else [])

    def __is_complement_part(self, token: Token, non_initiator_deps: Set[str]) -> bool:
        """Checks if a token can be part of a complement based on its dependency."""
        if token.dep_ in non_initiator_deps:
            return False
        if token.dep_ == 'conj' and token.pos_ in ['VERB', 'AUX']: return False
        if token.pos_ == "PRON" and "Rel" in token.morph.get("PronType", []): return False

        valid_deps = {"obj", "iobj", "obl", "obl:agent", "xcomp", "ccomp", "advcl", "advmod", "nmod", "amod", "nummod",
                      "appos", "dep", "case", "det", "flat", "flat:name", "fixed", "cop", "mark", "cc", "nsubj",
                      "nsubj:pass", "acl", "acl:relcl"}

        if token.dep_ in valid_deps: return True
        if token.dep_ == "conj": return True
        if token.dep_ == "punct" and self.__valid_punct(token): return True
        return False

    @staticmethod
    def __clean_connectors(element: TripleElement) -> TripleElement:
        """Removes leading coordinating conjunctions from a complement piece."""
        cleaned_element = TripleElement()
        tokens = element.get_all_tokens()
        if not tokens: return cleaned_element

        if tokens[0].dep_ == 'cc':
            tokens.pop(0)

        for piece in tokens:
            cleaned_element.add_piece(piece)

        if not cleaned_element.core and cleaned_element.pieces:
            cleaned_element.core = cleaned_element.pieces.pop(0)
        return cleaned_element

    @staticmethod
    def __find_conjunction_components(start_token: Token, visited_indices: set) -> List[TripleElement]:
        """Finds all components linked by a conjunction."""
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
        """Generic DFS to build a complement phrase."""
        complement = TripleElement(start_token)
        stack = deque([start_token])
        local_visited = visited_indices.copy()
        local_visited.add(start_token.i)

        extractor_instance = Extractor()

        ignore_deps = set()
        if ignore_conj:
            ignore_deps.add('conj')

        while stack:
            current_token = stack.pop()
            for child in sorted(current_token.children, key=lambda t: t.i):
                if child.i not in local_visited and child.dep_ not in ignore_deps:
                    # Uses an empty set for non_initiator_deps to capture all valid children.
                    if extractor_instance.__is_complement_part(child, set()):
                        complement.add_piece(child)
                        local_visited.add(child.i)
                        stack.append(child)
        return complement

    @staticmethod
    def __valid_punct(token: Token) -> bool:
        """Checks if a punctuation token is valid to be included in an element."""
        valid_punctuation = {"(", ")", "{", "}", "\"", "'", "[", "]", ","}
        return token.text in valid_punctuation
