from collections import deque
from typing import List, Optional, Set, Tuple, Any, Generator
from spacy.tokens import Span, Doc, Token


class TripleElement:
    """
    Representa um componente de uma extração (sujeito, relação ou complemento).
    É construído em torno de um token principal e pode incluir outros tokens
    que compõem o sintagma.
    """

    def __init__(self, token: Token = None, text: Optional[str] = None):
        """
        Inicializa o elemento.

        Args:
            token (Token, optional): O token principal (núcleo) do elemento.
            text (Optional[str], optional): Um texto pré-definido para o elemento.
                                            Útil para relações sintéticas como "é".
        """
        self.core: Optional[Token] = token
        self.pieces: List[Token] = []
        self._text: Optional[str] = text
        self.is_from_appositive = False  # Flag para identificar elementos de aposto

    def __str__(self):
        """Retorna a representação textual do elemento, limpando pontuações e conectores nas bordas."""
        if self._text:
            return self._text
        # get_output_tokens já retorna os tokens limpos e ordenados
        text = ' '.join([token.text for token in self.get_output_tokens()])
        return text.strip()

    def is_empty(self) -> bool:
        """Verifica se o elemento não contém tokens ou texto."""
        return not self.core and not self.pieces and not self._text

    def get_output_tokens(self) -> List[Token]:
        """
        Retorna a lista de tokens para a saída final, limpa de pontuações
        e conjunções coordenativas no início e no fim.
        """
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
        # Combina o núcleo e as peças, garantindo que não haja None
        tokens = [t for t in [self.core] + self.pieces if t is not None]

        # Garante tokens únicos e os ordena pelo índice na sentença
        seen_ids = set()
        unique_tokens = []
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
    """
    Representa uma única tripla Sujeito-Relação-Complemento (arg1, rel, arg2).
    """

    def __init__(self, subject: TripleElement = None, relation: TripleElement = None, complement: TripleElement = None):
        self.subject = subject
        self.relation = relation
        self.complement = complement

    def __iter__(self) -> Generator[tuple[str, str], Any, None]:
        """Permite a conversão da extração para um dicionário, facilitando a serialização."""
        yield 'arg1', str(self.subject) if self.subject and not self.subject.is_empty() else ''
        yield 'rel', str(self.relation) if self.relation and not self.relation.is_empty() else ''
        yield 'arg2', str(self.complement) if self.complement and not self.complement.is_empty() else ''

    def is_valid(self) -> bool:
        """
        Verifica se a extração é válida. Uma extração válida deve ter:
        1. Sujeito e Relação não vazios.
        2. A Relação deve conter um verbo.
        3. O Sujeito não pode ser apenas um pronome relativo ou conjunção.
        """
        if not self.subject or self.subject.is_empty() or not self.relation or self.relation.is_empty():
            return False

        # (Filtro de Coerência): Garante que a relação contenha um verbo.
        if not any(t.pos_ in ['VERB', 'AUX'] for t in self.relation.get_all_tokens()):
            return False

        # (Filtro de Lixo Sintático): Evita extrações onde o sujeito é apenas um pronome relativo.
        subject_tokens = self.subject.get_all_tokens()
        if len(subject_tokens) == 1 and subject_tokens[0].pos_ in ['PRON', 'SCONJ'] and 'Rel' in subject_tokens[0].morph.get("PronType", []):
            return False

        return True

    def to_tuple(self) -> Tuple[str, str, str]:
        """Retorna a extração como uma tupla de strings, útil para comparação e remoção de duplicatas."""
        return (
            str(self.subject) if self.subject else '',
            str(self.relation) if self.relation else '',
            str(self.complement) if self.complement else ''
        )


class ExtractorConfig:
    """Classe de configuração para controlar o comportamento do extrator."""

    def __init__(self,
                 coordinating_conjunctions: bool = True,
                 subordinating_conjunctions: bool = True,
                 appositive: bool = True,
                 appositive_transitivity: bool = True,
                 hidden_subjects: bool = False,
                 debug: bool = False):
        self.coordinating_conjunctions = coordinating_conjunctions
        self.subordinating_conjunctions = subordinating_conjunctions
        self.appositive = appositive
        self.appositive_transitivity = appositive_transitivity
        self.hidden_subjects = hidden_subjects
        self.debug = debug


class Extractor:
    """
    Classe principal para realizar a extração de informação aberta (OIE)
    a partir de um documento processado pelo spaCy.
    """
    # Dependências que compõem um sintagma nominal (sujeito ou complemento)
    _NOMINAL_PHRASE_DEPS = {"nummod", "advmod", "nmod", "amod", "dep", "det", "case", "flat", "flat:name", "punct"}

    # Dependências que podem fazer parte de uma locução verbal
    _RELATION_VERB_DEPS = {"aux", "aux:pass", "xcomp"}

    # Modificadores que podem se juntar à relação (ex: pronomes clíticos)
    _RELATION_MODIFIER_DEPS = {"expl:pv"}

    # Advérbios comuns que modificam o verbo e devem ser incluídos na relação
    _RELATION_ADVERBS = {"não", "ja", "ainda", "também", "nunca"}

    # Dependências que tipicamente iniciam um complemento
    _COMPLEMENT_HEAD_DEPS = {"obj", "iobj", "xcomp", "ccomp", "advcl", "obl"}

    # Dependências que NUNCA devem ser parte de um complemento (pois têm suas próprias funções)
    _COMPLEMENT_IGNORE_DEPS = {'nsubj', 'nsubj:pass', 'csubj', 'csubj:pass'}

    # A busca por complemento deve PARAR ao encontrá-las para evitar que o arg2 se estenda demais.
    _COMPLEMENT_BOUNDARY_DEPS = {"advcl", "ccomp", "mark"}

    # Dependências que identificam um sujeito
    _SUBJECT_DEPS = {"nsubj", "nsubj:pass", "csubj", "csubj:pass"}

    # Verbos que podem ter o sujeito lógico na posição de objeto (voz passiva sintética, etc.)
    _EXISTENTIAL_VERBS = {'haver', 'ocorrer', 'acontecer', 'existir', 'surgir'}

    def __init__(self, config: ExtractorConfig = None):
        self.config = config if config else ExtractorConfig()

    def get_extractions_from_doc(self, doc: Doc) -> List[Extraction]:
        """Processa um documento spaCy e retorna uma lista de todas as extrações encontradas."""
        extractions = []
        for sentence in doc.sents:
            extractions.extend(self.get_extractions_from_sentence(sentence))
        return extractions

    def get_extractions_from_sentence(self, sentence: Span) -> list[Extraction]:
        """
        Extrai triplas de uma única sentença.
        O processo principal itera sobre os tokens, identifica predicados (verbos),
        e a partir deles, busca por sujeitos, relações e complementos.
        """
        final_extractions = []
        processed_tokens = set()

        # 1. Extração baseada em predicados verbais
        for token in sentence:
            if token.i in processed_tokens:
                continue

            is_verb_head = token.pos_ in ['VERB', 'AUX']

            # Ignora verbos que estão em orações relativas, pois eles funcionam como
            # modificadores e a lógica atual não consegue resolver seu sujeito corretamente.
            is_in_relative_clause = token.dep_ in ['acl', 'acl:relcl']

            is_nominal_predicate_root = token.dep_ == 'ROOT' and token.pos_ in ['ADJ', 'NOUN'] and any(
                c.dep_ == 'cop' for c in token.children)

            # A condição principal agora impede o início da extração para verbos em orações relativas.
            if (is_verb_head and not is_in_relative_clause) or is_nominal_predicate_root:
                start_node = token

                if is_nominal_predicate_root:
                    copula_candidates = [c for c in token.children if c.dep_ == 'cop']
                    if not copula_candidates: continue
                    start_node = copula_candidates[0]

                subject_element = self.__find_subject(start_node)

                if subject_element is None and not self.config.hidden_subjects:
                    continue

                if subject_element is None and self.config.hidden_subjects:
                    subject_element = TripleElement()  # Sujeito oculto

                # Extrai a relação e os complementos associados
                for rel_extr, effective_verb in self.__extract_relation_and_conjunctions(subject_element, start_node):
                    final_extractions.extend(self.__extract_complements(rel_extr, effective_verb))
                    if rel_extr.relation and rel_extr.relation.core:
                        for t in rel_extr.relation.get_all_tokens():
                            processed_tokens.add(t.i)

        # 2. Extração baseada em apostos
        if self.config.appositive:
            appositive_extractions = self.__extract_from_appositives(sentence)
            if self.config.appositive_transitivity:
                # Aplica a regra de transitividade usando as extrações de aposto e as já encontradas
                final_extractions.extend(
                    self.__apply_appositive_transitivity(appositive_extractions, final_extractions))
            final_extractions.extend(appositive_extractions)

        # 3. Remove duplicatas e retorna extrações válidas
        unique_extractions = []
        seen = set()
        for extr in final_extractions:
            representation = extr.to_tuple()
            if representation not in seen and extr.is_valid():
                seen.add(representation)
                unique_extractions.append(extr)

        return unique_extractions

    def __find_subject(self, verb_token: Token) -> Optional[TripleElement]:
        """Encontra o sujeito de um determinado verbo, lidando com voz passiva, orações relativas e verbos existenciais."""
        search_node = verb_token
        is_passive = any(c.dep_ == 'aux:pass' for c in search_node.children)

        # Se o token for um auxiliar ou cópula, o sujeito estará ligado ao verbo principal
        if verb_token.dep_ in ['cop', 'aux', 'aux:pass']:
            search_node = verb_token.head
            is_passive = is_passive or any(c.dep_ == 'aux:pass' for c in search_node.children)

        # Busca por sujeito (nsubj, csubj)
        for child in search_node.children:
            if child.dep_ in self._SUBJECT_DEPS:
                # Se o sujeito for um pronome relativo, busca o seu antecedente
                if child.pos_ == 'PRON' and 'Rel' in child.morph.get("PronType", []):
                    return self.__dfs_for_nominal_phrase(child.head, is_subject=True)
                # Se for uma oração subjetiva, trata como um complemento
                if child.dep_.startswith('csubj'):
                    return self.__dfs_for_complement(child, set())
                return self.__dfs_for_nominal_phrase(child, is_subject=True)

        # Lógica para voz passiva e verbos existenciais (ex: "vende-se casas", "há vagas")
        if is_passive or search_node.lemma_ in self._EXISTENTIAL_VERBS:
            for child in search_node.children:
                if child.dep_ == 'obj':
                    return self.__dfs_for_nominal_phrase(child, is_subject=False)

        # Se o verbo estiver em uma oração adjetiva, o sujeito é o núcleo da oração principal
        if search_node.dep_ in ['acl', 'acl:relcl']:
            return self.__dfs_for_nominal_phrase(search_node.head, is_subject=True)

        return None

    def __extract_relation_and_conjunctions(self, subject: Optional[TripleElement], start_node: Token) -> List[
        Tuple[Extraction, Token]]:
        """Extrai a relação base e expande para relações coordenadas (conj)."""
        subject_tokens = {p.i for p in subject.get_all_tokens()} if subject else set()

        base_relation, effective_verb = self.__build_relation_element(start_node, subject_tokens)
        if not base_relation:
            return []

        extraction = Extraction(subject=subject, relation=base_relation)
        extractions_found = [(extraction, effective_verb)]

        # Lógica para tratar conjunções coordenativas de verbos.
        if self.config.coordinating_conjunctions:
            for child in effective_verb.children:
                if child.dep_ == 'conj' and child.pos_ in ['VERB', 'AUX']:
                    # Verifica se o verbo conjugado tem seu próprio sujeito. Se não, herda o da oração principal.
                    has_own_subject = any(c.dep_.startswith("nsubj") for c in child.children)
                    if not has_own_subject:
                        new_relation, new_effective_verb = self.__build_relation_element(child, set())
                        if new_relation:
                            new_extraction = Extraction(subject=subject, relation=new_relation)
                            extractions_found.append((new_extraction, new_effective_verb))
        return extractions_found

    def __extract_complements(self, extraction: Extraction, complement_root: Token) -> List[Extraction]:
        """Extrai os complementos a partir do verbo efetivo da relação."""
        if not extraction.relation or not extraction.relation.core:
            return [extraction] if extraction.subject and extraction.is_valid() else []

        # Tokens já usados no sujeito e na relação não podem fazer parte do complemento
        base_visited = {tok.i for tok in extraction.subject.get_all_tokens()} if extraction.subject else set()
        base_visited.update(tok.i for tok in extraction.relation.get_all_tokens())

        # Identifica as "cabeças" de cada complemento (ex: múltiplos objetos)
        complement_heads = []
        for child in sorted(complement_root.children, key=lambda t: t.i):
            if child.i in base_visited:
                continue
            if child.dep_ in self._COMPLEMENT_HEAD_DEPS:
                complement_heads.append(child)

        # Constrói cada complemento separadamente
        complement_parts = []
        processed_in_this_run = set()
        for head in complement_heads:
            if head.i in processed_in_this_run: continue

            # Ignora as outras cabeças de complemento para não misturar as extrações
            temp_visited = base_visited.union({h.i for h in complement_heads if h.i != head.i})
            component = self.__dfs_for_complement(head, temp_visited)

            if not component.is_empty():
                complement_parts.append(component)
                processed_in_this_run.update(t.i for t in component.get_all_tokens())

        final_extractions = []
        if not complement_parts:
            # Se não houver complementos, mas sujeito e relação são válidos, retorna a extração intransitiva
            if extraction.is_valid():
                final_extractions.append(extraction)
            return final_extractions

        # Cria uma extração com o complemento completo (todos os complementos juntos)
        full_complement = TripleElement()
        for part in complement_parts:
            full_complement.merge(part)

        if not full_complement.is_empty():
            complete_extraction = Extraction(extraction.subject, extraction.relation, full_complement)
            final_extractions.append(complete_extraction)

        # Se houver múltiplos complementos, decompõe em extrações menores.
        if len(complement_parts) > 1 and self.config.coordinating_conjunctions:
            for component in complement_parts:
                if not component.is_empty():
                    decomposed_extraction = Extraction(extraction.subject, extraction.relation, component)
                    final_extractions.append(decomposed_extraction)

        return final_extractions

    def __extract_from_appositives(self, sentence: Span) -> List[Extraction]:
        """
        Extrai fatos de relações de aposto, criando uma relação sintética "é".
        Ex: "João, o carpinteiro, ..." -> (João, é, o carpinteiro)
        """
        extractions = []
        for token in sentence:
            if token.dep_ == 'appos':
                subject_head = token.head
                # Evita extrair apostos de complementos de oração
                if subject_head.dep_ in ['ccomp', 'xcomp']: continue

                subject = self.__dfs_for_nominal_phrase(subject_head, is_subject=True, ignore_appos=True)
                complement = self.__dfs_for_nominal_phrase(token, is_subject=False)

                # Cria uma relação sintética "é"
                relation = TripleElement(text="é")

                if subject and complement:
                    extraction = Extraction(subject, relation, complement)
                    extraction.subject.is_from_appositive = True
                    extractions.append(extraction)
        return extractions

    def __apply_appositive_transitivity(self, appositive_extractions: List[Extraction],
                                        clausal_extractions: List[Extraction]) -> List[Extraction]:
        """
        Aplica a regra de transitividade com base nas extrações de aposto.
        Se (A é B) e (A faz C), infere (B faz C).
        """
        transitive_extractions = []
        for appos_extr in appositive_extractions:
            # A é o sujeito da extração de aposto (ex: "O diretor do hospital")
            # B é o complemento (ex: "Júlio")
            subj_A_core = appos_extr.subject.core
            subj_B = appos_extr.complement

            for clausal_extr in clausal_extractions:
                if clausal_extr.subject and clausal_extr.subject.core == subj_A_core:
                    # Cria a nova extração (B, faz, C)
                    new_extraction = Extraction(
                        subject=subj_B,
                        relation=clausal_extr.relation,
                        complement=clausal_extr.complement
                    )
                    transitive_extractions.append(new_extraction)
        return transitive_extractions

    @classmethod
    def __build_relation_element(cls, start_token: Token, visited_tokens: Set[int]) -> Tuple[
        Optional[TripleElement], Optional[Token]]:
        """Constrói o elemento da Relação a partir de um token verbal inicial."""
        relation = TripleElement(start_token)
        effective_verb = start_token

        # Usa uma pilha para busca em profundidade de partes do verbo
        stack = deque([start_token])
        local_visited = visited_tokens.copy()
        local_visited.add(start_token.i)

        # Se for uma cópula, o predicado nominal também faz parte da relação
        if start_token.dep_ == 'cop':
            effective_verb = start_token.head
            relation.add_piece(effective_verb)
            local_visited.add(effective_verb.i)

        while stack:
            current = stack.pop()
            if current not in relation.get_all_tokens():
                relation.add_piece(current)

            for child in current.children:
                if child.i in local_visited: continue

                is_verb_part = child.dep_ in cls._RELATION_VERB_DEPS and child.pos_ in ['VERB', 'AUX']
                is_rel_adverb = child.dep_ == 'advmod' and child.lemma_.lower() in cls._RELATION_ADVERBS
                is_extra_rel_dep = child.dep_ in cls._RELATION_MODIFIER_DEPS

                if is_verb_part:
                    stack.append(child)
                    local_visited.add(child.i)
                    if child.i > effective_verb.i:
                        effective_verb = child  # O verbo mais à direita é geralmente o principal
                elif is_rel_adverb or is_extra_rel_dep:
                    relation.add_piece(child)
                    local_visited.add(child.i)

        # Garante que o verbo principal (head) de uma cópula ou auxiliar seja incluído
        if effective_verb.dep_ in ['aux', 'aux:pass', 'cop'] and effective_verb.head not in relation.get_all_tokens():
            relation.add_piece(effective_verb.head)
            effective_verb = effective_verb.head

        # Uma relação válida deve conter um verbo
        has_verb = any(t.pos_ in ['VERB', 'AUX'] for t in relation.get_all_tokens())
        return (relation, effective_verb) if has_verb else (None, None)

    @classmethod
    def __dfs_for_nominal_phrase(cls, start_token: Token, is_subject: bool = False,
                                 ignore_appos: bool = False) -> TripleElement:
        """Realiza uma busca em profundidade para construir um sintagma nominal completo."""
        element = TripleElement(start_token)
        stack = deque([start_token])
        local_visited = {start_token.i}

        valid_deps = cls._NOMINAL_PHRASE_DEPS.copy()
        valid_deps.add("conj")
        valid_deps.add("cc")
        if not ignore_appos:
            valid_deps.add("appos")

        while stack:
            current_token = stack.pop()
            for child in sorted(current_token.children, key=lambda t: t.i):
                if child.i in local_visited: continue

                # Heurística para não incluir preposições que iniciam o sujeito (ex: "De o menino...")
                if is_subject and current_token.i == start_token.i and child.dep_ == 'case': continue

                is_valid_dep = child.dep_ in valid_deps
                is_non_verbal_conj = not (child.dep_ == "conj" and child.pos_ in ["VERB", "AUX"])

                if is_valid_dep and is_non_verbal_conj:
                    element.add_piece(child)
                    local_visited.add(child.i)
                    stack.append(child)
        return element

    @classmethod
    def __dfs_for_complement(cls, start_token: Token, visited_indices: set) -> TripleElement:
        """Realiza uma busca em profundidade para construir um complemento."""
        complement = TripleElement(start_token)
        stack = deque([start_token])
        local_visited = visited_indices.copy()
        local_visited.add(start_token.i)

        # Combina as dependências que devem ser ignoradas com as que delimitam uma cláusula
        boundary_and_ignore_deps = cls._COMPLEMENT_IGNORE_DEPS | cls._COMPLEMENT_BOUNDARY_DEPS

        while stack:
            current_token = stack.pop()
            for child in sorted(current_token.children, key=lambda t: t.i):
                if child.i not in local_visited and child.dep_ not in boundary_and_ignore_deps:
                    complement.add_piece(child)
                    local_visited.add(child.i)
                    stack.append(child)
        return complement
