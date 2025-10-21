# Documentação da Heurística de Extração do DPToie-Python

Heurística e os algoritmos usado no extrator de Informação Aberta (OIE), a partir de documentos processados pelo spaCy.
O extrator utiliza uma abordagem baseada em dependências sintáticas e busca em profundidade (DFS) para construir os elementos da tripla e gerenciar a complexidade de sentenças com coordenação e subordinação.

## 1. Heurística Geral de Avaliação de Extrações

A avaliação de uma extração é realizada principalmente pelo método `is_valid()` da classe `Extraction`. Uma extração é considerada válida se atender aos seguintes critérios:

*   **Presença de Sujeito e Relação**: A extração principal deve ter um sujeito (ou permitir um sujeito oculto, se configurado) e uma relação não vazios. Extrações que servem apenas como contêineres para sub-extrações válidas são uma exceção.
*   **Verbo na Relação**: A relação, a menos que seja sintética (como \"é\" em apostos), deve conter pelo menos um token com `pos_` de 'VERB' ou 'AUX'.
*   **Sujeito Não Relativo Solitário**: Um sujeito não pode ser composto unicamente por um pronome relativo (`PRON` ou `SCONJ` com `PronType=Rel`), pois estes atuam como conectores e seu antecedente é o verdadeiro sujeito.

Extrações duplicadas, identificadas pela sua representação em tupla (`to_tuple()`), são removidas ao final do processo para garantir a unicidade dos resultados.

## 2. Estrutura de Saída Esperada

As extrações são representadas pela classe `Extraction`, que pode ser convertida em uma tupla ou dicionário para serialização. A estrutura básica de uma extração é:

*   `subject`: Um objeto `TripleElement` representando o sujeito.
*   `relation`: Um objeto `TripleElement` representando a relação.
*   `complement`: Um objeto `TripleElement` representando o complemento.
*   `sub_extractions`: Uma lista de objetos `Extraction` para extrações aninhadas (resultantes de orações subordinadas).

Ao ser convertida para tupla (`to_tuple()`), a saída é `(sujeito_str, relacao_str, complemento_str, sub_extrações_tupla)`. Para dicionário (via `__iter__`), é `{'arg1': '...', 'rel': '...', 'arg2': '...', 'sub_extractions': [...]}`.

## 3. Processo de Extração Passo a Passo

O processo de extração principal (`get_extractions_from_sentence`) itera sobre os tokens de uma sentença, identificando os núcleos de predicados (verbos ou predicados nominais com cópula). A partir desses núcleos, a busca por sujeito, relação e complemento é iniciada.

### 3.1. Algoritmo para Encontrar o Sujeito (`__find_subject`)

O algoritmo para encontrar o sujeito de um verbo (`verb_token`) segue as seguintes heurísticas:

1.  **Verbo Principal**: Se o `verb_token` for um auxiliar (`aux`, `aux:pass`) ou cópula (`cop`), a busca pelo sujeito é redirecionada para o `head` desse token (o verbo principal ou o predicado nominal).
2.  **Dependências Diretas**: Procura por filhos do `search_node` (o verbo principal ou predicado) com as dependências `nsubj`, `nsubj:pass`, `csubj`, `csubj:pass`.
3.  **Pronomes Relativos**: Se um sujeito encontrado for um pronome relativo (`PRON` com `PronType=Rel`), o algoritmo tenta encontrar o antecedente desse pronome (geralmente o `head` do pronome) como o verdadeiro sujeito, usando `__dfs_for_nominal_phrase`.
4.  **Sujeito Oracional (`csubj`)**: Se o sujeito for um `csubj` (sujeito oracional), ele é tratado como um complemento, sendo construído via `__dfs_for_complement`.
5.  **Voz Passiva e Verbos Existenciais**: Para verbos na voz passiva (`aux:pass`) ou verbos existenciais (e.g., 'haver', 'ocorrer', 'existir'), o objeto (`obj`) pode ser interpretado como o sujeito lógico.
6.  **Orações Adjetivas**: Se o verbo estiver em uma oração adjetiva (`acl`, `acl:relcl`), o sujeito é inferido como o núcleo da oração principal (o `head` do verbo).
7.  **Sujeitos Ocultos/Impersonais**: Se nenhum sujeito explícito for encontrado e a configuração `hidden_subjects` estiver desativada, a extração é descartada, a menos que o verbo seja impessoal (3ª pessoa sem sujeito explícito).

**Exemplo:**
*   **Sentença**: \\\"O menino **comeu** a maçã.\\\" 
    *   `__find_subject(comeu)`: Encontra 'menino' com `dep_ = nsubj`.
*   **Sentença**: \\\"A maçã **foi comida** pelo menino.\\\" 
    *   `__find_subject(comida)`: `foi` é `aux:pass`. Redireciona para `comida`. `comida` tem `aux:pass`. Procura `obj`. Não encontra. (Neste caso, o sujeito é 'maçã' via `nsubj:pass`)
*   **Sentença**: \\\"**Há** muitas pessoas na festa.\\\" 
    *   `__find_subject(Há)`: `Há` é um verbo existencial. Procura `obj`. Encontra 'pessoas' com `dep_ = obj`. 'pessoas' é o sujeito.
*   **Sentença**: \\\"O homem **que** **comprou** o carro é rico.\\\" 
    *   `__find_subject(comprou)`: `comprou` tem `dep_ = acl:relcl`. Procura `nsubj`. Encontra 'que'. 'que' é pronome relativo. Busca o `head` de 'que', que é 'homem'. 'homem' é o sujeito.

### 3.2. Algoritmo para Construir a Relação (`__build_relation_element`)

A relação é construída a partir de um `start_token` (geralmente um verbo ou cópula) e expandida para incluir elementos que formam o predicado verbal. Utiliza uma busca em profundidade (DFS) para coletar tokens relacionados:

1.  **Núcleo da Relação**: O `start_token` é o núcleo inicial da relação.
2.  **Verbo Efetivo**: Para cópulas (`cop`), o `effective_verb` é o `head` da cópula (o predicado nominal). Caso contrário, é o `start_token`.
3.  **Componentes Verbais**: Inclui filhos com dependências `aux`, `aux:pass`, `xcomp` (partes de locuções verbais).
4.  **Modificadores da Relação**: Inclui advérbios específicos (`advmod` com lemas como \"não\", \"já\", \"ainda\", \"também\", \"nunca\") e pronomes clíticos (`expl:pv`).
5.  **Validação**: A relação final deve conter pelo menos um token com `pos_` de 'VERB' ou 'AUX' para ser considerada válida.

**Exemplo:**
*   **Sentença**: \\\"Ele **não está comendo** a maçã.\\\" 
    *   `start_token = comendo`.
    *   `__build_relation_element(comendo)`: Inclui `não` (advérbio) e `está` (auxiliar). 
    *   Relação: \\\"não está comendo\\\".
*   **Sentença**: \\\"A casa **foi construída** rapidamente.\\\" 
    *   `start_token = construída`.
    *   `__build_relation_element(construída)`: Inclui `foi` (auxiliar passivo).
    *   Relação: \\\"foi construída\\\".

### 3.3. Algoritmo para Extrair o Complemento (`__extract_complements`)

Este é um dos módulos mais complexos, responsável por identificar e agrupar os complementos de uma relação, incluindo o tratamento de coordenação e subordinação. As heurísticas são:

1.  **Identificação de Cabeças**: Procura por filhos do `complement_root` (o verbo efetivo da relação) com dependências que tipicamente iniciam um complemento: `obj`, `iobj`, `xcomp`, `obl`, `advmod`, `nmod`, `ROOT` (para predicados nominais de cópula).
2.  **Predicados Nominais**: Se a relação for uma cópula (`cop`), o `head` da cópula (o predicado nominal) é considerado a cabeça do complemento.
3.  **Coordenação de Complementos**: Para cada cabeça de complemento, o algoritmo busca por tokens coordenados (`conj`) para formar um sintagma nominal completo (ex: \"banana, pera e maçã\"). A preposição (`case`) do primeiro elemento coordenado é propagada para os demais, se eles não tiverem a própria.
4.  **Construção do Sintagma**: Utiliza `__dfs_for_nominal_phrase` ou `__dfs_for_complement` para construir o `TripleElement` para cada parte do complemento.
5.  **Orações Subordinadas (`ccomp`, `advcl`)**: Se um filho for uma oração subordinada (`ccomp`, `advcl`), o algoritmo verifica se ela possui um sujeito próprio:
    *   **Com Sujeito**: Se um sujeito for encontrado, a oração é tratada como uma sub-extração, e o `mark` (conjunção subordinativa, ex: \"que\") é adicionado ao complemento da extração principal.
    *   **Sem Sujeito**: Se nenhum sujeito for encontrado, a oração é tratada como um complemento normal, sendo construída via `__dfs_for_complement`.
6.  **Distribuição de Complementos Compartilhados**: Se houver múltiplas relações coordenadas (ex: \"comprou e vendeu\") e a última extração tiver um complemento, este complemento pode ser propagado para as extrações anteriores que não possuem complemento, desde que os verbos sejam de ação (`VERB`).
7.  **Decomposição**: Se múltiplos complementos forem encontrados para uma única relação, o extrator pode criar extrações separadas para cada complemento, se a configuração de conjunções coordenativas estiver ativada.

**Exemplo:**
*   **Sentença**: \\\"Ele **comprou** um carro e vendeu uma moto.\\\" 
    *   Extração 1: (Ele; comprou; um carro)
    *   Extração 2: (Ele; vendeu; uma moto)
    *   (Distribuição de complemento não se aplica aqui, pois cada verbo tem seu próprio complemento)
*   **Sentença**: \\\"Ele **gosta** de banana, pera e maçã.\\\" 
    *   `__extract_complements(gosta)`: Encontra 'banana' como cabeça. Busca `conj` e encontra 'pera' e 'maçã'. Propaga 'de' para 'pera' e 'maçã'.
    *   Extração: (Ele; gosta; de banana, pera e maçã)
    *   Se decomposto: (Ele; gosta; de banana), (Ele; gosta; de pera), (Ele; gosta; de maçã)
*   **Sentença**: \\\"Ele **disse** que **iria viajar**.\\\" 
    *   `__extract_complements(disse)`: Encontra 'iria' com `dep_ = ccomp`. `__find_subject(iria)` não encontra sujeito explícito (sujeito oculto 'ele'). Trata 'que iria viajar' como um complemento normal.
    *   Extração: (Ele; disse; que iria viajar)
*   **Sentença**: \\\"Ele **sabe** que **você** **mentiu**.\\\" 
    *   `__extract_complements(sabe)`: Encontra 'mentiu' com `dep_ = ccomp`. `__find_subject(mentiu)` encontra 'você'. Cria sub-extração.
    *   Extração: (Ele; sabe; que) com sub-extração: (você; mentiu; )

### 3.4. Busca em Profundidade (DFS) para Sintagmas

O extrator utiliza duas funções DFS principais para construir os elementos da tripla:

#### `__dfs_for_nominal_phrase`

Usada para construir sujeitos e complementos que são sintagmas nominais. Inicia a partir de um `start_token` e expande para seus filhos com as seguintes dependências:

*   `nummod`, `advmod`, `nmod`, `amod`, `dep`, `det`, `case`, `flat`, `flat:name`, `punct`.
*   `conj` e `cc` (se `ignore_conjunctions` for `False`).
*   `appos` (se `ignore_appos` for `False`).

**Heurística Específica**: Evita incluir preposições (`case`) que iniciam o próprio sujeito (ex: \"De o menino...\"), pois estas são geralmente erros de parsing ou construções que não devem fazer parte do sujeito.

#### `__dfs_for_complement`

Usada para construir complementos de forma mais geral, incluindo orações. Inicia a partir de um `start_token` e expande para seus filhos, mas com regras de parada mais rigorosas:

*   Inclui todos os filhos que não estão em `_COMPLEMENT_IGNORE_DEPS` (`nsubj`, `nsubj:pass`, `csubj`, `csubj:pass`) ou `_COMPLEMENT_BOUNDARY_DEPS` (`mark`). Essas dependências indicam o início de uma nova cláusula ou um elemento que não deve ser parte do complemento atual.

## 4. Módulos de Processamento

### 4.1. Módulo de Aposto (`__extract_from_appositives`)

Este módulo identifica relações de aposto e as transforma em extrações de OIE com uma relação sintética.

*   **Identificação**: Procura por tokens com `dep_ = 'appos'`.
*   **Sujeito**: O `head` do token `appos` é o sujeito da extração.
*   **Relação**: Uma relação sintética \"é\" é criada.
*   **Complemento**: O próprio token `appos` e seus modificadores formam o complemento.
*   **Filtragem**: Apostos que são complementos de orações (`ccomp`, `xcomp`) são ignorados para evitar extrações redundantes ou incorretas.

**Exemplo:**
*   **Sentença**: \\\"João, **o carpinteiro**, construiu a casa.\\\"
    *   `token = carpinteiro` (`dep_ = appos`, `head = João`)
    *   Extração: (João; é; o carpinteiro)

### 4.2. Módulo de Transitividade de Aposto (`__apply_appositive_transitivity`)

Este módulo aplica uma regra de inferência para gerar novas extrações baseadas em apostos e extrações clausais existentes.

*   **Regra**: Se (A é B) e (A faz C), então infere-se (B faz C).
*   **Processo**: Itera sobre as extrações de aposto (A é B). Para cada uma, busca extrações clausais onde o sujeito é A (A faz C). Se encontrada, cria uma nova extração com B como sujeito, a mesma relação e o mesmo complemento de (A faz C).

**Exemplo:**
*   **Extração de Aposto**: (João; é; o carpinteiro)
*   **Extração Clausal**: (João; construiu; a casa)
*   **Extração Transitiva Inferida**: (O carpinteiro; construiu; a casa)

### 4.3. Módulo de Conjunção Coordenativa (`__process_conjunction`, `_is_valid_verbal_conjunction`)

Este módulo lida com verbos coordenados e a distribuição de complementos compartilhados.

*   **Identificação de Verbos Coordenados**: A função `_is_valid_verbal_conjunction` verifica se um token é um verbo (`VERB` ou `AUX`) com dependência `conj`.
    *   **Heurística**: Prioriza conectores como \"e\" e \"ou\". Ignora verbos coordenados que possuem seu próprio sujeito explícito, pois estes iniciariam uma nova extração independente.
*   **Processamento**: Para cada verbo coordenado válido, uma nova extração é criada com o mesmo sujeito da extração principal.
*   **Distribuição de Complementos**: Conforme descrito em 3.3, se a última extração de uma sequência de verbos coordenados tiver um complemento, este pode ser propagado para as extrações anteriores que não possuem complemento, desde que os verbos sejam de ação.

**Exemplo:**
*   **Sentença**: \\\"Ele **leu** e **escreveu** um livro.\\\" 
    *   `__process_conjunction(leu)`: Encontra 'escreveu' como `conj` de 'leu'.
    *   Extração 1: (Ele; leu; )
    *   Extração 2: (Ele; escreveu; um livro)
    *   Após distribuição: Extração 1: (Ele; leu; um livro), Extração 2: (Ele; escreveu; um livro)

### 4.4. Módulo de Oração Subordinada (`__extract_complements`)

Este módulo trata orações subordinadas (`ccomp`, `advcl`) de forma particular, gerando sub-extrações quando apropriado.

*   **Identificação**: Procura por filhos do verbo principal com dependências `ccomp` (complemento de oração) ou `advcl` (cláusula adverbial).
*   **Com Sujeito Próprio**: Se a oração subordinada tiver um sujeito explícito (`__find_subject` retorna um sujeito), ela é processada recursivamente por `__process_conjunction` para gerar uma ou mais `sub_extractions`. O `mark` (conjunção subordinativa, ex: \"que\", \"se\") é adicionado ao complemento da extração principal.
*   **Sem Sujeito Próprio**: Se a oração subordinada não tiver um sujeito explícito, ela é tratada como um complemento normal da extração principal, sendo construída via `__dfs_for_complement`.

**Exemplo:**
*   **Sentença**: \\\"Ele **disse** que **iria viajar**.\\\" 
    *   `disse` tem `iria` como `ccomp`. `iria` não tem sujeito explícito. 
    *   Extração: (Ele; disse; que iria viajar)
*   **Sentença**: \\\"Ele **sabe** que **você** **mentiu**.\\\" 
    *   `sabe` tem `mentiu` como `ccomp`. `mentiu` tem `você` como sujeito. 
    *   Extração: (Ele; sabe; que) com sub-extração: (você; mentiu; )

## 5. Sanitização (`TripleElement.get_output_tokens`)

A sanitização é o processo de limpeza dos tokens que compõem um `TripleElement` (sujeito, relação, complemento) antes de sua representação textual final. Isso garante que a saída seja legível e livre de pontuações e conectores desnecessários nas bordas.

As heurísticas de sanitização incluem:

1.  **Remoção de Parênteses/Colchetes Externos**: Se o primeiro e o último token forem um par de parênteses, colchetes ou chaves (ex: `(`, `)`), eles são removidos.
2.  **Remoção de Pontuação e Conectores Iniciais**: Remove pontuações (exceto parênteses/colchetes) e conjunções coordenativas (`dep_ = 'cc'`) do início do elemento.
3.  **Remoção de Pontuação Finais**: Remove pontuações do final do elemento, exceto aquelas que formam pares com as iniciais (já tratadas no passo 1).

**Exemplo:**
*   **Tokens Originais**: `[('(', 'PUNCT'), ('o', 'DET'), ('carpinteiro', 'NOUN'), (',', 'PUNCT'), ('e', 'CCONJ'), ('etc', 'ADV'), ('.', 'PUNCT'), (')', 'PUNCT')]`
*   **Após Sanitização**: `['o', 'carpinteiro', ',', 'e', 'etc']`
    *   `(` e `)` externos removidos.
    *   `.` final removido.
    *   `,` e `e` internos são mantidos, pois não estão nas bordas após a remoção dos parênteses externos.

Este processo garante que as triplas extraídas sejam concisas e representem a informação de forma clara.\"\n