# Análise do Extrator de Informação Aberta (OIE)

Este documento detalha a heurística e os algoritmos empregados pelo script Python para extração de triplas (sujeito; relação; objeto). A análise abrange o processo de identificação de cada componente da tripla, o tratamento de estruturas linguísticas complexas e os procedimentos de sanitização.

## Tabela de Heurísticas de Extração

A tabela a seguir detalha o passo a passo do processo de extração, descrevendo as regras (heurísticas) e os atributos do spaCy (dependências, classes gramaticais) que justificam cada decisão do algoritmo.

### **Parte 1: Encontrando o Sujeito (`__find_subject`)**

O processo inicia a partir de um token verbal (predicado) e busca seu argumento principal (sujeito).

| Heurística / Passo | Atributos spaCy Relevantes (Token Filho do Verbo) | Descrição e Justificativa | Exemplo |
| :--- | :--- | :--- | :--- |
| **1. Sujeito Nominal Direto** | `dep_` in `_SUBJECT_DEPS` (`nsubj`, `csubj`, etc.) | O algoritmo busca um filho direto do verbo (ou de seu núcleo) com uma dependência de sujeito. Esta é a forma mais canônica de identificar um sujeito. | "<u>O menino</u> **chutou** a bola."<br> `chutou` -> `menino` (`dep_: nsubj`) |
| **2. Sujeito em Voz Passiva** | `dep_ == 'aux:pass'` (no verbo) e `dep_ == 'obj'` (no sujeito lógico) | Se o verbo possui um auxiliar passivo (`foi`, `é`), o sujeito gramatical (`nsubj:pass`) é o agente da passiva. O script, no entanto, busca o `obj` do verbo principal, que é o sujeito lógico (paciente). | "<u>As casas</u> **foram vendidas**."<br> `vendidas` -> `casas` (`dep_: obj`) |
| **3. Sujeito em Verbos Existenciais** | `verbo.lemma_` in `_EXISTENTIAL_VERBS` (`haver`, `ocorrer`) e `dep_ == 'obj'` | Para verbos que indicam existência, o sujeito lógico frequentemente aparece como objeto direto na análise sintática. | "**Havia** <u>muitos problemas</u>."<br> `Havia` -> `problemas` (`dep_: obj`) |
| **4. Resolução de Pronome Relativo** | `pos_ == 'PRON'` e `'Rel' in morph.get("PronType")` | Se o sujeito encontrado é um pronome relativo (ex: "que"), o verdadeiro sujeito é o `head` (antecedente) desse pronome. | "O menino <u>que</u> **correu**..."<br> `correu` -> `que` (`dep_: nsubj`). O algoritmo sobe para o `head` de "que", que é "menino". |
| **5. Sujeito de Oração Adjetiva** | `verbo.dep_` in `['acl', 'acl:relcl']` | Se o verbo faz parte de uma oração que modifica um nome (adjetiva), o sujeito desse verbo é o nome que está sendo modificado (`verbo.head`). | "Vi o menino **correndo** na rua."<br> O sujeito de `correndo` (`dep_: acl`) é seu `head`, "menino". |
| **6. Sujeito Oculto / Impessoal** | `config.hidden_subjects == True` ou Verbo Impessoal | Se configurado, ou se o verbo é impessoal (3ª pessoa sem `nsubj`), o extrator cria um sujeito vazio para capturar o evento. | "**Choverá** amanhã."<br> Sujeito é criado como vazio: `( ; choverá; amanhã)` |
| **7. Sujeito Oracional** | `dep_ == 'csubj'` | Quando o sujeito é uma oração inteira, ele é tratado como um complemento, usando a função `__dfs_for_complement` para extrair todo o sintagma. | "**Vencer o jogo** é importante."<br> O sujeito de `é` é o sintagma `Vencer o jogo`. |

---

### **Parte 2: Encontrando a Relação (`__build_relation_element`)**

A relação é construída em torno do verbo, expandindo-se para incluir locuções verbais, advérbios e partículas.

| Heurística / Passo | Atributos spaCy Relevantes (Token Filho do Verbo) | Descrição e Justificativa | Exemplo |
| :--- | :--- | :--- | :--- |
| **1. Núcleo Verbal** | `token.pos_` in `['VERB', 'AUX']` | O ponto de partida da extração é sempre um verbo ou auxiliar. | "Ele **comeu**." -> Relação: `comeu` |
| **2. Expansão para Auxiliares** | `dep_` in `_RELATION_VERB_DEPS` (`aux`, `aux:pass`) | Inclui verbos auxiliares que formam tempos compostos ou voz passiva, criando uma locução verbal. A busca é feita via DFS. | "Ele **tinha comido**."<br> `comido` -> `tinha` (`dep_: aux`). Relação: `tinha comido` |
| **3. Expansão para Verbo de Ligação (Cópula)** | `dep_ == 'cop'` | O verbo de ligação (`ser`, `estar`) é o início da relação. O `head` da cópula (o predicativo) será tratado como a raiz do complemento. | "Ele **é** feliz."<br> Relação: `é`. O `head` "feliz" será a raiz do complemento. |
| **4. Inclusão de Advérbios Relevantes** | `dep_ == 'advmod'` e `lemma_` in `_RELATION_ADVERBS` | Advérbios específicos como "não", "já", "nunca" são semanticamente cruciais para a relação e são incorporados a ela. | "Ele **não comeu** ainda."<br> `comeu` -> `não` (`dep_: advmod`). Relação: `não comeu` |
| **5. Inclusão de Partículas Pronominais** | `dep_` in `_RELATION_MODIFIER_DEPS` (`expl:pv`) | Partículas como o "se" em verbos pronominais são incluídas na relação. | "**Vende-se** a casa."<br> `Vende` -> `se` (`dep_: expl:pv`). Relação: `Vende-se` |
| **6. Expansão para `xcomp`** | `dep_ == 'xcomp'` | Inclui complementos oracionais com sujeito não expresso, que funcionam como parte da locução verbal. | "Ele **quer sair**."<br> `quer` -> `sair` (`dep_: xcomp`). Relação: `quer sair` |

---

### **Parte 3: Encontrando o Complemento (`__extract_complements`)**

Após definir sujeito e relação, o algoritmo busca os argumentos restantes (objetos, adjuntos).

| Heurística / Passo | Atributos spaCy Relevantes (Token Filho do Verbo) | Descrição e Justificativa | Exemplo |
| :--- | :--- | :--- | :--- |
| **1. Identificação das "Cabeças" do Complemento** | `dep_` in `_COMPLEMENT_HEAD_DEPS` (`obj`, `iobj`, `obl`, `xcomp`, `advmod`) | O algoritmo identifica todos os filhos diretos do verbo que iniciam um complemento. Isso permite capturar múltiplos objetos e adjuntos. | "Ele deu <u>um livro</u> <u>para Maria</u>."<br> `deu` -> `livro` (`dep_: obj`), `Maria` (`dep_: obl`). |
| **2. Predicativo do Sujeito** | `relação.core.dep_ == 'cop'` | Se a relação é uma cópula, o complemento principal é o `head` dessa cópula (o predicativo do sujeito). | "Ele é <u>um bom médico</u>."<br> `é` (`dep_: cop`) -> `head`: `médico`. Complemento: `um bom médico`. |
| **3. Construção do Sintagma** | `__dfs_for_complement` | A partir de cada "cabeça", uma busca em profundidade (DFS) é realizada para montar o sintagma completo, agregando modificadores (`amod`, `nmod`, `det`, etc.). | "Ele comprou <u>um carro azul novo</u>."<br> A partir de `carro` (`obj`), o DFS anexa `um`, `azul` e `novo`. |
| **4. Parada em Limites de Oração** | `dep_` in `_COMPLEMENT_BOUNDARY_DEPS` (`mark`) | A busca DFS para ao encontrar marcadores de subordinação como "que" ou "se". Isso evita que o complemento se estenda para a oração seguinte. | "Eu disse <u>isso</u> **antes que** ele chegasse."<br> O complemento de `disse` para em `isso`, não incluindo "antes que...". |
| **5. Ignorar Dependências de Sujeito** | `dep_` in `_COMPLEMENT_IGNORE_DEPS` (`nsubj`, `csubj`) | A busca ignora explicitamente tokens que são sujeitos de outras orações para não os anexar erroneamente como complemento. | "Ele quer <u>que o menino vença</u>."<br> O complemento de `quer` é a oração inteira, mas o DFS não incluirá `menino` como parte de um sintagma nominal simples. |

---

## Módulos de Expansão e Inferência

O extrator utiliza módulos específicos para lidar com estruturas complexas, gerando extrações mais ricas ou inferindo novas triplas.

### **Módulo de Aposto (`__extract_from_appositives`)**

- **Heurística:** Identifica uma relação de equivalência a partir de apostos explicativos.
- **Gatilho:** Um token com `dep_ == 'appos'`.
- **Processo:**
    1.  O `head` do token com `dep_ == 'appos'` é definido como **Sujeito**.
    2.  O token `appos` e seu sintagma são definidos como **Complemento**.
    3.  Uma **Relação** sintética "é" é criada.
- **Exemplo:** "Júlio, <u>o diretor do hospital</u>, anunciou a decisão."
    - `diretor` (`dep_: appos`), `head`: `Júlio`.
    - **Extração:** (Júlio; é; o diretor do hospital)

### **Módulo de Transitividade de Aposto (`__apply_appositive_transitivity`)**

- **Heurística:** Se (A é B) e (A faz C), então infere-se que (B faz C).
- **Gatilho:** `config.appositive_transitivity == True`.
- **Processo:**
    1.  Pega uma extração de aposto, como `(A; é; B)`.
    2.  Busca em todas as outras extrações um padrão onde `A` é o sujeito, como `(A; R; C)`.
    3.  Se encontrado, gera uma nova extração substituindo `A` por `B`: `(B; R; C)`.
- **Exemplo:**
    - Extração 1 (Aposto): `(Júlio; é; o diretor)`
    - Extração 2 (Cláusula): `(O diretor; demitiu; o funcionário)`
    - **Extração Inferida:** `(Júlio; demitiu; o funcionário)`

### **Módulo de Conjunção Coordenativa (`__process_conjunction`, etc.)**

- **Heurística:** Desmembra e distribui argumentos em estruturas coordenadas.
- **Gatilho:** Tokens com `dep_ == 'conj'` e conectores com `dep_ == 'cc'` ("e", "ou").
- **Processo:**
    1.  **Verbos Coordenados:** "Ele <u>lavou</u> e <u>passou</u> a roupa."
        - O algoritmo identifica `passou` como `conj` de `lavou`.
        - Cria duas estruturas de Relação: `(Ele; lavou)` e `(Ele; passou)`.
        - O complemento "a roupa", ligado a `passou`, é distribuído para `lavou`, que não tem complemento próprio.
        - **Extrações:** `(Ele; lavou; a roupa)`, `(Ele; passou; a roupa)`.
    2.  **Complementos Coordenados:** "Ele comprou <u>bananas</u>, <u>peras</u> e <u>maçãs</u>."
        - O algoritmo identifica `peras` e `maçãs` como `conj` de `bananas`.
        - Se configurado, gera extrações decompostas para cada item.
        - **Extrações:** `(Ele; comprou; bananas, peras e maçãs)`, `(Ele; comprou; bananas)`, `(Ele; comprou; peras)`, `(Ele; comprou; maçãs)`.

### **Módulo de Oração Subordinada (`__extract_complements`)**

- **Heurística:** Trata orações subordinadas (`ccomp`, `advcl`) como complementos complexos ou como extrações aninhadas.
- **Gatilho:** Um token com `dep_` em `_SUBORDINATE_CLAUSE_DEPS`.
- **Processo:**
    1.  O algoritmo tenta encontrar um sujeito para o verbo da oração subordinada.
    2.  **Caso 1: Sujeito Encontrado.** A oração é tratada como uma **sub-extração**.
        - **Exemplo:** "Ele disse <u>que o menino chegou</u>."
        - O verbo `chegou` (`dep_: ccomp`) tem seu próprio sujeito, `menino`.
        - **Extração Principal:** `(Ele; disse; que)`
        - **Sub-extração:** `(o menino; chegou; )`
    3.  **Caso 2: Sem Sujeito Encontrado.** A oração inteira é tratada como um único complemento verbal.
        - **Exemplo:** "Ele saiu <u>sem avisar</u>."
        - O verbo `avisar` (`dep_: advcl`) não tem sujeito explícito.
        - **Extração:** `(Ele; saiu; sem avisar)`

---

## Construção de Sintagmas (DFS)

A montagem dos elementos da tripla é feita com algoritmos de busca em profundidade (DFS) que navegam pela árvore de dependências.

| Função | Propósito | Dependências Seguidas | Lógica de Parada / Exclusão |
| :--- | :--- | :--- | :--- |
| **`__dfs_for_nominal_phrase`** | Construir sintagmas nominais (Sujeitos, Objetos). | `amod`, `nmod`, `det`, `case`, `nummod`, `flat`, `conj`, `appos`. | Ignora `conj` verbais. Tem uma regra especial para não incluir preposição (`case`) no início de um sujeito. |
| **`__dfs_for_complement`** | Construir complementos de forma mais genérica. | Quase todas as dependências. | Para explicitamente ao encontrar `_COMPLEMENT_BOUNDARY_DEPS` (`mark`). Ignora `_COMPLEMENT_IGNORE_DEPS` (`nsubj`, `csubj`) para não invadir outras orações. |

---

## Sanitização e Validação Final

Antes de retornar o resultado, as extrações passam por um processo de limpeza e validação.

| Etapa | Função Responsável | Descrição | Exemplo |
| :--- | :--- | :--- | :--- |
| **1. Limpeza de Saída** | `TripleElement.get_output_tokens` | Remove pontuações e conjunções coordenativas (`cc`) do início e do fim do texto de cada elemento da tripla. | `", e o menino ,"` se torna `"o menino"`. |
| **2. Validação da Extração** | `Extraction.is_valid` | Verifica se uma extração é semanticamente coerente. As regras são: <br> 1. Deve ter Sujeito e Relação. <br> 2. A Relação deve conter um verbo (a menos que seja sintética como "é"). <br> 3. O Sujeito não pode ser apenas um pronome relativo. | A tripla `(que; comeu; a maçã)` seria invalidada pela regra 3. |
| **3. Remoção de Duplicatas** | `get_extractions_from_sentence` | Converte cada extração válida em uma tupla imutável e a armazena em um `set` para garantir que apenas extrações únicas sejam retornadas. | Se `(Ele; comprou; bananas)` for gerada duas vezes, apenas uma será mantida. |