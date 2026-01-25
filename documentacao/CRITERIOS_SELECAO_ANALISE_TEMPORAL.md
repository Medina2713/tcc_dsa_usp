# Critérios de Seleção para Análise Temporal

Este documento descreve os **critérios de escolha** e o **código** do script que seleciona os melhores produtos (SKUs) para análise de séries temporais, usado no sistema de previsão de estoque e elencação do TCC.

---

## 1. Objetivo e contexto

**Modelos preveem ESTOQUE (saldo), não vendas.** SARIMA, ARIMA, Holt-Winters e Média Móvel são treinados na série histórica de **saldo de estoque** (`historico_estoque`). A previsão informa **unidades em estoque** futuras e, na ferramenta de elencação, **sinaliza necessidade de reposição** (terceiro pilar): estoque previsto baixo → priorizar repor; estoque previsto alto → menor urgência.

Os critérios abaixo selecionam SKUs com **qualidade de série** adequada para esses modelos (mais observações, variabilidade útil, poucas lacunas, estoque relevante). O pipeline TCC ainda ranqueia por **desempenho dos modelos** (MAE) para escolher os 10 melhores.

O script `previsoes/selecionar_top_skus_analise_temporal.py` identifica os **N SKUs com os melhores dados** para modelagem temporal (SARIMA, comparação de modelos, etc.). O foco não é “maior giro” ou “maior venda”, e sim **qualidade dos dados da série histórica de estoque**: mais observações, variabilidade útil, poucas lacunas e estoque relevante.

**Diferença em relação a outros scripts:**

| Script / Pipeline | Critério principal | Uso típico |
|-------------------|--------------------|------------|
| **`gerar_figuras_tcc.py`** (pipeline TCC) | Exploratório → 300 candidatos → métricas → 10 melhores por MAE | Figuras TCC, Tabela 2, relatórios, **elencação final** (ranking) |
| `selecionar_top_skus_analise_temporal.py` | **Qualidade dos dados** (score) para série temporal | Análise SARIMA, comparação de modelos, estudos temporais |
| `comparacao_top_skus.py` | **Giro de estoque** (vendas / estoque médio) | Comparação de modelos nos mais movimentados |
| `teste_sarima_produto.py` | **Um único** “melhor” SKU (observações × CV × média) | Teste rápido em um produto |

---

## 2. Critérios de Qualidade

Cada SKU recebe métricas calculadas sobre sua série de estoque. Essas métricas alimentam um **score de qualidade** e **filtros mínimos**.

### 2.1 Número de observações (`n_observacoes`)

- **O que é:** Quantidade de registros (dias) com saldo de estoque para o SKU.
- **Por que importa:** SARIMA e modelos temporais precisam de história suficiente (ex.: ≥ 30 dias). Mais observações em geral permitem estimar melhor tendência e sazonalidade.
- **No score:** Quanto maior, melhor. Entra com peso **0,3** no score combinado.

### 2.2 Coeficiente de variação (`coeficiente_variacao`, CV)

- **O que é:** CV = desvio padrão / média do estoque. Mede a variabilidade relativa da série.
- **Por que importa:** Séries quase constantes (CV ≈ 0) tendem a virar “random walk” em SARIMA e trazem pouco sinal para previsão. CV moderado indica movimento de estoque (entradas/saídas) útil para o modelo.
- **No score:** Quanto maior o CV, maior a parcela desse critério. Entra como `cv * 100` com peso **0,25**.

### 2.3 Estoque médio (`estoque_medio`)

- **O que é:** Média diária do saldo de estoque do SKU.
- **Por que importa:** SKUs com estoque médio muito baixo (ex.: &lt; 1) ou sempre zero quase não variam e pouco ajudam na análise temporal. Priorizamos itens com nível de estoque “significativo”.
- **No score:** Quanto maior o estoque médio, melhor. Peso **0,2**.
- **Filtro:** `min_estoque_medio` (padrão 1,0) — SKUs abaixo disso são excluídos antes do ranking.

### 2.4 Continuidade temporal — lacunas (`percentual_lacunas`)

- **O que é:** Para cada SKU, a série é “expandida” para frequência diária (`asfreq('D')`) com forward fill. Contam-se os dias sem observação (lacunas). O percentual de lacunas é `(dias sem dado / total de dias no período) * 100`.
- **Por que importa:** Muitos buracos na série atrapalham SARIMA e quebram a hipótese de série regular. Menos lacunas = série mais contínua = melhor para análise temporal.
- **No score:** Quanto **menor** o percentual de lacunas, melhor. Usa-se `(100 - percentual_lacunas)` com peso **0,15**.
- **Filtro:** `max_percentual_lacunas` (padrão 50%) — SKUs com mais lacunas que isso são excluídos.

### 2.5 Densidade de observações (`densidade_observacoes`)

- **O que é:** `n_observacoes / periodo_dias`, em que `periodo_dias` é o número de dias entre a primeira e a última data do SKU.
- **Por que importa:** Dois SKUs podem ter o mesmo `n_observacoes`, mas um espalhado em poucos meses e outro em anos. A densidade favorece séries em que os dados estão menos “espalhados” no tempo.
- **No score:** `densidade_observacoes * 100` com peso **0,1**.

---

## 3. Fórmula do score de qualidade

O **score de qualidade** é uma combinação linear ponderada:

```
score_qualidade =
    n_observacoes           * 0.30   +
    (cv * 100)              * 0.25   +
    estoque_medio           * 0.20   +
    (100 - percentual_lacunas) * 0.15 +
    (densidade_observacoes * 100) * 0.10
```

Os pesos (0,3; 0,25; 0,2; 0,15; 0,1) priorizam, em ordem: **volume de dados**, **variabilidade**, **nível de estoque**, **continuidade** e **densidade**. O ranking final é feito por `score_qualidade` em ordem decrescente.

---

## 4. Filtros mínimos

Antes de aplicar o score, só entram no ranking SKUs que passam em **todos** os filtros:

| Parâmetro | Padrão | Regra |
|-----------|--------|-------|
| `min_observacoes` | 30 | `n_observacoes >= 30` |
| `min_estoque_medio` | 1.0 | `estoque_medio >= 1.0` |
| `max_percentual_lacunas` | 50.0 | `percentual_lacunas <= 50%` |

Ou seja: pelo menos 30 dias de dado, estoque médio ≥ 1 e no máximo 50% de dias sem observação no período.

---

## 5. Estrutura do código

### 5.1 Função `calcular_metricas_qualidade_temporal(df, sku)`

**Responsabilidade:** Calcular todas as métricas de qualidade para **um** SKU.

**Entrada:**

- `df`: DataFrame de estoque com colunas `data`, `sku`, `estoque_atual` (ou `saldo` mapeado para `estoque_atual`).
- `sku`: Código do produto.

**Passos principais:**

1. Filtrar `df` pelo `sku` e ordenar por `data`.
2. Definir `data` como índice para manipulação temporal.
3. Calcular:
   - `n_observacoes`, `estoque_medio`, `estoque_std`, `estoque_min`, `estoque_max`
   - `cv = estoque_std / estoque_medio` (ou 0 se média 0)
   - Série diária com `asfreq('D', method='ffill')`, contar `n_lacunas` e `percentual_lacunas`
   - `periodo_dias` e `densidade_observacoes = n_observacoes / periodo_dias`
4. Calcular `score_qualidade` pela fórmula acima.
5. Retornar um `dict` com todas as métricas (incl. `score_qualidade`).

**Saída:** `dict` ou `None` se não houver dados para o SKU.

---

### 5.2 Função `selecionar_top_skus_analise_temporal(...)`

**Responsabilidade:** Carregar dados, calcular métricas para todos os SKUs, aplicar filtros, ordenar por score e retornar o **top N** SKUs.

**Parâmetros:**

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `caminho_dados` | str | `DB/historico_estoque_atual_processado.csv` | Caminho do CSV processado (saída do data wrangling) |
| `top_n` | int | 10 | Quantidade de SKUs a selecionar |
| `min_observacoes` | int | 30 | Mínimo de observações |
| `min_estoque_medio` | float | 1.0 | Estoque médio mínimo |
| `max_percentual_lacunas` | float | 50.0 | Percentual máximo de lacunas |

**Passos principais:**

1. Carregar o CSV, converter `data` para `datetime`, garantir coluna `estoque_atual`.
2. Para cada SKU, chamar `calcular_metricas_qualidade_temporal` e juntar numa tabela de métricas.
3. Aplicar os três filtros (`min_observacoes`, `min_estoque_medio`, `max_percentual_lacunas`).
4. Ordenar por `score_qualidade` decrescente e tomar os `top_n` primeiros.
5. Atribuir `ranking` 1, 2, …, N.
6. Escrever:
   - `resultados/top_{top_n}_skus_analise_temporal.csv` — tabela com métricas e ranking.
   - `resultados/lista_top_{top_n}_skus.txt` — lista numerada dos SKUs.

**Retorno:** `pd.DataFrame` com o top N (ou `None` em caso de erro / nenhum SKU elegível).

---

## 6. Uso

### 6.1 Pré-requisito

Os dados de estoque precisam estar no formato processado (saída do data wrangling). Se ainda não rodou:

```bash
python data_wrangling/dw_historico.py
```

O script de seleção espera, por padrão, `DB/historico_estoque_atual_processado.csv`. Ajuste `caminho_dados` se usar outro arquivo ou caminho.

### 6.2 Executar

```bash
python previsoes/selecionar_top_skus_analise_temporal.py
```

Isso usa os padrões (`top_n=10`, filtros acima). Para mudar critérios, edite a chamada em `main()`:

```python
resultado = selecionar_top_skus_analise_temporal(
    caminho_dados='DB/historico_estoque_atual_processado.csv',
    top_n=10,
    min_observacoes=30,
    min_estoque_medio=1.0,
    max_percentual_lacunas=50.0
)
```

### 6.3 Saídas

- **`resultados/top_10_skus_analise_temporal.csv`**  
  Colunas incluem: `ranking`, `sku`, `n_observacoes`, `periodo_dias`, `densidade_observacoes`, `estoque_medio`, `estoque_std`, `estoque_min`, `estoque_max`, `coeficiente_variacao`, `n_lacunas`, `percentual_lacunas`, `score_qualidade`.

- **`resultados/lista_top_10_skus.txt`**  
  Lista simples dos SKUs ordenados pelo ranking, para uso em scripts downstream (ex.: comparação de modelos, elencação).

---

## 7. Pipeline TCC: 300 candidatos → 10 melhores (`gerar_figuras_tcc.py`)

O **gerador de figuras TCC** usa seleção em duas etapas: (1) **candidatos** pela análise exploratória; (2) **10 melhores** por desempenho dos modelos (MAE), após filtrar testes constantes e resultados insatisfatórios.

### 7.1 Seleção dos 300 candidatos (análise exploratória)

- **Fonte:** `analises/analise_exploratoria_sazonalidade.py`, função `_top_n_eligible(stats_sku, n=300, ...)`.
- **Critérios:** Mesmos da análise exploratória para o top 10:
  - `pct_zeros` ≤ 30% (máx. 30% de dias com estoque zerado).
  - `estoque_medio_geral` ≥ 1,0 (ou relaxado para 0,5 / 0,2 se necessário).
  - `cv_mensal` ≥ 1e-6 (exclui séries praticamente constantes).
- **Ordenação:** Por `diferenca_alta_outros` (maior variação entre meses de alta temporada Out/Dez e demais).
- **Saída:** Lista de até **300** SKUs usados na **Fase 1** do pipeline.

### 7.2 Fase 1: Rodada de métricas (sem figuras)

- Para cada um dos **300 candidatos**, roda comparação de modelos (SARIMA m=30, ARIMA, Média Móvel, Holt-Winters). **Não** gera figuras nem relatórios por SKU.
- Salva `resultados/candidatos_300_metricas.csv` (SKU, Modelo, MAE, RMSE, MAPE, teste_constante).

### 7.3 Fase 2: Filtros e escolha dos 10 melhores

- **Exclui** SKUs com **série de teste constante** (`teste_constante`), pois MAE/RMSE/MAPE zerados não são comparáveis.
- **Exclui** SKUs com **resultados insatisfatórios**: todos os modelos com MAE praticamente iguais (diferença &lt; 0,01), pois não há diferenciação entre modelos.
- **Ranqueia** os elegíveis pelo **menor MAE** (melhor modelo do SKU).
- **Seleciona** os **10 melhores** (menor MAE).

### 7.4 Fase 3: Figuras, relatórios e elencação para os 10 melhores

- Gera `comparacao_modelos_*.png`, `relatorio_comparacao_*.txt`, **figura5**, **figura6**, **figura7** e **Tabela 2** somente para os **10** selecionados.
- **Figuras 5–7** usam o **melhor dos 10** (SKU com menor MAE) como representativo, evitando figuras com resultados insatisfatórios ou teste constante.
- **Elencação final:** Calcula R(t), U(t) e GP(t) (soma das previsões de **estoque**) para os 10 melhores, gera o ranking, salva `resultados/elencacao_final.csv` e **retorna** o DataFrame do ranking. Esse é o **valor final da ferramenta de elencação** (priorização para reposição).

### 7.5 Resumo do pipeline

| Etapa | O que faz |
|-------|-----------|
| Exploratória | Top 300 candidatos (zeros ≤ 30%, estoque ok, cv_mensal); Fig 1–4 |
| Fase 1 | Métricas para os 300 (sem figuras); salva `candidatos_300_metricas.csv` |
| Fase 2 | Filtra constante/insatisfatório; ranqueia por MAE; escolhe 10 |
| Fase 3 | Figuras, relatórios, Fig 5–7, Tabela 2 e **elencação final** (ranking → `elencacao_final.csv`) só para os 10 |

**Referência:** `gerar_figuras_tcc.py`, `analises/analise_exploratoria_sazonalidade.py` (`_top_n_eligible`).

---

## 8. Resumo

- **Pipeline TCC:** 300 candidatos (exploratório) → métricas → filtros (sem constante/insatisfatório) → 10 melhores por MAE → figuras, Tabela 2 e **elencação final** (ranking R(t), U(t), GP(t)) só para os 10. O script **retorna** o DataFrame do ranking.
- **Modelos preveem estoque (saldo), não vendas.** GP(t) = soma das previsões de estoque; o terceiro pilar da elencação **sinaliza necessidade de reposição**.
- **Script alternativo** `selecionar_top_skus_analise_temporal.py`: escolha por **score de qualidade** (observações, CV, estoque, lacunas, densidade). Não usa rodada de modelos.
- Nenhum dos dois escolhe por giro ou venda; o foco é **qualidade da série** e, no pipeline TCC, **desempenho dos modelos**.

---

**Referências:**  
- Pipeline TCC: `gerar_figuras_tcc.py`, `analises/analise_exploratoria_sazonalidade.py`  
- Score de qualidade: `previsoes/selecionar_top_skus_analise_temporal.py`  

**Última atualização:** 25/01/26
