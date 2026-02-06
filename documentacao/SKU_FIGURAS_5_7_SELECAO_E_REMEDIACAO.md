# Seleção do SKU para Figuras 5–7: Como é Escolhido, Problema da Constância e Remediação

## 1. Objetivo deste documento

Este documento descreve em detalhes como o SKU usado nas Figuras 5, 6 e 7 (comparação de modelos de previsão) é escolhido, por que surge o problema de séries quase constantes e como remediá-lo, mantendo coerência com o restante do trabalho e priorizando um SKU que evidencie as diferenças entre os modelos.

---

## 2. Como o SKU de análise é escolhido atualmente

O fluxo de seleção tem três fases principais.

### 2.1 Fase 1: Pool de candidatos (top 300)

- **Origem:** Análise exploratória (`analise_exploratoria_sazonalidade.py`).
- **Função:** `_top_n_eligible(stats_sku, n=300, ...)`.
- **Critérios:**
  - `pct_zeros <= 30%` (até 30% de dias zerados);
  - `estoque_medio_geral >= 1.0`;
  - `cv_mensal >= 1e-6` (variabilidade mensal não nula);
  - Ordenação por `diferenca_alta_outros` (maior variação sazonal primeiro).
- **Saída:** lista de até 300 SKUs usada como pool para comparação de modelos.

### 2.2 Fase 2: Filtros e top 10 elegíveis

- **Função:** `_rodar_comparacao_300_selecionar_10()` em `gerar_figuras_tcc.py`.
- **Para cada candidato da fase 1:**
  1. **Exclusão por teste constante:** se `teste_constante == True`, o SKU é descartado.
  2. **Exclusão por métricas insatisfatórias:** se `diff_mae = max(MAE) - min(MAE) < 0.01`, o SKU é descartado (todos os modelos praticamente iguais).
  3. Os demais entram na lista de elegíveis, ranqueados pelo menor MAE entre todos os modelos.
- **Saída:** top 10 SKUs com menor MAE entre os elegíveis.

### 2.3 Fase 3: Escolha do SKU para Figuras 5–7

- **Critério atual:** `best_of_10 = top10_resultados[0]['sku']`.
- **Interpretação:** o SKU com menor MAE entre os 10 melhores é usado para gerar as Figuras 5, 6 e 7.
- **Função:** `salvar_figuras_tcc_multiplos_skus(..., sku_figura4=str(best_of_10))`.
- **Observação:** o parâmetro `sku_figura4` leva o nome do SKU da Figura 4, mas hoje recebe `best_of_10` (SKU da comparação), não necessariamente o SKU representativo da análise exploratória.

### 2.4 Definição de “teste constante”

Em `comparacao_modelos_previsao.py`:

```python
teste_constante = (s_teste < 0.01 or range_teste < 0.01)
```

- `s_teste`: desvio padrão da série de teste.
- `range_teste`: amplitude (max - min) da série de teste.
- **Considerado constante:** desvio &lt; 0,01 **ou** amplitude &lt; 0,01.

---

## 3. Por que estamos tendo o problema da constância

### 3.1 Lacuna no critério de constância

O critério atual considera constante apenas séries com `range_teste < 0.01` ou `s_teste < 0.01`.  

Exemplo real (SKU 2922):

- **Série de teste:** 28 dias com valor 13, 2 dias com valor 12.
- **range_teste = 1** → não é excluído por `range_teste < 0.01`.
- **s_teste ≈ 0,26** → não é excluído por `s_teste < 0.01`.
- **CV_teste ≈ 2%** (variabilidade baixa).

Assim, séries quase constantes (ex.: 93% dos dias iguais) podem passar no filtro de constância e ainda terem MAE baixo, o que as favorece na seleção por menor MAE.

### 3.2 Por que o critério “menor MAE” favorece séries quase constantes

Em séries quase constantes:

1. ARIMA, SARIMA e Holt-Winters tendem a convergir para previsão quase constante (ex.: último valor).
2. As previsões dos três modelos ficam praticamente idênticas.
3. O MAE fica baixo, pois qualquer modelo que repita o valor dominante acerta a maioria dos pontos.
4. O SKU com menor MAE costuma ser justamente um desses SKUs com teste quase constante.

### 3.3 Por que o filtro `diff_mae < 0.01` não evita isso

O filtro usa `diff_mae = max(MAE) - min(MAE)` entre **todos** os modelos (incluindo Média Móvel).  

No SKU 2922:

- SARIMA, ARIMA e Holt-Winters: MAE ≈ 0,067.
- Média Móvel: MAE ≈ 0,78.
- `diff_mae ≈ 0,71 > 0.01` → o SKU continua elegível.

Ou seja: o filtro exige que **algum** modelo seja diferente, mas não garante que os modelos das Figuras 5–7 (SARIMA, ARIMA, Holt-Winters) sejam diferentes entre si. Nos SKUs quase constantes, esses três costumam ter MAE praticamente iguais.

### 3.4 Resultado visual nas Figuras 5–7

- Holt-Winters, ARIMA e SARIMA prevêem quase o mesmo valor em todos os dias.
- As três figuras exibem curvas quase idênticas.
- A comparação entre modelos perde sentido.

---

## 4. Coerência do trabalho: Fig 4 vs Fig 5–7

### 4.1 SKU da Figura 4

- **Origem:** Análise exploratória.
- **Função:** `_escolher_sku_representativo()`.
- **Critério:** maior `diferenca_alta_outros` (variação sazonal), com `pct_zeros <= 30%`, `estoque_medio >= 1.0`, `cv_mensal >= 1e-6`.
- **Objetivo:** ilustrar sazonalidade (evolução temporal por SKU).

### 4.2 SKU das Figuras 5–7 (atual)

- **Origem:** Comparação de modelos.
- **Critério:** menor MAE entre os top 10 elegíveis.
- **Objetivo:** ilustrar desempenho dos modelos de previsão.

### 4.3 Incoerência atual

O SKU da Figura 4 pode ser diferente do SKU das Figuras 5–7. Além disso, o critério “menor MAE” tende a escolher SKUs quase constantes, o que prejudica a comparação visual entre Holt-Winters, ARIMA e SARIMA.

Para manter coerência, é desejável:

- Usar o mesmo SKU nas Figuras 4 e 5–7 quando possível; e
- Priorizar SKUs em que os modelos realmente se diferenciem.

---

## 5. Como remediar: SKU que evidencie as diferenças entre modelos

### 5.1 Princípios da solução

1. **Variabilidade na série de teste:** garantir que o período de teste tenha variação relevante (CV ou amplitude acima de um limite).
2. **Diferença entre os modelos das Fig 5–7:** priorizar SKUs em que Holt-Winters, ARIMA e SARIMA tenham MAE distintos.
3. **Coerência com a análise exploratória:** alinhar, quando fizer sentido, o SKU das Fig 5–7 ao SKU representativo da Figura 4 (ou a outro critério exploratório).
4. **Filtro adicional para quase constância:** excluir séries em que o teste seja quase constante, mesmo que não atinjam o critério estrito atual.

### 5.2 Opção A: Filtrar séries quase constantes na série de teste

**Idéia:** além de `teste_constante`, criar um filtro para séries quase constantes.

**Critério sugerido:**

- Excluir SKU se `CV_teste < X%` (ex.: 5% ou 10%).
- Ou: excluir se `range_teste / mean_teste < Y` (ex.: amplitude &lt; 10% da média).

**Efeito:** reduz a chance de escolher SKUs em que todos os modelos tenham desempenho similar por causa da baixa variabilidade do teste.

### 5.3 Opção B: Priorizar variabilidade entre os três modelos das Fig 5–7

**Idéia:** entre os elegíveis, priorizar SKUs em que Holt-Winters, ARIMA e SARIMA tenham MAE claramente diferentes.

**Critério sugerido:**

- Para cada SKU elegível, calcular:
  - `mae_top3 = [MAE(Holt-Winters), MAE(ARIMA), MAE(SARIMA)]`
  - `diff_mae_top3 = max(mae_top3) - min(mae_top3)`
  - Ou: `cv_mae_top3 = std(mae_top3) / mean(mae_top3)`
- Ranquear elegíveis por `diff_mae_top3` (ou `cv_mae_top3`) decrescente.
- Escolher o SKU com maior diferença entre os três modelos, dentro de um grupo aceitável de MAE (ex.: entre os 10 ou 20 melhores por MAE).

**Efeito:** aumenta a chance de obter figuras com curvas visualmente distintas entre os modelos.

### 5.4 Opção C: Usar o SKU representativo da Figura 4 (quando elegível)

**Idéia:** usar o mesmo SKU da Figura 4 nas Figuras 5–7, desde que ele seja elegível e não quase constante.

**Critério sugerido:**

1. Obter `sku_representativo` da análise exploratória (Figura 4).
2. Verificar se esse SKU está em `top10_resultados` (e portanto foi processado na comparação).
3. Verificar se `CV_teste >= X%` e `diff_mae_top3 >= epsilon`.
4. Se sim: usar `sku_representativo` para Fig 5–7.
5. Se não: usar fallback (ex.: Opção B ou A+B).

**Efeito:** mantém coerência narrativa entre análise exploratória e comparação de modelos.

### 5.5 Opção D: Combinar critérios (recomendação)

**Fluxo sugerido:**

1. **Filtro de quase constância:** entre os candidatos, manter apenas aqueles com `CV_teste >= 5%` (ou critério equivalente).
2. **Pool elegível:** aplicar os filtros atuais (`teste_constante`, `diff_mae >= 0.01`).
3. **Score composto:** para cada elegível, calcular algo como:
   - `score = w1 * (-MAE_normalizado) + w2 * diff_mae_top3_normalizado`
   - onde `MAE_normalizado` e `diff_mae_top3_normalizado` são escalas padronizadas.
4. **Prioridade de coerência:** se o SKU da Figura 4 estiver no pool elegível e tiver `CV_teste >= 5%` e `diff_mae_top3 >= epsilon`, usá-lo para Fig 5–7.
5. **Fallback:** caso contrário, escolher o SKU com maior `diff_mae_top3` entre os elegíveis (ou com maior score composto).

---

## 6. Resumo das alterações sugeridas (sem implementar código)

### 6.1 No critério de constância

- Manter `teste_constante = (s_teste < 0.01 or range_teste < 0.01)`.
- Adicionar filtro de quase constância: ex.: excluir se `CV_teste < 5%` (valor ajustável).

### 6.2 Na seleção do SKU para Fig 5–7

- Substituir o critério único “menor MAE” por uma lógica que priorize `diff_mae_top3` (ou variabilidade entre Holt-Winters, ARIMA e SARIMA).
- Incluir prioridade para o SKU da Figura 4 quando elegível e não quase constante.

### 6.3 Documentação e narrativa do TCC

- Descrever explicitamente o critério de seleção do SKU das Figuras 5–7.
- Justificar a escolha de um SKU com variabilidade suficiente para evidenciar diferenças entre os modelos.
- Mencionar que séries quase constantes foram excluídas ou despriorizadas para evitar figuras pouco informativas.

---

## 7. Conclusão

O problema atual não é erro de cálculo, mas de critério de seleção:

- O SKU é escolhido pelo menor MAE.
- Séries quase constantes tendem a ter MAE baixo e passam pelos filtros atuais.
- Em séries quase constantes, Holt-Winters, ARIMA e SARIMA produzem previsões quase idênticas, gerando figuras redundantes.

A solução passa por:

1. Filtrar séries de teste quase constantes.
2. Priorizar SKUs em que os três modelos das Fig 5–7 tenham desempenhos distintos (`diff_mae_top3` elevado).
3. Usar o SKU da Figura 4 quando possível, mantendo coerência entre análise exploratória e comparação de modelos.

Com isso, as Figuras 5–7 passam a ilustrar de forma mais clara as diferenças entre os modelos, mantendo a coerência metodológica do trabalho.
