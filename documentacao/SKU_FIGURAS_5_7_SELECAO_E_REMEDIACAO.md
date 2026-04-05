# Seleção do SKU para Figuras 5–7: Como é Escolhido, Problema da Constância e Remediação

## 1. Objetivo deste documento

Descreve como o SKU das **Figuras 5, 6 e 7** é escolhido no pipeline atual, por que séries quase constantes eram problemáticas, e quais **remediações já estão implementadas** em `gerar_figuras_tcc.py`.

---

## 2. Como o SKU de análise é escolhido atualmente

### 2.1 Fase 1: Pool de candidatos (top 300)

- **Origem:** Análise exploratória (`analise_exploratoria_sazonalidade.py`).
- **Critérios:** `pct_zeros <= 30%`, `estoque_medio_geral >= 1.0`, `cv_mensal >= 1e-6`, ordenação por `diferenca_alta_outros`.
- **Saída:** até 300 SKUs; métricas guardadas em `resultados/candidatos_300_metricas.csv`.

### 2.2 Fase 2: Filtros e top 10 elegíveis

Em `_rodar_comparacao_300_selecionar_10()` (`gerar_figuras_tcc.py`):

1. Exclui `teste_constante` (critério em `comparacao_modelos_previsao.py`).
2. Exclui séries de teste com **CV &lt; `CV_TESTE_MIN`** (padrão 5%) — quase constantes.
3. Exclui séries com **amplitude** (`range_teste`) **&lt; `RANGE_TESTE_MIN`** (padrão 20 unidades) — escala ilegível nos gráficos.
4. Exige `diff_mae = max(MAE) - min(MAE) >= EPSILON_MAE_IGUAL` (padrão 0,01) entre **todos** os modelos.
5. Ordena por **menor melhor MAE** entre modelos; mantém os **10** primeiros elegíveis.

### 2.3 Fase 3: Escolha do SKU para Figuras 5–7 (**implementado**)

**Não** se usa apenas `top10_resultados[0]`.

- Calcula-se `diff_mae_top3` = max(MAE) − min(MAE) entre **Holt-Winters**, **ARIMA** e **SARIMA mensal (m=30)**.
- Considera-se um **pool** dos até **30** melhores elegíveis por MAE.
- Entre SKUs com `diff_mae_top3 >= EPSILON_DIFF_MAE_TOP3` (padrão 0,5), ordena-se por **maior** `diff_mae_top3` (desempate: menor MAE).
- Se nenhum atingir o epsilon, usa-se o pool inteiro com a mesma ordenação.
- **Coerência com a Figura 4:** se o SKU representativo da exploratória estiver entre os candidatos ordenados e o seu MAE não for mais de **10% pior** que o melhor do grupo, **prefere-se** esse SKU para as Figuras 5–7.

O parâmetro `sku_figura4` em `salvar_figuras_tcc_multiplos_skus` recebe o SKU **assim escolhido** (que pode coincidir com o da figura 4 quando a regra de preferência se aplica).

### 2.4 Definição de “teste constante”

Em `comparacao_modelos_previsao.py`:

```python
teste_constante = (s_teste < 0.01 or range_teste < 0.01)
```

---

## 3. Problema histórico (por que a remediação foi necessária)

Em SKUs com série de **teste** quase constante ou de **escala** muito pequena:

- Holt-Winters, ARIMA e SARIMA tendem a previsões quase idênticas.
- O “menor MAE” sozinho favorecia esses casos.
- O filtro `diff_mae` entre **todos** os modelos (incluindo Média Móvel) podia mascarar igualdade entre os três modelos das figuras.

Os filtros **`CV_teste`**, **`range_teste`**, **`diff_mae_top3`** e a **preferência pelo SKU da figura 4** endereçam isso.

---

## 4. Coerência Figura 4 vs Figuras 5–7

- **Figura 4:** SKU com maior variação sazonal (`diferenca_alta_outros`), para ilustrar padrão sazonal.
- **Figuras 5–7:** SKU escolhido para **evidenciar diferenças** entre modelos (com restrições de MAE e preferência de narrativa quando o SKU da figura 4 é competitivo).

Podem ser SKUs diferentes; o código **alinha quando possível** pela regra dos 10%.

---

## 5. Evidências e parâmetros gravados

Após a seleção, `modelos/evidencias_orientadora_tcc.py` grava CSVs (ex.: `criterio_selecao_figuras_5_7.json`, `evidencia_arima_sarima_por_sku.csv`) em `resultados/tabelas_tcc/`. Os valores numéricos dos parâmetros (`CV_TESTE_MIN`, `RANGE_TESTE_MIN`, `EPSILON_DIFF_MAE_TOP3`, etc.) vêm das constantes no topo de `gerar_figuras_tcc.py`.

---

## 6. Conclusão

O problema não era erro de cálculo, e sim **critério de seleção**. A lógica atual combina **filtros de qualidade da série de teste**, **diferenciação entre os três modelos exibidos** e **opcionalmente** o mesmo SKU da análise exploratória, para figuras mais informativas e texto do TCC alinhado ao código.

---

**Última atualização:** 05/04/2026
