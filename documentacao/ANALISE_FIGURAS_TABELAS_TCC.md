# Análise: Figuras e Tabelas do TCC vs. Código Atual

Este documento verifica se o repositório **consegue gerar todas as figuras e tabelas** exigidas na parte escrita do TCC e descreve **as alterações já implementadas**.

**Funcionamento atual (resumo):** O script **`gerar_figuras_tcc.py`** gera figuras 1–7, Tabela 2 e o **valor final da ferramenta de elencação** (ranking R(t), U(t), GP(t)). Pipeline: análise exploratória → 300 candidatos → métricas (sem figuras) → filtros (constante/insatisfatório) → 10 melhores por MAE → figuras, relatórios e elencação só para os 10. Figuras 5–7 usam o **melhor dos 10** (menor MAE). Os **modelos preveem estoque (saldo)**, não vendas; GP(t) = soma das previsões de estoque; o terceiro pilar **sinaliza necessidade de reposição**. Saídas: `resultados/figuras_tcc/`, `resultados/tabelas_tcc/`, `resultados/elencacao_final.csv`. Ver `documentacao/COMO_GERAR_FIGURAS_TCC.md` e `documentacao/CRITERIOS_SELECAO_ANALISE_TEMPORAL.md`.

---

## Resumo Executivo (atualizado)

| Item | TCC | Situação atual | Consegue preencher? |
|------|-----|----------------|---------------------|
| **Figura 1** | Evolução temporal do estoque total agregado | `figura_01_evolucao_estoque_total.png` em `resultados/figuras_exploratoria/` | **Sim** |
| **Figura 2** | Distribuição mensal do estoque (boxplots) | `figura_02_distribuicao_mensal.png` | **Sim** |
| **Figura 3** | Estoque médio por mês | `figura_03_estoque_medio_mes.png` | **Sim** |
| **Figura 4** | Série temporal de SKU com maior variação sazonal | `figura_04_serie_temporal_sku.png` — SKU = top‑1 `diferenca_alta_outros` | **Sim** |
| **Figura 5** | Previsão com Holt-Winters | `figura_05_holt_winters_{sku}.png` em `resultados/figuras_modelos/` | **Sim** |
| **Figura 6** | Previsão com ARIMA | `figura_06_arima_{sku}.png` | **Sim** |
| **Figura 7** | Previsão com SARIMA | `figura_07_sarima_{sku}.png` e `previsao_sarima_{sku}.png` | **Sim** |
| **Tabela 1** | Explicação da base de dados | `validacao/gerar_tabelas_tcc.py` → `resultados/tabelas_tcc/tabela_01_base_dados.csv` | **Sim** |
| **Tabela 2** | Desempenho dos modelos (MAE, RMSE, MAPE) | `tabela_02_desempenho_modelos.csv` (comparação ou consolidado) | **Sim** |

---

## 1. Figuras da Análise Exploratória (Figuras 1–4)

### O que o TCC pede

- **Figura 1:** Evolução temporal do **estoque total agregado** ao longo do período.
- **Figura 2:** **Distribuição mensal** do estoque (boxplots por mês).
- **Figura 3:** **Estoque médio por mês** (agregando independente do ano).
- **Figura 4:** **Série temporal de um SKU representativo**, selecionado entre os de **maior variação sazonal**.

### O que o código faz hoje

**Script:** `analises/analise_exploratoria_sazonalidade.py`

- A função `visualizar_padroes_sazonais` gera **uma única figura** 2×2 (`analise_sazonalidade_padroes.png`) com:
  1. **Subplot 1:** Evolução temporal do estoque total diário → equivale à **Figura 1**.
  2. **Subplot 2:** Boxplot por mês → equivale à **Figura 2**.
  3. **Subplot 3:** Estoque médio por mês (barras) → equivale à **Figura 3**.
  4. **Subplot 4:** Série temporal de um SKU → conceito da **Figura 4**.

**Problemas identificados:**

1. **Figuras 1–4 em uma só:** O TCC trata “Figura 1”, “Figura 2”, etc. como **figuras distintas**. Hoje tudo está em um único PNG. Seria necessário **salvar quatro figuras separadas** (ou ao menos exportar subfiguras individuais) para atender ao formato do texto.

2. **Figura 4 – Critério do SKU:** O TCC pede SKU “**selecionado entre aqueles que apresentaram maior variação sazonal**”. No código:
   - `analise_por_sku_individual` calcula `diferenca_alta_outros` e ordena por ela (maior variação sazonal em primeiro).
   - Já existe, portanto, o **ranking** de SKUs por variação sazonal.
   - Porém `visualizar_padroes_sazonais` é chamada **sem** `sku_exemplo` e **sem** `stats_sku`. O subplot 4 usa o **default**: SKU com **mais observações** (`df.groupby('sku').size().idxmax()`), **não** o de maior variação sazonal.
   - Ou seja: **os dados para escolher o SKU certo existem**, mas a **visualização não os utiliza**.

3. **Pasta de saída:** O script salva `analise_sazonalidade_padroes.png` e `relatorio_analise_sazonalidade.txt` no **diretório corrente** (não em `resultados/`). O README prevê resultados em `resultados/`; isso é apenas organização.

**Conclusão (Figuras 1–4):**

- **Conteúdo:** Figuras 1, 2 e 3 estão **implementadas** (como subplots). A Figura 4 também, mas com **critério de SKU errado**.
- **Ajustes necessários (sem mexer ainda, só verificação):**
  - Gerar **Figuras 1, 2 e 3** como arquivos PNG separados (ou equivalentes).
  - Passar para `visualizar_padroes_sazonais` o **SKU com maior `diferenca_alta_outros`** (primeiro de `stats_sku`) para o subplot da Figura 4 e, se desejado, salvar esse subplot como **Figura 4** em arquivo próprio.

---

## 2. Figuras da Modelagem Preditiva (Figuras 5–7)

### O que o TCC pede

- **Figura 5:** Previsão do estoque com o **modelo Holt-Winters** (apenas esse modelo).
- **Figura 6:** Previsão do estoque com o **modelo ARIMA** (apenas esse modelo).
- **Figura 7:** Previsão do estoque com o **modelo SARIMA** (apenas esse modelo).

Cada figura deve ilustrar “Previsão do Estoque com o Modelo X”, i.e. histórico + previsão **por modelo**, em figuras **separadas**.

### O que o código faz hoje

**Scripts relevantes:**

1. **`modelos/comparacao_modelos_previsao.py`**
   - Compara SARIMA (anual e mensal), ARIMA, média móvel e **Holt-Winters** (ExponentialSmoothing).
   - Gera **uma única** figura `comparacao_modelos_{sku}.png` com **todos** os modelos no mesmo gráfico (treino + teste + previsões).
   - **Não** gera figuras individuais por modelo (Holt-Winters só, ARIMA só, SARIMA só).

2. **`previsoes/teste_sarima_produto.py`**
   - Gera `resultados/previsao_sarima_{sku}.png` com **apenas** SARIMA (histórico + previsão).
   - Formato adequado para **Figura 7**.

3. **`modelos/comparacao_top_skus_otimizado.py`**
   - Salva apenas JSON e CSV (métricas por modelo por SKU). **Não gera figuras.**

**Problemas identificados:**

1. **Figura 5 (Holt-Winters):** Não há figura **exclusiva** de previsão Holt-Winters. Só aparece no gráfico comparativo.
2. **Figura 6 (ARIMA):** Idem — não há figura **exclusiva** ARIMA.
3. **Figura 7 (SARIMA):** **Atendida** por `teste_sarima_produto.py` (ou uso equivalente do módulo SARIMA para gerar um PNG só com SARIMA).

**Conclusão (Figuras 5–7):**

- **Figura 7:** **Sim** — já existe geração de figura “só SARIMA”.
- **Figuras 5 e 6:** **Não** — o código não gera figuras **separadas** “só Holt-Winters” e “só ARIMA”. Seria preciso **novas rotinas de plot** (ou reaproveitar a lógica de `comparacao_modelos_previsao` e `teste_sarima_produto`) para exportar duas figuras adicionais, uma por modelo.

---

## 3. Tabela 1 – Explicação da Base de Dados

### O que o TCC pede

**Tabela 1.** Explicação da base de dados utilizada.

Colunas sugeridas: **Variável** | **Descrição da variável** | **Código e rótulo da variável**.

O texto do TCC deixa a tabela em branco (“...”); o preenchimento é manual.

### O que o código faz hoje

- **Fontes:** `historico_estoque` (sku, created_at, saldo) e `venda_produtos` (sku, created_at, quantidade, valor_unitario, custo_unitario, margem_proporcional, etc.).
- **Uso:** `data_wrangling/dw_historico.py`, `validacao/validar_extracao_vendas.py`, `validacao/calcular_metricas_elencacao.py`, leituras em `previsoes/` e `modelos/`.
- Não há **script** que gere a Tabela 1; ela é **documental**.

**Conclusão (Tabela 1):**

- **Sim** — dá para preencher a Tabela 1 com as variáveis das tabelas `historico_estoque` e `venda_produtos`, suas descrições e códigos/formatos. Basta redigir a partir do esquema e do uso no código. Nenhuma alteração de código é necessária para “gerar” a tabela; o que existe já sustenta o preenchimento.

---

## 4. Tabela 2 – Desempenho dos Modelos (MAE, RMSE, MAPE)

### O que o TCC pede

**Tabela 2.** Desempenho dos modelos de previsão por métrica de erro.

Resumo dos resultados **médios** (ou análogos) para os principais modelos, usando **MAE**, **RMSE** e **MAPE**.

### O que o código faz hoje

- **`comparacao_modelos_previsao.py`:** Calcula MAE, RMSE e MAPE por modelo (por SKU). Imprime e grava em `relatorio_comparacao_{sku}.txt`. Não gera CSV em formato de “tabela resumo” para o TCC.
- **`comparacao_top_skus_otimizado.py`:**
  - Salva `metricas_{sku}.csv` (por SKU e modelo) e `metricas_consolidadas.csv` (todos os SKUs e modelos).
  - Gera `relatorio_consolidado.txt` com **“ESTATISTICAS POR MODELO”**: MAE médio, RMSE médio, MAPE médio por modelo.
- Ou seja: **os dados existem** (por modelo e, no consolidado, médias por modelo).

**Conclusão (Tabela 2):**

- **Sim** — é possível montar a Tabela 2 a partir de:
  - `resultados/resultados_comparacao/metricas_consolidadas.csv`, ou
  - `relatorio_consolidado.txt` (bloco “ESTATISTICAS POR MODELO”).
- Basta **agregar** por modelo (média de MAE, RMSE, MAPE) e **formatar** em tabela (Excel, LaTeX, etc.). Um script auxiliar poderia exportar diretamente uma tabela “Modelo | MAE | RMSE | MAPE”, mas os dados já permitem o preenchimento.

---

## 5. Checklist por Item

| # | Item | Geração automática? | Dados existem? | Ação para preencher |
|---|------|---------------------|----------------|----------------------|
| 1 | **Figura 1** – Evolução estoque total | Subplot apenas | Sim | Salvar subplot como figura única |
| 2 | **Figura 2** – Boxplots mensais | Subplot apenas | Sim | Salvar subplot como figura única |
| 3 | **Figura 3** – Estoque médio por mês | Subplot apenas | Sim | Salvar subplot como figura única |
| 4 | **Figura 4** – Série SKU maior variação sazonal | Subplot com SKU errado | Sim (stats_sku) | Usar top-1 `diferenca_alta_outros` e opcionalmente separar figura |
| 5 | **Figura 5** – Holt-Winters | Não | Sim (previsões) | Criar rotina de figura só Holt-Winters |
| 6 | **Figura 6** – ARIMA | Não | Sim (previsões) | Criar rotina de figura só ARIMA |
| 7 | **Figura 7** – SARIMA | Sim (`previsao_sarima_*.png`) | Sim | Usar como está |
| 8 | **Tabela 1** – Base de dados | Não (manual) | Sim (código/schema) | Preencher manualmente |
| 9 | **Tabela 2** – MAE, RMSE, MAPE | CSV/relatório (não tabela formatada) | Sim | Agregar por modelo e formatar |

---

## 6. Resumo das Lacunas

1. **Figuras 1–4:** Conteúdo presente em um único PNG; TCC exige figuras **separadas**. Figura 4 usa SKU por **número de observações**, não por **maior variação sazonal**.
2. **Figuras 5 e 6:** Não existem figuras **exclusivas** para Holt-Winters e ARIMA; só o comparativo geral e a figura só SARIMA.
3. **Tabela 1:** Preenchimento manual; dados disponíveis no projeto.
4. **Tabela 2:** Dados disponíveis (CSV/relatório); falta só agregar e formatar como tabela do TCC.

---

## 7. Conclusão

- **Tabela 1 e Tabela 2:** **Sim** — é possível preencher ambas. Tabela 1 manualmente; Tabela 2 a partir dos CSVs/relatórios existentes.
- **Figura 7:** **Sim** — já existe geração de figura “Previsão do Estoque com o Modelo SARIMA”.
- **Figuras 1–4:** **Parcial** — os gráficos existem, mas em figura única e (no caso da Fig 4) com critério de SKU incorreto. Ajustes no script de análise exploratória permitiriam atender ao TCC.
- **Figuras 5 e 6:** **Não** — o repositório hoje **não** gera figuras exclusivas para Holt-Winters e ARIMA. Seria necessário implementar essa geração (ou estender a comparação de modelos para exportar figuras por modelo).

**Em resumo:** As alterações foram **implementadas**. Todas as figuras e tabelas do TCC podem ser geradas pelos scripts atuais.

---

## 8. Estrutura de saída (pastas específicas)

- **`resultados/figuras_exploratoria/`** — Figuras 1–4 e figura combinada (`analise_sazonalidade_padroes.png`).
- **`resultados/figuras_modelos/`** — Figuras 5, 6, 7, `comparacao_modelos_{sku}.png`, `previsao_sarima_{sku}.png`.
- **`resultados/tabelas_tcc/`** — Tabela 1 (base de dados) e Tabela 2 (desempenho dos modelos).
- **`resultados/resultados_comparacao/`** — Saídas do `comparacao_top_skus_otimizado` (JSON, CSV consolidado).
- **`resultados/`** — Relatórios TXT, métricas de elencação, etc.

---

**Referência:** Parte escrita do TCC (figuras e tabelas citadas).  
**Última atualização:** 25/01/26 (alterações implementadas)
