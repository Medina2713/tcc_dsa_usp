# Análise: Figuras e Tabelas do TCC vs. Código Atual

Este documento descreve **como o repositório gera hoje** as figuras e tabelas usadas no TCC e onde encontrar as saídas.

**Estado atual (resumo):** O script **`gerar_figuras_tcc.py`** (raiz do repositório) orquestra o fluxo completo: Tabela 1, data wrangling, figuras 1–4 (modo TCC), fase de comparação em até 300 SKUs, seleção dos 10 melhores, figuras 5–7, Tabela 2, elencação final e **CSVs de evidência** em `resultados/tabelas_tcc/`. Os modelos preveem **estoque (saldo)**; GP(t) na elencação é a soma das previsões de estoque. Documentação complementar: `documentacao/COMO_GERAR_FIGURAS_TCC.md`, `documentacao/CRITERIOS_SELECAO_ANALISE_TEMPORAL.md`, `documentacao/SKU_FIGURAS_5_7_SELECAO_E_REMEDIACAO.md`, `documentacao/RESPOSTAS_ORIENTADORA_ANALISE_RESULTADOS.md`.

---

## Resumo executivo

| Item | Onde é gerado / ficheiro | Notas |
|------|---------------------------|--------|
| **Figura 1–4** | `resultados/figuras_tcc/figura1.png` … `figura4.png` | `analises/analise_exploratoria_sazonalidade.py` com `usar_nomes_tcc=True` via pipeline; SKU da figura 4 = maior variação sazonal (critérios em `CRITERIOS_…`). |
| **Figura 5–7** | `resultados/figuras_tcc/figura5.png` … `figura7.png` | Uma figura por modelo (Holt-Winters, ARIMA, SARIMA **mensal m=30**) para **um** SKU; ver seleção abaixo. |
| **Tabela 1** | `resultados/tabelas_tcc/tabela_01_base_dados.md` (+ CSV se aplicável) | `validacao/gerar_tabelas_tcc.py` — chamada no início de `gerar_figuras_tcc.py`. |
| **Tabela 2** | `resultados/tabelas_tcc/tabela_02_desempenho_modelos.csv` | Consolidada a partir da comparação nos 10 SKUs selecionados. |
| **Elencação final** | `resultados/elencacao_final.csv` | Ranking R(t), U(t), GP(t) para os 10 melhores. |
| **Evidências (discussão / orientadora)** | `evidencia_arima_sarima_por_sku.csv`, `taxa_vitoria_modelos_resumo.csv`, `vitoria_modelo_por_sku.csv`, `medias_por_modelo_*.csv`, `criterio_selecao_figuras_5_7.json`, etc. | `modelos/evidencias_orientadora_tcc.py` (invocado por `gerar_figuras_tcc.py`). Regeneração parcial: `python validacao/gerar_evidencias_de_candidatos_csv.py`. |

---

## 1. Figuras 1–4 (análise exploratória)

- **Script:** `analises/analise_exploratoria_sazonalidade.py`, acionado pelo pipeline com dados em `DB/historico_estoque_atual_processado.csv` (ou equivalente gerado pelo wrangling).
- **Saída TCC:** ficheiros **separados** `figura1.png`–`figura4.png` em `resultados/figuras_tcc/`, não apenas um painel 2×2 em pasta alternativa.
- **Figura 4:** SKU representativo por **maior** `diferenca_alta_outros` (variação sazonal), com filtros de zeros e estoque (alinhado ao texto do TCC).

Execução isolada (equivalente ao trecho exploratório do pipeline):

```bash
python analises/analise_exploratoria_sazonalidade.py --tcc
```

---

## 2. Figuras 5–7 (comparação de modelos)

- **Lógica principal:** `modelos/comparacao_modelos_previsao.py` — compara vários modelos; em modo TCC gera PNGs **por modelo** (figuras 5–7) para o SKU indicado.
- **Pipeline TCC:** após métricas em até 300 candidatos e filtros (`teste_constante`, `diff_mae` entre todos os modelos, `CV_teste` mínimo, `range_teste` mínimo, etc.), o SKU das figuras 5–7 **não** é apenas “o primeiro do top 10 por MAE”: escolhe-se no **pool** (até 30 melhores por MAE) o SKU com maior **`diff_mae_top3`** (diferença de MAE entre Holt-Winters, ARIMA e SARIMA mensal), com **preferência pelo SKU da figura 4** se estiver no conjunto e com MAE até 10% pior que o melhor candidato — ver constantes em `gerar_figuras_tcc.py` e `documentacao/SKU_FIGURAS_5_7_SELECAO_E_REMEDIACAO.md`.
- **SARIMA anual (m=365):** só entra na comparação se o treino tiver **≥ 730 dias** (`MIN_DIAS_SARIMA_ANUAL` em `comparacao_modelos_previsao.py`). As figuras 5–7 do TCC usam **SARIMA mensal (m=30)** na figura 7.

---

## 3. Tabela 1 — base de dados

Gerada por código (`validacao/gerar_tabelas_tcc.py`), invocada automaticamente por `gerar_figuras_tcc.py`. Descreve variáveis das bases `historico_estoque` e `venda_produtos` conforme usadas no projeto.

---

## 4. Tabela 2 — desempenho dos modelos

Produzida no fluxo de comparação dos **10** SKUs finais; ficheiro em `resultados/tabelas_tcc/`. Médias agregadas e “taxas de vitória” por SKU aparecem também nos CSVs de evidência listados no resumo executivo.

---

## 5. Ficheiros auxiliares e gráficos fora do modo TCC

- **`resultados/figuras_exploratoria/`** — execuções sem nomes `figura1`…`figura4` (nomenclatura antiga/alternativa).
- **`resultados/figuras_modelos/`** — comparações e figuras por SKU quando se corre `comparacao_modelos_previsao.py` diretamente (nomes podem incluir o código do SKU).
- **`resultados/resultados_comparacao/`** — saídas de `comparacao_top_skus_otimizado.py` (JSON/CSV por SKU, consolidados).

Para o **capítulo do TCC** em formato padronizado, use **`resultados/figuras_tcc/`** e as tabelas em **`resultados/tabelas_tcc/`** após `python gerar_figuras_tcc.py`.

---

## 6. PDF vs. Markdown

Ficheiros como `REspostas_orientadora.pdf` na raiz (se existirem) são **exportações manuais** a partir dos `.md` em `documentacao/`; o pipeline Python **não** regenera PDF automaticamente.

---

**Última atualização:** 05/04/2026
