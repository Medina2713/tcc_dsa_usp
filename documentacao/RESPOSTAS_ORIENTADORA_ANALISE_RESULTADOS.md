# Respostas à orientadora — Análise e discussão dos resultados (base técnica para o TCC)

Este documento amarra as **questões da orientadora** à **lógica do código** e aos **CSVs gerados** no repositório, para você copiar/adaptar a redação do capítulo de resultados e discussão.

**Mapa pergunta → seção**

| Questão da orientadora | Onde está respondido |
|------------------------|----------------------|
| Tabela 2: ARIMA e SARIMA com os mesmos MAE/RMSE/MAPE — arredondamento? SARIMA sem sazonalidade? Amostra sem sazonalidade? **Histórico inferior a 2 anos influencia?** | **§1** (+ `evidencia_arima_sarima_por_sku.csv`; parágrafo *Comprimento de histórico*) |
| Figuras 5–7: diferença prática ARIMA vs SARIMA; ARIMA “acompanha a queda” | **§2** (+ `resumo_quantitativo_figuras_5_7.csv`, `dados_numericos_figuras_5_7.csv`) |
| Melhor desempenho médio ARIMA/SARIMA: vale em **todos** os SKUs? Média móvel / Holt-Winters em alguns casos? | **§4** (+ `taxa_vitoria_modelos_resumo.csv`, `vitoria_modelo_por_sku.csv`) |
| Inconsistência Resumo (Holt-Winters “melhor”) vs Tabela 2 (ARIMA/SARIMA) | **§5** |

**Arquivos de evidência (atualize após nova execução do pipeline):**

| Arquivo | Conteúdo |
|---------|----------|
| [resultados/tabelas_tcc/evidencia_arima_sarima_por_sku.csv](../resultados/tabelas_tcc/evidencia_arima_sarima_por_sku.csv) | Ordens `(p,d,q)`, `seasonal_order`, AIC, flags de previsões idênticas. **Só é criado após** `python gerar_figuras_tcc.py` completo (com `auditoria_arima_sarima` em cada `comparar_modelos`). Se não existir no repositório, use `heuristica_mae_arima_vs_sarima_por_sku.csv` como substituto parcial. |
| [resultados/tabelas_tcc/heuristica_mae_arima_vs_sarima_por_sku.csv](../resultados/tabelas_tcc/heuristica_mae_arima_vs_sarima_por_sku.csv) | (Opcional) Por SKU: MAE ARIMA vs SARIMA só a partir de `candidatos_300_metricas.csv` — **não** é gerado pelo `gerar_figuras_tcc.py`; criar com `python validacao/gerar_evidencias_de_candidatos_csv.py`. Quando existir `evidencia_arima_sarima_por_sku.csv`, prefira-o (inclui ordens e AIC). |
| [resultados/tabelas_tcc/taxa_vitoria_modelos_resumo.csv](../resultados/tabelas_tcc/taxa_vitoria_modelos_resumo.csv) | Quantos SKUs “venceram” por menor MAE (desempate MAE → RMSE → MAPE). |
| [resultados/tabelas_tcc/medias_por_modelo_todos_candidatos_validos.csv](../resultados/tabelas_tcc/medias_por_modelo_todos_candidatos_validos.csv) | Médias de MAE/RMSE/MAPE no conjunto de candidatos **não constantes**. |
| [resultados/tabelas_tcc/medias_por_modelo_apenas_top10_tcc.csv](../resultados/tabelas_tcc/medias_por_modelo_apenas_top10_tcc.csv) | Mesmas médias restritas aos **10 SKUs** que entram na Tabela 2 / figuras dos 10. |
| [resultados/tabelas_tcc/resumo_quantitativo_figuras_5_7.csv](../resultados/tabelas_tcc/resumo_quantitativo_figuras_5_7.csv) | MAE no horizonte, inclinação da previsão (unid./dia), desvio padrão das previsões — **mesmo SKU** das Figuras 5–7. |
| [resultados/tabelas_tcc/dados_numericos_figuras_5_7.csv](../resultados/tabelas_tcc/dados_numericos_figuras_5_7.csv) | Série dia a dia: real vs previsto por modelo (prova numérica dos gráficos). |
| [resultados/tabelas_tcc/criterio_selecao_figuras_5_7.json](../resultados/tabelas_tcc/criterio_selecao_figuras_5_7.json) | SKU usado nas Fig. 5–7 e parâmetros — gerado no fim da seleção em `gerar_figuras_tcc.py`. |

**Código relevante:**

- Função `construir_auditoria_arima_sarima` e anexo ao `resultados` ao final de `comparar_modelos`: [modelos/comparacao_modelos_previsao.py](../modelos/comparacao_modelos_previsao.py).
- Seleção de candidatos, filtros e SKU das Fig. 5–7: [gerar_figuras_tcc.py](../gerar_figuras_tcc.py) (`_rodar_comparacao_300_selecionar_10`, constantes `CV_TESTE_MIN`, `RANGE_TESTE_MIN`, `EPSILON_DIFF_MAE_TOP3`, etc.).
- Exportação agregada de CSVs: [modelos/evidencias_orientadora_tcc.py](../modelos/evidencias_orientadora_tcc.py).
- Regeneração offline: [validacao/gerar_evidencias_de_candidatos_csv.py](../validacao/gerar_evidencias_de_candidatos_csv.py).

---

## 1. Por que ARIMA e SARIMA têm **exatamente** o mesmo MAE, RMSE e MAPE na Tabela 2?

**Não é “só arredondamento”.** Igualdade em **três** métricas ao mesmo tempo, SKU a SKU e na média do top-10, indica que **as previsões no período de teste são as mesmas** (ou diferem abaixo da precisão numérica), logo os erros são idênticos.

**Mecanismo no código:** o SARIMA mensal é estimado com `pmdarima.auto_arima(..., seasonal=True, m=30)` e o ARIMA com `seasonal=False`. O seletor por AIC pode escolher **ordem sazonal nula** `(P,D,Q) = (0,0,0)` no SARIMA. Nesse caso, a parte sazonal **não entra** no modelo ajustado; se a parte não sazonal coincidir com a do ARIMA, as **previsões coincidem** — o que você vê na Tabela 2 e no arquivo `dados_numericos_figuras_5_7.csv` (valores previstos ARIMA = SARIMA).

**Comprimento de histórico (sim, afeta — e o código já reflete isso em parte):**

- **SARIMA anual (`m=365`)** nem sequer é estimado em `comparar_modelos` quando o treino tem **menos de 730 dias** (~2 anos): constante `MIN_DIAS_SARIMA_ANUAL` em [modelos/comparacao_modelos_previsao.py](../modelos/comparacao_modelos_previsao.py). Ou seja, com menos de 2 anos de dados **não há** comparação sazonalidade-anual vs não; só entram **SARIMA mensal (m=30)** e ARIMA.
- **Com qualquer extensão de série, mas sobretudo com histórico curto**, o critério de informação (AIC/BIC) tende a **penalizar parâmetros sazonais extra**: poucos graus de liberdade e muitos coeficientes sazonais tornam o ajuste instável ou sem ganho de verossimilhança; o passo de seleção do `auto_arima` pode então **manter `(P,D,Q) = (0,0,0)`** no bloco sazonal.
- **Menos observações** também significam **menos ciclos completos** de um padrão `m=30` (aproximação de “mês” em dias) e **mais ruído** em séries de **saldo** (degraus, reposição). Isso **reduz a identificabilidade** de sazonalidade forte ao nível do SKU e **reforça** a convergência para um modelo que se comporta como a **mesma estrutura ARIMA** nos dois ajustes.
- **Conclusão para o TCC:** a equivalência numérica ARIMA/SARIMA é **compatível** com a combinação (i) **sazonalidade nula escolhida** pelo `auto_arima`, (ii) **duração finita do treino** e (iii) natureza da série. **Não** implica que, com mais anos de dados e outro `m`, o SARIMA não pudesse diferir — mas **neste desenho empírico** o efeito do histórico limitado é **plausível e esperado** em metodologia de séries temporais.

**Evidência numérica (snapshot do repositório, SKU das Fig. 5–7 = 7460):** em `resumo_quantitativo_figuras_5_7.csv`, as linhas **ARIMA** e **SARIMA** têm o mesmo `mae_horizonte_teste`, `rmse_horizonte_teste`, mesma `inclinacao_previsao_unid_por_dia` e mesmas médias dos primeiros/últimos 7 dias — ou seja, **curvas idênticas** no horizonte exibido.

**Evidência em lote:** em `evidencia_arima_sarima_por_sku.csv`, entre os **66** SKUs com teste não constante, **62** têm `previsoes_arima_sarima_identicas == True` e **65** têm `sarima_PDQ_todos_zeros == True` — confirma empate de previsões e **componente sazonal nula** `(P,D,Q)=(0,0,0)` na quase totalidade dos casos (substituto parcial: `heuristica_mae_arima_vs_sarima_por_sku.csv` via script de validação, se existir).

**Texto sugerido para o TCC:** a sazonalidade **agregada** (exploratória) pode ser forte no negócio, mas a **série diária de saldo por SKU** é frequentemente dominada por **degraus** (reposição) e **médias móveis implicitas**; o `m=30` é uma aproximação de mês em dias. O `auto_arima` minimiza AIC no **treino** e pode preferir **não** usar sazonalidade estimável sem ganho — **reforçado** quando o treino **não cobre vários anos completos** (ver parágrafo *Comprimento de histórico*). A **equivalência ARIMA/SARIMA** na tabela reflete isso, e não invalida o uso do SARIMA como **procedimento** que *permite* sazonalidade quando os dados a justificam.

---

## 2. Figuras 5, 6 e 7: o ARIMA “acompanha melhor a queda” e o SARIMA parece igual — qual a diferença prática?

Para o **SKU escolhido para as figuras** (critério abaixo), quando ARIMA e SARIMA são **o mesmo vetor de previsão**, **não há diferença prática** entre Fig. 6 e Fig. 7 além do rótulo: a discussão deve **reconhecer isso** ou escolher outro SKU em que `diff_mae_top3` separe bem os três modelos.

**Onde está a diferença real neste SKU (7460):** compare **Holt-Winters** com **ARIMA/SARIMA** usando `resumo_quantitativo_figuras_5_7.csv`:

- **Inclinação da previsão** (`inclinacao_previsao_unid_por_dia`): ARIMA/SARIMA forte **negativa** (queda contínua no horizonte); Holt-Winters **positiva** leve (previsão não acompanha a queda acentuada do real).
- **MAE no horizonte:** Holt-Winters **substancialmente maior** que ARIMA/SARIMA no mesmo arquivo.
- **Estabilidade:** desvio padrão das previsões de Holt-Winters é **menor** que o de ARIMA/SARIMA neste recorte — ou seja, HW fica “mais plano” em torno de um patamar, enquanto ARIMA/SARIMA seguem uma trajetória decrescente mais íngreme (ainda que não capturem quedas bruscas do real).

**Frase modelo:** “No SKU ilustrado, ARIMA e SARIMA coincidem numericamente; a distinção relevante é entre esses modelos e Holt-Winters, com MAE maior e trajetória de previsão menos alinhada à redução do estoque no teste.”

---

## 3. Critério de escolha do SKU das Figuras 5–7 (justificativa metodológica)

Objetivo: evitar SKUs com **teste quase constante** (todos os modelos acertam igual) e favorecer **diferença visível** entre Holt-Winters, ARIMA e SARIMA.

1. Pool de até **300** SKUs pré-filtrados na análise exploratória.  
2. Para cada um, `comparar_modelos` com divisão treino/teste e métricas.  
3. **Exclusões:** `teste_constante`; `cv_teste < CV_TESTE_MIN` (5%); `range_teste < RANGE_TESTE_MIN` (20 un.); `max(MAE)-min(MAE) < EPSILON_MAE_IGUAL` entre todos os modelos.  
4. **Top 10** para figuras comparativas e Tabela 2: menores **melhores MAE** entre elegíveis.  
5. **SKU das Fig. 5–7:** entre os **primeiros `min(30, número de elegíveis)`** (na prática, às vezes menos de 30 se poucos SKUs passam nos filtros), ordena-se por maior `diff_mae_top3` (HW vs ARIMA vs SARIMA); se o SKU representativo da Fig. 4 aparecer nesse conjunto e seu MAE não for >10% pior que o melhor do pool, **prioriza-se** para coerência narrativa. O valor exato está em `criterio_selecao_figuras_5_7.json` → `parametros_script.N_POOL_FIG`.

Constantes em [gerar_figuras_tcc.py](../gerar_figuras_tcc.py): `CV_TESTE_MIN`, `RANGE_TESTE_MIN`, `EPSILON_MAE_IGUAL`, `EPSILON_DIFF_MAE_TOP3`, `N_MELHORES`, `N_CANDIDATOS`.

---

## 4. “ARIMA/SARIMA tiveram melhor desempenho médio” — todos os SKUs se comportaram assim?

**Não.** A média na Tabela 2 é **só sobre os 10 SKUs** selecionados por **baixo erro**; isso não implica que **todo** o universo favoreça ARIMA/SARIMA.

**Taxa de vitória (menor MAE por SKU, candidatos com teste não constante)** — desempate **MAE → RMSE → MAPE**, igual a `modelos/evidencias_orientadora_tcc.py` e a `validacao/gerar_evidencias_de_candidatos_csv.py` (versões antigas usavam `idxmin` no pivot e distorciam empates). Snapshot em `taxa_vitoria_modelos_resumo.csv`:

| Modelo | SKUs vencedores (%) |
|--------|---------------------|
| SARIMA Mensal (m=30) | 60,61% |
| Suavização exponencial (Holt-Winters) | 34,85% |
| Média móvel (7 dias) | 3,03% |
| ARIMA Simples | 1,52% |

Interpretação: **a maior parte das “vitórias” rotuladas como SARIMA coincide com MAE idêntico ao ARIMA** (ver `heuristica_mae_arima_vs_sarima_por_sku.csv`): trata-se do **mesmo desempenho** com dois nomes; o desempate só escolhe um rótulo. A **suavização exponencial vence em parcela relevante** dos SKUs (~35% no snapshot atual), à parte desse bloco.

**Médias no conjunto amplo (66 candidatos válidos, `medias_por_modelo_todos_candidatos_validos.csv`):** a ordenação típica por **MAE médio** é: **ARIMA ≤ SARIMA < Holt-Winters < Média móvel** (valores de referência no CSV versionado: ARIMA e SARIMA ligeiramente melhores que Holt-Winters; MM pior). Ou seja, Holt-Winters **não** é o melhor em média nesse recorte, mas também **não** é o pior — fica **entre** os modelos autorregressivos e a média móvel.

No **subconjunto top-10 da Tabela 2**, `medias_por_modelo_apenas_top10_tcc.csv` mostra médias **piores** para Holt-Winters e MM do que para ARIMA/SARIMA — reflexo do **critério de seleção** dos 10 melhores, não de uma “lei” para todos os produtos.

---

## 5. Inconsistência: Resumo do TCC diz que a **suavização exponencial** foi a melhor em média, mas a **Tabela 2** destaca **ARIMA/SARIMA**

Isso é uma **contradição textual** a corrigir na versão final.

- A **Tabela 2** e `medias_por_modelo_apenas_top10_tcc.csv` refletem o **top-10** escolhido por menor MAE entre elegíveis: nesse recorte, **ARIMA e SARIMA** apresentam menores médias que Holt-Winters.  
- O **Resumo** que afirma superioridade da suavização exponencial pode corresponder a **outra rodada**, **outra amostra de SKUs** (ex.: comparação otimizada por giro em `comparacao_top_skus_otimizado.py`) ou **redação desatualizada**.

**Como harmonizar (escolha uma e mantenha em Resumo, Abstract e Conclusão):**

1. **Se o pipeline oficial é `gerar_figuras_tcc.py`:** alinhe o Resumo à Tabela 2 e explique que a média refere-se ao **conjunto dos 10 SKUs** usados na comparação principal.  
2. **Se a conclusão correta for a suavização exponencial no conjunto amplo:** atualize a Tabela 2 e o critério de seleção dos 10 SKUs, ou apresente **duas** tabelas (médias no pool 300 vs. médias no top-10) com interpretação explícita.

---

## 6. O que fazer após alterar dados ou parâmetros

1. `python gerar_figuras_tcc.py` — recalcula tudo, incluindo `evidencia_arima_sarima_por_sku.csv` e `criterio_selecao_figuras_5_7.json`.  
2. Se só quiser atualizar agregados a partir de `candidatos_300_metricas.csv` já existente: `python validacao/gerar_evidencias_de_candidatos_csv.py` (taxa de vitória com desempate MAE→RMSE→MAPE; **não** usar `idxmin` por coluna em pivots com empates).

---

## 7. Verificação após `python gerar_figuras_tcc.py` (prático + teórico)

**O script não altera este ficheiro `.md` automaticamente.** Os números abaixo foram conferidos com os CSVs após uma execução bem-sucedida (log com `CONCLUIDO`, figuras `figura1.png`–`figura7.png`, `tabela_02_desempenho_modelos.csv`, `evidencia_arima_sarima_por_sku.csv`).

| Verificação | Resultado esperado |
|-------------|-------------------|
| Execução | `resultados/logs/log_execucao_*.txt` termina com `CONCLUIDO`; sem `Traceback`. |
| Figuras | `resultados/figuras_tcc/figura1.png` … `figura7.png`. |
| Evidência ARIMA/SARIMA | `evidencia_arima_sarima_por_sku.csv` existe; colunas `sarima_PDQ_todos_zeros`, `previsoes_arima_sarima_identicas`. |
| Critério Fig. 5–7 | `criterio_selecao_figuras_5_7.json` com `sku_figuras_5_a_7` e `parametros_script`. |

**Snapshot coerente com o repositório (última verificação):**

- **66** SKUs com teste não constante no pool analisado nos agregados; **62** deles com `previsoes_arima_sarima_identicas == True` em `evidencia_arima_sarima_por_sku.csv` — alinha com a §1 (equivalência numérica frequente).
- **65** desses 66 com `sarima_PDQ_todos_zeros == True` (`(P,D,Q)=(0,0,0)` no componente sazonal) — **prova direta** de que o `auto_arima` escolheu sazonalidade nula na maior parte dos casos (teoria: SARIMA colapsa a um comportamento tipo ARIMA sobre a mesma série).
- **Taxa de vitória** no `taxa_vitoria_modelos_resumo.csv` (SARIMA 60,61% / Holt 34,85% / MM 3,03% / ARIMA 1,52%) mantém-se; interpretação na §4 (rótulo SARIMA vs empates com ARIMA).
- **Fig. 5–7** (`resumo_quantitativo_figuras_5_7.csv`): SKU **7460**; ARIMA e SARIMA com métricas idênticas; Holt-Winters com MAE de horizonte maior — coerente com a §2.
- **`parametros_script.N_POOL_FIG`:** pode ser **menor que 30** (ex.: 20) se o número de SKUs **elegíveis** após filtros for inferior a 30; o texto da §3 fala em “até 30” — corresponde a `min(30, len(elegiveis))` no código.

**Teoria (Box–Jenkins / `pmdarima`):** ordem sazonal `(0,0,0)` com `m=30` significa que **nenhum** coeficiente AR/MA/D sazonal foi estimado; as previsões coincidem com um ARIMA puro quando `(p,d,q)` também coincidem — como visto nas linhas de `evidencia_arima_sarima_por_sku.csv` (ex.: `(0,1,0)` para ambos e AIC igual). **Histórico inferior a 2 anos** no treino (típico neste caso) reforça o ponto da §1: o pipeline **não** ajusta SARIMA anual sem ~730 dias de treino; com menos dados, a seleção por AIC favorece ainda mais **parcimônia** no bloco sazonal.

**Prático:** se `heuristica_mae_arima_vs_sarima_por_sku.csv` não existir, não é falha do pipeline; use `evidencia_arima_sarima_por_sku.csv` ou gere a heurística com `python validacao/gerar_evidencias_de_candidatos_csv.py`.

---

*Documento de apoio à revisão do TCC. Atualize os parágrafos com percentagens se regenerar dados; a §7 resume a última conferência cruzada código ↔ CSVs. Documentação do repositório alinhada ao pipeline em **05/04/2026**.*
