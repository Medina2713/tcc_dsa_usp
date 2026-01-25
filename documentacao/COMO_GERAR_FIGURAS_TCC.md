# Como Gerar as Figuras do TCC

Este documento explica como gerar cada uma das **figuras** (figura1 … figura7) exigidas no TCC, a **Tabela 2** e o **valor final da ferramenta de elencação**. As figuras seguem a nomenclatura do texto: **figura1**, **figura2**, …, **figura7** e são salvas em `resultados/figuras_tcc/`.

---

## 0. Funcionamento atual e razões

- **Modelos preveem ESTOQUE (saldo), não vendas.** SARIMA, ARIMA, Holt-Winters e Média Móvel são treinados na série histórica de **saldo de estoque** (`historico_estoque`). A saída é previsão de **unidades em estoque** por dia. **GP(t)** na elencação = **soma dessas previsões de estoque** no horizonte (ex.: 30 dias).
- **Terceiro pilar da elencação:** A previsão de estoque serve para **sinalizar necessidade de reposição**. Estoque previsto baixo ou tendendo a zero → maior prioridade para repor; estoque previsto alto → menor urgência. Ou seja, a ferramenta responde “preciso repor unidades?” (e com que prioridade).
- **Pipeline 300 → 10:** Seleciona até 300 candidatos (exploratório), roda **métricas** para todos (sem figuras), filtra testes constantes e resultados insatisfatórios, ranqueia por MAE e escolhe os **10 melhores**. Figuras, relatórios e Tabela 2 são gerados **apenas** para esses 10; figuras 5–7 usam o **melhor dos 10** (menor MAE) como SKU representativo.
- **Elencação final:** Ao fim do pipeline, o script calcula **R(t)**, **U(t)** e **GP(t)** para os 10 melhores, gera o **ranking de elencação**, salva `resultados/elencacao_final.csv` e **retorna** o DataFrame desse ranking.
- **Limpeza:** Antes de cada rodada, o script **remove** figuras, tabelas e relatórios de comparação de execuções anteriores (mantém apenas logs com timestamp).
- **CPU:** Uso limitado a ~80% (quando `psutil` está disponível) para não travar outras aplicações.

---

## 1. Pré-requisitos

1. **Dados processados**  
   - Execute o data wrangling para gerar o histórico processado:
     ```bash
     python data_wrangling/dw_historico.py
     ```
   - O script espera `DB/historico_estoque_atual.csv` (ou o arquivo configurado) e gera `DB/historico_estoque_atual_processado.csv` (ou equivalente).

2. **Dependências**  
   - Instale as dependências do projeto:
     ```bash
     pip install -r requirements_sarima.txt
     ```

3. **Execução a partir da raiz**  
   - Rode os comandos a partir da **raiz do repositório** (onde estão `README.md`, `data_wrangling/`, `analises/`, etc.).

---

## 2. Opção A: Script único (recomendado)

O script **`gerar_figuras_tcc.py`** (na raiz do repositório) gera todas as figuras em sequência.

```bash
python gerar_figuras_tcc.py
```

**O que ele faz:**

1. **Limpa** diretórios de saída (figuras, tabelas, relatórios de comparação) de rodadas anteriores.
2. Executa **data wrangling** (se o CSV processado não existir).
3. Cria `resultados/figuras_tcc/` e roda a **análise exploratória** → gera **figura1**, **figura2**, **figura3**, **figura4** (agregados + SKU representativo com maior variação sazonal, zeros ≤ 30%).
4. **Fase 1:** Seleciona até **300 candidatos** (zeros ≤ 30%, estoque ok, cv_mensal) e roda **métricas** (ARIMA, SARIMA m=30, Holt-Winters, Média Móvel) para todos, **sem** figuras nem relatórios. Salva `resultados/candidatos_300_metricas.csv`.
5. **Fase 2:** Filtra testes constantes e resultados insatisfatórios (métricas idênticas entre modelos); ranqueia por **melhor MAE** e escolhe os **10 melhores**.
6. **Fase 3:** Gera **figura5**, **figura6**, **figura7** (SKU com **menor MAE** dos 10), **Tabela 2**, relatórios e gráficos individuais **apenas** para os 10 melhores.
7. **Elencação final:** Calcula R(t), U(t), GP(t) para os 10 melhores, gera o ranking, salva `resultados/elencacao_final.csv` e **retorna** o DataFrame do ranking (valor final da ferramenta de elencação).

Você pode deixar o script rodando; a Fase 1 (até 300 candidatos) pode levar dezenas de minutos; a Fase 3 (10 SKUs) é mais rápida. Ao final, as figuras, a tabela e o CSV de elencação estarão nos diretórios indicados. O script **retorna** o DataFrame do ranking; ao rodar pela linha de comando, exibe uma mensagem com o caminho do CSV e o número de linhas. Veja `documentacao/CRITERIOS_SELECAO_ANALISE_TEMPORAL.md` para os critérios de seleção.

---

## 3. Opção B: Gerar cada figura separadamente

Se quiser rodar **passo a passo** (por script ou para debug):

### 3.1 Figuras 1–4 (análise exploratória)

- **Script:** `analises/analise_exploratoria_sazonalidade.py`
- **Modo TCC:** use o flag `--tcc` para salvar **figura1.png** … **figura4.png** em `resultados/figuras_tcc/`:

```bash
python analises/analise_exploratoria_sazonalidade.py --tcc
```

**O que cada figura é:**

| Arquivo     | Conteúdo |
|------------|----------|
| **figura1** | Evolução temporal do **estoque total agregado** (todos os produtos). |
| **figura2** | **Distribuição mensal** do estoque (boxplots por mês). |
| **figura3** | **Estoque médio por mês** (agregado, independente do ano). |
| **figura4** | **Série temporal** do **SKU representativo** (maior variação sazonal). |

Sem `--tcc`, as figuras vão para `resultados/figuras_exploratoria/` com nomes `figura_01_...`, etc.

---

### 3.2 Figuras 5–7 (modelos de previsão)

- **Script:** `modelos/comparacao_modelos_previsao.py`
- O script escolhe um SKU (por variabilidade, etc.), treina Holt-Winters, ARIMA e SARIMA, e gera as figuras de previsão.
- Para **nomenclatura TCC** (figura5.png … figura7.png em `resultados/figuras_tcc/`), use `--tcc`. Para usar o **mesmo SKU** da figura4, use `--sku=CODIGO`:
  ```bash
  python modelos/comparacao_modelos_previsao.py --tcc --sku=9788538072362
  ```
  (Substitua pelo SKU representativo obtido na análise exploratória.)
- Ou rode o script mestre `gerar_figuras_tcc.py`, que já usa o SKU da figura4 e salva figura5–7 em `resultados/figuras_tcc/`.

**O que cada figura é:**

| Arquivo     | Conteúdo |
|------------|----------|
| **figura5** | Previsão do estoque com o modelo **Holt-Winters** (SKU representativo escolhido automaticamente). |
| **figura6** | Previsão do estoque com o modelo **ARIMA** (SKU representativo escolhido automaticamente). |
| **figura7** | Previsão do estoque com o modelo **SARIMA** (SKU representativo escolhido automaticamente). |

**Nota:** O SKU representativo é escolhido automaticamente como aquele com **maior variabilidade nas métricas** entre modelos, evitando SKUs onde todos os modelos têm métricas idênticas (problema comum quando modelos convergem para Random Walk ou quando a série de teste é constante).

Quando o modo TCC está ativo, essas figuras são salvas em `resultados/figuras_tcc/` como **figura5.png**, **figura6.png**, **figura7.png**. O script mestre `gerar_figuras_tcc.py` ativa esse modo e usa o **mesmo** SKU representativo da figura4.

---

## 4. Onde ficam as figuras e demais saídas

- **Figuras:** `resultados/figuras_tcc/` — `figura1.png`, `figura2.png`, …, `figura7.png` (referências no texto do TCC).
- **Tabela 2:** `resultados/tabelas_tcc/tabela_02_desempenho_modelos.csv`
- **Elencação final:** `resultados/elencacao_final.csv` (ranking com R(t), U(t), GP(t), score_elencacao). É o **valor final da ferramenta de elencação** retornado pelo script.
- **Candidatos 300:** `resultados/candidatos_300_metricas.csv`
- **Logs:** `resultados/logs/log_execucao_YYYYMMDD_HHMMSS.txt`

---

## 5. Agrupamento e SKU representativo

- **Figuras 1–3:** resultados **agregados** (todos os produtos ou totais/médias).
- **Figura 4:** **um** SKU representativo (maior variação sazonal).
- **Figuras 5–7:** **um** SKU representativo (escolhido automaticamente com maior variabilidade nas métricas), uma figura por modelo.

**Importante:** As figuras 5–7 mostram apenas **um único SKU** por figura (não múltiplos subplots). O SKU é escolhido automaticamente para ter métricas mais expressivas e diferenciadas entre os modelos, evitando casos onde todos os modelos têm métricas idênticas.

---

## 6. Resumo rápido

| Objetivo | Comando |
|----------|---------|
| Gerar **todas** as figuras (figura1–7), Tabela 2 e **elencação final** | `python gerar_figuras_tcc.py` |
| Gerar **só** figura1–4 (exploratória) em modo TCC | `python analises/analise_exploratoria_sazonalidade.py --tcc` |
| Gerar figura5–7 | Rodar o script de comparação em modo TCC (ou via `gerar_figuras_tcc.py`) |

---

**Referência:** Figuras do TCC (nomenclatura e conteúdo), pipeline 300→10, elencação e critérios em `documentacao/CRITERIOS_SELECAO_ANALISE_TEMPORAL.md`.  
**Última atualização:** 25/01/26
