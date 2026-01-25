# Sistema de Previsão de Estoque e Elencação - TCC MBA DSA USP

Sistema completo de **previsão de estoque** (saldo) e **elencação de produtos** para reposição, usando modelos SARIMA, ARIMA, Holt-Winters e Média Móvel. Os modelos preveem **unidades em estoque**, não vendas; o **terceiro pilar** da elencação usa a previsão para **sinalizar necessidade de reposição** (estoque previsto baixo → priorizar repor).

## 📁 Estrutura do Repositório

```
.
├── gerar_figuras_tcc.py       # Script mestre TCC: figuras 1–7, Tabela 2, elencação final
├── requirements_sarima.txt    # Dependências Python
├── README.md                  # Este arquivo
├── DB/                        # Dados (historico_estoque, venda_produtos)
│
├── data_wrangling/            # Preparação e limpeza de dados
│   ├── dw_historico.py        # Script principal de data wrangling
│   └── README.md
│
├── analises/                  # Análises exploratórias
│   ├── analise_exploratoria_sazonalidade.py
│   ├── analise_box_jenkins_sarima.py
│   └── README_ANALISE_EXPLORATORIA.md
│
├── modelos/                   # Modelos de previsão e comparação
│   ├── comparacao_modelos_previsao.py
│   ├── comparacao_top_skus_otimizado.py
│   └── README_COMPARACAO_MODELOS.md
│
├── validacao/                 # Scripts de validação e testes
│   ├── validar_extracao_vendas.py
│   ├── calcular_metricas_elencacao.py
│   ├── gerar_tabelas_tcc.py   # Tabela 1 (base de dados) e Tabela 2 (desempenho)
│   └── validacao_walk_forward_sarima.py
│
├── previsoes/                 # Scripts de previsão
│   ├── sarima_estoque.py      # Módulo SARIMA (previsão de ESTOQUE)
│   ├── teste_sarima_produto.py
│   └── teste_elencacao_3_skus.py
│
├── exemplos/                  # Exemplos de uso
│   ├── exemplo_uso_sarima.py
│   └── exemplo_elencacao_completa.py
│
├── documentacao/              # Documentação completa
│   ├── COMO_GERAR_FIGURAS_TCC.md
│   ├── CRITERIOS_SELECAO_ANALISE_TEMPORAL.md
│   ├── DOCUMENTACAO_GERAL_SISTEMA.md
│   ├── GUIA_RAPIDO.md
│   ├── README_SARIMA.md
│   └── DOCUMENTACAO_TECNICA_FERRAMENTAS.md
│
├── dados/                     # Dados processados intermediários
│
└── resultados/                # Figuras, tabelas, elencação, logs
    ├── figuras_tcc/           # figura1.png … figura7.png
    ├── tabelas_tcc/           # tabela_02_desempenho_modelos.csv
    ├── elencacao_final.csv    # Ranking R(t), U(t), GP(t) — valor final da ferramenta
    ├── figuras_modelos/       # comparacao_modelos_*.png
    ├── candidatos_300_metricas.csv
    ├── metricas_elencacao.csv
    ├── resultado_elencacao_*.csv
    └── logs/
```

## 🚀 Início Rápido

### 1. Instalar Dependências

```bash
pip install -r requirements_sarima.txt
```

### 2. Dados de Entrada

Coloque na pasta `DB/`:
- `DB/historico_estoque_atual.csv` — histórico de estoque (sku, created_at, saldo)
- `DB/venda_produtos_atual.csv` — histórico de vendas (para R(t), U(t) na elencação)

### 3. Pipeline TCC (recomendado)

Gera **todas** as figuras (1–7), **Tabela 2** e o **valor final da ferramenta de elencação**:

```bash
python gerar_figuras_tcc.py
```

O script executa data wrangling (se necessário), análise exploratória (figura1–4), pipeline 300 candidatos → 10 melhores (métricas, filtros, figuras 5–7, Tabela 2) e **elencação final** (R(t), U(t), GP(t) → ranking). Salva `resultados/elencacao_final.csv` e **retorna** o DataFrame do ranking. Veja `documentacao/COMO_GERAR_FIGURAS_TCC.md` e `documentacao/CRITERIOS_SELECAO_ANALISE_TEMPORAL.md`.

**Funcionamento e razões:** Os modelos preveem **estoque (saldo)**, não vendas. GP(t) = soma das previsões de estoque; o terceiro pilar **sinaliza necessidade de reposição**. Limpeza de saídas anteriores antes de cada rodada; CPU limitado a ~80% (psutil).

### 4. Outros scripts

#### Data wrangling (isolado)
```bash
python data_wrangling/dw_historico.py
```

#### Análise Exploratória (Figuras 1–4, modo TCC)
```bash
python analises/analise_exploratoria_sazonalidade.py --tcc
```

#### Selecionar Top SKUs para Análise Temporal
```bash
python previsoes/selecionar_top_skus_analise_temporal.py
```
Ver `documentacao/CRITERIOS_SELECAO_ANALISE_TEMPORAL.md`.

#### Teste de Elencação (3 SKUs)
```bash
python previsoes/teste_elencacao_3_skus.py
```

#### Calcular Métricas de Elencação
```bash
python validacao/calcular_metricas_elencacao.py
```

#### Comparar Modelos (Figuras 5–7, Tabela 2)
```bash
python modelos/comparacao_modelos_previsao.py
```
Um SKU: gera Fig 5 (Holt-Winters), 6 (ARIMA), 7 (SARIMA) em `resultados/figuras_modelos/` e Tabela 2 em `resultados/tabelas_tcc/`.

```bash
python modelos/comparacao_top_skus_otimizado.py
```
Vários SKUs: resultados em `resultados/resultados_comparacao/` e Tabela 2 (médias por modelo) em `resultados/tabelas_tcc/`.

*(O pipeline principal `gerar_figuras_tcc.py` já gera figuras 1–7, Tabela 2 e elencação final; ver seção 3.)*

#### Gerar Tabelas do TCC (Metodologia)
```bash
python validacao/gerar_tabelas_tcc.py
```
- **Tabela 1:** Explicação da base de dados (variáveis, descrição, código e rótulo). Sempre gerada em `resultados/tabelas_tcc/`.
- **Tabela 2:** Desempenho dos modelos (MAE, RMSE, MAPE). Usa saída de `comparacao_modelos_previsao` ou `comparacao_top_skus_otimizado` se já executados.

#### Validar Extração de Dados
```bash
python validacao/validar_extracao_vendas.py
```

## 📊 Principais Funcionalidades

- ✅ **Previsão de estoque** (SARIMA, ARIMA, Holt-Winters, Média Móvel) — modelos preveem **estoque (saldo)**, não vendas; terceiro pilar da elencação **sinaliza reposição**
- ✅ **Identificação de padrões sazonais** - Análise de sazonalidade (outubro/dezembro)
- ✅ **Comparação de modelos** - SARIMA, ARIMA, Médias Móveis, Suavização Exponencial
- ✅ **Métricas de desempenho** - MAE, RMSE, MAPE, R², MAE%, RMSE%, Bias
- ✅ **Cálculo de elencação** - Rentabilidade (R(t)), Nível de Urgência (U(t)), Giro Futuro (GP(t))
- ✅ **Validação walk-forward** - Validação temporal dos modelos
- ✅ **Processamento otimizado** - Salvamento incremental, sistema de checkpoint

## 📚 Documentação

Consulte a pasta `documentacao/` para documentação detalhada:

- **COMO_GERAR_FIGURAS_TCC.md** — Como gerar figuras 1–7, Tabela 2 e elencação final; funcionamento e razões do pipeline
- **CRITERIOS_SELECAO_ANALISE_TEMPORAL.md** — Critérios de seleção de SKUs; pipeline 300→10; modelos preveem estoque, terceiro pilar = reposição
- **DOCUMENTACAO_GERAL_SISTEMA.md** — Visão geral do sistema, fluxo de elencação, GP(t) = previsão de estoque
- **README_SARIMA.md** — Módulo SARIMA (previsão de **estoque**)
- **DOCUMENTACAO_TECNICA_FERRAMENTAS.md** — Ferramentas estatísticas (Box-Jenkins, etc.)
- **GUIA_RAPIDO.md** — Guia rápido de uso
- **EXPLICACAO_RESULTADOS_SARIMA.md** — Interpretação de resultados SARIMA
- **RESUMO_VALIDACAO_VENDAS.md** — Validação das métricas de elencação
- **ANALISE_FIGURAS_TABELAS_TCC.md** — Verificação figuras/tabelas vs. TCC

## 🛠️ Uso do Módulo SARIMA

### Importar o Módulo

```python
from sarima_estoque import PrevisorEstoqueSARIMA
```

### Exemplo Básico

```python
# Inicializar previsor
previsor = PrevisorEstoqueSARIMA(horizonte_previsao=30, frequencia='D')

# Carregar dados
df_estoque = pd.read_csv('DB/historico_estoque_atual.csv')

# Processar previsões
resultados = previsor.processar_lote(df_estoque, lista_skus=['SKU1', 'SKU2'])
```

Veja `exemplos/exemplo_uso_sarima.py` para mais exemplos.

## 📝 Estrutura de Dados

### Entrada (DB/)

**historico_estoque_atual.csv:**
- `sku`: Código do produto
- `created_at`: Data (datetime)
- `saldo`: Quantidade em estoque

**venda_produtos_atual.csv:**
- `sku`: Código do produto
- `created_at`: Data da venda (datetime)
- `quantidade`: Quantidade vendida
- `valor_unitario`: Preço de venda unitário
- `custo_unitario`: Custo unitário
- `margem_proporcional`: Margem proporcional (%)

### Saída (resultados/)

Todos os resultados são salvos na pasta `resultados/`:
- **CSV**: Métricas, rankings, previsões
- **PNG**: Gráficos de previsões e análises
- **TXT**: Relatórios em texto

## 🔍 Métricas de Elencação

Conforme Tabela 2.2, o sistema calcula:

1. **R(t) - Rentabilidade**: Média (Valor Unitário - Custo Unitário)
2. **U(t) - Nível de Urgência**: Estoque Atual / Venda Média Diária Histórica
3. **GP(t) - Giro Futuro Previsto**: Soma das Previsões SARIMA (próximos N dias)

## ⚙️ Configuração

### Ajustar Horizonte de Previsão

```python
previsor = PrevisorEstoqueSARIMA(horizonte_previsao=15)  # 15 dias
```

### Ajustar Sazonalidade SARIMA

No arquivo `sarima_estoque.py`, linha ~130:
```python
m=30,  # Sazonalidade mensal (30 dias)
```

## 📌 Notas Importantes

- **Dados mínimos**: Cada SKU precisa de pelo menos 30 observações para treinar o modelo
- **Processamento**: Scripts longos (comparação de modelos) têm sistema de checkpoint
- **Resultados**: Todos os resultados são salvos na pasta `resultados/`
- **Performance**: Scripts otimizados para processamento incremental

## 🐛 Troubleshooting

### Erro: "Dados insuficientes"
- SKU precisa de pelo menos 30 observações históricas

### Erro: "index must be monotonic"
- Dados de estoque têm datas duplicadas ou fora de ordem
- Execute `data_wrangling/dw_historico.py` para limpar dados

### Erro: "ModuleNotFoundError: No module named 'pmdarima'"
- Instale dependências: `pip install -r requirements_sarima.txt`

## 📄 Licença

TCC MBA Data Science & Analytics - USP

## 👤 Autor

Medina2713

---

**Última atualização**: 02/01/26
