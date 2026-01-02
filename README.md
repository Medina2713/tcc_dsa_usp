# Sistema de Previsão de Demanda - TCC MBA DSA USP

Sistema completo de previsão de demanda para gestão de estoque usando modelos SARIMA e técnicas de análise de séries temporais.

## 📁 Estrutura do Repositório

```
.
├── sarima_estoque.py          # Módulo principal SARIMA (importar nos scripts)
├── requirements_sarima.txt    # Dependências Python
├── README.md                  # Este arquivo
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
│   └── validacao_walk_forward_sarima.py
│
├── previsoes/                 # Scripts de previsão
│   ├── teste_sarima_produto.py
│   └── teste_elencacao_3_skus.py
│
├── exemplos/                  # Exemplos de uso
│   ├── exemplo_uso_sarima.py
│   └── exemplo_elencacao_completa.py
│
├── documentacao/              # Documentação completa
│   ├── GUIA_RAPIDO.md
│   ├── README_SARIMA.md
│   └── DOCUMENTACAO_TECNICA_FERRAMENTAS.md
│
├── dados/                     # Dados processados intermediários (gerados pelos scripts)
│
└── resultados/                # Todos os resultados (CSV, PNG, relatórios)
    ├── metricas_elencacao.csv
    ├── resultado_elencacao_*.csv
    ├── previsao_sarima_*.png
    └── relatorio_*.txt
```

## 🚀 Início Rápido

### 1. Instalar Dependências

```bash
pip install -r requirements_sarima.txt
```

### 2. Preparar Dados

Os dados de entrada devem estar na pasta `DB/`:
- `DB/historico_estoque_atual.csv` - Histórico de estoque
- `DB/venda_produtos_atual.csv` - Histórico de vendas

Processar dados para formato SARIMA:
```bash
python data_wrangling/dw_historico.py
```

### 3. Executar Análises

#### Análise Exploratória de Sazonalidade
```bash
python analises/analise_exploratoria_sazonalidade.py
```
Resultados salvos em: `resultados/analise_sazonalidade_*.png` e `resultados/relatorio_analise_sazonalidade.txt`

#### Previsão para um Produto
```bash
python previsoes/teste_sarima_produto.py
```
Resultados salvos em: `resultados/previsao_sarima_[SKU].png`

#### Teste de Elencação (3 SKUs)
```bash
python previsoes/teste_elencacao_3_skus.py
```
Resultados salvos em: `resultados/resultado_elencacao_3_skus.csv`

#### Calcular Métricas de Elencação
```bash
python validacao/calcular_metricas_elencacao.py
```
Resultados salvos em: `resultados/metricas_elencacao.csv`

#### Comparar Modelos (Top 10 SKUs)
```bash
python modelos/comparacao_top_skus_otimizado.py
```
Resultados salvos em: `resultados/resultados_comparacao/`

### 4. Validar Extração de Dados

```bash
python validacao/validar_extracao_vendas.py
```

## 📊 Principais Funcionalidades

- ✅ **Previsão de demanda usando SARIMA** - Modelos automáticos com auto_arima
- ✅ **Identificação de padrões sazonais** - Análise de sazonalidade (outubro/dezembro)
- ✅ **Comparação de modelos** - SARIMA, ARIMA, Médias Móveis, Suavização Exponencial
- ✅ **Métricas de desempenho** - MAE, RMSE, MAPE, R², MAE%, RMSE%, Bias
- ✅ **Cálculo de elencação** - Rentabilidade (R(t)), Nível de Urgência (U(t)), Giro Futuro (GP(t))
- ✅ **Validação walk-forward** - Validação temporal dos modelos
- ✅ **Processamento otimizado** - Salvamento incremental, sistema de checkpoint

## 📚 Documentação

Consulte a pasta `documentacao/` para documentação detalhada:

- **GUIA_RAPIDO.md** - Guia rápido de uso
- **README_SARIMA.md** - Documentação técnica do módulo SARIMA
- **DOCUMENTACAO_TECNICA_FERRAMENTAS.md** - Guia completo de ferramentas estatísticas
- **EXPLICACAO_RESULTADOS_SARIMA.md** - Interpretação de resultados SARIMA
- **RESUMO_VALIDACAO_VENDAS.md** - Validação das métricas de elencação

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

**Última atualização**: 2024
