# Documentação Geral do Sistema de Previsão de Demanda e Elencação

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [Estrutura do Sistema](#estrutura-do-sistema)
3. [Arquivos e Módulos](#arquivos-e-módulos)
4. [Como Usar o Sistema](#como-usar-o-sistema)
5. [Fluxo de Elencação de Produtos](#fluxo-de-elencacao-de-produtos)
6. [Exemplos Práticos](#exemplos-práticos)
7. [Configurações e Parâmetros](#configurações-e-parâmetros)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Visão Geral

Este sistema foi desenvolvido para **previsão de demanda** e **elencação de produtos** (priorização) em um ambiente de e-commerce de brinquedos. O sistema combina:

- **Modelos SARIMA** para previsão de demanda futura
- **Métricas de negócio** (Rentabilidade, Urgência, Giro) para elencação
- **Análises exploratórias** para identificar padrões sazonais
- **Comparação de modelos** para validação estatística

### Objetivo Principal

Priorizar produtos para compra/reposição com base em três pilares:
1. **Rentabilidade (R(t))**: Valor financeiro (margem de contribuição)
2. **Nível de Urgência (U(t))**: Tempo que o estoque atual dura
3. **Giro Futuro Previsto (GP(t))**: Previsão SARIMA de demanda futura

---

## 📁 Estrutura do Sistema

```
.
├── sarima_estoque.py              # Módulo principal SARIMA
├── requirements_sarima.txt        # Dependências Python
│
├── data_wrangling/                # Preparação de dados
│   ├── dw_historico.py           # Processa histórico de estoque
│   └── README.md
│
├── analises/                      # Análises exploratórias
│   ├── analise_exploratoria_sazonalidade.py
│   ├── analise_box_jenkins_sarima.py
│   └── README_ANALISE_EXPLORATORIA.md
│
├── modelos/                       # Modelos de previsão
│   ├── comparacao_modelos_previsao.py
│   ├── comparacao_top_skus_otimizado.py
│   └── README_COMPARACAO_MODELOS.md
│
├── validacao/                     # Validação e testes
│   ├── validar_extracao_vendas.py
│   ├── calcular_metricas_elencacao.py
│   └── validacao_walk_forward_sarima.py
│
├── previsoes/                     # Scripts de previsão
│   ├── teste_sarima_produto.py
│   └── teste_elencacao_3_skus.py
│
├── exemplos/                      # Exemplos de uso
│   ├── exemplo_uso_sarima.py
│   └── exemplo_elencacao_completa.py
│
├── documentacao/                  # Documentação completa
│
├── dados/                         # Dados processados (gerados)
│
└── resultados/                    # Todos os resultados (CSV, PNG, TXT)
```

---

## 📄 Arquivos e Módulos

### 🔧 Módulo Principal

#### `sarima_estoque.py`
**Descrição**: Módulo principal com a classe `PrevisorEstoqueSARIMA` para previsão de demanda usando modelos SARIMA.

**Classe Principal**: `PrevisorEstoqueSARIMA`

**Métodos Principais**:
- `preparar_serie_temporal()`: Prepara série temporal de estoque por SKU
- `treinar_modelo()`: Treina modelo SARIMA usando auto_arima
- `prever()`: Gera previsões futuras
- `processar_lote()`: Processa múltiplos SKUs

**Como usar**:
```python
from sarima_estoque import PrevisorEstoqueSARIMA

previsor = PrevisorEstoqueSARIMA(horizonte_previsao=30, frequencia='D')
resultados = previsor.processar_lote(df_estoque, lista_skus=['SKU1', 'SKU2'])
```

---

### 📊 Data Wrangling

#### `data_wrangling/dw_historico.py`
**Descrição**: Processa histórico de estoque para formato adequado ao SARIMA.

**Funções Principais**:
- `carregar_dados()`: Carrega CSV de histórico
- `limpar_dados()`: Remove registros inválidos
- `agregar_por_dia()`: Agrega múltiplos registros do mesmo dia
- `criar_series_completas()`: Preenche lacunas temporais
- `processar_historico_estoque()`: Função principal que orquestra o processamento

**Entrada**: `DB/historico_estoque_atual.csv`
**Saída**: `DB/historico_estoque_atual_processado.csv`

**Como usar**:
```bash
python data_wrangling/dw_historico.py
```

---

### 📈 Análises Exploratórias

#### `analises/analise_exploratoria_sazonalidade.py`
**Descrição**: Identifica padrões sazonais nos dados (ex: picos em outubro/dezembro).

**Saídas**:
- Gráficos de sazonalidade: `resultados/analise_sazonalidade_padroes.png`
- Relatório: `resultados/relatorio_analise_sazonalidade.txt`

**Como usar**:
```bash
python analises/analise_exploratoria_sazonalidade.py
```

#### `analises/analise_box_jenkins_sarima.py`
**Descrição**: Análise Box-Jenkins completa para identificar parâmetros SARIMA manualmente.

**Uso**: Para análise técnica avançada dos modelos.

---

### 🤖 Modelos de Previsão

#### `modelos/comparacao_modelos_previsao.py`
**Descrição**: Compara múltiplos modelos de previsão (SARIMA, ARIMA, Médias Móveis, Suavização Exponencial).

**Métricas Calculadas**: MAE, RMSE, MAPE, R², MAE%, RMSE%, Bias

**Saídas**: 
- Gráficos de comparação: `resultados/comparacao_modelos_[SKU].png`
- Relatórios: `resultados/relatorio_comparacao_[SKU].txt`

#### `modelos/comparacao_top_skus_otimizado.py`
**Descrição**: Compara modelos para os top N SKUs com maior giro de estoque.

**Características**:
- ✅ Processamento incremental (salva por SKU)
- ✅ Sistema de checkpoint (pode retomar processamento)
- ✅ Otimizado para performance

**Saídas**: `resultados/resultados_comparacao/`
- `resultado_[SKU].json`
- `metricas_[SKU].csv`
- `relatorio_consolidado.txt`

**Como usar**:
```bash
python modelos/comparacao_top_skus_otimizado.py
```

---

### ✅ Validação

#### `validacao/validar_extracao_vendas.py`
**Descrição**: Valida se o sistema consegue extrair corretamente métricas do arquivo de vendas.

**Valida**:
- Estrutura do CSV de vendas
- Cálculo de Rentabilidade (R(t))
- Cálculo de Nível de Urgência (U(t))
- Quantidade vendida total

**Como usar**:
```bash
python validacao/validar_extracao_vendas.py
```

#### `validacao/calcular_metricas_elencacao.py`
**Descrição**: Calcula todas as métricas necessárias para elencação a partir de `venda_produtos_atual.csv`.

**Métricas Calculadas**:
- Rentabilidade (R(t)) = Média (Valor Unitário - Custo Unitário)
- Margem Proporcional média
- Quantidade Vendida Total
- Venda Média Diária Histórica
- Nível de Urgência (U(t)) = Estoque Atual / Venda Média Diária

**Saída**: `resultados/metricas_elencacao.csv`

**Como usar**:
```bash
python validacao/calcular_metricas_elencacao.py
```

#### `validacao/validacao_walk_forward_sarima.py`
**Descrição**: Validação walk-forward dos modelos SARIMA (validação temporal).

**Uso**: Para validar robustez dos modelos ao longo do tempo.

---

### 🔮 Previsões

#### `previsoes/teste_sarima_produto.py`
**Descrição**: Testa SARIMA em um produto específico (seleciona automaticamente o melhor SKU).

**Saídas**:
- Gráfico: `resultados/previsao_sarima_[SKU].png`
- Informações do modelo no console

**Como usar**:
```bash
python previsoes/teste_sarima_produto.py
```

#### `previsoes/teste_elencacao_3_skus.py`
**Descrição**: Testa sistema completo de elencação para os 3 SKUs com maior movimentação.

**Fluxo**:
1. Identifica top 3 SKUs por quantidade vendida
2. Calcula métricas de vendas (R(t))
3. Gera previsões SARIMA (GP(t))
4. Calcula Nível de Urgência (U(t))
5. Gera ranking de elencação

**Saída**: `resultados/resultado_elencacao_3_skus.csv`

**Como usar**:
```bash
python previsoes/teste_elencacao_3_skus.py
```

---

### 💡 Exemplos

#### `exemplos/exemplo_uso_sarima.py`
**Descrição**: Exemplos práticos de uso do módulo SARIMA.

**Conteúdo**:
- Uso básico do previsor
- Processamento de lote
- Visualização de resultados

#### `exemplos/exemplo_elencacao_completa.py`
**Descrição**: Exemplo completo de fórmula de elencação (com dados simulados).

**Conteúdo**:
- Cálculo de margem de contribuição
- Cálculo de giro de estoque
- Cálculo de risco de ruptura
- Score final de elencação

---

## 🚀 Como Usar o Sistema

### Pré-requisitos

1. **Instalar dependências**:
```bash
pip install -r requirements_sarima.txt
```

2. **Dados de entrada** (na pasta `DB/`):
   - `historico_estoque_atual.csv`: Histórico de estoque (colunas: `sku`, `created_at`, `saldo`)
   - `venda_produtos_atual.csv`: Histórico de vendas (colunas: `sku`, `created_at`, `quantidade`, `valor_unitario`, `custo_unitario`, `margem_proporcional`)

### Fluxo Básico

1. **Preparar dados**:
```bash
python data_wrangling/dw_historico.py
```

2. **Análise exploratória** (opcional):
```bash
python analises/analise_exploratoria_sazonalidade.py
```

3. **Calcular métricas de elencação**:
```bash
python validacao/calcular_metricas_elencacao.py
```

4. **Gerar previsões e elencação**:
```bash
python previsoes/teste_elencacao_3_skus.py
```

---

## 🎯 Fluxo de Elencação de Produtos

A elencação de produtos é o processo de **priorização** baseado em três métricas principais:

### 1. Rentabilidade (R(t))

**Fórmula**: R(t) = Média (Valor de Venda Unitário - Custo de Aquisição Unitário)

**Fonte**: `venda_produtos_atual.csv`

**Cálculo**:
```python
# Agregado por SKU
rentabilidade = valor_unitario_medio - custo_unitario_medio
```

**Interpretação**: Quanto maior, maior a margem de contribuição do produto.

---

### 2. Nível de Urgência (U(t))

**Fórmula**: U(t) = Estoque Atual / Venda Média Diária Histórica

**Fontes**: 
- Estoque atual: `historico_estoque_atual.csv` (último saldo por SKU)
- Venda média diária: `venda_produtos_atual.csv` (média dos últimos 365 dias)

**Cálculo**:
```python
# Venda média diária (últimos 365 dias)
venda_media_diaria = vendas.groupby('sku')['quantidade'].mean()

# Nível de urgência
nivel_urgencia = estoque_atual / venda_media_diaria
```

**Interpretação**: 
- Menor valor = maior urgência (estoque vai acabar logo)
- Ex: U(t) = 5 dias significa que o estoque dura apenas 5 dias na velocidade atual

---

### 3. Giro Futuro Previsto (GP(t))

**Fórmula**: GP(t) = Soma das Previsões SARIMA para os próximos N dias

**Fonte**: Modelo SARIMA treinado com `historico_estoque_atual.csv`

**Cálculo**:
```python
from sarima_estoque import PrevisorEstoqueSARIMA

previsor = PrevisorEstoqueSARIMA(horizonte_previsao=30)
previsao = previsor.prever(serie_temporal, modelo=modelo_treinado)
giro_futuro_previsto = previsao.sum()  # Soma das previsões
```

**Interpretação**: Demanda total prevista para os próximos N dias.

---

### Score Final de Elencação

O score final combina as três métricas com pesos:

```python
score_elencacao = (
    peso_rentabilidade * rentabilidade_normalizada +
    peso_urgencia * urgencia_normalizada +
    peso_giro * giro_normalizado
)
```

**Pesos padrão** (podem ser ajustados):
- `peso_rentabilidade = 0.4` (40%)
- `peso_urgencia = 0.3` (30%)
- `peso_giro = 0.3` (30%)

**Ranking**: Produtos ordenados por score (maior = maior prioridade)

---

## 📝 Exemplos Práticos

### Exemplo 1: Elencação Completa (3 SKUs)

```bash
# Executa teste completo de elencação
python previsoes/teste_elencacao_3_skus.py
```

**Resultado**: 
- Ranking de 3 SKUs
- Todas as métricas calculadas
- CSV salvo em `resultados/resultado_elencacao_3_skus.csv`

---

### Exemplo 2: Calcular Métricas para Todos os SKUs

```bash
# Calcula métricas de elencação
python validacao/calcular_metricas_elencacao.py
```

**Resultado**: `resultados/metricas_elencacao.csv` com:
- Rentabilidade (R(t))
- Margem proporcional média
- Quantidade vendida total
- Venda média diária
- Nível de urgência (U(t))

---

### Exemplo 3: Previsão SARIMA para um Produto

```python
from sarima_estoque import PrevisorEstoqueSARIMA
import pandas as pd

# Carregar dados
df_estoque = pd.read_csv('DB/historico_estoque_atual.csv')
df_estoque['created_at'] = pd.to_datetime(df_estoque['created_at'])
df_estoque['data'] = df_estoque['created_at']
df_estoque['estoque_atual'] = df_estoque['saldo']

# Inicializar previsor
previsor = PrevisorEstoqueSARIMA(horizonte_previsao=30)

# Gerar previsão para um SKU
sku = '9786555521368'
serie = previsor.preparar_serie_temporal(df_estoque, sku)
modelo = previsor.treinar_modelo(serie, sku)
previsao = previsor.prever(serie, modelo=modelo)

print(f"Previsão para próximos 30 dias: {previsao.sum():.0f} unidades")
```

---

### Exemplo 4: Elencação Customizada (Múltiplos SKUs)

```python
import pandas as pd
from sarima_estoque import PrevisorEstoqueSARIMA
from validacao.calcular_metricas_elencacao import calcular_metricas_completas

# 1. Calcular métricas de vendas
df_metricas = calcular_metricas_completas(salvar_resultado=False)

# 2. Selecionar SKUs de interesse
skus_interesse = ['SKU1', 'SKU2', 'SKU3']

# 3. Gerar previsões SARIMA
previsor = PrevisorEstoqueSARIMA(horizonte_previsao=30)
df_estoque = pd.read_csv('DB/historico_estoque_atual.csv')
# ... (preparar dados)

previsoes = {}
for sku in skus_interesse:
    serie = previsor.preparar_serie_temporal(df_estoque, sku)
    modelo = previsor.treinar_modelo(serie, sku)
    previsao = previsor.prever(serie, modelo=modelo)
    previsoes[sku] = previsao.sum()

# 4. Calcular scores e ranking
# ... (combinar métricas e gerar ranking)
```

---

## ⚙️ Configurações e Parâmetros

### SARIMA

**Arquivo**: `sarima_estoque.py` (linha ~130)

**Parâmetros importantes**:
```python
m=30,  # Sazonalidade mensal (30 dias)
horizonte_previsao=30,  # Previsão para 30 dias
max_p=5, max_d=2, max_q=5,  # Limites de busca de parâmetros
```

**Ajustes**:
- Para sazonalidade anual: `m=365`
- Para sazonalidade semanal: `m=7`
- Para previsão mais longa: `horizonte_previsao=60`

---

### Elencação

**Pesos da fórmula** (em `teste_elencacao_3_skus.py`):
```python
peso_rentabilidade = 0.4  # 40%
peso_urgencia = 0.3       # 30%
peso_giro = 0.3           # 30%
```

**Ajustes**: Modifique os pesos conforme a importância para seu negócio:
- Se margem é crítica: aumente `peso_rentabilidade`
- Se evitar ruptura é crítico: aumente `peso_urgencia`
- Se demanda futura é importante: aumente `peso_giro`

---

### Data Wrangling

**Arquivo**: `data_wrangling/dw_historico.py`

**Parâmetros**:
```python
min_observacoes=30,  # Mínimo de observações por SKU
criar_serie_completa=True,  # Preencher lacunas temporais
```

---

## 🐛 Troubleshooting

### Erro: "Dados insuficientes"

**Causa**: SKU tem menos de 30 observações históricas.

**Solução**: 
- Verificar dados do SKU
- Reduzir `min_observacoes` (não recomendado)
- Excluir SKU da análise

---

### Erro: "index must be monotonic"

**Causa**: Datas duplicadas ou fora de ordem no histórico.

**Solução**:
```bash
python data_wrangling/dw_historico.py
```
Isso processa e limpa os dados.

---

### Erro: "ModuleNotFoundError: No module named 'pmdarima'"

**Causa**: Dependências não instaladas.

**Solução**:
```bash
pip install -r requirements_sarima.txt
```

---

### Previsões muito conservadoras (Random Walk)

**Causa**: Série temporal não tem padrões claros ou é muito irregular.

**Solução**:
- Verificar qualidade dos dados
- Ajustar parâmetros do auto_arima
- Considerar outros modelos (ARIMA simples, médias móveis)

---

### SKUs com urgência = 0 (estoque zerado)

**Interpretação**: Estoque atual é zero, urgência máxima.

**Ação**: Priorizar reposição imediata desses SKUs.

---

## 📚 Documentação Adicional

Consulte a pasta `documentacao/` para:
- **GUIA_RAPIDO.md**: Guia rápido de uso
- **README_SARIMA.md**: Documentação técnica detalhada do SARIMA
- **DOCUMENTACAO_TECNICA_FERRAMENTAS.md**: Ferramentas estatísticas
- **EXPLICACAO_RESULTADOS_SARIMA.md**: Interpretação de resultados
- **RESUMO_VALIDACAO_VENDAS.md**: Validação das métricas

---

## 🔄 Fluxo Completo Recomendado

### Para Análise Completa:

1. **Preparar dados**:
   ```bash
   python data_wrangling/dw_historico.py
   ```

2. **Análise exploratória**:
   ```bash
   python analises/analise_exploratoria_sazonalidade.py
   ```

3. **Validar extração de dados**:
   ```bash
   python validacao/validar_extracao_vendas.py
   ```

4. **Calcular métricas**:
   ```bash
   python validacao/calcular_metricas_elencacao.py
   ```

5. **Gerar elencação**:
   ```bash
   python previsoes/teste_elencacao_3_skus.py
   ```

6. **Comparar modelos** (opcional, demorado):
   ```bash
   python modelos/comparacao_top_skus_otimizado.py
   ```

### Para Uso Rápido (Produção):

1. **Calcular métricas atualizadas**:
   ```bash
   python validacao/calcular_metricas_elencacao.py
   ```

2. **Gerar previsões e ranking**:
   ```bash
   python previsoes/teste_elencacao_3_skus.py
   ```

3. **Consultar resultados**:
   - `resultados/resultado_elencacao_3_skus.csv`: Ranking final
   - `resultados/metricas_elencacao.csv`: Todas as métricas

---

## 📊 Estrutura de Dados

### Entrada

**DB/historico_estoque_atual.csv**:
- `sku`: Código do produto (string)
- `created_at`: Data/hora (datetime)
- `saldo`: Quantidade em estoque (numeric)

**DB/venda_produtos_atual.csv**:
- `sku`: Código do produto (string)
- `created_at`: Data/hora da venda (datetime)
- `quantidade`: Quantidade vendida (numeric)
- `valor_unitario`: Preço de venda (numeric)
- `custo_unitario`: Custo de aquisição (numeric)
- `margem_proporcional`: Margem proporcional % (numeric)

### Saída

**resultados/resultado_elencacao_[N]_skus.csv**:
- `sku`: Código do produto
- `quantidade_vendida_total`: Soma de quantidade vendida
- `rentabilidade_Rt`: Rentabilidade (R$)
- `margem_proporcional_media`: Margem proporcional média (%)
- `estoque_atual`: Estoque atual (unidades)
- `nivel_urgencia_Ut`: Nível de urgência (dias)
- `giro_futuro_previsto_GPt`: Giro futuro previsto (soma previsões)
- `estoque_medio_previsto`: Estoque médio previsto
- `score_elencacao`: Score final de elencação
- `ranking`: Posição no ranking (1 = maior prioridade)

---

## 📞 Suporte

Para dúvidas ou problemas:
1. Consulte a documentação na pasta `documentacao/`
2. Verifique os exemplos em `exemplos/`
3. Execute scripts de validação para diagnosticar problemas

---

**Última atualização**: 2024  
**Versão**: 1.0

