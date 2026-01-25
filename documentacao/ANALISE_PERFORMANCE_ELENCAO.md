# Análise de Performance: Script de Elencação

## 📊 Situação Atual

**Tempo de processamento**: ~40 minutos por SKU  
**Gargalo principal**: Treinamento do modelo SARIMA via `auto_arima`

---

## 🔍 Análise Detalhada dos Gargalos

### 1. **Treinamento SARIMA (auto_arima) - ~95% do tempo**

#### Problema Identificado

O treinamento do modelo SARIMA é o maior gargalo. No arquivo `previsoes/sarima_estoque.py` (linhas 143-167):

```python
modelo = auto_arima(
    serie,
    seasonal=True,
    m=30,
    stepwise=True,
    suppress_warnings=True,
    error_action='ignore',
    max_p=5,      # ⚠️ ALTO: testa 0,1,2,3,4,5
    max_d=2,      # ⚠️ ALTO: testa 0,1,2
    max_q=5,      # ⚠️ ALTO: testa 0,1,2,3,4,5
    max_P=2,      # ⚠️ ALTO: testa 0,1,2
    max_D=1,      # OK
    max_Q=2,      # ⚠️ ALTO: testa 0,1,2
    information_criterion='aic',
    trace=False,
    n_jobs=-1     # ⚠️ Pode não estar funcionando corretamente
)
```

#### Cálculo de Combinações

Com os parâmetros atuais:
- **Não-sazonal**: (5+1) × (2+1) × (5+1) = **108 combinações base**
- **Sazonal**: (2+1) × (1+1) × (2+1) = **18 combinações sazonais**
- **Total teórico**: 108 × 18 = **1.944 combinações possíveis**

O `stepwise=True` reduz isso, mas ainda testa **centenas de combinações** por SKU.

#### Impacto no Tempo

- Cada combinação testada requer:
  1. Ajuste do modelo (MLE - Maximum Likelihood Estimation)
  2. Cálculo de AIC/BIC
  3. Validação de estacionariedade
- **Tempo médio por combinação**: ~2-5 segundos
- **Total estimado**: 200-500 combinações × 3s = **10-25 minutos por SKU**

---

### 2. **Carregamento Repetido de Dados CSV**

#### Problema Identificado

No arquivo `previsoes/teste_elencacao_3_skus.py`:

- **Linha 32**: `df_vendas = pd.read_csv(caminho_vendas, low_memory=False)` (função `identificar_top_skus_movimentacao`)
- **Linha 63**: `df_vendas = pd.read_csv(caminho_vendas, low_memory=False)` (função `calcular_metricas_vendas`)
- **Linha 103**: `df_vendas = pd.read_csv(caminho_vendas, low_memory=False)` (função `calcular_venda_media_diaria`)
- **Linha 134**: `df_estoque = pd.read_csv(caminho_estoque, low_memory=False)` (função `calcular_nivel_urgencia`)
- **Linha 175**: `df_estoque = pd.read_csv(caminho_estoque, low_memory=False)` (função `gerar_previsoes_sarima`)

#### Impacto

- **Arquivo de vendas**: Carregado **3 vezes** (se ~32k linhas, ~50-100MB)
- **Arquivo de estoque**: Carregado **2 vezes** (se ~100k linhas, ~20-50MB)
- **Tempo total desperdiçado**: ~30-60 segundos por execução

---

### 3. **Processamento Sequencial (Não Paralelo)**

#### Problema Identificado

No arquivo `previsoes/teste_elencacao_3_skus.py` (linha 194):

```python
for sku in skus:  # ⚠️ Processa um SKU por vez
    # ... treina modelo SARIMA ...
```

Cada SKU é processado **sequencialmente**, mesmo que o sistema tenha múltiplos cores disponíveis.

#### Impacto

- **CPU ociosa**: Se há 8 cores, apenas 1 está sendo usado
- **Tempo total**: 3 SKUs × 40 min = **120 minutos** (poderia ser ~40-50 min com paralelização)

---

### 4. **Preparação de Série Temporal Repetida**

#### Problema Identificado

No arquivo `previsoes/sarima_estoque.py` (linha 44-81), a função `preparar_serie_temporal`:

1. Filtra DataFrame completo por SKU
2. Converte datas
3. Ordena
4. Remove duplicatas
5. Cria índice temporal
6. Preenche frequência
7. Remove NaN

Isso é feito **a cada chamada**, mesmo que os dados não tenham mudado.

#### Impacto

- **Tempo por preparação**: ~1-3 segundos
- **Repetido**: 3 vezes por SKU (preparação, treino, previsão)
- **Total desperdiçado**: ~3-9 segundos por SKU

---

### 5. **Falta de Cache de Modelos Treinados**

#### Problema Identificado

O sistema não salva modelos treinados. Se o mesmo SKU for processado novamente, o modelo é **retreinado do zero**.

#### Impacto

- **Retreinamento desnecessário**: Se processar os mesmos SKUs, perde 40 min por SKU novamente
- **Sem persistência**: Modelos não podem ser reutilizados

---

### 6. **Paralelização do auto_arima Não Eficiente**

#### Problema Identificado

O parâmetro `n_jobs=-1` no `auto_arima` **não paraleliza a busca de parâmetros**. Ele paraleliza apenas:
- Testes de estacionariedade (ADF)
- Algumas operações internas

A **busca stepwise é sequencial** por design.

#### Impacto

- **CPU subutilizada**: Mesmo com `n_jobs=-1`, apenas 1-2 cores são usadas efetivamente

---

## 🚀 Oportunidades de Otimização

### **Categoria 1: Otimizações de Código (Sem GPU)**

#### 1.1. Reduzir Parâmetros do auto_arima ⭐⭐⭐⭐⭐

**Impacto**: **Alto** (redução de 60-80% no tempo)

**Mudança proposta**:
```python
# ATUAL (1.944 combinações teóricas)
max_p=5, max_d=2, max_q=5, max_P=2, max_D=1, max_Q=2

# OTIMIZADO (108 combinações teóricas)
max_p=3, max_d=1, max_q=3, max_P=1, max_D=1, max_Q=1
```

**Justificativa**:
- Modelos SARIMA raramente precisam de ordens > 3
- A maioria dos modelos reais usa (1,1,1) ou (2,1,2)
- Redução de **18x menos combinações** a testar

**Tempo estimado**: 40 min → **8-12 min por SKU**

---

#### 1.2. Cache de Modelos Treinados ⭐⭐⭐⭐

**Impacto**: **Muito Alto** (para reprocessamento)

**Implementação**:
- Salvar modelos treinados em `pickle` ou `joblib`
- Verificar se modelo já existe antes de treinar
- Reutilizar modelo se dados não mudaram

**Tempo estimado**: 40 min → **0 min** (se cache existe)

---

#### 1.3. Carregamento Único de Dados ⭐⭐⭐

**Impacto**: **Médio** (redução de 30-60 segundos)

**Mudança proposta**:
- Carregar dados uma vez no início
- Passar DataFrames como parâmetros entre funções
- Evitar múltiplos `pd.read_csv()`

**Tempo estimado**: Economia de **30-60 segundos** por execução

---

#### 1.4. Preparação de Série Temporal em Cache ⭐⭐⭐

**Impacto**: **Médio** (redução de 3-9 segundos por SKU)

**Mudança proposta**:
- Preparar todas as séries temporais uma vez
- Armazenar em dicionário `{sku: serie}`
- Reutilizar séries preparadas

**Tempo estimado**: Economia de **3-9 segundos por SKU**

---

#### 1.5. Processamento Paralelo de SKUs ⭐⭐⭐⭐

**Impacto**: **Alto** (redução proporcional ao número de cores)

**Implementação**:
- Usar `multiprocessing.Pool` ou `concurrent.futures.ProcessPoolExecutor`
- Processar múltiplos SKUs simultaneamente
- Limitar número de processos ao número de cores

**Tempo estimado**: 
- 3 SKUs sequenciais: 120 min
- 3 SKUs paralelos (4 cores): **~40-50 min**

---

#### 1.6. Usar BIC em vez de AIC ⭐⭐

**Impacto**: **Baixo-Médio** (pode reduzir complexidade dos modelos)

**Mudança proposta**:
```python
information_criterion='bic'  # Penaliza mais modelos complexos
```

**Justificativa**:
- BIC tende a escolher modelos mais simples
- Modelos mais simples = menos tempo de treino
- Pode reduzir 10-20% do tempo

---

#### 1.7. Limitar Tamanho da Série Temporal ⭐⭐

**Impacto**: **Médio** (redução de 20-30% no tempo)

**Mudança proposta**:
- Usar apenas últimos N dias (ex: 365 dias)
- Reduz tamanho da série = menos cálculos

**Tempo estimado**: Redução de **20-30%** no tempo de treino

---

### **Categoria 2: Otimizações com GPU**

#### 2.1. GPU para auto_arima? ❌ **NÃO RECOMENDADO**

**Análise**:
- `pmdarima` (auto_arima) **não suporta GPU**
- É baseado em `statsmodels` e `scipy`, que são CPU-only
- A busca de parâmetros é sequencial por design

**Conclusão**: **Não é viável usar GPU para auto_arima diretamente**

---

#### 2.2. Alternativas com GPU ⭐⭐⭐

**Opção A: Usar TensorFlow/PyTorch para Previsão**

**Implementação**:
- Treinar modelo LSTM/GRU na GPU
- Usar apenas para previsão (não para busca de parâmetros)
- Manter SARIMA para validação

**Vantagens**:
- Treino LSTM na GPU: **10-50x mais rápido** que CPU
- Pode processar múltiplos SKUs em batch

**Desvantagens**:
- Requer reimplementação significativa
- LSTM pode não capturar sazonalidade tão bem quanto SARIMA
- Requer mais dados para treinar

**Tempo estimado**: 40 min → **2-5 min por SKU** (apenas treino LSTM)

---

**Opção B: Usar RAPIDS cuDF para Processamento de Dados**

**Implementação**:
- Substituir `pandas` por `cudf` (GPU DataFrame)
- Processar agregações e merges na GPU
- Manter auto_arima na CPU

**Vantagens**:
- Agregações e merges **10-100x mais rápidos** na GPU
- Carregamento de dados mais rápido

**Desvantagens**:
- Requer GPU NVIDIA com CUDA
- Não acelera o auto_arima (principal gargalo)
- Ganho limitado (~5-10% do tempo total)

**Tempo estimado**: 40 min → **38-39 min por SKU** (ganho mínimo)

---

**Opção C: Usar Dask para Paralelização Distribuída**

**Implementação**:
- Usar `dask` para processar múltiplos SKUs em paralelo
- Pode usar GPU workers se disponível
- Distribuir carga entre múltiplos processos/GPUs

**Vantagens**:
- Escala horizontalmente (múltiplas GPUs/máquinas)
- Processa muitos SKUs simultaneamente

**Desvantagens**:
- Complexidade de setup
- Overhead de comunicação
- Ainda não acelera auto_arima individual

**Tempo estimado**: 100 SKUs → **~40-50 min total** (vs 66 horas sequencial)

---

## 📈 Comparação de Estratégias

### **Estratégia 1: Otimizações Simples (Sem GPU)**

| Otimização | Redução de Tempo | Complexidade | Prioridade |
|------------|------------------|--------------|------------|
| Reduzir parâmetros auto_arima | 60-80% | Baixa | ⭐⭐⭐⭐⭐ |
| Cache de modelos | 100% (reprocessamento) | Média | ⭐⭐⭐⭐ |
| Carregamento único de dados | 1-2% | Baixa | ⭐⭐⭐ |
| Preparação de série em cache | 2-5% | Baixa | ⭐⭐⭐ |
| Processamento paralelo | 50-75% (múltiplos SKUs) | Média | ⭐⭐⭐⭐ |

**Tempo estimado final**: 40 min → **5-8 min por SKU** (primeira execução)  
**Tempo estimado final**: 40 min → **0-1 min por SKU** (com cache)

---

### **Estratégia 2: Otimizações + GPU (LSTM)**

| Componente | Tempo Atual | Tempo com GPU | Ganho |
|------------|-------------|---------------|-------|
| Treino SARIMA | 40 min | 40 min | 0% |
| Treino LSTM | N/A | 2-5 min | - |
| Previsão | 1 min | 0.1 min | 90% |

**Tempo estimado final**: 40 min → **2-5 min por SKU** (LSTM)  
**Nota**: Requer validação de que LSTM tem qualidade similar ao SARIMA

---

### **Estratégia 3: Híbrida (Otimizações + Paralelização)**

| Componente | Tempo Atual | Tempo Otimizado | Ganho |
|------------|-------------|------------------|-------|
| 1 SKU sequencial | 40 min | 8 min | 80% |
| 3 SKUs sequenciais | 120 min | 24 min | 80% |
| 3 SKUs paralelos (4 cores) | 120 min | 8-12 min | 90-93% |

**Tempo estimado final**: **8-12 min para 3 SKUs** (primeira execução)

---

## 🎯 Recomendações Prioritárias

### **Prioridade 1: Implementar Imediatamente** ⭐⭐⭐⭐⭐

1. **Reduzir parâmetros do auto_arima**
   - Mudança: `max_p=3, max_d=1, max_q=3, max_P=1, max_D=1, max_Q=1`
   - Impacto: **60-80% de redução** no tempo
   - Esforço: **5 minutos** (alterar 1 linha)

2. **Cache de modelos treinados**
   - Impacto: **100% de redução** em reprocessamento
   - Esforço: **30-60 minutos** (implementar sistema de cache)

---

### **Prioridade 2: Implementar em Seguida** ⭐⭐⭐⭐

3. **Processamento paralelo de SKUs**
   - Impacto: **50-75% de redução** para múltiplos SKUs
   - Esforço: **1-2 horas** (implementar multiprocessing)

4. **Carregamento único de dados**
   - Impacto: **1-2% de redução** + código mais limpo
   - Esforço: **30 minutos** (refatorar funções)

---

### **Prioridade 3: Considerar no Futuro** ⭐⭐⭐

5. **Preparação de série em cache**
   - Impacto: **2-5% de redução**
   - Esforço: **30 minutos**

6. **Limitar tamanho da série temporal**
   - Impacto: **20-30% de redução**
   - Esforço: **15 minutos**

---

### **Prioridade 4: Avaliar Alternativas** ⭐⭐

7. **LSTM com GPU** (se qualidade for aceitável)
   - Impacto: **80-90% de redução** (mas requer validação)
   - Esforço: **1-2 semanas** (reimplementação significativa)

8. **RAPIDS cuDF** (se GPU disponível)
   - Impacto: **5-10% de redução** (ganho limitado)
   - Esforço: **2-4 horas**

---

## 📊 Estimativa de Ganho Total

### **Cenário Conservador (Apenas Otimizações Simples)**

| Otimização | Ganho Individual | Ganho Acumulado |
|------------|-------------------|-----------------|
| Baseline | 40 min/SKU | 40 min/SKU |
| Reduzir parâmetros | -70% | 12 min/SKU |
| Cache de modelos | -100% (reprocessamento) | 0 min/SKU |
| Carregamento único | -1% | 11.9 min/SKU |
| Preparação em cache | -3% | 11.5 min/SKU |
| **TOTAL** | | **~11-12 min/SKU** (primeira vez) |

**Ganho total**: **70-72% de redução**

---

### **Cenário Otimista (Otimizações + Paralelização)**

| Otimização | Ganho Individual | Ganho Acumulado |
|------------|-------------------|-----------------|
| Baseline | 40 min/SKU | 40 min/SKU |
| Reduzir parâmetros | -70% | 12 min/SKU |
| Processamento paralelo (4 cores) | -75% (3 SKUs) | 3 min/SKU |
| Cache de modelos | -100% (reprocessamento) | 0 min/SKU |
| **TOTAL** | | **~3-4 min/SKU** (primeira vez, 3 SKUs) |

**Ganho total**: **90-92% de redução**

---

## 🔧 Detalhamento Técnico das Otimizações

### **1. Reduzir Parâmetros do auto_arima**

**Arquivo**: `previsoes/sarima_estoque.py` (linha 143-167)

**Mudança**:
```python
# ANTES
max_p=5, max_d=2, max_q=5, max_P=2, max_D=1, max_Q=2

# DEPOIS
max_p=3, max_d=1, max_q=3, max_P=1, max_D=1, max_Q=1
```

**Justificativa Estatística**:
- Modelos ARIMA raramente precisam de ordens > 3
- A maioria dos modelos reais usa (1,1,1) ou (2,1,2)
- Sazonalidade mensal (m=30) raramente precisa de P, D, Q > 1

**Risco**: Baixo (pode perder modelos muito complexos, mas improvável)

---

### **2. Cache de Modelos**

**Implementação proposta**:
```python
import pickle
from pathlib import Path

def carregar_modelo_cache(sku, cache_dir="cache_modelos"):
    cache_path = Path(cache_dir) / f"modelo_{sku}.pkl"
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None

def salvar_modelo_cache(sku, modelo, cache_dir="cache_modelos"):
    Path(cache_dir).mkdir(exist_ok=True)
    cache_path = Path(cache_dir) / f"modelo_{sku}.pkl"
    with open(cache_path, 'wb') as f:
        pickle.dump(modelo, f)
```

**Validação de cache**:
- Verificar hash dos dados de entrada
- Se dados mudaram, retreinar modelo
- Se dados não mudaram, reutilizar modelo

---

### **3. Processamento Paralelo**

**Implementação proposta**:
```python
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

def processar_skus_paralelo(skus, df_estoque, n_workers=None):
    if n_workers is None:
        n_workers = min(len(skus), cpu_count())
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(processar_sku, sku, df_estoque): sku 
            for sku in skus
        }
        
        resultados = {}
        for future in futures:
            sku = futures[future]
            try:
                resultados[sku] = future.result()
            except Exception as e:
                print(f"Erro ao processar {sku}: {e}")
    
    return resultados
```

**Considerações**:
- Cada processo precisa de cópia dos dados
- Overhead de comunicação entre processos
- Ideal para 4-8 SKUs simultâneos

---

### **4. Carregamento Único de Dados**

**Refatoração proposta**:
```python
def gerar_elencacao_completa():
    # Carregar dados UMA VEZ
    df_vendas = pd.read_csv("DB/venda_produtos_atual.csv", low_memory=False)
    df_estoque = pd.read_csv("DB/historico_estoque_atual.csv", low_memory=False)
    
    # Preparar dados
    df_vendas = preparar_dados_vendas(df_vendas)
    df_estoque = preparar_dados_estoque(df_estoque)
    
    # Passar DataFrames para funções
    top_skus = identificar_top_skus_movimentacao(df_vendas)
    metricas = calcular_metricas_vendas(df_vendas, top_skus)
    previsoes = gerar_previsoes_sarima(df_estoque, top_skus)
    
    # ...
```

---

## ⚠️ Riscos e Considerações

### **Risco 1: Redução de Parâmetros Pode Piorar Qualidade**

**Mitigação**:
- Validar modelos otimizados vs. modelos completos
- Comparar métricas (MAE, RMSE, MAPE)
- Se qualidade degradar > 5%, ajustar parâmetros incrementalmente

---

### **Risco 2: Cache de Modelos Pode Ficar Desatualizado**

**Mitigação**:
- Implementar sistema de versionamento de dados
- Invalidar cache quando dados mudarem
- Adicionar timestamp aos arquivos de cache

---

### **Risco 3: Paralelização Pode Consumir Muita Memória**

**Mitigação**:
- Limitar número de workers ao número de cores
- Processar em batches se muitos SKUs
- Monitorar uso de memória

---

## 📝 Resumo Executivo

### **Problema Principal**
- Treinamento SARIMA via `auto_arima` consome ~95% do tempo
- Parâmetros altos (max_p=5, max_q=5) testam muitas combinações
- Processamento sequencial não aproveita múltiplos cores

### **Solução Recomendada (Imediata)**
1. **Reduzir parâmetros do auto_arima**: `max_p=3, max_q=3, max_P=1, max_Q=1`
   - Ganho: **60-80% de redução** no tempo
   - Esforço: **5 minutos**

2. **Implementar cache de modelos**
   - Ganho: **100% de redução** em reprocessamento
   - Esforço: **30-60 minutos**

3. **Processamento paralelo de SKUs**
   - Ganho: **50-75% de redução** para múltiplos SKUs
   - Esforço: **1-2 horas**

### **Resultado Esperado**
- **Primeira execução**: 40 min → **8-12 min por SKU**
- **Reprocessamento**: 40 min → **0-1 min por SKU** (com cache)
- **3 SKUs paralelos**: 120 min → **8-12 min total**

### **Sobre GPU**
- **auto_arima não suporta GPU** (baseado em statsmodels/scipy)
- **Alternativa LSTM na GPU**: Viável, mas requer reimplementação
- **RAPIDS cuDF**: Ganho limitado (~5-10%) pois não acelera auto_arima

### **Conclusão**
**Melhor estratégia**: Otimizações de código (sem GPU) podem reduzir tempo em **70-90%** com esforço baixo-médio. GPU só seria útil se migrar para LSTM, o que requer validação de qualidade.

---

**Data da Análise**: 2024  
**Versão**: 1.0

