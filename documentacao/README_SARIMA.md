# Previsão de Estoque com SARIMA (Auto-ARIMA)

## 📋 Visão Geral

Este módulo implementa **previsões de estoque (saldo)** futuro usando o modelo **SARIMA** (Seasonal AutoRegressive Integrated Moving Average) com busca automática de parâmetros via `pmdarima.auto_arima`.

Desenvolvido para o **TCC do MBA em Data Science & Analytics** - Ferramenta de Gestão de Estoque para E-commerce de Brinquedos.

**Importante:** Os modelos preveem **estoque (unidades em estoque)**, não vendas. A previsão é usada na elencação para **sinalizar necessidade de reposição**: estoque previsto baixo → priorizar repor; estoque previsto alto → menor urgência. GP(t) = soma das previsões de estoque no horizonte.

---

## 🎯 Objetivo

Gerar previsões de **estoque** para os próximos **7 a 15 dias** por produto (SKU), que serão utilizadas como **terceiro pilar** (GP(t)) na fórmula de elencação da ferramenta de reposição de estoque.

---

## 🔑 Por que Auto-ARIMA?

O **Auto-ARIMA** resolve o problema de escolher manualmente os parâmetros do SARIMA (`p, d, q` x `P, D, Q, s`) para cada produto:

- ✅ **Automatizado**: Testa múltiplas combinações e escolhe a melhor
- ✅ **Escalável**: Funciona para centenas/milhares de produtos
- ✅ **Inteligente**: Usa critérios estatísticos (AIC, BIC) para seleção
- ✅ **Eficiente**: Algoritmo stepwise acelera a busca

---

## 📦 Instalação

```bash
pip install -r requirements_sarima.txt
```

### Dependências principais:
- `pmdarima`: Auto-ARIMA
- `pandas`: Manipulação de dados
- `numpy`: Computação numérica
- `matplotlib`: Visualização (opcional)

---

## 🚀 Uso Básico

### Exemplo 1: Um único produto

```python
from sarima_estoque import PrevisorEstoqueSARIMA
import pandas as pd

# 1. Prepare seus dados (formato: DataFrame com colunas 'data', 'sku', 'estoque_atual')
df_estoque = pd.DataFrame({
    'data': pd.date_range('2024-01-01', periods=90, freq='D'),
    'sku': 'BRINQUEDO_001',
    'estoque_atual': [100, 95, 90, ...]  # seus dados reais
})

# 2. Inicialize o previsor
previsor = PrevisorEstoqueSARIMA(horizonte_previsao=7, frequencia='D')

# 3. Prepare a série temporal
serie = previsor.preparar_serie_temporal(df_estoque, sku='BRINQUEDO_001')

# 4. Treine o modelo (auto_arima busca parâmetros automaticamente)
modelo = previsor.treinar_modelo(serie, sku='BRINQUEDO_001')

# 5. Gere a previsão
previsao = previsor.prever(serie, modelo=modelo)

print(previsao)
```

### Exemplo 2: Múltiplos produtos (lote)

```python
# Processa todos os SKUs de uma vez
resultados = previsor.processar_lote(df_estoque, lista_skus=['SKU1', 'SKU2', 'SKU3'])

# Resultado: DataFrame com previsões para todos os produtos
print(resultados)
```

---

## 📊 Formato dos Dados

Seu DataFrame de entrada deve ter a seguinte estrutura:

| data | sku | estoque_atual |
|------|-----|---------------|
| 2024-01-01 | BRINQUEDO_001 | 100 |
| 2024-01-02 | BRINQUEDO_001 | 95 |
| 2024-01-03 | BRINQUEDO_001 | 90 |
| ... | ... | ... |

**Requisitos:**
- Coluna `data`: Datetime (formato datetime)
- Coluna `sku`: String (código do produto)
- Coluna `estoque_atual`: Numérico (unidades em estoque)
- **Mínimo de 30 observações** por SKU para treinar o modelo

---

## 🔧 Integração com API

Para integrar com seus dados reais via API:

```python
# Exemplo de estrutura (adaptar conforme sua API)
import requests

def obter_dados_estoque_api(data_inicio, data_fim):
    url = "sua_api/historico_estoque"
    params = {'data_inicio': data_inicio, 'data_fim': data_fim}
    response = requests.get(url, params=params)
    dados = response.json()
    
    df = pd.DataFrame(dados)
    df['data'] = pd.to_datetime(df['data'])
    
    return df

# Uso
df_estoque = obter_dados_estoque_api('2024-01-01', '2024-06-30')
previsor = PrevisorEstoqueSARIMA(horizonte_previsao=15)
resultados = previsor.processar_lote(df_estoque)
```

---

## 📈 Integração com Fórmula de Elencação

As previsões podem ser integradas à sua fórmula de elencação:

```python
# Exemplo: Calcular score de risco de ruptura
for sku in resultados['sku'].unique():
    df_sku = resultados[resultados['sku'] == sku]
    estoque_medio_previsto = df_sku['estoque_previsto'].mean()
    
    # Score de risco (quanto menor o estoque previsto, maior o risco)
    risco_ruptura = 1 / (1 + estoque_medio_previsto)  # Normalizado [0, 1]
    
    # Sua fórmula completa (exemplo)
    score_final = (
        0.4 * margem_contribuicao +
        0.3 * giro_estoque +
        0.3 * risco_ruptura  # <-- previsão SARIMA aqui
    )
```

---

## 🎓 Conceitos Importantes (para seu TCC)

### 1. Estacionariedade
O SARIMA requer séries **estacionárias** (sem tendência forte). O `auto_arima` resolve isso automaticamente através da **diferenciação** (`d` e `D`).

### 2. Sazonalidade
O parâmetro `m=7` assume sazonalidade **semanal** (7 dias). Ajuste conforme seu padrão:
- `m=7`: Sazonalidade semanal
- `m=30`: Sazonalidade mensal
- `m=365`: Sazonalidade anual

### 3. Parâmetros do SARIMA
- **(p, d, q)**: Componente não-sazonal (AR, diferenciação, MA)
- **(P, D, Q, s)**: Componente sazonal (s = período sazonal)

O `auto_arima` escolhe esses valores automaticamente testando múltiplas combinações.

### 4. Critérios de Seleção
- **AIC** (Akaike Information Criterion): Equilibra ajuste e complexidade (padrão)
- **BIC** (Bayesian Information Criterion): Penaliza mais modelos complexos
- **AICc**: Versão corrigida do AIC para amostras pequenas

---

## ⚠️ Limitações e Considerações

1. **Dados mínimos**: Requer pelo menos **30 observações** por SKU
2. **Séries muito curtas**: Para menos de 30 pontos, considere métodos mais simples (média móvel)
3. **Produtos novos**: Sem histórico suficiente, use métodos alternativos
4. **Eventos externos**: SARIMA não captura promoções/eventos especiais (requer modelagem adicional)

---

## 📝 Exemplos Completos

Execute o arquivo `exemplo_uso_sarima.py` para ver exemplos práticos:

```bash
python exemplo_uso_sarima.py
```

---

## 🔍 Troubleshooting

### Erro: "Dados insuficientes"
- **Causa**: Menos de 30 observações
- **Solução**: Use histórico maior ou métodos alternativos para produtos novos

### Erro: "Modelo não convergiu"
- **Causa**: Série muito irregular ou com muitos outliers
- **Solução**: Limpe dados, remova outliers, ou use métodos mais robustos

### Previsões sempre iguais
- **Causa**: Modelo muito simples (pode ser apenas média)
- **Solução**: Verifique se há padrões na série; considere ajustar limites de busca

---

## 📚 Referências para TCC

1. **Hyndman & Athanasopoulos** - Forecasting: Principles and Practice (Cap. 8: ARIMA models)
2. **Box & Jenkins** - Time Series Analysis: Forecasting and Control (clássico)
3. **pmdarima documentation**: https://alkaline-ml.com/pmdarima/

---

## 📧 Suporte

Para dúvidas técnicas sobre implementação, consulte:
- Documentação do `pmdarima`: https://alkaline-ml.com/pmdarima/
- Stack Overflow: Tag `sarima` ou `pmdarima`
- Fórum do curso de Data Science

---

**Desenvolvido para TCC MBA Data Science & Analytics — USP** · *Documentação revista em 05/04/2026.*


