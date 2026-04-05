# Comparação de Modelos de Previsão de Estoque

## 📋 Visão Geral

Este script compara diferentes modelos de previsão temporal (**SARIMA**, **ARIMA**, **Holt-Winters**, **Média Móvel**) para **previsão de estoque (saldo)** por SKU. Os modelos são treinados na série histórica de **saldo de estoque** (`historico_estoque`); a saída é previsão de **unidades em estoque**, não vendas. A previsão é usada na elencação (GP(t)) para **sinalizar necessidade de reposição**. Usado pelo pipeline TCC (`gerar_figuras_tcc.py`) para figuras 5–7 e Tabela 2.

## 🎯 Modelos Comparados

1. **SARIMA com Sazonalidade Anual (m=365)**
   - Captura padrões que se repetem anualmente
   - **Só é estimado** se o período de treino tiver **≥ 730 dias** (~2 anos); caso contrário o script regista que o SARIMA anual foi omitido (séries curtas não sustentam bem `m=365`)

2. **SARIMA com Sazonalidade Mensal (m=30)**
   - Captura padrões mensais
   - Já testado anteriormente

3. **ARIMA Simples (sem sazonalidade)**
   - Modelo básico sem componente sazonal
   - Útil como baseline

4. **Média Móvel Simples**
   - Modelo mais simples
   - Prevê a média dos últimos N valores
   - Baseline mínimo esperado

5. **Suavização Exponencial (Holt-Winters)**
   - Captura tendência e sazonalidade
   - Útil para padrões suaves

## 📊 Métricas de Avaliação

### MAE (Mean Absolute Error)
- **Fórmula**: MAE = (1/n) × Σ |y_real - y_previsto|
- **Interpretação**: Erro médio absoluto
- **Melhor**: Menor valor
- **Unidade**: Mesma unidade dos dados

### RMSE (Root Mean Squared Error)
- **Fórmula**: RMSE = √[(1/n) × Σ (y_real - y_previsto)²]
- **Interpretação**: Penaliza erros grandes mais que erros pequenos
- **Melhor**: Menor valor
- **Unidade**: Mesma unidade dos dados

### MAPE (Mean Absolute Percentage Error)
- **Fórmula**: MAPE = (1/n) × Σ |y_real - y_previsto| / |y_real| × 100
- **Interpretação**: Erro percentual médio
- **Melhor**: Menor valor
- **Unidade**: Porcentagem (%)

## 🚀 Uso

```bash
python comparacao_modelos_previsao.py
```

## 📈 Estrutura do Script

### PARTE 1: Cálculo de MAPE
**Função:** `calcular_mape()`

Calcula erro percentual médio absoluto.

### PARTE 2: Divisão Treino/Teste
**Função:** `dividir_serie_temporal()`

Divide série em 80% treino / 20% teste (mantém ordem temporal).

### PARTE 3: Treinamento de Modelos

#### 3A: SARIMA Anual
- Período sazonal: 365 dias
- Parâmetros reduzidos para economizar memória

#### 3B: SARIMA Mensal
- Período sazonal: 30 dias

#### 3C: ARIMA Simples
- Sem componente sazonal

#### 3D: Média Móvel
- Janela: 7 dias

#### 3E: Suavização Exponencial
- Holt-Winters com sazonalidade (se dados suficientes)

### PARTE 4: Avaliação
**Função:** `avaliar_modelo()`

Calcula MAE, RMSE e MAPE para cada modelo.

### PARTE 5: Comparação Completa
**Função:** `comparar_modelos()`

Orquestra todo o processo de comparação.

### PARTE 6: Visualização
**Função:** `visualizar_comparacao()`

Gera gráficos comparativos com 2 painéis:
- Visão geral (treino + teste + previsões)
- Zoom no período de teste

### PARTE 7: Relatório
**Função:** `gerar_relatorio_comparacao()`

Gera relatório textual com métricas e melhor modelo por métrica.

## 📝 Arquivos Gerados

1. **comparacao_modelos_[SKU].png** — em `resultados/` (visão comparativa de todos os modelos)
2. **Com `--tcc` ou via `gerar_figuras_tcc.py`:** **figura5.png**, **figura6.png**, **figura7.png** em `resultados/figuras_tcc/` (um modelo por ficheiro)
3. **relatorio_comparacao_[SKU].txt**
   - Relatório textual completo
   - Métricas de todos os modelos
   - Melhor modelo por métrica

## 🎓 Interpretação dos Resultados

### Escolha do Melhor Modelo

**Estratégia recomendada:**
1. **Primeiro**: Verificar MAPE (mais fácil de interpretar)
2. **Segundo**: Verificar MAE (erro absoluto)
3. **Terceiro**: Verificar RMSE (se houver outliers)

**Exemplo:**
- MAPE < 10%: Excelente
- MAPE 10-20%: Bom
- MAPE 20-30%: Aceitável
- MAPE > 30%: Precisa melhorar

### Limitações Conhecidas

**SARIMA Anual (m=365):**
- Requer muito mais dados (preferencialmente 2+ anos)
- Consome muita memória
- Pode falhar em séries curtas
- Se falhar, não é um problema - outros modelos podem ser melhores

**Média Móvel:**
- Modelo muito simples
- Esperado ter desempenho pior
- Útil como baseline

## ⚙️ Parâmetros Ajustáveis

- **`horizonte_previsao`**: Número de períodos a prever (padrão: 30)
- **`proporcao_treino`**: Proporção para treino (padrão: 0.8 = 80%)
- **`janela`** (média móvel): Tamanho da janela (padrão: 7 dias)

## 🔧 Troubleshooting

### SARIMA Anual falha com erro de memória
- **Causa**: Sazonalidade anual requer muita memória
- **Solução**: 
  - Normal (esperado para séries curtas)
  - Outros modelos ainda funcionam
  - Para séries muito longas (>2 anos), pode funcionar

### AIC aparece como "bound method"
- **Corrigido no código**: Agora extrai valor corretamente

### Modelo muito lento
- SARIMA pode levar alguns minutos
- Normal para Auto-ARIMA (testa múltiplas combinações)

## 📚 Referências

- **MAE, RMSE, MAPE**: Métricas padrão de avaliação de modelos de previsão
- **SARIMA**: Modelo avançado com sazonalidade
- **ARIMA**: Modelo básico de séries temporais
- **Holt-Winters**: Método clássico de suavização exponencial

---

**Desenvolvido para TCC MBA Data Science & Analytics — USP**

**Última atualização:** 05/04/2026

