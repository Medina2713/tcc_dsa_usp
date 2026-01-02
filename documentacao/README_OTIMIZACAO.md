# Script Otimizado: Comparação de Modelos para Top SKUs

## 🚀 Melhorias Implementadas

### 1. Salvamento Incremental
- ✅ **Salva resultados por SKU** conforme processa
- ✅ **Sistema de checkpoint** - pode retomar de onde parou
- ✅ **Arquivos individuais** - JSON e CSV por SKU
- ✅ **Relatório final consolidado** - gera no final

### 2. Otimizações de Performance

#### Auto-ARIMA Otimizado:
- **Parâmetros reduzidos**: `max_p=3, max_d=1, max_q=3` (antes: 5,2,5)
- **Sazonais reduzidos**: `max_P=1, max_D=1, max_Q=1` (antes: 2,1,2)
- **1 core apenas**: `n_jobs=1` (evita problemas de memória)
- **Sem SARIMA anual**: Remove modelo que consome muita memória

#### Processamento Eficiente:
- **Cache de dados**: Carrega dados uma vez só
- **Operações vetorizadas**: Usa pandas/numpy otimizado
- **Filtragem inteligente**: Filtra dados antes de processar

### 3. Métricas Estatísticas Completas

Todas as métricas são calculadas:
- ✅ **MAE** (Mean Absolute Error)
- ✅ **RMSE** (Root Mean Squared Error)
- ✅ **MAPE** (Mean Absolute Percentage Error)
- ✅ **R²** (Coeficiente de Determinação)
- ✅ **MAE%** (MAE percentual)
- ✅ **RMSE%** (RMSE percentual)
- ✅ **Bias** (Desvio médio - viés sistemático)
- ✅ **MAE Normalizado** (dividido pelo range)

### 4. Sistema de Checkpoint

O script salva automaticamente quais SKUs já foram processados:
- Arquivo: `resultados_comparacao/checkpoint_skus.json`
- Pode interromper e retomar
- Evita reprocessar SKUs já processados

## 📁 Estrutura de Arquivos Gerados

```
resultados_comparacao/
├── checkpoint_skus.json              # Checkpoint (SKUs processados)
├── resultado_[SKU].json              # Resultados detalhados (JSON)
├── metricas_[SKU].csv                # Métricas (CSV)
├── relatorio_consolidado.txt         # Relatório final (texto)
└── metricas_consolidadas.csv         # Todas métricas (CSV)
```

## 🚀 Como Usar

### Execução Normal:
```bash
python comparacao_top_skus_otimizado.py
```

### Retomar Processamento:
```bash
# Se interromper, apenas execute novamente
# O script automaticamente pula SKUs já processados
python comparacao_top_skus_otimizado.py
```

### Limpar e Começar do Zero:
```bash
# Delete a pasta resultados_comparacao/
rm -rf resultados_comparacao/  # Linux/Mac
# ou
rmdir /s resultados_comparacao  # Windows
```

## ⚙️ Configurações Ajustáveis

No código, você pode ajustar:

1. **Número de SKUs**: Altere `top_n` na função `main()`
2. **Horizonte de previsão**: Altere `horizonte_previsao=30`
3. **Proporção treino/teste**: Altere `proporcao_treino=0.8`
4. **Parâmetros Auto-ARIMA**: Ajuste limites em `comparar_modelos_otimizado()`

## 📊 Tempo Estimado

**Antes (versão antiga)**: 4+ horas para 10 SKUs  
**Agora (otimizado)**: 
- ~15-30 minutos por SKU (dependendo da série)
- Total: ~2.5-5 horas para 10 SKUs
- **Mas salva incrementalmente** - pode interromper a qualquer momento

## 🎯 Comparação com Versão Anterior

| Aspecto | Versão Antiga | Versão Otimizada |
|---------|---------------|------------------|
| Salvamento | Só no final | Incremental (por SKU) |
| Checkpoint | Não | Sim |
| SARIMA Anual | Sim (lento) | Não (remove) |
| Parâmetros ARIMA | 5,2,5 / 2,1,2 | 3,1,3 / 1,1,1 |
| Métricas | Básicas | Completas (8 métricas) |
| Retomada | Não | Sim |
| Tempo/SKU | ~30-40 min | ~15-30 min |

## 🔧 Troubleshooting

### Script muito lento ainda?
- Reduza ainda mais os parâmetros do auto_arima
- Processe menos SKUs por vez
- Use apenas modelos mais rápidos (remova exponencial)

### Erro de memória?
- Já otimizado para 1 core
- SARIMA anual foi removido
- Se persistir, processe 1 SKU por vez

### Checkpoint não funciona?
- Verifique permissões de escrita
- Delete `checkpoint_skus.json` para resetar

## 📝 Notas Importantes

1. **SARIMA Anual removido**: Consumia muita memória. Use SARIMA Mensal (m=30) que já captura padrões mensais.

2. **Parâmetros reduzidos**: Trade-off entre qualidade e velocidade. Para análise inicial, é suficiente.

3. **Salvamento incremental**: Sempre salva após cada SKU. Se interromper, apenas execute novamente.

4. **Métricas completas**: Agora todas as métricas estatísticas são calculadas e salvas.

---

**Desenvolvido para TCC MBA Data Science & Analytics - 2024**

