# 🚀 Guia Rápido - SARIMA para Previsão de Estoque

**Funcionamento atual:** Os modelos (SARIMA, ARIMA, Holt-Winters, Média Móvel) preveem **estoque (saldo)**, não vendas. A previsão é usada na elencação (GP(t)) para **sinalizar necessidade de reposição**. Pipeline TCC: `python gerar_figuras_tcc.py` gera figuras 1–7, Tabela 2 e **elencação final** (ranking R(t), U(t), GP(t)). Ver `documentacao/COMO_GERAR_FIGURAS_TCC.md`.

---

## Instalação (1 minuto)

```bash
pip install pmdarima pandas numpy
```

## Uso Básico (3 minutos)

### Passo 1: Prepare seus dados

Seu DataFrame deve ter estas colunas:
- `data`: Datas (datetime)
- `sku`: Código do produto
- `estoque_atual`: Quantidade em estoque

### Passo 2: Execute o código

```python
from sarima_estoque import PrevisorEstoqueSARIMA
import pandas as pd

# Seus dados (substitua pelo seu DataFrame real)
df_estoque = pd.DataFrame({
    'data': pd.date_range('2024-01-01', periods=90, freq='D'),
    'sku': 'MEU_PRODUTO',
    'estoque_atual': [100, 95, 90, 85, ...]  # seus dados aqui
})

# Cria o previsor
previsor = PrevisorEstoqueSARIMA(horizonte_previsao=7)

# Prepara série temporal
serie = previsor.preparar_serie_temporal(df_estoque, sku='MEU_PRODUTO')

# Treina modelo (auto_arima faz a mágica!)
modelo = previsor.treinar_modelo(serie, sku='MEU_PRODUTO')

# Gera previsão
previsao = previsor.prever(serie, modelo=modelo)

print(previsao)  # Sua previsão para os próximos 7 dias!
```

### Passo 3: Para múltiplos produtos

```python
# Processa todos os SKUs de uma vez
resultados = previsor.processar_lote(df_estoque)

# Resultado: DataFrame com previsões para todos os produtos
print(resultados)
```

---

## ⚡ Exemplo Completo em 30 segundos

Execute o arquivo de exemplo (a partir da raiz do repositório):

```bash
python exemplos/exemplo_uso_sarima.py
```

Isso vai:
1. ✅ Gerar dados simulados
2. ✅ Treinar modelos SARIMA
3. ✅ Gerar previsões
4. ✅ Mostrar resultados

---

## 📊 Integração com sua Fórmula de Elencação

```python
from exemplo_elencacao_completa import calcular_score_elencacao, calcular_score_risco_ruptura

# Usa a previsão SARIMA no cálculo de risco
estoque_previsto = previsao.mean()  # Média da previsão
risco = calcular_score_risco_ruptura(estoque_previsto, estoque_minimo=30)

# Combina com outros fatores
score_final = calcular_score_elencacao(
    margem_contribuicao=0.6,
    giro_estoque=0.5,
    risco_ruptura=risco
)
```

---

## ❓ Perguntas Frequentes

### Quantos dados eu preciso?

**Mínimo:** 30 observações (dias) por produto  
**Recomendado:** 60-90 dias ou mais

### O modelo funciona para produtos novos?

Não. Para produtos novos (sem histórico), use métodos alternativos:
- Média móvel simples
- Previsão baseada em produtos similares
- Métodos estatísticos mais simples

### Posso ajustar os parâmetros do SARIMA?

Sim! O `auto_arima` escolhe automaticamente, mas você pode customizar. Veja `exemplo_uso_sarima.py` → `exemplo_parametros_avancados()`

### Como integrar com minha API?

Veja `exemplo_uso_sarima.py` → `exemplo_com_dados_reais_api()` para ver a estrutura.

---

## 🆘 Problemas Comuns

### Erro: "Dados insuficientes"

**Causa:** Menos de 30 observações  
**Solução:** Use mais dados históricos ou métodos alternativos

### Modelo demora muito para treinar

**Causa:** Muitos produtos ou muitos parâmetros testados  
**Solução:** 
- Limite a busca: `max_p=3, max_q=3`
- Processe em lote menor
- Use `n_jobs=-1` para paralelizar (já está no código)

### Previsões não fazem sentido

**Causa:** Série muito irregular ou com outliers  
**Solução:** 
- Limpe os dados (remova outliers)
- Verifique se há padrões sazonais
- Tente ajustar período sazonal (`m`)

---

## 📚 Próximos Passos

1. ✅ Execute `exemplos/exemplo_uso_sarima.py` para ver exemplos
2. ✅ Leia `README_SARIMA.md` para entender melhor os conceitos
3. ✅ Veja `exemplo_elencacao_completa.py` para integração completa
4. ✅ Adapte para seus dados reais da API

---

**Dúvidas?** Consulte a documentação completa em `README_SARIMA.md` ou a documentação do `pmdarima`: https://alkaline-ml.com/pmdarima/


