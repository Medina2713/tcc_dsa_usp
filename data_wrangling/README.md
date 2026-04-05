# Data Wrangling - Preparação de Dados para SARIMA

Este diretório contém scripts para preparar dados brutos de **histórico de estoque** (`historico_estoque`: sku, created_at, saldo) para uso com modelos de **previsão de estoque** (SARIMA, ARIMA, Holt-Winters, etc.). O pipeline TCC (`gerar_figuras_tcc.py`) procura `DB/historico_estoque_atual.csv` ou `DB/historico_estoque.csv`, gera **`DB/historico_estoque_atual_processado.csv`** e executa o wrangling automaticamente quando necessário. Os defaults internos de `dw_historico.py` podem ainda referir-se a `historico_estoque.csv`; o pipeline da raiz usa os ficheiros `*_atual_*` quando presentes.

## 📁 Arquivos

- **`dw_historico.py`**: Script principal para processar dados do histórico de estoque

## 🔧 Uso

### Processar Histórico de Estoque

```python
from data_wrangling.dw_historico import processar_historico_estoque

# Processa e salva automaticamente
df_processado = processar_historico_estoque(
    caminho_entrada='DB/historico_estoque.csv',
    caminho_saida='DB/historico_estoque_processado.csv',
    min_observacoes=30,
    criar_serie_completa=True
)
```

Ou execute diretamente:

```bash
python data_wrangling/dw_historico.py
```

## 📊 Pipeline de Processamento

O script `dw_historico.py` executa os seguintes passos:

1. **Carregar dados**: Lê o arquivo CSV original
2. **Limpar dados**: Remove registros inválidos (SKU nulo, saldo negativo, data inválida)
3. **Agregar por dia**: Agrupa múltiplos registros do mesmo SKU no mesmo dia (usa último saldo do dia)
4. **Criar série completa**: Preenche gaps nas séries temporais (datas faltantes)
5. **Filtrar SKUs**: Mantém apenas SKUs com número mínimo de observações (padrão: 30)
6. **Formatar**: Ajusta formato para o módulo SARIMA (colunas: data, sku, estoque_atual)

## 📋 Formato de Entrada

O arquivo `historico_estoque.csv` deve conter:
- **sku**: Código do produto
- **created_at**: Data/hora do registro
- **saldo**: Quantidade em estoque

## 📋 Formato de Saída

O arquivo processado terá:
- **data**: Data (datetime, apenas data, sem hora)
- **sku**: Código do produto (string)
- **estoque_atual**: Quantidade em estoque (numérico)

## ⚙️ Parâmetros

- `min_observacoes`: Número mínimo de observações por SKU (padrão: 30)
- `criar_serie_completa`: Se True, preenche gaps nas séries temporais (padrão: True)
- `data_inicio`: Data de início para série completa (opcional)
- `data_fim`: Data de fim para série completa (opcional)

## 📝 Notas

- O script agrega múltiplos registros do mesmo SKU no mesmo dia usando o **último saldo do dia**
- Gaps nas séries temporais são preenchidos com **forward fill** (último valor conhecido)
- SKUs com menos de 30 observações são filtrados (requisito mínimo para SARIMA)

