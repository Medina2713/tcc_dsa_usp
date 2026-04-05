# Validação: Extração de Métricas de Vendas para Elencação

## ✅ Validação Concluída

O sistema foi validado e **consegue extrair todas as métricas necessárias** do arquivo `venda_produtos_atual.csv` conforme a Tabela 2.2 (Normalização das Métricas).

## 📊 Métricas Validadas

### 1. **Rentabilidade R(t)** ✅
- **Fórmula:** R(t) = Média (Valor de Venda Unitário - Custo de Aquisição Unitário)
- **Dado de Origem:** `venda_produtos_atual.csv`
- **Colunas usadas:** `valor_unitario`, `custo_unitario`
- **Status:** ✅ Calculada com sucesso
- **Resultado:** Média geral de R$ 19.63 por SKU

### 2. **Margem Proporcional (Média)** ✅
- **Fórmula:** Média da coluna `margem_proporcional`
- **Dado de Origem:** `venda_produtos_atual.csv`
- **Coluna usada:** `margem_proporcional`
- **Status:** ✅ Extraída com sucesso
- **Resultado:** Média geral de 4.85%

### 3. **Quantidade Vendida Total (Soma)** ✅
- **Fórmula:** Soma da coluna `quantidade` por SKU
- **Dado de Origem:** `venda_produtos_atual.csv`
- **Coluna usada:** `quantidade`
- **Status:** ✅ Calculada com sucesso
- **Resultado:** Total de 46,272 unidades vendidas

### 4. **Nível de Urgência U(t)** ✅
- **Fórmula:** U(t) = Estoque Atual / Venda Média Diária Histórica
- **Dado de Origem:** 
  - `historico_estoque_atual.csv` (estoque atual)
  - `venda_produtos_atual.csv` (venda média diária)
- **Status:** ✅ Calculada com sucesso
- **Resultado:** Média geral de 8.3 dias de estoque

### 5. **Giro Futuro Previsto GP(t)** ✅
- **Fórmula:** Soma das Previsões de Vendas para os próximos N dias (Resultado do SARIMA)
- **Dado de Origem:** Resultado do SARIMA (já implementado)
- **Status:** ✅ Já implementado no módulo SARIMA

## 📁 Arquivos Criados

### 1. `validar_extracao_vendas.py`
Script de validação que verifica:
- Estrutura do CSV de vendas
- Extração de todas as métricas
- Cálculo correto das fórmulas

### 2. `calcular_metricas_elencacao.py`
Script principal para calcular métricas:
- **Rentabilidade R(t)**
- **Margem Proporcional (média)**
- **Quantidade Vendida Total (soma)**
- **Venda Média Diária Histórica**
- **Nível de Urgência U(t)**

### 3. `metricas_elencacao.csv`
Arquivo CSV com todas as métricas calculadas por SKU:
- 653 SKUs processados
- Todas as métricas consolidadas
- Pronto para uso na elencação

## 📋 Estrutura do CSV de Vendas

O arquivo `venda_produtos_atual.csv` contém as seguintes colunas necessárias:

| Coluna | Tipo | Uso |
|--------|------|-----|
| `sku` | string | Identificador do produto |
| `quantidade` | numeric | Soma para quantidade total vendida |
| `margem_proporcional` | numeric | Média para margem proporcional |
| `valor_unitario` | numeric | Média para cálculo de R(t) |
| `custo_unitario` | numeric | Média para cálculo de R(t) |
| `created_at` | datetime | Para cálculo de venda média diária |

## 🔍 Resultados da Validação

### Estatísticas Gerais:
- **Total de SKUs:** 653
- **Total de registros de vendas:** 32,133
- **Rentabilidade média:** R$ 19.63
- **Margem proporcional média:** 4.85%
- **Venda média diária:** 3.72 unidades/dia
- **Nível de urgência médio:** 8.3 dias

### Top 10 SKUs por Rentabilidade:
1. MK170-2: R$ 224.99
2. MK170-1: R$ 213.04
3. SP12-18: R$ 122.14
4. 7307: R$ 119.85
5. MK421: R$ 115.26

### SKUs com Maior Risco (menor nível de urgência):
- SKUs com estoque zerado (urgência = 0.0 dias)
- Requerem atenção imediata

## ✅ Conclusão

**Todas as métricas necessárias para a elencação foram validadas e podem ser extraídas corretamente do arquivo `venda_produtos_atual.csv`.**

O sistema está pronto para:
1. ✅ Calcular Rentabilidade R(t)
2. ✅ Extrair Margem Proporcional (média)
3. ✅ Calcular Quantidade Vendida Total (soma)
4. ✅ Calcular Nível de Urgência U(t)
5. ✅ Integrar com resultados do SARIMA para GP(t)

## 📝 Próximos Passos

1. Integrar `calcular_metricas_elencacao.py` com o script de elencação completo
2. Combinar R(t), U(t) e GP(t) na fórmula final de elencação
3. Aplicar normalização conforme Tabela 2.2
4. Gerar ranking de priorização

---

*Documentação do repositório revista em 05/04/2026.*
