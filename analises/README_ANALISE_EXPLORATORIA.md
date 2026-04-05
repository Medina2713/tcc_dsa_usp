# Análise Exploratória: Padrões Sazonais em Dados de Estoque

## 📋 Visão Geral

Este script realiza análise exploratória detalhada para identificar padrões sazonais nos dados de **estoque (saldo)**, especialmente em períodos de maior movimentação (outubro e dezembro para brinquedos). Gera **figuras 1–4** do TCC (evolução estoque total, distribuição mensal, estoque médio por mês, série do SKU representativo) e alimenta o **pipeline TCC** (`gerar_figuras_tcc.py`) com top 10 e **top 300 candidatos** para comparação de modelos. Uso em modo TCC: `--tcc` (salva figura1.png … figura4.png em `resultados/figuras_tcc/`).

## 🎯 Objetivos

1. Verificar padrões visuais nas séries temporais
2. Analisar agregados mensais (médias, totais)
3. Comparar meses específicos (outubro vs dezembro vs outros)
4. Identificar produtos com padrões sazonais mais evidentes
5. Visualizar resultados com gráficos informativos

## 📦 Dependências

```bash
pip install pandas numpy matplotlib seaborn
```

## 🚀 Uso

A partir da **raiz do repositório**:

```bash
python analises/analise_exploratoria_sazonalidade.py
python analises/analise_exploratoria_sazonalidade.py --tcc
```

## 📊 Estrutura do Script

### PARTE 1: Carregamento de Dados

**Função:** `carregar_dados_processados()`

**O que faz:**
- Carrega o arquivo CSV processado de histórico de estoque
- Valida estrutura dos dados
- Exibe estatísticas básicas (total de registros, período, SKUs únicos)

**Input:** Caminho para arquivo CSV processado  
**Output:** DataFrame com colunas: `data`, `sku`, `estoque_atual`

---

### PARTE 2: Criação de Variáveis Temporais

**Função:** `adicionar_variaveis_temporais()`

**O que faz:**
- Extrai componentes temporais da data:
  - **ano, mês, dia**: Componentes básicos
  - **dia_semana**: 0=Segunda, 6=Domingo
  - **trimestre**: 1-4
  - **semana_ano**: Semana do ano (1-52)
  - **mes_nome**: Nome abreviado do mês (Jan, Fev, etc.)
- Cria flag `mes_alta_temporada`: True para outubro (10) e dezembro (12)

**Por que é importante:**
- Permite agrupar dados por diferentes períodos temporais
- Facilita comparações entre meses/trimestres
- Flag de alta temporada facilita análises comparativas

---

### PARTE 3: Análise de Agregados Mensais

**Função:** `analise_agregados_mensais()`

**O que faz:**
- Agrega dados por mês (ignorando ano) calculando:
  - **estoque_total**: Soma de todo estoque no mês
  - **estoque_medio**: Média de estoque por registro
  - **estoque_desvio**: Desvio padrão (variabilidade)
  - **observacoes**: Quantidade de registros
  - **skus_unicos**: Quantidade de SKUs únicos
- Compara meses de alta temporada (Out/Dez) vs outros meses

**Métricas calculadas:**
```
Estoque médio (Out/Dez) vs Estoque médio (outros)
Diferença absoluta e percentual
```

**Interpretação:**
- **Se estoque maior em Out/Dez**: Empresa prepara estoque para alta demanda (esperado)
- **Se estoque menor em Out/Dez**: Alta rotatividade (vendas rápidas)

---

### PARTE 4: Análise de Produtos Individuais (SKUs)

**Função:** `analise_por_sku_individual()`

**O que faz:**
- Para cada SKU, calcula:
  - **estoque_medio_geral**: Média geral de estoque
  - **estoque_medio_out_dez**: Média apenas nos meses Out/Dez
  - **estoque_medio_outros**: Média nos outros meses
  - **cv_mensal**: Coeficiente de variação entre meses (variabilidade)
  - **diferenca_alta_outros**: Diferença percentual entre alta temporada e outros meses
- Identifica produtos com maior variação sazonal

**Por que é importante:**
- Identifica quais produtos têm padrão sazonal mais claro
- Produtos com alta diferença são candidatos para modelos SARIMA com sazonalidade
- Ajuda a priorizar produtos para análise mais detalhada

---

### PARTE 5: Visualização dos Padrões Sazonais

**Função:** `visualizar_padroes_sazonais()`

**Gráficos gerados:**

1. **Evolução Temporal: Estoque Total Diário**
   - Linha temporal mostrando estoque total agregado ao longo do tempo
   - Permite visualizar tendências gerais e variações

2. **Boxplot por Mês**
   - Distribuição de estoque em cada mês
   - Mostra mediana, quartis e outliers
   - Outubro e Dezembro destacados em vermelho

3. **Estoque Médio por Mês (Bar Chart)**
   - Médias mensais comparadas visualmente
   - Outubro e Dezembro destacados em vermelho

4. **Série Temporal de Produto Específico**
   - Análise detalhada de um SKU individual
   - Destaque para pontos em Outubro/Dezembro

**Output:** Arquivo PNG `analise_sazonalidade_padroes.png`

---

### PARTE 6: Geração de Relatório Completo

**Função:** `gerar_relatorio_completo()`

**O que faz:**
- Compila todas as análises em relatório textual
- Inclui:
  - Resumo executivo (período, quantidade de dados)
  - Tabela completa de agregados mensais
  - Comparação estatística Out/Dez vs outros
  - Top 10 produtos com maior variação sazonal
  - Conclusões e interpretações

**Output:** Arquivo TXT `relatorio_analise_sazonalidade.txt`

---

## 📈 Interpretação dos Resultados

### Sinal de Padrão Sazonal Forte:

✅ **Estoque médio Out/Dez > Estoque médio outros meses**
- Indica preparação para alta demanda
- Padrão sazonal presente e capturável

✅ **Alta diferença percentual (>30%)**
- Padrão suficientemente forte para modelos sazonais

✅ **Produtos com alta variação individual**
- SKUs específicos mostram padrão claro

### Sinal de Padrão Sazonal Fraco:

⚠️ **Diferença pequena (<20%)**
- Padrão pode não ser estatisticamente significativo
- Ruído pode mascarar padrão

⚠️ **Alta variabilidade entre produtos**
- Alguns produtos têm padrão, outros não
- Modelos podem precisar ser específicos por produto

---

## 🔍 Resultados Esperados

Baseado na análise realizada, você deve encontrar:

1. **Agregados Mensais**: Tabela mostrando estatísticas por mês
2. **Comparação Out/Dez**: Diferença percentual entre alta temporada e outros meses
3. **Top Produtos**: Lista de SKUs com maior variação sazonal
4. **Visualizações**: 4 gráficos mostrando diferentes perspectivas
5. **Relatório**: Documento textual completo com todas as informações

---

## 📝 Arquivos Gerados

1. **analise_sazonalidade_padroes.png**
   - Gráficos visuais com 4 painéis
   - Formato: PNG, alta resolução (300 DPI)

2. **relatorio_analise_sazonalidade.txt**
   - Relatório textual completo
   - Formato: TXT, UTF-8

---

## 🎓 Uso para TCC

Este script fornece evidências quantitativas e visuais sobre padrões sazonais:

1. **Justificativa para modelos sazonais**: Se houver padrão claro, justifica uso de SARIMA com sazonalidade
2. **Identificação de produtos**: Produtos com padrão mais claro podem ter modelos específicos
3. **Validação de hipóteses**: Confirma ou refuta hipótese de sazonalidade em Out/Dez
4. **Documentação**: Relatório pode ser incluído no TCC como análise exploratória

---

## ⚙️ Parâmetros Ajustáveis

No código, você pode ajustar:

- **`top_n`** em `analise_por_sku_individual()`: Quantos produtos destacar (padrão: 10)
- **`sku_exemplo`** em `visualizar_padroes_sazonais()`: SKU específico para análise detalhada
- **Meses de alta temporada**: Alterar `[10, 12]` para outros meses se necessário

---

## 🔧 Troubleshooting

### Erro: "FileNotFoundError"
- Verifique se o arquivo `DB/historico_estoque_atual_processado.csv` existe
- Execute primeiro o script de data wrangling

### Gráficos não aparecem
- Verifique se matplotlib está instalado: `pip install matplotlib seaborn`
- Arquivo PNG será salvo mesmo se não aparecer na tela

### Memória insuficiente
- Para datasets muito grandes, o script faz amostragem no boxplot
- Ajuste `min(50000, len(df_plot))` se necessário

---

## 📚 Referências

- **Análise Exploratória de Dados**: Primeiro passo antes de modelagem
- **Visualizações Temporais**: Gráficos de séries temporais
- **Agregações**: Técnicas de agrupamento e sumarização

---

**Desenvolvido para TCC MBA Data Science & Analytics — USP** · *Documentação revista em 05/04/2026.*

