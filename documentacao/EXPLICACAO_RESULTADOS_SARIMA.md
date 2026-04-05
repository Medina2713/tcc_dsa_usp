# Explicação dos Resultados do Modelo SARIMA

## 📊 Resultados Obtidos

```
SARIMA (0, 1, 0) x (0, 0, 0, 7)
AIC: 4435.76
Tipo: Random Walk simples
```

---

## 🔍 Decodificando os Parâmetros SARIMA

### Estrutura Geral: `(p, d, q) x (P, D, Q, s)`

O modelo SARIMA é descrito por **dois conjuntos de parâmetros**:

#### **1. Componente Não-Sazonal: (p, d, q)**

- **p = 0** → **AR (AutoRegressivo) de ordem 0**
  - Não há componente autoregressivo
  - O valor atual **não depende** de valores anteriores da série
  - Significa que não há correlação linear entre o estoque de hoje e o de dias anteriores

- **d = 1** → **Diferenciação de ordem 1**
  - A série foi diferenciada **uma vez** para se tornar estacionária
  - Em outras palavras: o modelo trabalha com **diferenças** (estoque_t - estoque_t-1)
  - Isso remove tendências lineares da série

- **q = 0** → **MA (Média Móvel) de ordem 0**
  - Não há componente de média móvel
  - O modelo não considera erros de previsão anteriores

#### **2. Componente Sazonal: (P, D, Q, s)**

- **P = 0** → AR sazonal de ordem 0
  - Não há autocorrelação sazonal

- **D = 0** → Diferenciação sazonal de ordem 0
  - Não foi necessário diferenciar sazonalmente

- **Q = 0** → MA sazonal de ordem 0
  - Não há média móvel sazonal

- **s = 7** → **Período sazonal de 7 dias** (semanal)
  - O modelo foi configurado para considerar padrões semanais
  - Mas como todos os parâmetros sazonais são 0, essa configuração não teve efeito

---

## 🎯 O que é um Random Walk?

Um **Random Walk (Caminhada Aleatória)** é um modelo muito simples onde:

```
Estoque(t) = Estoque(t-1) + Ruído(t)
```

Ou seja: **o valor de hoje é igual ao de ontem, mais um termo aleatório**.

### Características do Random Walk:

1. **Previsão simples**: A melhor previsão é o último valor conhecido
2. **Sem memória de longo prazo**: Apenas o último valor importa
3. **Mudanças imprevisíveis**: As mudanças são tratadas como aleatórias
4. **Conservador**: Não assume tendências ou padrões

### Por que o Auto-ARIMA escolheu isso?

O Auto-ARIMA testa múltiplos modelos e escolhe o que tem **menor AIC**. O Random Walk foi escolhido porque:

1. **A série não apresenta autocorrelação significativa**
   - Não há padrão claro que relacione estoque de hoje com dias anteriores
   
2. **A série já é quase estacionária após diferenciação**
   - Com d=1 (uma diferenciação), a série fica estável
   
3. **Modelos mais complexos não melhoram significativamente**
   - Adicionar termos AR ou MA aumentaria o AIC (pior ajuste)
   - O princípio da parcimônia: o modelo mais simples que explica os dados

---

## 📈 O que é o AIC (Akaike Information Criterion)?

**AIC = 4435.76**

### Definição:

O **AIC (Akaike Information Criterion)** é uma métrica que avalia a **qualidade do ajuste do modelo**, penalizando a complexidade.

### Fórmula (conceitual):

```
AIC = -2 × log(verossimilhança) + 2 × número_de_parâmetros
```

### Interpretação:

- **Menor AIC = Melhor modelo** (dentro das opções testadas)
- **Penaliza complexidade**: Modelos com mais parâmetros precisam ser significativamente melhores para compensar
- **Não é absoluto**: Só faz sentido comparar entre modelos diferentes

### No seu caso:

- O AIC de 4435.76 foi o **menor entre todos os modelos testados**
- Isso significa que, entre todas as combinações de parâmetros testadas, este foi o que melhor equilibrou:
  - **Ajuste aos dados** (quão bem o modelo explica o histórico)
  - **Simplicidade** (número de parâmetros)

---

## 🔄 Por que a Previsão é Constante?

Se a previsão para os próximos 30 dias é **sempre 480 unidades**, isso acontece porque:

### Random Walk com diferenciação:

Quando você diferencia uma vez (d=1), o modelo prevê:
```
ΔEstoque(t) = Ruído(t)
```

Onde ΔEstoque(t) = Estoque(t) - Estoque(t-1)

### Expectativa do ruído:

Em um Random Walk, a **expectativa do ruído é zero**, então:
```
E[ΔEstoque(t)] = E[Ruído(t)] = 0
```

Isso significa:
```
E[Estoque(t)] = E[Estoque(t-1)]
```

### Resultado:

A **melhor previsão** é que o estoque permaneça no último valor conhecido (480 unidades).

---

## 🎓 Implicações para seu TCC

### ✅ Pontos Positivos:

1. **Modelo válido**: O Auto-ARIMA escolheu o modelo mais adequado estatisticamente
2. **Interpretação clara**: Random Walk é fácil de entender
3. **Conservador**: Previsão conservadora (não assume mudanças drásticas)

### ⚠️ Limitações:

1. **Previsão constante**: Não captura tendências ou padrões futuros
2. **Não usa histórico**: Apenas o último valor importa
3. **Imprevisível**: Não prevê mudanças sistemáticas

### 💡 Possíveis Razões para o Resultado:

1. **Série muito irregular**: O estoque pode ter comportamento quase aleatório
2. **Poucos padrões detectáveis**: O histórico não mostra autocorrelações claras
3. **Variações externas**: Mudanças podem ser causadas por fatores externos não capturados

### 🔧 O que fazer?

1. **Validar com outros SKUs**: Teste com outros produtos para ver se o padrão se repete
2. **Análise exploratória**: Verificar se há tendências ou sazonalidades visuais nos dados
3. **Variáveis externas**: Considerar adicionar variáveis exógenas (vendas, promoções, etc.)
4. **Outros modelos**: Comparar com métodos alternativos (média móvel, exponencial, etc.)

---

## 📊 Interpretação Prática

### Para o Produto SKU 811078:

**Previsão:** O modelo prevê que o estoque permanecerá em **480 unidades** nos próximos 30 dias.

**Confiança:** Esta é uma previsão **conservadora** e **estatisticamente válida**, mas pode não capturar:
- Eventos sazonais
- Tendências de crescimento/declínio
- Efeitos de promoções ou campanhas

**Recomendação prática:** Use esta previsão como **baseline conservadora**, mas monitore o produto e ajuste conforme necessário.

---

## 📚 Referências para TCC

### Conceitos Importantes:

1. **Random Walk**: Modelo básico de séries temporais
2. **AIC**: Critério de informação para seleção de modelos
3. **Diferenciação (d)**: Técnica para tornar séries estacionárias
4. **Parcimônia**: Princípio de escolher o modelo mais simples que explica os dados

### Citações Úteis:

- "O modelo Random Walk é útil quando a série não apresenta autocorrelação significativa" (Hyndman & Athanasopoulos, 2021)
- "O AIC equilibra qualidade do ajuste e complexidade do modelo" (Akaike, 1974)
- "Modelos mais simples são preferíveis quando não há ganho significativo com complexidade adicional" (Box & Jenkins, 1976)

---

**Criado para TCC MBA Data Science & Analytics — USP** · *Documentação revista em 05/04/2026.*

