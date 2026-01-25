# 🚀 Guia Rápido: Como Explicar Cada Ferramenta

**Contexto do projeto:** Os modelos (SARIMA, ARIMA, Holt-Winters, Média Móvel) preveem **estoque (saldo)**, não vendas. A previsão alimenta o **terceiro pilar** da elencação (GP(t) = soma das previsões de estoque) e **sinaliza necessidade de reposição**. Pipeline TCC: `gerar_figuras_tcc.py` gera figuras 1–7, Tabela 2 e elencação final. Ver `COMO_GERAR_FIGURAS_TCC.md` e `CRITERIOS_SELECAO_ANALISE_TEMPORAL.md`.

---

## 📋 Estrutura de Explicação (Use para TODAS as ferramentas)

Para cada ferramenta, siga esta estrutura de 5 pontos:

1. **O QUE É** (1 frase)
2. **POR QUE USAR** (2-3 razões)
3. **COMO FUNCIONA** (mecanismo básico)
4. **RESULTADO ESPERADO** (o que obtemos)
5. **INTERPRETAÇÃO** (como ler os resultados)

---

## 🔍 FERRAMENTAS DE IDENTIFICAÇÃO

### 1. Teste ADF (Estacionariedade)

**O QUE É:**
"Verifica se a série de estoque é estacionária, ou seja, se a média e variância são constantes ao longo do tempo."

**POR QUE USAR:**
- Modelos SARIMA requerem séries estacionárias
- Se não estacionária, previsões podem ser enviesadas
- Indica necessidade de diferenciação (parâmetro `d`)

**COMO FUNCIONA:**
- Testa hipótese: série tem raiz unitária (não estacionária) vs. é estacionária
- Calcula estatística ADF e p-value
- Auto-ARIMA aplica diferenciação automaticamente se necessário

**RESULTADO:**
- P-value do teste ADF
- Conclusão: estacionária ou não

**INTERPRETAÇÃO:**
- **p-value < 0.05**: Série é estacionária ✅ (pode prosseguir)
- **p-value ≥ 0.05**: Série não estacionária ❌ (diferenciação necessária)

**FRASE-CHAVE:**
"O teste ADF garante que nossa série atende ao requisito fundamental do SARIMA: estacionariedade."

---

### 2. ACF e PACF

**O QUE É:**
"ACF mede correlação entre valores da série em diferentes lags. PACF mede correlação direta, removendo efeitos intermediários."

**POR QUE USAR:**
- Identifica ordem dos parâmetros `p` (AR) e `q` (MA)
- Detecta padrões sazonais (picos em lags específicos)
- Ajuda a entender estrutura temporal dos dados

**COMO FUNCIONA:**
- Calcula correlação para cada lag (1, 2, 3, ...)
- Visualiza em gráficos com intervalos de confiança
- Identifica onde há "corte abrupto" (indica ordem)

**RESULTADO:**
- Gráficos de ACF e PACF
- Lags significativos identificados

**INTERPRETAÇÃO:**
- **PACF corta no lag k**: `p = k` (ordem AR)
- **ACF corta no lag k**: `q = k` (ordem MA)
- **Picos em lags 7, 14, 21**: Sazonalidade semanal (`m = 7`)

**FRASE-CHAVE:**
"ACF e PACF são como 'impressões digitais' da série, revelando sua estrutura temporal e ajudando a definir os parâmetros do modelo."

---

### 3. Decomposição Sazonal

**O QUE É:**
"Separa a série em três componentes: tendência (movimento de longo prazo), sazonalidade (padrões repetitivos) e resíduo (ruído aleatório)."

**POR QUE USAR:**
- Confirma presença de sazonalidade
- Identifica período sazonal (ex: 30 dias = mensal)
- Calcula força da sazonalidade (quão importante é)
- Visualiza componentes separadamente

**COMO FUNCIONA:**
- Modelo aditivo: Série = Tendência + Sazonalidade + Resíduo
- Calcula cada componente usando médias móveis
- Força = Var(Sazonalidade) / [Var(Sazonalidade) + Var(Resíduo)]

**RESULTADO:**
- Componentes separados (tendência, sazonalidade, resíduo)
- Força da sazonalidade (0 a 1)

**INTERPRETAÇÃO:**
- **Força > 0.5**: Sazonalidade forte (importante modelar) ✅
- **Força < 0.5**: Sazonalidade fraca (pode ignorar)
- **Período identificado**: Define parâmetro `m` do SARIMA

**FRASE-CHAVE:**
"A decomposição sazonal confirma que há padrões repetitivos em nossos dados de estoque, especialmente em outubro e dezembro, justificando o uso de SARIMA em vez de ARIMA simples."

---

## ⚙️ FERRAMENTAS DE ESTIMAÇÃO

### 4. Auto-ARIMA

**O QUE É:**
"Algoritmo que automaticamente encontra os melhores parâmetros (p, d, q) x (P, D, Q, s) para o modelo SARIMA, testando múltiplas combinações."

**POR QUE USAR:**
- **Escalabilidade**: Testar manualmente é impraticável (1000+ combinações por SKU)
- **Objetividade**: Usa critérios estatísticos (AIC) em vez de intuição
- **Eficiência**: Algoritmo stepwise reduz tempo de busca
- **Reprodutibilidade**: Mesmo processo para todos os SKUs

**COMO FUNCIONA:**
1. Começa com modelo simples
2. Testa adicionar/remover parâmetros
3. Escolhe combinação com menor AIC
4. Para quando não há melhoria

**RESULTADO:**
- Modelo SARIMA com parâmetros otimizados
- Valor de AIC (menor = melhor)

**INTERPRETAÇÃO:**
- **AIC menor**: Modelo melhor (equilibra ajuste e complexidade)
- **Parâmetros encontrados**: Ex: (2,1,1) x (1,1,1,30) = ARIMA(2,1,1) com sazonalidade (1,1,1) de período 30

**FRASE-CHAVE:**
"Auto-ARIMA permite processar centenas de produtos automaticamente, encontrando o melhor modelo para cada um baseado em critérios estatísticos objetivos, não em tentativa e erro."

---

### 5. Critério AIC

**O QUE É:**
"Critério que compara modelos equilibrando qualidade do ajuste e complexidade, prevenindo overfitting."

**POR QUE USAR:**
- **Previne overfitting**: Modelos muito complexos ajustam bem aos dados de treino mas falham em prever
- **Comparação objetiva**: Permite escolher entre modelos diferentes
- **Amplamente aceito**: Padrão na literatura estatística

**COMO FUNCIONA:**
- Fórmula: AIC = -2 × log(Likelihood) + 2 × k
- Penaliza número de parâmetros (k)
- Menor AIC = melhor modelo

**RESULTADO:**
- Valor de AIC para cada modelo testado

**INTERPRETAÇÃO:**
- **Menor AIC**: Melhor modelo ✅
- **Diferença > 2**: Modelo significativamente melhor
- **Diferença < 2**: Modelos equivalentes

**FRASE-CHAVE:**
"O AIC garante que escolhemos um modelo que se ajusta bem aos dados sem ser excessivamente complexo, prevenindo overfitting e garantindo boas previsões futuras."

---

## 🔬 FERRAMENTAS DE DIAGNÓSTICO

### 6. Teste de Ljung-Box

**O QUE É:**
"Verifica se os resíduos do modelo são não correlacionados (ruído branco), ou seja, se o modelo capturou toda a informação disponível."

**POR QUE USAR:**
- **Suposição do SARIMA**: Resíduos devem ser ruído branco
- **Valida qualidade do modelo**: Se resíduos têm padrão, modelo pode melhorar
- **Garante adequação**: Modelo adequado quando resíduos são aleatórios

**COMO FUNCIONA:**
- Testa autocorrelação dos resíduos em múltiplos lags
- Calcula estatística Q e p-value
- H₀: Resíduos são não correlacionados

**RESULTADO:**
- Estatística Ljung-Box
- P-value do teste

**INTERPRETAÇÃO:**
- **p-value > 0.05**: Resíduos são ruído branco ✅ (modelo adequado)
- **p-value ≤ 0.05**: Resíduos são correlacionados ❌ (modelo pode melhorar)

**FRASE-CHAVE:**
"O teste de Ljung-Box valida que nosso modelo capturou todos os padrões disponíveis nos dados. Se os resíduos são aleatórios, significa que não há mais informação a ser extraída."

---

### 7. Testes de Normalidade

**O QUE É:**
"Verifica se os resíduos seguem distribuição normal, usando três testes diferentes para robustez."

**POR QUE USAR:**
- **Suposição do SARIMA**: Resíduos devem ser normais para intervalos de confiança válidos
- **Robustez**: Três testes diferentes aumentam confiança na conclusão
- **Validação completa**: Cada teste funciona melhor em diferentes situações

**COMO FUNCIONA:**
- **Shapiro-Wilk**: Para amostras pequenas/médias
- **Jarque-Bera**: Testa assimetria e curtose
- **Anderson-Darling**: Teste robusto, detecta desvios nas caudas

**RESULTADO:**
- P-values dos três testes
- Conclusão sobre normalidade

**INTERPRETAÇÃO:**
- **Todos p-values > 0.05**: Resíduos são normais ✅
- **Algum p-value ≤ 0.05**: Resíduos podem não ser normais ⚠️
- **Nota**: Resíduos não normais não invalidam o modelo, mas podem afetar intervalos de confiança

**FRASE-CHAVE:**
"Usamos três testes de normalidade diferentes para garantir robustez. Mesmo que resíduos não sejam perfeitamente normais, isso não invalida o modelo, mas nos alerta sobre a precisão dos intervalos de confiança."

---

### 8. Teste de Heterocedasticidade (ARCH)

**O QUE É:**
"Verifica se a variância dos resíduos é constante ao longo do tempo (homocedasticidade)."

**POR QUE USAR:**
- **Suposição do SARIMA**: Variância constante dos resíduos
- **Intervalos de confiança**: Heterocedasticidade pode torná-los incorretos
- **Eventos especiais**: Pode indicar períodos de maior volatilidade (ex: Black Friday)

**COMO FUNCIONA:**
- Testa se variância dos resíduos muda ao longo do tempo
- Usa teste LM (Lagrange Multiplier)
- H₀: Homocedasticidade (variância constante)

**RESULTADO:**
- P-value do teste ARCH
- Conclusão sobre homocedasticidade

**INTERPRETAÇÃO:**
- **p-value > 0.05**: Homocedástico ✅ (variância constante)
- **p-value ≤ 0.05**: Heterocedástico ❌ (variância não constante)

**FRASE-CHAVE:**
"O teste ARCH garante que a variabilidade dos erros do modelo é constante, o que é necessário para intervalos de confiança confiáveis. Se detectarmos heterocedasticidade, podemos considerar modelos GARCH."

---

## ✅ FERRAMENTAS DE VALIDAÇÃO

### 9. Validação Walk-Forward

**O QUE É:**
"Método de validação que respeita a ordem temporal: treina com dados do passado e testa em dados futuros, expandindo a janela de treino progressivamente."

**POR QUE USAR:**
- **Ordem temporal**: Séries temporais têm ordem, não podemos embaralhar
- **Simula uso real**: Treina com passado, prevê futuro (como será usado)
- **Testa estabilidade**: Verifica se modelo é consistente ao longo do tempo
- **Método correto**: Padrão para validação de séries temporais

**COMO FUNCIONA:**
```
Fold 1: Treina M1-M6, Testa M7
Fold 2: Treina M1-M7, Testa M8  ← Expandiu!
Fold 3: Treina M1-M8, Testa M9  ← Expandiu!
```

**RESULTADO:**
- Métricas por fold (MAE, RMSE, MAPE)
- Análise de estabilidade

**INTERPRETAÇÃO:**
- **Métricas consistentes**: Modelo estável ✅
- **Métricas variam muito**: Modelo instável ⚠️
- **Média das métricas**: Performance esperada do modelo

**FRASE-CHAVE:**
"Walk-forward é o método correto para validar séries temporais porque respeita a ordem dos dados e simula exatamente como o modelo será usado na prática: treinar com histórico e prever o futuro."

---

## 🛠️ FERRAMENTAS DE TRATAMENTO

### 10. Tratamento de Outliers

**O QUE É:**
"Identifica e trata valores que se desviam significativamente do padrão normal, como picos de demanda em eventos especiais (Dia das Crianças, Black Friday)."

**POR QUE USAR:**
- **Distorcem modelo**: Outliers podem fazer modelo aprender padrões incorretos
- **Afetam previsões**: Picos podem fazer modelo superestimar demanda futura
- **Eventos especiais**: Em e-commerce, eventos geram picos que não são padrão normal

**COMO FUNCIONA:**
- **IQR**: Identifica valores fora de Q1-1.5×IQR a Q3+1.5×IQR
- **Z-Score**: Identifica valores além de 3 desvios padrão
- **Tratamento**: Suavização (substitui por média móvel) preserva informação

**RESULTADO:**
- Série com outliers tratados
- Estatísticas sobre outliers detectados

**INTERPRETAÇÃO:**
- **Outliers detectados**: Valores que se desviam do padrão
- **Série tratada**: Pronta para modelagem sem distorções

**FRASE-CHAVE:**
"Tratamos outliers porque eventos especiais como Black Friday geram picos que não representam o padrão normal de demanda. Se não tratados, o modelo pode superestimar demanda futura baseado nesses eventos únicos."

---

## 📊 MÉTRICAS DE AVALIAÇÃO

### 11. MAE (Mean Absolute Error)

**O QUE É:**
"Erro médio absoluto entre valores reais e previstos."

**POR QUE USAR:**
- Fácil de interpretar (mesma unidade dos dados)
- Robusto a outliers
- Intuitivo para stakeholders

**INTERPRETAÇÃO:**
- MAE = 5: Erro médio de 5 unidades
- Menor = melhor

---

### 12. RMSE (Root Mean Squared Error)

**O QUE É:**
"Erro quadrático médio, dando mais peso a erros grandes."

**POR QUE USAR:**
- Penaliza erros grandes (importante para estoque)
- Padrão na literatura
- Propriedades matemáticas úteis

**INTERPRETAÇÃO:**
- RMSE ≥ MAE (sempre)
- Diferença grande indica presença de erros grandes
- Menor = melhor

---

### 13. MAPE (Mean Absolute Percentage Error)

**O QUE É:**
"Erro percentual médio, útil para comparar modelos em diferentes escalas."

**POR QUE USAR:**
- Comparável entre SKUs diferentes
- Fácil de comunicar ("erro de 10%")
- Útil para negócio

**INTERPRETAÇÃO:**
- < 10%: Excelente
- 10-20%: Bom
- 20-50%: Razoável
- > 50%: Precisa melhorar

---

## 🎯 ESTRATÉGIA DE APRESENTAÇÃO

### Ordem Recomendada

1. **Problema**: Por que precisamos prever estoque?
2. **Metodologia**: Box-Jenkins (padrão-ouro)
3. **Identificação**: ADF, ACF/PACF, Decomposição
4. **Estimação**: Auto-ARIMA com AIC
5. **Diagnóstico**: Ljung-Box, Normalidade, ARCH
6. **Validação**: Walk-Forward
7. **Tratamento**: Outliers
8. **Resultados**: Métricas e previsões

### Frases de Transição

- "Para garantir rigor estatístico, seguimos a metodologia Box-Jenkins..."
- "Antes de estimar o modelo, precisamos identificar suas características..."
- "Após estimar, validamos se o modelo é adequado através de testes de diagnóstico..."
- "Para garantir que o modelo funciona bem em dados novos, usamos validação walk-forward..."

---

## ✅ CHECKLIST ANTES DE APRESENTAR

- [ ] Sei explicar o que cada ferramenta faz (1 frase)
- [ ] Sei justificar por que é necessária (2-3 razões)
- [ ] Entendo como funciona (mecanismo básico)
- [ ] Consigo interpretar os resultados
- [ ] Tenho exemplos práticos do nosso projeto
- [ ] Conheço as referências principais
- [ ] Sei responder perguntas comuns

---

**Guia criado para TCC MBA Data Science & Analytics - 2024**

