# Organização do Repositório - Estrutura Proposta

## 📁 Estrutura de Pastas Criada

```
CODE/
├── analises/              # Scripts de análise exploratória
├── modelos/               # Scripts de comparação de modelos
├── validacao/             # Scripts de validação (walk-forward, etc)
├── previsoes/             # Scripts de previsão e testes
├── documentacao/          # Documentação (READMEs, guias)
├── exemplos/              # Exemplos de uso
├── data_wrangling/        # Scripts de preparação de dados
├── DB/                    # Dados (CSV)
└── resultados_comparacao/ # Resultados (criado pelo script otimizado)
```

## 📋 Mapeamento de Arquivos

### analises/
- `analise_exploratoria_sazonalidade.py`
- `analise_box_jenkins_sarima.py`
- `analise_sazonalidade_padroes.png`
- `relatorio_analise_sazonalidade.txt`
- `README_ANALISE_EXPLORATORIA.md`

### modelos/
- `comparacao_modelos_previsao.py`
- `comparacao_top_skus.py`
- `comparacao_top_skus_otimizado.py` ⭐ **NOVO - Use este!**
- `comparacao_modelos_*.png`
- `relatorio_comparacao_*.txt`
- `README_COMPARACAO_MODELOS.md`
- `README_OTIMIZACAO.md` ⭐ **NOVO**
- `resultados_comparacao/` (pasta criada pelo script)

### validacao/
- `validacao_walk_forward_sarima.py`
- `teste_tempo_processamento.py`
- `tratamento_outliers_sarima.py`

### previsoes/
- `teste_sarima_produto.py`
- `previsao_sarima_*.png`

### documentacao/
- `README_SARIMA.md`
- `GUIA_RAPIDO.md`
- `GUIA_RAPIDO_EXPLICACAO_FERRAMENTAS.md`
- `EXPLICACAO_RESULTADOS_SARIMA.md`
- `DOCUMENTACAO_TECNICA_FERRAMENTAS.md`
- `ANALISE_BOX_JENKINS.md`
- `CHECKLIST_BOX_JENKINS.md`
- `explicacao_ferramentas_sarima.pdf`

### exemplos/
- `exemplo_uso_sarima.py`
- `exemplo_elencacao_completa.py`

### data_wrangling/
- `dw_historico.py`
- `exemplo_uso.py`
- `README.md`

### Arquivos Principais (raiz)
- **`gerar_figuras_tcc.py`** — Script mestre TCC: figuras 1–7, Tabela 2, **elencação final** (ranking R(t), U(t), GP(t)). Pipeline 300→10 melhores. Modelos preveem **estoque (saldo)**, não vendas; terceiro pilar **sinaliza reposição**. Ver `documentacao/COMO_GERAR_FIGURAS_TCC.md`.
- `sarima_estoque.py` — Módulo SARIMA (em `previsoes/`)
- `requirements_sarima.txt` — Dependências
- `ORGANIZACAO_REPOSITORIO.md` — Este arquivo

## 🎯 Scripts Principais

### Pipeline TCC (figuras, tabela, elencação):
1. **`gerar_figuras_tcc.py`** ⭐ **RECOMENDADO**
   - Gera figuras 1–7, Tabela 2, `elencacao_final.csv`
   - Pipeline: exploratória → 300 candidatos → 10 melhores → figuras e elencação
   - Retorna DataFrame do ranking (valor final da ferramenta de elencação)

### Para comparação de modelos (uso direto):
1. **`modelos/comparacao_modelos_previsao.py`** — usado pelo pipeline TCC
2. **`modelos/comparacao_top_skus_otimizado.py`** — versão otimizada, checkpoint
3. `modelos/comparacao_top_skus.py` — versão antiga

### Para Análise Exploratória:
- `analise_exploratoria_sazonalidade.py`

### Para Testes Individuais:
- `teste_sarima_produto.py`

## 📝 Notas

- **Não movemos arquivos automaticamente** - você pode mover manualmente se quiser
- **Estrutura de pastas é opcional** - scripts funcionam na raiz também
- **Pastas criadas são apenas organização** - não afeta funcionalidade

---

**Use `gerar_figuras_tcc.py` para figuras TCC e elencação final.** Para comparação de modelos isolada: `comparacao_modelos_previsao.py` ou `comparacao_top_skus_otimizado.py`. ✅

