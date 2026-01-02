"""
Script para organizar estrutura do repositório
Cria estrutura de pastas lógica e organiza arquivos
"""

import shutil
from pathlib import Path

# Estrutura de pastas proposta
ESTRUTURA = {
    'analises': [
        'analise_exploratoria_sazonalidade.py',
        'analise_box_jenkins_sarima.py',
        'analise_sazonalidade_padroes.png',
        'relatorio_analise_sazonalidade.txt',
        'README_ANALISE_EXPLORATORIA.md',
    ],
    'modelos': [
        'comparacao_modelos_previsao.py',
        'comparacao_top_skus.py',
        'comparacao_top_skus_otimizado.py',
        'comparacao_modelos_*.png',
        'relatorio_comparacao_*.txt',
        'README_COMPARACAO_MODELOS.md',
        'resultados_comparacao',
    ],
    'validacao': [
        'validacao_walk_forward_sarima.py',
        'teste_tempo_processamento.py',
        'tratamento_outliers_sarima.py',
    ],
    'previsoes': [
        'teste_sarima_produto.py',
        'previsao_sarima_*.png',
    ],
    'documentacao': [
        'README_SARIMA.md',
        'GUIA_RAPIDO.md',
        'GUIA_RAPIDO_EXPLICACAO_FERRAMENTAS.md',
        'EXPLICACAO_RESULTADOS_SARIMA.md',
        'DOCUMENTACAO_TECNICA_FERRAMENTAS.md',
        'ANALISE_BOX_JENKINS.md',
        'CHECKLIST_BOX_JENKINS.md',
        'explicacao_ferramentas_sarima.pdf',
    ],
    'exemplos': [
        'exemplo_uso_sarima.py',
        'exemplo_elencacao_completa.py',
    ],
}

def organizar():
    """Organiza arquivos nas pastas"""
    print("Organizando repositorio...")
    
    # Cria pastas
    for pasta in ESTRUTURA.keys():
        Path(pasta).mkdir(exist_ok=True)
        print(f"[OK] Pasta criada/verificada: {pasta}/")
    
    print("\n[INFO] Estrutura de pastas criada.")
    print("[INFO] Arquivos devem ser movidos manualmente para manter controle.")
    print("\nEstrutura proposta:")
    for pasta, arquivos in ESTRUTURA.items():
        print(f"\n  {pasta}/")
        for arq in arquivos[:3]:  # Mostra primeiros 3
            print(f"    - {arq}")
        if len(arquivos) > 3:
            print(f"    ... e mais {len(arquivos) - 3} arquivos")

if __name__ == "__main__":
    organizar()

