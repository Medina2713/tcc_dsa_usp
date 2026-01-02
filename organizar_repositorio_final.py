"""
Script para organizar repositório final com estrutura clara
Cria estrutura de pastas e organiza arquivos para facilitar uso
"""

import shutil
from pathlib import Path
import glob

# Estrutura de pastas
ESTRUTURA = {
    'dados': [],  # Para dados processados intermediários
    'resultados': [],  # Para todos os resultados (CSV, PNG, relatórios)
    'analises': [],  # Scripts de análise exploratória
    'modelos': [],  # Scripts de modelos e comparação
    'validacao': [],  # Scripts de validação e testes
    'previsoes': [],  # Scripts de previsão
    'documentacao': [],  # Documentação
    'exemplos': [],  # Exemplos de uso
}

# Mapeamento de arquivos para pastas
MAPEAMENTO_ARQUIVOS = {
    'analises': [
        'analise_exploratoria_sazonalidade.py',
        'analise_box_jenkins_sarima.py',
        'README_ANALISE_EXPLORATORIA.md',
        'ANALISE_BOX_JENKINS.md',
        'CHECKLIST_BOX_JENKINS.md',
    ],
    'modelos': [
        'comparacao_modelos_previsao.py',
        'comparacao_top_skus.py',
        'comparacao_top_skus_otimizado.py',
        'README_COMPARACAO_MODELOS.md',
        'README_OTIMIZACAO.md',
    ],
    'validacao': [
        'validacao_walk_forward_sarima.py',
        'teste_tempo_processamento.py',
        'tratamento_outliers_sarima.py',
        'validar_extracao_vendas.py',
        'calcular_metricas_elencacao.py',
    ],
    'previsoes': [
        'teste_sarima_produto.py',
        'teste_elencacao_3_skus.py',
    ],
    'exemplos': [
        'exemplo_uso_sarima.py',
        'exemplo_elencacao_completa.py',
    ],
    'documentacao': [
        'README_SARIMA.md',
        'GUIA_RAPIDO.md',
        'GUIA_RAPIDO_EXPLICACAO_FERRAMENTAS.md',
        'EXPLICACAO_RESULTADOS_SARIMA.md',
        'DOCUMENTACAO_TECNICA_FERRAMENTAS.md',
        'explicacao_ferramentas_sarima.pdf',
        'ORGANIZACAO_REPOSITORIO.md',
        'RESUMO_MELHORIAS.md',
        'RESUMO_VALIDACAO_VENDAS.md',
    ],
    'resultados': [
        '*.csv',
        '*.png',
        '*.txt',
        'metricas_elencacao.csv',
        'resultado_elencacao_3_skus.csv',
        'metricas_vendas_para_elencacao.csv',
        'relatorio_*.txt',
        'previsao_sarima_*.png',
        'comparacao_modelos_*.png',
        'analise_sazonalidade_*.png',
    ],
}

# Arquivos que ficam na raiz
ARQUIVOS_RAIZ = [
    'sarima_estoque.py',
    'requirements_sarima.txt',
    'README.md',
    '.gitignore',
]


def criar_estrutura_pastas():
    """Cria estrutura de pastas"""
    print("Criando estrutura de pastas...")
    for pasta in ESTRUTURA.keys():
        Path(pasta).mkdir(exist_ok=True)
        print(f"  [OK] {pasta}/")
    print()


def mover_arquivos_resultados():
    """Move arquivos de resultados para pasta resultados/"""
    print("Movendo arquivos de resultados...")
    
    resultados_path = Path('resultados')
    movidos = 0
    
    # Padrões para arquivos de resultado
    padroes = [
        '*.csv',
        '*.png',
        'relatorio_*.txt',
        'metricas_*.csv',
        'resultado_*.csv',
        'previsao_*.png',
        'comparacao_*.png',
        'analise_*.png',
    ]
    
    for padrao in padroes:
        arquivos = glob.glob(padrao)
        for arquivo in arquivos:
            origem = Path(arquivo)
            # Não move se já está na pasta resultados ou se é arquivo importante da raiz
            if origem.parent == resultados_path or origem.name in ARQUIVOS_RAIZ:
                continue
            
            # Não move arquivos que estão em subpastas (já organizados)
            if len(origem.parts) > 1 and origem.parts[0] in ESTRUTURA.keys():
                continue
            
            try:
                destino = resultados_path / origem.name
                shutil.move(str(origem), str(destino))
                print(f"  [OK] {arquivo} -> resultados/")
                movidos += 1
            except Exception as e:
                print(f"  [AVISO] {arquivo}: {str(e)}")
    
    print(f"\n[OK] {movidos} arquivos de resultados movidos\n")


def organizar_arquivos_scripts():
    """Move scripts para pastas apropriadas"""
    print("Organizando scripts...")
    
    movidos = 0
    
    for pasta, arquivos in MAPEAMENTO_ARQUIVOS.items():
        if pasta == 'resultados':  # Já tratado acima
            continue
            
        pasta_path = Path(pasta)
        
        for arquivo in arquivos:
            # Se tem wildcard, usa glob
            if '*' in arquivo:
                matches = glob.glob(arquivo)
                for match in matches:
                    origem = Path(match)
                    if origem.parent == pasta_path or origem.parent.name in ESTRUTURA.keys():
                        continue
                    try:
                        destino = pasta_path / origem.name
                        shutil.move(str(origem), str(destino))
                        print(f"  [OK] {match} -> {pasta}/")
                        movidos += 1
                    except Exception as e:
                        print(f"  [AVISO] {match}: {str(e)}")
            else:
                origem = Path(arquivo)
                if not origem.exists():
                    continue
                if origem.parent == pasta_path or origem.parent.name in ESTRUTURA.keys():
                    continue
                try:
                    destino = pasta_path / origem.name
                    shutil.move(str(origem), str(destino))
                    print(f"  [OK] {arquivo} -> {pasta}/")
                    movidos += 1
                except Exception as e:
                    print(f"  [AVISO] {arquivo}: {str(e)}")
    
    print(f"\n[OK] {movidos} scripts organizados\n")


def organizar():
    """Função principal"""
    print("=" * 80)
    print("ORGANIZACAO FINAL DO REPOSITORIO")
    print("=" * 80)
    print()
    
    criar_estrutura_pastas()
    mover_arquivos_resultados()
    organizar_arquivos_scripts()
    
    print("=" * 80)
    print("ORGANIZACAO CONCLUIDA!")
    print("=" * 80)
    print("\nEstrutura criada:")
    print("  dados/          - Dados processados intermediários")
    print("  resultados/     - Todos os resultados (CSV, PNG, relatórios)")
    print("  analises/       - Scripts de análise exploratória")
    print("  modelos/        - Scripts de modelos e comparação")
    print("  validacao/      - Scripts de validação e testes")
    print("  previsoes/      - Scripts de previsão")
    print("  documentacao/   - Documentação completa")
    print("  exemplos/       - Exemplos de uso")
    print("\nArquivos na raiz:")
    for arq in ARQUIVOS_RAIZ:
        if Path(arq).exists():
            print(f"  - {arq}")


if __name__ == "__main__":
    organizar()

