"""
Script para organizar arquivos nas pastas apropriadas
Move arquivos para estrutura organizada
"""

import shutil
from pathlib import Path
import glob

# Mapeamento de arquivos para pastas
MAPEAMENTO = {
    'analises': [
        'analise_exploratoria_sazonalidade.py',
        'analise_box_jenkins_sarima.py',
        'analise_sazonalidade_padroes.png',
        'relatorio_analise_sazonalidade.txt',
        'README_ANALISE_EXPLORATORIA.md',
        'ANALISE_BOX_JENKINS.md',
        'CHECKLIST_BOX_JENKINS.md',
    ],
    'modelos': [
        'comparacao_modelos_previsao.py',
        'comparacao_top_skus.py',
        'comparacao_top_skus_otimizado.py',
        'README_COMPARACAO_MODELOS.md',
    ],
    'validacao': [
        'validacao_walk_forward_sarima.py',
        'teste_tempo_processamento.py',
        'tratamento_outliers_sarima.py',
    ],
    'previsoes': [
        'teste_sarima_produto.py',
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
        'README_OTIMIZACAO.md',
    ],
    'exemplos': [
        'exemplo_uso_sarima.py',
        'exemplo_elencacao_completa.py',
    ],
}

# Arquivos que devem ficar na raiz
ARQUIVOS_RAIZ = [
    'sarima_estoque.py',
    'requirements_sarima.txt',
    'organizar_repositorio.py',
    'organizar_arquivos.py',
]

# Padrões glob (arquivos com wildcards)
PADROES = {
    'modelos': ['comparacao_modelos_*.png', 'relatorio_comparacao_*.txt'],
    'previsoes': ['previsao_sarima_*.png'],
}


def organizar_arquivos(dry_run=False):
    """
    Organiza arquivos movendo para pastas apropriadas.
    
    Parameters:
    -----------
    dry_run : bool
        Se True, apenas mostra o que seria feito (não move)
    """
    print("=" * 80)
    print("ORGANIZACAO DE ARQUIVOS")
    print("=" * 80)
    
    if dry_run:
        print("\n[MODO DRY-RUN] - Apenas mostrando o que seria feito\n")
    
    movidos = 0
    erros = 0
    
    # Move arquivos específicos
    for pasta, arquivos in MAPEAMENTO.items():
        pasta_path = Path(pasta)
        pasta_path.mkdir(exist_ok=True)
        
        print(f"\n[{pasta.upper()}]")
        for arquivo in arquivos:
            origem = Path(arquivo)
            destino = pasta_path / arquivo
            
            if origem.exists():
                if not dry_run:
                    try:
                        shutil.move(str(origem), str(destino))
                        print(f"  [OK] {arquivo} -> {pasta}/")
                        movidos += 1
                    except Exception as e:
                        print(f"  [ERRO] {arquivo}: {str(e)}")
                        erros += 1
                else:
                    print(f"  [MOVED] {arquivo} -> {pasta}/")
            else:
                # Tenta com padrão glob
                encontrado = False
                for padrao in PADROES.get(pasta, []):
                    matches = glob.glob(padrao)
                    if matches and arquivo in [m.replace('*', '') for m in matches if arquivo.replace('*', '') in m]:
                        encontrado = True
                        break
                if not encontrado:
                    print(f"  [NAO ENCONTRADO] {arquivo}")
    
    # Move arquivos com padrões glob
    print(f"\n[PADROES GLOB]")
    for pasta, padroes in PADROES.items():
        pasta_path = Path(pasta)
        pasta_path.mkdir(exist_ok=True)
        
        for padrao in padroes:
            matches = glob.glob(padrao)
            for match in matches:
                origem = Path(match)
                destino = pasta_path / origem.name
                
                # Não move se já está na pasta destino
                if origem.parent == pasta_path:
                    continue
                
                if origem.exists():
                    if not dry_run:
                        try:
                            shutil.move(str(origem), str(destino))
                            print(f"  [OK] {match} -> {pasta}/")
                            movidos += 1
                        except Exception as e:
                            print(f"  [ERRO] {match}: {str(e)}")
                            erros += 1
                    else:
                        print(f"  [MOVED] {match} -> {pasta}/")
    
    print("\n" + "=" * 80)
    if dry_run:
        print("DRY-RUN CONCLUIDO - Nenhum arquivo foi movido")
        print("Execute sem --dry-run para mover os arquivos")
    else:
        print(f"ORGANIZACAO CONCLUIDA!")
        print(f"  Arquivos movidos: {movidos}")
        if erros > 0:
            print(f"  Erros: {erros}")
    print("=" * 80)
    
    # Lista arquivos que devem ficar na raiz
    print("\nArquivos que permanecem na raiz:")
    for arq in ARQUIVOS_RAIZ:
        if Path(arq).exists():
            print(f"  - {arq}")


if __name__ == "__main__":
    import sys
    
    dry_run = '--dry-run' in sys.argv or '-n' in sys.argv
    
    organizar_arquivos(dry_run=dry_run)

