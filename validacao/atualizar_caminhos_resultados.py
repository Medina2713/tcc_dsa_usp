"""
Script para atualizar caminhos de saída nos scripts para usar pasta resultados/
"""

import re
from pathlib import Path

# Scripts que precisam ser atualizados
SCRIPTS_ATUALIZAR = [
    'modelos/comparacao_modelos_previsao.py',
    'modelos/comparacao_top_skus.py',
    'modelos/comparacao_top_skus_otimizado.py',
    'analises/analise_exploratoria_sazonalidade.py',
    'analises/analise_box_jenkins_sarima.py',
    'validacao/validacao_walk_forward_sarima.py',
    'validacao/tratamento_outliers_sarima.py',
    'validacao/teste_tempo_processamento.py',
    'exemplos/exemplo_uso_sarima.py',
]

def atualizar_caminho_arquivo(caminho_script):
    """Atualiza caminhos de saída em um script"""
    if not Path(caminho_script).exists():
        return False
    
    with open(caminho_script, 'r', encoding='utf-8') as f:
        conteudo = f.read()
    
    conteudo_original = conteudo
    mudancas = 0
    
    # Padrões para substituir
    padroes = [
        # plt.savefig('arquivo.png')
        (r"plt\.savefig\(['\"](\w+\.png)['\"]", r"plt.savefig('resultados/\1'"),
        # plt.savefig("arquivo.png")
        (r'plt\.savefig\(["\']([^"\']+\.png)["\']', lambda m: f"plt.savefig('resultados/{Path(m.group(1)).name}'"),
        # to_csv('arquivo.csv')
        (r"\.to_csv\(['\"](\w+\.csv)['\"]", r".to_csv('resultados/\1'"),
        # nome_arquivo = 'arquivo.png'
        (r"nome_arquivo\s*=\s*['\"](\w+\.(png|csv|txt))['\"]", r"nome_arquivo = 'resultados/\1'"),
        # caminho_saida = 'arquivo.csv'
        (r"caminho_saida\s*=\s*['\"](\w+\.(png|csv|txt))['\"]", r"caminho_saida = 'resultados/\1'"),
    ]
    
    # Para cada padrão
    for padrao, substituicao in padroes:
        if callable(substituicao):
            conteudo = re.sub(padrao, substituicao, conteudo)
        else:
            novo_conteudo = re.sub(padrao, substituicao, conteudo)
            if novo_conteudo != conteudo:
                mudancas += 1
                conteudo = novo_conteudo
    
    # Adiciona criação da pasta resultados/ no início das funções principais
    if 'Path("resultados").mkdir(exist_ok=True)' not in conteudo and ('savefig' in conteudo or 'to_csv' in conteudo):
        # Adiciona import e criação de pasta no início do arquivo ou função
        if 'from pathlib import Path' not in conteudo:
            # Adiciona após imports
            conteudo = re.sub(
                r'(import\s+\w+\s*\n)',
                r'\1from pathlib import Path\n',
                conteudo,
                count=1
            )
        
        # Adiciona criação de pasta antes de salvar
        conteudo = re.sub(
            r'(plt\.savefig|\.to_csv)',
            r'Path("resultados").mkdir(exist_ok=True)\n    \1',
            conteudo,
            count=1
        )
    
    if conteudo != conteudo_original:
        with open(caminho_script, 'w', encoding='utf-8') as f:
            f.write(conteudo)
        return True
    
    return False


def main():
    """Atualiza todos os scripts"""
    print("Atualizando caminhos de saida nos scripts...")
    print("=" * 80)
    
    atualizados = 0
    for script in SCRIPTS_ATUALIZAR:
        if atualizar_caminho_arquivo(script):
            print(f"[OK] {script}")
            atualizados += 1
        else:
            print(f"[SKIP] {script} (sem mudancas necessarias)")
    
    print(f"\n[OK] {atualizados} scripts atualizados")
    print("=" * 80)
    print("\nNOTA: Esta e uma primeira passagem.")
    print("      Revise os scripts manualmente para garantir que os caminhos estao corretos.")


if __name__ == "__main__":
    main()

