"""
Gera as Tabelas do TCC (Metodologia)
TCC MBA Data Science & Analytics - E-commerce de Brinquedos

- Tabela 1: Explicacao da Base de Dados Utilizada (variaveis, descricao, codigo/rotulo)
- Tabela 2: Desempenho dos Modelos (MAE, RMSE, MAPE) - a partir de metricas existentes

Execute apos rodar analise exploratoria e comparacao de modelos.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

DIR_TABELAS = Path('resultados/tabelas_tcc')
DIR_TABELAS.mkdir(parents=True, exist_ok=True)


def gerar_tabela_01_base_dados():
    """
    Tabela 1 - Explicacao da Base de Dados Utilizada.
    Conforme requisitado na metodologia do TCC.
    """
    linhas = [
        # historico_estoque
        ('sku', 'Identificador unico do produto (Stock Keeping Unit)', 'sku (str)'),
        ('created_at', 'Data e hora do registro do saldo de estoque', 'created_at (datetime)'),
        ('saldo', 'Quantidade em estoque no momento do registro', 'saldo (int)'),
        # venda_produtos
        ('sku', 'Identificador unico do produto (mesmo que historico_estoque)', 'sku (str)'),
        ('created_at', 'Data e hora da transacao de venda', 'created_at (datetime)'),
        ('quantidade', 'Quantidade de unidades vendidas na transacao', 'quantidade (int)'),
        ('valor_unitario', 'Preco de venda unitario do produto', 'valor_unitario (float)'),
        ('custo_unitario', 'Custo de aquisicao unitario do produto', 'custo_unitario (float)'),
        ('margem_proporcional', 'Margem de contribuicao em percentual ((valor - custo) / valor * 100)', 'margem_proporcional (float)'),
    ]

    # Remove duplicata de sku/created_at (estao em duas tabelas) - mantemos as duas linhas com contexto
    df = pd.DataFrame(linhas, columns=['Variavel', 'Descricao da Variavel', 'Codigo e Rotulo da Variavel'])

    # Adiciona coluna de origem para clareza
    origem = []
    for i, r in df.iterrows():
        if i < 3:
            origem.append('historico_estoque')
        else:
            origem.append('venda_produtos')
    df.insert(0, 'Fonte', origem)

    caminho_csv = DIR_TABELAS / 'tabela_01_base_dados.csv'
    df.to_csv(caminho_csv, index=False, encoding='utf-8-sig', sep=';')
    print(f"[OK] Tabela 1 salva: {caminho_csv}")

    # Tambem gera versao em Markdown para visualizacao
    caminho_md = DIR_TABELAS / 'tabela_01_base_dados.md'
    with open(caminho_md, 'w', encoding='utf-8') as f:
        f.write("# Tabela 1 - Explicacao da Base de Dados Utilizada\n\n")
        f.write("Fonte: Dados originais da pesquisa.\n\n")
        f.write("| Fonte | Variavel | Descricao da Variavel | Codigo e Rotulo da Variavel |\n")
        f.write("|-------|----------|------------------------|-----------------------------|\n")
        for _, r in df.iterrows():
            f.write(f"| {r['Fonte']} | {r['Variavel']} | {r['Descricao da Variavel']} | {r['Codigo e Rotulo da Variavel']} |\n")
    print(f"[OK] Tabela 1 (MD) salva: {caminho_md}")

    return df


def gerar_tabela_02_desempenho_modelos():
    """
    Tabela 2 - Desempenho dos Modelos de Previsao por Metrica de Erro.
    Prioridade:
    1. resultados/tabelas_tcc/tabela_02_desempenho_modelos.csv (gerado por comparacao_modelos)
    2. metricas_consolidadas.csv (comparacao_top_skus_otimizado)
    """
    # 1. Tabela 2 ja gerada pelo script de comparacao (um SKU)
    existente = DIR_TABELAS / 'tabela_02_desempenho_modelos.csv'
    if existente.exists():
        print(f"[OK] Tabela 2 ja existe: {existente}")
        return pd.read_csv(existente, sep=';')

    # 2. Metricas consolidadas (varios SKUs)
    consolidado = Path('resultados/resultados_comparacao/metricas_consolidadas.csv')
    if consolidado.exists():
        df = pd.read_csv(consolidado)
        if 'modelo' in df.columns and 'mae' in df.columns:
            resumo = df.groupby('modelo').agg({
                'mae': 'mean',
                'rmse': 'mean',
                'mape': 'mean'
            }).round(4).reset_index()
            resumo.columns = ['Modelo', 'MAE', 'RMSE', 'MAPE']
            resumo['MAPE'] = resumo['MAPE'].astype(str) + '%'
            resumo.to_csv(existente, index=False, encoding='utf-8-sig', sep=';')
            print(f"[OK] Tabela 2 (consolidada) salva: {existente}")
            return resumo

    print("[AVISO] Nenhuma metrica de modelos encontrada. Execute comparacao_modelos_previsao.py ou comparacao_top_skus_otimizado.py antes.")
    return None


def main():
    print("=" * 80)
    print("GERACAO DAS TABELAS DO TCC (METODOLOGIA)")
    print("=" * 80)
    print(f"\nSaida: {DIR_TABELAS.absolute()}\n")

    gerar_tabela_01_base_dados()
    print()
    gerar_tabela_02_desempenho_modelos()

    print("\n" + "=" * 80)
    print("CONCLUIDO")
    print("=" * 80)


if __name__ == "__main__":
    main()
