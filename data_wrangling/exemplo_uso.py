"""
Exemplo de uso do script de Data Wrangling
"""

from dw_historico import processar_historico_estoque

if __name__ == "__main__":
    print("Processando dados históricos de estoque...\n")
    
    # Processa os dados
    df_processado = processar_historico_estoque(
        caminho_entrada='../DB/historico_estoque.csv',
        caminho_saida='../DB/historico_estoque_processado.csv',
        min_observacoes=30,
        criar_serie_completa=True
    )
    
    print("\n" + "="*70)
    print("Exemplo de dados processados:")
    print("="*70)
    print(df_processado.head(20))
    
    print("\n" + "="*70)
    print("Estatísticas por SKU (top 10 por quantidade de observações):")
    print("="*70)
    stats = df_processado.groupby('sku').agg({
        'estoque_atual': ['count', 'mean', 'min', 'max']
    }).sort_values(('estoque_atual', 'count'), ascending=False).head(10)
    stats.columns = ['observacoes', 'media', 'minimo', 'maximo']
    print(stats)

