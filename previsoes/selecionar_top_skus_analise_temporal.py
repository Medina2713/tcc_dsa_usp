"""
Script para Selecionar os 10 Melhores Produtos para Análise Temporal
TCC MBA Data Science & Analytics

Este script identifica os SKUs com os melhores dados para análise temporal,
baseado em critérios de qualidade de dados:
- Número de observações (mais = melhor)
- Variabilidade (coeficiente de variação)
- Estoque médio significativo
- Continuidade temporal (menos lacunas)

Autor: Medina2713
Data: 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def calcular_metricas_qualidade_temporal(df, sku):
    """
    Calcula métricas de qualidade temporal para um SKU.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame com dados de estoque
    sku : str
        Código do SKU
        
    Returns:
    --------
    dict
        Métricas de qualidade
    """
    df_sku = df[df['sku'] == sku].copy()
    
    if len(df_sku) == 0:
        return None
    
    # Ordena por data
    df_sku = df_sku.sort_values('data')
    df_sku = df_sku.set_index('data')
    
    # Métricas básicas
    n_observacoes = len(df_sku)
    estoque_medio = df_sku['estoque_atual'].mean()
    estoque_std = df_sku['estoque_atual'].std()
    estoque_min = df_sku['estoque_atual'].min()
    estoque_max = df_sku['estoque_atual'].max()
    
    # Coeficiente de variação (variabilidade)
    cv = estoque_std / estoque_medio if estoque_medio > 0 else 0
    
    # Continuidade temporal (lacunas)
    # Calcula diferença entre datas consecutivas
    df_sku_resampled = df_sku['estoque_atual'].asfreq('D', method='ffill')
    n_lacunas = df_sku_resampled.isna().sum()
    percentual_lacunas = (n_lacunas / len(df_sku_resampled)) * 100 if len(df_sku_resampled) > 0 else 100
    
    # Período de dados
    periodo_dias = (df_sku.index.max() - df_sku.index.min()).days
    densidade_observacoes = n_observacoes / periodo_dias if periodo_dias > 0 else 0
    
    # Score combinado para qualidade temporal
    # Favoriza: muitas observações, alta variabilidade, estoque significativo, poucas lacunas
    score = (
        n_observacoes * 0.3 +           # Peso: número de observações
        cv * 100 * 0.25 +               # Peso: variabilidade
        estoque_medio * 0.2 +           # Peso: estoque médio
        (100 - percentual_lacunas) * 0.15 +  # Peso: continuidade (menos lacunas = melhor)
        densidade_observacoes * 100 * 0.1    # Peso: densidade de observações
    )
    
    return {
        'sku': sku,
        'n_observacoes': n_observacoes,
        'periodo_dias': periodo_dias,
        'densidade_observacoes': densidade_observacoes,
        'estoque_medio': estoque_medio,
        'estoque_std': estoque_std,
        'estoque_min': estoque_min,
        'estoque_max': estoque_max,
        'coeficiente_variacao': cv,
        'n_lacunas': n_lacunas,
        'percentual_lacunas': percentual_lacunas,
        'score_qualidade': score
    }


def selecionar_top_skus_analise_temporal(
    caminho_dados='DB/historico_estoque_atual_processado.csv',
    top_n=10,
    min_observacoes=30,
    min_estoque_medio=1.0,
    max_percentual_lacunas=50.0
):
    """
    Seleciona os N melhores SKUs para análise temporal.
    
    Parameters:
    -----------
    caminho_dados : str
        Caminho do arquivo com dados processados
    top_n : int
        Número de SKUs a selecionar
    min_observacoes : int
        Número mínimo de observações
    min_estoque_medio : float
        Estoque médio mínimo
    max_percentual_lacunas : float
        Percentual máximo de lacunas permitido
        
    Returns:
    --------
    pd.DataFrame
        DataFrame com top SKUs e suas métricas
    """
    print("=" * 80)
    print("SELECAO DOS MELHORES PRODUTOS PARA ANALISE TEMPORAL")
    print("=" * 80)
    
    # Carrega dados
    print(f"\n[1/4] Carregando dados: {caminho_dados}")
    if not Path(caminho_dados).exists():
        print(f"[ERRO] Arquivo não encontrado: {caminho_dados}")
        print(f"       Execute primeiro: python data_wrangling/dw_historico.py")
        return None
    
    df = pd.read_csv(caminho_dados)
    df['data'] = pd.to_datetime(df['data'])
    
    # Garante coluna estoque_atual
    if 'estoque_atual' not in df.columns and 'saldo' in df.columns:
        df['estoque_atual'] = df['saldo']
    
    print(f"      [OK] {len(df):,} registros carregados")
    print(f"      [OK] {df['sku'].nunique()} SKUs disponíveis")
    
    # Lista todos os SKUs
    skus_unicos = df['sku'].unique().tolist()
    print(f"\n[2/4] Calculando métricas de qualidade para {len(skus_unicos)} SKUs...")
    
    # Calcula métricas para cada SKU
    metricas_skus = []
    for i, sku in enumerate(skus_unicos, 1):
        if i % 100 == 0:
            print(f"      Processando SKU {i}/{len(skus_unicos)}...")
        
        metricas = calcular_metricas_qualidade_temporal(df, sku)
        if metricas:
            metricas_skus.append(metricas)
    
    if len(metricas_skus) == 0:
        print("[ERRO] Nenhuma métrica calculada!")
        return None
    
    df_metricas = pd.DataFrame(metricas_skus)
    print(f"      [OK] Métricas calculadas para {len(df_metricas)} SKUs")
    
    # Filtra por critérios mínimos
    print(f"\n[3/4] Aplicando filtros de qualidade...")
    print(f"      - Mínimo de observações: {min_observacoes}")
    print(f"      - Estoque médio mínimo: {min_estoque_medio}")
    print(f"      - Percentual máximo de lacunas: {max_percentual_lacunas}%")
    
    df_filtrado = df_metricas[
        (df_metricas['n_observacoes'] >= min_observacoes) &
        (df_metricas['estoque_medio'] >= min_estoque_medio) &
        (df_metricas['percentual_lacunas'] <= max_percentual_lacunas)
    ].copy()
    
    n_antes = len(df_metricas)
    n_depois = len(df_filtrado)
    print(f"      [OK] {n_antes} SKUs → {n_depois} SKUs após filtros")
    
    if len(df_filtrado) == 0:
        print("[AVISO] Nenhum SKU atende aos critérios mínimos!")
        print("        Tente relaxar os filtros.")
        return None
    
    # Ordena por score de qualidade
    df_filtrado = df_filtrado.sort_values('score_qualidade', ascending=False)
    
    # Seleciona top N
    df_top = df_filtrado.head(top_n).copy()
    df_top['ranking'] = range(1, len(df_top) + 1)
    
    print(f"\n[4/4] Top {top_n} SKUs selecionados!")
    print("=" * 80)
    
    # Exibe resultados
    print(f"\n{'Rank':<6} {'SKU':<20} {'Obs':<8} {'CV':<8} {'Est.Médio':<12} {'Lacunas%':<12} {'Score':<12}")
    print("-" * 80)
    
    for _, row in df_top.iterrows():
        print(f"{int(row['ranking']):<6} "
              f"{str(row['sku']):<20} "
              f"{int(row['n_observacoes']):<8} "
              f"{row['coeficiente_variacao']:<8.3f} "
              f"{row['estoque_medio']:<12.2f} "
              f"{row['percentual_lacunas']:<12.1f} "
              f"{row['score_qualidade']:<12.2f}")
    
    print("-" * 80)
    
    # Detalhes por SKU
    print("\n" + "=" * 80)
    print("DETALHES DOS TOP SKUs")
    print("=" * 80)
    
    for _, row in df_top.iterrows():
        print(f"\nRank {int(row['ranking'])}: SKU {row['sku']}")
        print(f"  Observações: {int(row['n_observacoes'])}")
        print(f"  Período: {int(row['periodo_dias'])} dias")
        print(f"  Densidade: {row['densidade_observacoes']:.2f} observações/dia")
        print(f"  Estoque médio: {row['estoque_medio']:.2f} unidades")
        print(f"  Estoque (min/max): {row['estoque_min']:.0f} / {row['estoque_max']:.0f}")
        print(f"  Coeficiente de variação: {row['coeficiente_variacao']:.3f}")
        print(f"  Lacunas: {int(row['n_lacunas'])} ({row['percentual_lacunas']:.1f}%)")
        print(f"  Score de qualidade: {row['score_qualidade']:.2f}")
    
    # Salva resultado
    Path("resultados").mkdir(exist_ok=True)
    caminho_saida = f"resultados/top_{top_n}_skus_analise_temporal.csv"
    df_top.to_csv(caminho_saida, index=False)
    print(f"\n[OK] Resultado salvo em: {caminho_saida}")
    
    # Salva lista simples de SKUs
    caminho_lista = f"resultados/lista_top_{top_n}_skus.txt"
    with open(caminho_lista, 'w', encoding='utf-8') as f:
        f.write(f"Top {top_n} SKUs para Análise Temporal\n")
        f.write(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        for _, row in df_top.iterrows():
            f.write(f"{int(row['ranking'])}. {row['sku']}\n")
    print(f"[OK] Lista de SKUs salva em: {caminho_lista}")
    
    print("\n" + "=" * 80)
    print("SELECAO CONCLUIDA!")
    print("=" * 80)
    
    return df_top


def main():
    """Função principal"""
    resultado = selecionar_top_skus_analise_temporal(
        caminho_dados='DB/historico_estoque_atual_processado.csv',
        top_n=10,
        min_observacoes=30,
        min_estoque_medio=1.0,
        max_percentual_lacunas=50.0
    )
    
    if resultado is not None:
        print(f"\n✅ {len(resultado)} SKUs selecionados com sucesso!")
        print("\nPróximos passos:")
        print("  1. Use a lista de SKUs para análises temporais")
        print("  2. Execute modelos de previsão nos SKUs selecionados")
        print("  3. Consulte: resultados/top_10_skus_analise_temporal.csv")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERRO] Erro durante execução: {str(e)}")
        import traceback
        traceback.print_exc()

