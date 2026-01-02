"""
Comparação de Modelos para Top 10 SKUs com Maior Giro de Estoque
TCC MBA Data Science & Analytics - E-commerce de Brinquedos

Este script:
1. Identifica os 10 SKUs com maior giro de estoque (baseado em vendas)
2. Executa comparação de modelos de previsão para cada SKU
3. Gera relatório consolidado com resultados

Autor: Medina2713
Data: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from comparacao_modelos_previsao import comparar_modelos, visualizar_comparacao, gerar_relatorio_comparacao
from sarima_estoque import PrevisorEstoqueSARIMA

plt.style.use('seaborn-v0_8-darkgrid')


def calcular_giro_estoque(df_vendas, df_estoque, periodo_dias=30):
    """
    PARTE 1: CÁLCULO DE GIRO DE ESTOQUE
    
    Calcula giro de estoque para cada SKU baseado em:
    - Quantidade vendida no período
    - Estoque médio no período
    
    Giro = Quantidade Vendida / Estoque Médio
    
    Parameters:
    -----------
    df_vendas : pd.DataFrame
        DataFrame com dados de vendas (colunas: sku, quantidade, created_at)
    df_estoque : pd.DataFrame
        DataFrame com dados de estoque (colunas: sku, estoque_atual, data)
    periodo_dias : int
        Período para cálculo (padrão: 30 dias)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame com giro de estoque por SKU
    """
    print("=" * 80)
    print("PARTE 1: CALCULO DE GIRO DE ESTOQUE")
    print("=" * 80)
    
    # Processa vendas
    df_vendas['created_at'] = pd.to_datetime(df_vendas['created_at'], format='mixed', errors='coerce')
    df_vendas = df_vendas[df_vendas['created_at'].notna()].copy()
    
    # Período recente (últimos N dias)
    data_fim = df_vendas['created_at'].max()
    data_inicio = data_fim - timedelta(days=periodo_dias)
    
    vendas_recentes = df_vendas[
        (df_vendas['created_at'] >= data_inicio) & 
        (df_vendas['created_at'] <= data_fim)
    ].copy()
    
    print(f"\nPeriodo de analise: {data_inicio.date()} ate {data_fim.date()} ({periodo_dias} dias)")
    
    # Agrega vendas por SKU
    vendas_por_sku = vendas_recentes.groupby('sku')['quantidade'].sum().reset_index()
    vendas_por_sku.columns = ['sku', 'quantidade_vendida']
    
    print(f"SKUs com vendas no periodo: {len(vendas_por_sku)}")
    
    # Processa estoque
    df_estoque['data'] = pd.to_datetime(df_estoque['data'])
    estoque_recente = df_estoque[
        (df_estoque['data'] >= data_inicio) & 
        (df_estoque['data'] <= data_fim)
    ].copy()
    
    # Estoque médio por SKU
    estoque_medio_por_sku = estoque_recente.groupby('sku')['estoque_atual'].mean().reset_index()
    estoque_medio_por_sku.columns = ['sku', 'estoque_medio']
    
    # Merge e calcula giro
    giro = vendas_por_sku.merge(estoque_medio_por_sku, on='sku', how='inner')
    
    # Giro = Quantidade Vendida / Estoque Médio
    # Adiciona 1 ao estoque médio para evitar divisão por zero
    giro['giro_estoque'] = giro['quantidade_vendida'] / (giro['estoque_medio'] + 1)
    
    # Ordena por giro (maior primeiro)
    giro = giro.sort_values('giro_estoque', ascending=False)
    
    print(f"\nTop 10 SKUs por giro de estoque:")
    print(giro.head(10)[['sku', 'quantidade_vendida', 'estoque_medio', 'giro_estoque']].to_string(index=False))
    
    return giro


def selecionar_top_skus(giro, top_n=10, min_observacoes=200):
    """
    PARTE 2: SELEÇÃO DOS TOP SKUs
    
    Seleciona top N SKUs com maior giro, garantindo que tenham dados suficientes.
    
    Parameters:
    -----------
    giro : pd.DataFrame
        DataFrame com giro de estoque
    top_n : int
        Número de SKUs a selecionar
    min_observacoes : int
        Número mínimo de observações necessário
        
    Returns:
    --------
    list
        Lista de SKUs selecionados
    """
    print("\n" + "=" * 80)
    print("PARTE 2: SELECAO DOS TOP SKUs")
    print("=" * 80)
    
    top_skus = giro.head(top_n * 2)['sku'].tolist()  # Pega mais para ter opções
    
    print(f"\nSelecionando {top_n} SKUs dos {len(top_skus)} com maior giro...")
    
    return top_skus[:top_n]


def processar_skus_em_lote(df_estoque, lista_skus):
    """
    PARTE 3: PROCESSAMENTO EM LOTE
    
    Processa múltiplos SKUs executando comparação de modelos para cada um.
    
    Parameters:
    -----------
    df_estoque : pd.DataFrame
        DataFrame completo de estoque
    lista_skus : list
        Lista de SKUs para processar
        
    Returns:
    --------
    dict
        Dicionário com resultados de cada SKU
    """
    print("\n" + "=" * 80)
    print("PARTE 3: PROCESSAMENTO EM LOTE")
    print("=" * 80)
    
    previsor = PrevisorEstoqueSARIMA()
    resultados_completos = {}
    
    for i, sku in enumerate(lista_skus, 1):
        print(f"\n{'='*80}")
        print(f"Processando SKU {i}/{len(lista_skus)}: {sku}")
        print(f"{'='*80}")
        
        try:
            # Prepara série temporal
            serie = previsor.preparar_serie_temporal(df_estoque, sku=sku)
            
            if len(serie) < 200:
                print(f"[AVISO] SKU {sku}: Dados insuficientes ({len(serie)} observacoes). Pulando...")
                continue
            
            # Compara modelos
            resultados = comparar_modelos(serie, sku, horizonte_previsao=30, proporcao_treino=0.8)
            
            if len(resultados.get('metricas', [])) > 0:
                resultados_completos[sku] = resultados
                print(f"[OK] SKU {sku} processado com sucesso")
            else:
                print(f"[AVISO] SKU {sku}: Nenhuma metrica gerada")
                
        except Exception as e:
            print(f"[ERRO] SKU {sku}: {str(e)}")
            continue
    
    return resultados_completos


def gerar_relatorio_consolidado(resultados_completos, giro):
    """
    PARTE 4: RELATÓRIO CONSOLIDADO
    
    Gera relatório consolidado comparando resultados de todos os SKUs.
    
    Parameters:
    -----------
    resultados_completos : dict
        Dicionário com resultados de cada SKU
    giro : pd.DataFrame
        DataFrame com giro de estoque
    """
    print("\n" + "=" * 80)
    print("PARTE 4: GERACAO DE RELATORIO CONSOLIDADO")
    print("=" * 80)
    
    if len(resultados_completos) == 0:
        print("[AVISO] Nenhum resultado para consolidar")
        return
    
    # Compila métricas de todos os SKUs
    todas_metricas = []
    
    for sku, resultado in resultados_completos.items():
        metricas_sku = resultado.get('metricas', [])
        for metrica in metricas_sku:
            metrica['sku'] = sku
            todas_metricas.append(metrica)
    
    df_metricas = pd.DataFrame(todas_metricas)
    
    if len(df_metricas) == 0:
        print("[AVISO] Nenhuma metrica disponivel")
        return
    
    # Adiciona giro de estoque
    giro_dict = dict(zip(giro['sku'], giro['giro_estoque']))
    df_metricas['giro_estoque'] = df_metricas['sku'].map(giro_dict)
    
    # Melhor modelo por SKU (menor MAE)
    melhores_modelos = []
    for sku in df_metricas['sku'].unique():
        df_sku = df_metricas[df_metricas['sku'] == sku]
        melhor = df_sku.loc[df_sku['mae'].idxmin()]
        melhores_modelos.append(melhor.to_dict())
    
    df_melhores = pd.DataFrame(melhores_modelos)
    df_melhores = df_melhores.sort_values('giro_estoque', ascending=False)
    
    # Estatísticas por modelo
    stats_por_modelo = df_metricas.groupby('modelo').agg({
        'mae': ['mean', 'std'],
        'rmse': ['mean', 'std'],
        'mape': ['mean', 'std']
    }).round(2)
    
    # Gera relatório
    relatorio = []
    relatorio.append("=" * 80)
    relatorio.append("RELATORIO CONSOLIDADO: COMPARACAO DE MODELOS - TOP 10 SKUs")
    relatorio.append("=" * 80)
    relatorio.append(f"\nData: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    relatorio.append(f"SKUs processados: {len(resultados_completos)}")
    
    relatorio.append("\n" + "-" * 80)
    relatorio.append("MELHOR MODELO POR SKU (ordenado por giro de estoque)")
    relatorio.append("-" * 80)
    relatorio.append(df_melhores[['sku', 'giro_estoque', 'modelo', 'mae', 'rmse', 'mape']].to_string(index=False))
    
    relatorio.append("\n" + "-" * 80)
    relatorio.append("ESTATISTICAS POR MODELO (media e desvio padrao)")
    relatorio.append("-" * 80)
    relatorio.append(str(stats_por_modelo))
    
    relatorio.append("\n" + "-" * 80)
    relatorio.append("RESUMO POR MODELO")
    relatorio.append("-" * 80)
    
    for modelo in df_metricas['modelo'].unique():
        df_modelo = df_metricas[df_metricas['modelo'] == modelo]
        relatorio.append(f"\n{modelo}:")
        relatorio.append(f"  - MAE medio: {df_modelo['mae'].mean():.2f} (+/- {df_modelo['mae'].std():.2f})")
        relatorio.append(f"  - RMSE medio: {df_modelo['rmse'].mean():.2f} (+/- {df_modelo['rmse'].std():.2f})")
        relatorio.append(f"  - MAPE medio: {df_modelo['mape'].mean():.2f}% (+/- {df_modelo['mape'].std():.2f}%)")
        relatorio.append(f"  - Quantidade de SKUs: {len(df_modelo['sku'].unique())}")
    
    # Salva relatório
    texto = "\n".join(relatorio)
    nome_arquivo = 'relatorio_consolidado_top_skus.txt'
    with open(nome_arquivo, 'w', encoding='utf-8') as f:
        f.write(texto)
    
    print(f"\n[OK] Relatorio consolidado salvo: {nome_arquivo}")
    
    # Exibe resumo
    print("\n" + "-" * 80)
    print("RESUMO CONSOLIDADO")
    print("-" * 80)
    print(texto[-1000:])  # Últimas linhas
    
    return df_melhores, stats_por_modelo


def visualizar_resultados_consolidados(resultados_completos, df_melhores):
    """
    PARTE 5: VISUALIZAÇÃO CONSOLIDADA
    
    Cria visualizações comparando resultados entre SKUs.
    
    Parameters:
    -----------
    resultados_completos : dict
        Resultados de todos os SKUs
    df_melhores : pd.DataFrame
        DataFrame com melhores modelos por SKU
    """
    print("\n" + "=" * 80)
    print("PARTE 5: VISUALIZACAO CONSOLIDADA")
    print("=" * 80)
    
    if len(resultados_completos) == 0:
        print("[AVISO] Nenhum resultado para visualizar")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Gráfico 1: MAE por modelo (boxplot)
    ax1 = axes[0, 0]
    todas_metricas = []
    for sku, resultado in resultados_completos.items():
        for metrica in resultado.get('metricas', []):
            todas_metricas.append(metrica)
    
    df_metricas = pd.DataFrame(todas_metricas)
    if len(df_metricas) > 0:
        modelos_unicos = df_metricas['modelo'].unique()
        dados_boxplot = [df_metricas[df_metricas['modelo'] == m]['mae'].values for m in modelos_unicos]
        ax1.boxplot(dados_boxplot, labels=modelos_unicos)
        ax1.set_title('Distribuicao de MAE por Modelo', fontsize=12, fontweight='bold')
        ax1.set_ylabel('MAE')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
    
    # Gráfico 2: MAPE por modelo (boxplot)
    ax2 = axes[0, 1]
    if len(df_metricas) > 0:
        dados_boxplot = [df_metricas[df_metricas['modelo'] == m]['mape'].values for m in modelos_unicos]
        ax2.boxplot(dados_boxplot, labels=modelos_unicos)
        ax2.set_title('Distribuicao de MAPE por Modelo', fontsize=12, fontweight='bold')
        ax2.set_ylabel('MAPE (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
    
    # Gráfico 3: Melhor modelo por SKU (bar chart)
    ax3 = axes[1, 0]
    if len(df_melhores) > 0:
        modelos_contagem = df_melhores['modelo'].value_counts()
        ax3.bar(range(len(modelos_contagem)), modelos_contagem.values, color='steelblue', alpha=0.7)
        ax3.set_xticks(range(len(modelos_contagem)))
        ax3.set_xticklabels(modelos_contagem.index, rotation=45, ha='right')
        ax3.set_title('Frequencia: Melhor Modelo por SKU', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Quantidade de SKUs')
        ax3.grid(True, alpha=0.3, axis='y')
    
    # Gráfico 4: MAE vs Giro de Estoque
    ax4 = axes[1, 1]
    if len(df_melhores) > 0:
        ax4.scatter(df_melhores['giro_estoque'], df_melhores['mae'], alpha=0.6, s=100)
        ax4.set_xlabel('Giro de Estoque')
        ax4.set_ylabel('MAE (Melhor Modelo)')
        ax4.set_title('MAE vs Giro de Estoque', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    nome_arquivo = 'comparacao_consolidada_top_skus.png'
    plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Grafico consolidado salvo: {nome_arquivo}")
    plt.close()


def main():
    """
    FUNÇÃO PRINCIPAL
    
    Executa análise completa dos top 10 SKUs com maior giro de estoque.
    """
    print("=" * 80)
    print("COMPARACAO DE MODELOS: TOP 10 SKUs COM MAIOR GIRO DE ESTOQUE")
    print("=" * 80)
    
    # Carrega dados
    print("\nCarregando dados...")
    df_vendas = pd.read_csv('DB/venda_produtos_atual.csv', low_memory=False)
    df_estoque = pd.read_csv('DB/historico_estoque_atual_processado.csv')
    df_estoque['data'] = pd.to_datetime(df_estoque['data'])
    
    print(f"[OK] Vendas: {len(df_vendas):,} registros")
    print(f"[OK] Estoque: {len(df_estoque):,} registros")
    
    # Calcula giro de estoque
    giro = calcular_giro_estoque(df_vendas, df_estoque, periodo_dias=30)
    
    # Seleciona top 10
    top_skus = selecionar_top_skus(giro, top_n=10, min_observacoes=200)
    print(f"\n[OK] Top 10 SKUs selecionados: {len(top_skus)} SKUs")
    
    # Processa em lote
    resultados_completos = processar_skus_em_lote(df_estoque, top_skus)
    
    print(f"\n[OK] SKUs processados com sucesso: {len(resultados_completos)}")
    
    if len(resultados_completos) == 0:
        print("\n[AVISO] Nenhum SKU foi processado com sucesso")
        return
    
    # Gera visualizações individuais
    print("\n" + "=" * 80)
    print("GERANDO VISUALIZACOES INDIVIDUAIS...")
    print("=" * 80)
    for sku, resultado in resultados_completos.items():
        try:
            visualizar_comparacao(resultado)
            gerar_relatorio_comparacao(resultado)
        except Exception as e:
            print(f"[AVISO] Erro ao gerar visualizacao para {sku}: {str(e)}")
    
    # Gera relatório consolidado
    df_melhores, stats_por_modelo = gerar_relatorio_consolidado(resultados_completos, giro)
    
    # Visualização consolidada
    visualizar_resultados_consolidados(resultados_completos, df_melhores)
    
    print("\n" + "=" * 80)
    print("PROCESSAMENTO CONCLUIDO!")
    print("=" * 80)
    print("\nArquivos gerados:")
    print("  - relatorio_consolidado_top_skus.txt")
    print("  - comparacao_consolidada_top_skus.png")
    print("  - comparacao_modelos_[SKU].png (para cada SKU)")
    print("  - relatorio_comparacao_[SKU].txt (para cada SKU)")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERRO] Erro durante execucao: {str(e)}")
        import traceback
        traceback.print_exc()

