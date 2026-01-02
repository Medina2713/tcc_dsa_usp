"""
Teste SARIMA: Previsão de Demanda para Produto Selecionado
TCC MBA Data Science & Analytics

Este script:
1. Processa os dados históricos de estoque
2. Identifica o produto (SKU) com mais observações e variações
3. Treina modelo SARIMA
4. Gera previsão para o próximo mês (30 dias)
5. Visualiza resultados

Autor: Medina2713
Data: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data_wrangling.dw_historico import processar_historico_estoque
from sarima_estoque import PrevisorEstoqueSARIMA


def identificar_melhor_sku(df):
    """
    Identifica o SKU com mais observações e maior variabilidade.
    
    Usa uma métrica combinada: número de observações * coeficiente de variação
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame processado com dados de estoque
        
    Returns:
    --------
    str
        SKU selecionado
    dict
        Estatísticas do SKU
    """
    print("\n" + "=" * 70)
    print("Identificando melhor SKU para teste...")
    print("=" * 70)
    
    # Calcula estatísticas por SKU
    stats = df.groupby('sku')['estoque_atual'].agg([
        'count',
        'mean',
        'std',
        'min',
        'max'
    ]).reset_index()
    
    # Filtra SKUs com média muito baixa (menos de 1 unidade em média)
    # SKUs com estoque quase sempre zero não são bons para SARIMA
    stats = stats[stats['mean'] >= 1.0].copy()
    
    if len(stats) == 0:
        # Se nenhum SKU atende o critério, relaxa para média >= 0.5
        stats = df.groupby('sku')['estoque_atual'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).reset_index()
        stats = stats[stats['mean'] >= 0.5].copy()
    
    # Coeficiente de variação (desvio padrão / média)
    stats['cv'] = stats['std'] / stats['mean']
    stats['cv'] = stats['cv'].fillna(0)
    
    # Score combinado: observações * coeficiente de variação * média
    # Isso favorece SKUs com muitas observações, alta variabilidade E estoque significativo
    stats['score'] = stats['count'] * stats['cv'] * stats['mean']
    
    # Ordena por score
    stats = stats.sort_values('score', ascending=False)
    
    # SKU selecionado
    sku_selecionado = stats.iloc[0]['sku']
    
    print(f"\n[OK] SKU Selecionado: {sku_selecionado}")
    print(f"\nEstatisticas do SKU:")
    print(f"   - Observacoes: {int(stats.iloc[0]['count'])}")
    print(f"   - Media de estoque: {stats.iloc[0]['mean']:.2f}")
    print(f"   - Desvio padrao: {stats.iloc[0]['std']:.2f}")
    print(f"   - Coeficiente de variacao: {stats.iloc[0]['cv']:.3f}")
    print(f"   - Minimo: {stats.iloc[0]['min']:.0f}")
    print(f"   - Maximo: {stats.iloc[0]['max']:.0f}")
    print(f"   - Score: {stats.iloc[0]['score']:.2f}")
    
    # Mostra top 5
    print(f"\nTop 5 SKUs por score (observacoes x variabilidade):")
    print(stats.head(5)[['sku', 'count', 'cv', 'score']].to_string(index=False))
    
    return sku_selecionado, stats.iloc[0].to_dict()


def visualizar_resultado(serie_historica, previsao, sku, stats):
    """
    Cria visualização comparando histórico e previsão.
    
    Parameters:
    -----------
    serie_historica : pd.Series
        Série temporal histórica
    previsao : pd.Series
        Previsão futura
    sku : str
        Código do SKU
    stats : dict
        Estatísticas do modelo
    """
    plt.figure(figsize=(14, 8))
    
    # Plot histórico (últimos 90 dias para melhor visualização)
    serie_plot = serie_historica.iloc[-90:] if len(serie_historica) > 90 else serie_historica
    
    plt.plot(serie_plot.index, serie_plot.values, 
             label='Histórico Real', color='#2E86AB', linewidth=2, alpha=0.8)
    
    # Plot previsão
    plt.plot(previsao.index, previsao.values, 
             label='Previsão (Próximo Mês)', color='#A23B72', linewidth=2.5, 
             linestyle='--', marker='o', markersize=4)
    
    # Linha divisória
    ultima_data = serie_historica.index[-1]
    plt.axvline(x=ultima_data, color='gray', linestyle=':', alpha=0.7, linewidth=2)
    plt.text(ultima_data, plt.ylim()[1] * 0.95, 'Fim do\nHistórico', 
             ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Configurações do gráfico
    plt.title(f'Previsão de Estoque - SKU: {sku}\nSARIMA ({stats.get("modelo", "N/A")})', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Estoque (unidades)', fontsize=12)
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Formata eixo X
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Salva gráfico
    from pathlib import Path
    Path("resultados").mkdir(exist_ok=True)
    nome_arquivo = f'resultados/previsao_sarima_{sku}.png'
    plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Grafico salvo: {nome_arquivo}")
    
    plt.close()


def calcular_estatisticas_previsao(serie_historica, previsao):
    """
    Calcula estatísticas da previsão.
    
    Parameters:
    -----------
    serie_historica : pd.Series
        Série histórica
    previsao : pd.Series
        Previsão
        
    Returns:
    --------
    dict
        Estatísticas
    """
    stats = {
        'previsao_media': previsao.mean(),
        'previsao_min': previsao.min(),
        'previsao_max': previsao.max(),
        'previsao_total': previsao.sum(),
        'estoque_atual': serie_historica.iloc[-1],
        'variacao_percentual': ((previsao.mean() - serie_historica.iloc[-1]) / serie_historica.iloc[-1] * 100) if serie_historica.iloc[-1] > 0 else 0
    }
    
    return stats


def teste_completo():
    """
    Executa teste completo: processa dados, seleciona SKU, treina SARIMA e prevê.
    """
    print("=" * 70)
    print("TESTE SARIMA: Previsao de Demanda para Proximo Mes")
    print("=" * 70)
    
    # 1. Processar dados (ou carregar se já processados)
    print("\nPASSO 1: Processando dados historicos...")
    # Usa arquivo atualizado
    try:
        # Tenta carregar dados já processados do arquivo atualizado
        df_processado = pd.read_csv('DB/historico_estoque_atual_processado.csv')
        df_processado['data'] = pd.to_datetime(df_processado['data'])
        print("   [OK] Dados processados (atualizados) encontrados! Carregando...")
    except FileNotFoundError:
        # Se não existir, processa arquivo atualizado
        print("   [AVISO] Dados processados nao encontrados. Processando arquivo atualizado...")
        df_processado = processar_historico_estoque(
            caminho_entrada='DB/historico_estoque_atual.csv',
            caminho_saida='DB/historico_estoque_atual_processado.csv',
            min_observacoes=30,
            criar_serie_completa=True
        )
    
    print(f"   [OK] {len(df_processado):,} registros carregados")
    print(f"   [OK] {df_processado['sku'].nunique()} SKUs disponiveis")
    
    # 2. Identificar melhor SKU
    sku_selecionado, stats_sku = identificar_melhor_sku(df_processado)
    
    # 3. Preparar série temporal
    print("\n" + "=" * 70)
    print("PASSO 2: Preparando serie temporal...")
    print("=" * 70)
    
    previsor = PrevisorEstoqueSARIMA(horizonte_previsao=30, frequencia='D')
    serie = previsor.preparar_serie_temporal(df_processado, sku=sku_selecionado)
    
    print(f"\n[OK] Serie temporal preparada:")
    print(f"   - Periodo: {serie.index[0].date()} ate {serie.index[-1].date()}")
    print(f"   - Observacoes: {len(serie)}")
    print(f"   - Estoque atual: {serie.iloc[-1]:.0f} unidades")
    print(f"   - Estoque medio: {serie.mean():.2f} unidades")
    
    # 4. Treinar modelo SARIMA
    print("\n" + "=" * 70)
    print("PASSO 3: Treinando modelo SARIMA...")
    print("=" * 70)
    print("   Aguarde... Isso pode levar alguns minutos...")
    print("   (Auto-ARIMA esta testando multiplas combinacoes de parametros)\n")
    
    modelo = previsor.treinar_modelo(serie, sku=sku_selecionado)
    
    if modelo is None:
        print("   [ERRO] Erro ao treinar modelo. Tente com outro SKU.")
        return
    
    modelo_info = f"{modelo.order} x {modelo.seasonal_order}"
    print(f"\n[OK] Modelo treinado com sucesso!")
    print(f"   - Parametros: {modelo_info}")
    try:
        aic = getattr(modelo, 'aic', None)
        if aic is not None:
            aic_value = aic() if callable(aic) else aic
            print(f"   - AIC: {aic_value:.2f}")
    except:
        pass
    
    # 5. Gerar previsão
    print("\n" + "=" * 70)
    print("PASSO 4: Gerando previsao para proximo mes (30 dias)...")
    print("=" * 70)
    
    previsao = previsor.prever(serie, modelo=modelo)
    
    if previsao is None:
        print("   [ERRO] Erro ao gerar previsao.")
        return
    
    # Estatísticas
    stats_previsao = calcular_estatisticas_previsao(serie, previsao)
    
    print(f"\n[OK] Previsao gerada com sucesso!")
    print(f"\nEstatisticas da Previsao:")
    print(f"   - Estoque atual: {stats_previsao['estoque_atual']:.0f} unidades")
    print(f"   - Estoque medio previsto (30 dias): {stats_previsao['previsao_media']:.2f} unidades")
    print(f"   - Variacao: {stats_previsao['variacao_percentual']:+.1f}%")
    print(f"   - Minimo previsto: {stats_previsao['previsao_min']:.0f} unidades")
    print(f"   - Maximo previsto: {stats_previsao['previsao_max']:.0f} unidades")
    
    # Tabela de previsão
    print(f"\nPrevisao Detalhada (Proximos 30 dias):")
    print("-" * 50)
    df_previsao = pd.DataFrame({
        'Data': previsao.index,
        'Estoque Previsto': previsao.values.round(0).astype(int)
    })
    # Mostra primeiras 10 e últimas 10
    print(df_previsao.head(10).to_string(index=False))
    print("   ...")
    print(df_previsao.tail(10).to_string(index=False))
    
    # 6. Visualizar
    print("\n" + "=" * 70)
    print("PASSO 5: Criando visualizacao...")
    print("=" * 70)
    
    stats_modelo = {'modelo': modelo_info}
    visualizar_resultado(serie, previsao, sku_selecionado, stats_modelo)
    
    # Resumo final
    print("\n" + "=" * 70)
    print("TESTE CONCLUIDO COM SUCESSO!")
    print("=" * 70)
    print(f"\nProduto analisado: {sku_selecionado}")
    print(f"Previsao para: {previsao.index[0].date()} ate {previsao.index[-1].date()}")
    print(f"Modelo: SARIMA {modelo_info}")
    print(f"Estoque medio previsto: {stats_previsao['previsao_media']:.1f} unidades")
    print(f"Variacao em relacao ao atual: {stats_previsao['variacao_percentual']:+.1f}%")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        teste_completo()
    except Exception as e:
        print(f"\n[ERRO] Erro durante execucao: {str(e)}")
        import traceback
        traceback.print_exc()

