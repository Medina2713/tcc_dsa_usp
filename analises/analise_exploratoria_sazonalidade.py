"""
Análise Exploratória: Padrões Sazonais em Dados de Estoque
TCC MBA Data Science & Analytics - E-commerce de Brinquedos

Este script realiza análise exploratória detalhada para identificar padrões sazonais
nos dados de estoque, especialmente relacionado a períodos de maior demanda (outubro e dezembro).

Objetivos:
1. Verificar padrões visuais nas séries temporais
2. Analisar agregados mensais (médias, totais)
3. Comparar meses específicos (outubro vs dezembro vs outros)
4. Identificar produtos com padrões sazonais mais evidentes
5. Visualizar resultados com gráficos informativos

Autor: Medina2713
Data: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import sys
import time
import warnings
warnings.filterwarnings('ignore')


def _log(msg, flush=True):
    """Imprime e faz flush para atualizacao imediata do log."""
    print(msg, flush=flush)

# Pastas de saida (TCC)
DIR_FIGURAS = Path('resultados/figuras_exploratoria')
DIR_RESULTADOS = Path('resultados')
DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)

# Limites para escolha do SKU representativo (Fig 4 e Fig 5-7)
MAX_PCT_ZEROS_REPRESENTATIVO = 30.0   # Nunca escolher SKU com mais que 30% de dias zerados
MIN_ESTOQUE_MEDIO_REPRESENTATIVO = 1.0

# Configuração de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def carregar_dados_processados(caminho='DB/historico_estoque_atual_processado.csv'):
    """
    PARTE 1: CARREGAMENTO DE DADOS
    
    Carrega os dados processados de histórico de estoque.
    Estes dados já foram limpos, agregados por dia e formatados para análise temporal.
    
    Parameters:
    -----------
    caminho : str
        Caminho para o arquivo CSV processado
        
    Returns:
    --------
    pd.DataFrame
        DataFrame com colunas: data, sku, estoque_atual
    """
    _log("=" * 80)
    _log("PARTE 1: CARREGAMENTO DE DADOS")
    _log("=" * 80)
    _log(f"\nCarregando dados de: {caminho}")
    
    df = pd.read_csv(caminho)
    df['data'] = pd.to_datetime(df['data'])
    
    _log(f"[OK] Dados carregados: {len(df):,} registros")
    _log(f"[OK] Periodo: {df['data'].min().date()} ate {df['data'].max().date()}")
    _log(f"[OK] SKUs unicos: {df['sku'].nunique()}")
    _log(f"[OK] Total de dias: {(df['data'].max() - df['data'].min()).days}")
    
    return df


def adicionar_variaveis_temporais(df):
    """
    PARTE 2: CRIAÇÃO DE VARIÁVEIS TEMPORAIS
    
    Adiciona variáveis derivadas da data para facilitar análise sazonal:
    - Ano, Mês, Dia do mês
    - Dia da semana
    - Trimestre
    - Semana do ano
    
    Essas variáveis permitem agrupar e analisar padrões por diferentes períodos.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame com coluna 'data'
        
    Returns:
    --------
    pd.DataFrame
        DataFrame com variáveis temporais adicionadas
    """
    _log("\n" + "=" * 80)
    _log("PARTE 2: CRIACAO DE VARIAVEIS TEMPORAIS")
    _log("=" * 80)
    
    df = df.copy()
    
    # Variáveis básicas de tempo
    df['ano'] = df['data'].dt.year
    df['mes'] = df['data'].dt.month
    df['dia'] = df['data'].dt.day
    df['dia_semana'] = df['data'].dt.dayofweek  # 0=Segunda, 6=Domingo
    df['trimestre'] = df['data'].dt.quarter
    df['semana_ano'] = df['data'].dt.isocalendar().week
    
    # Nome do mês (para visualização)
    df['mes_nome'] = df['data'].dt.strftime('%b')  # Jan, Feb, etc.
    
    # Flag para meses de alta temporada (outubro e dezembro)
    df['mes_alta_temporada'] = df['mes'].isin([10, 12])  # Outubro e Dezembro
    
    _log("\n[OK] Variaveis temporais criadas:")
    _log("     - ano, mes, dia, dia_semana, trimestre, semana_ano")
    _log("     - mes_nome, mes_alta_temporada (flag para out/dez)")
    
    return df


def analise_agregados_mensais(df):
    """
    PARTE 3: ANÁLISE DE AGREGADOS MENSAIS
    
    Calcula estatísticas agregadas por mês para identificar padrões sazonais.
    
    Métricas calculadas:
    - Total de estoque (soma)
    - Média de estoque por SKU
    - Desvio padrão (variabilidade)
    - Contagem de observações
    
    Esta análise permite comparar os meses e identificar quando há mais/menos estoque.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame com variáveis temporais
        
    Returns:
    --------
    pd.DataFrame
        Estatísticas agregadas por mês
    """
    _log("\n" + "=" * 80)
    _log("PARTE 3: ANALISE DE AGREGADOS MENSAIS")
    _log("=" * 80)
    
    # Agrega por mês (ignorando ano para ver padrão geral)
    agregados = df.groupby('mes').agg({
        'estoque_atual': ['sum', 'mean', 'std', 'count'],
        'sku': 'nunique'
    }).round(2)
    
    # Flatten column names
    agregados.columns = ['estoque_total', 'estoque_medio', 'estoque_desvio', 
                        'observacoes', 'skus_unicos']
    agregados = agregados.reset_index()
    
    # Nome do mês
    meses_nomes = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
                   7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'}
    agregados['mes_nome'] = agregados['mes'].map(meses_nomes)
    
    _log("\nEstatisticas agregadas por mes:")
    _log(agregados.to_string(index=False))
    
    # Comparação específica: Outubro e Dezembro vs outros meses
    _log("\n" + "-" * 80)
    _log("COMPARACAO: Meses de alta temporada (Out/Dez) vs Outros")
    _log("-" * 80)
    
    estoque_alta = df[df['mes_alta_temporada'] == True]['estoque_atual'].mean()
    estoque_outros = df[df['mes_alta_temporada'] == False]['estoque_atual'].mean()
    
    _log(f"Estoque medio (Out/Dez): {estoque_alta:.2f}")
    _log(f"Estoque medio (outros meses): {estoque_outros:.2f}")
    _log(f"Diferenca: {estoque_alta - estoque_outros:.2f} ({((estoque_alta/estoque_outros - 1) * 100):.1f}%)")
    
    return agregados


def analise_por_sku_individual(df, top_n=5):
    """
    PARTE 4: ANÁLISE DE PRODUTOS INDIVIDUAIS
    
    Analisa padrões sazonais em produtos específicos (SKUs).
    Identifica produtos que mostram padrões sazonais mais claros.
    
    Métricas por SKU:
    - Média de estoque por mês
    - Variação entre meses (coeficiente de variação)
    - Diferença entre meses de alta e baixa temporada
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame com variáveis temporais
    top_n : int
        Número de produtos a destacar no resumo
        
    Returns:
    --------
    pd.DataFrame
        Estatísticas por SKU
    """
    _log("\n" + "=" * 80)
    _log("PARTE 4: ANALISE DE PRODUTOS INDIVIDUAIS (SKUs)")
    _log("=" * 80)
    
    skus = df['sku'].unique()
    n_skus = len(skus)
    _log(f"\nCalculando estatisticas por SKU ({n_skus} produtos)...")
    log_cada = max(1, n_skus // 10)
    
    stats_sku = []
    t_inicio_parte4 = time.time()
    tempos_sku_parte4 = []
    
    for i, sku in enumerate(skus):
        t_sku_inicio = time.time()
        
        if (i + 1) % log_cada == 0 or i == 0 or i == n_skus - 1:
            pct = 100.0 * (i + 1) / n_skus
            # Estimativa de tempo restante
            if i > 0 and len(tempos_sku_parte4) > 0:
                tempo_medio = sum(tempos_sku_parte4) / len(tempos_sku_parte4)
                restantes = n_skus - i - 1
                tempo_estimado = tempo_medio * restantes
                _log(f"  [Parte 4] SKU {i + 1}/{n_skus} ({pct:.1f}%) | Tempo estimado restante: {tempo_estimado:.1f}s")
            else:
                _log(f"  [Parte 4] SKU {i + 1}/{n_skus} ({pct:.1f}%)")
        
        df_sku = df[df['sku'] == sku].copy()
        
        # Estatísticas por mês
        estoque_por_mes = df_sku.groupby('mes')['estoque_atual'].mean()
        
        # Estatísticas gerais
        n_zero = (df_sku['estoque_atual'] == 0).sum()
        pct_zeros = 100.0 * n_zero / len(df_sku) if len(df_sku) > 0 else 100.0
        stats = {
            'sku': sku,
            'estoque_medio_geral': df_sku['estoque_atual'].mean(),
            'estoque_medio_out_dez': df_sku[df_sku['mes_alta_temporada']]['estoque_atual'].mean(),
            'estoque_medio_outros': df_sku[~df_sku['mes_alta_temporada']]['estoque_atual'].mean(),
            'cv_mensal': estoque_por_mes.std() / estoque_por_mes.mean() if estoque_por_mes.mean() > 0 else 0,
            'diferenca_alta_outros': 0,
            'pct_zeros': pct_zeros,
        }
        
        # Diferença entre alta temporada e outros meses
        if not pd.isna(stats['estoque_medio_out_dez']) and not pd.isna(stats['estoque_medio_outros']):
            if stats['estoque_medio_outros'] > 0:
                stats['diferenca_alta_outros'] = ((stats['estoque_medio_out_dez'] - stats['estoque_medio_outros']) 
                                                  / stats['estoque_medio_outros'] * 100)
        
        stats_sku.append(stats)
        
        t_sku_fim = time.time()
        dt_sku = t_sku_fim - t_sku_inicio
        tempos_sku_parte4.append(dt_sku)
    
    dt_total_parte4 = time.time() - t_inicio_parte4
    _log(f"  [Parte 4] Concluido: {n_skus} SKUs processados em {dt_total_parte4:.1f}s.")
    df_stats_sku = pd.DataFrame(stats_sku)
    
    # Ordena por maior diferença entre alta temporada e outros meses
    df_stats_sku = df_stats_sku.sort_values('diferenca_alta_outros', ascending=False, 
                                            na_position='last')
    
    n_eleg = (df_stats_sku['pct_zeros'] <= MAX_PCT_ZEROS_REPRESENTATIVO).sum()
    _log(f"\nAnalisados {len(df_stats_sku)} SKUs")
    _log(f"Elegiveis para representativo (zeros <= {MAX_PCT_ZEROS_REPRESENTATIVO}%): {n_eleg}")
    _log(f"\nTop {top_n} produtos com maior variacao entre alta temporada e outros meses:")
    cols = ['sku', 'estoque_medio_geral', 'pct_zeros', 'estoque_medio_out_dez', 'estoque_medio_outros', 'diferenca_alta_outros']
    _log(df_stats_sku.head(top_n)[cols].to_string(index=False))
    
    return df_stats_sku


def _escolher_sku_representativo(stats_sku, min_estoque_medio=1.0, max_pct_zeros=50.0, min_cv_mensal=1e-6):
    """
    Escolhe o SKU representativo para Fig 4 e Fig 5-7: maior variacao sazonal
    entre os que tem estoque medio relevante, POUCOS ZEROS (pct_zeros <= max_pct_zeros)
    e variacao mensal nao nula (cv_mensal >= min_cv_mensal).
    NUNCA relaxa o limite de zeros; SKUs com muitos zeros ou serie praticamente constante nao sao elegiveis.
    """
    if stats_sku is None or len(stats_sku) == 0:
        return None
    df = stats_sku.copy()
    if 'pct_zeros' not in df.columns:
        df['pct_zeros'] = 100.0
    if 'cv_mensal' not in df.columns:
        df['cv_mensal'] = 0.0
    elegiveis = df[df['pct_zeros'] <= max_pct_zeros]
    elegiveis = elegiveis[elegiveis['cv_mensal'].fillna(0) >= min_cv_mensal]
    if len(elegiveis) == 0:
        return None
    for limiar_em in (min_estoque_medio, 0.5, 0.2):
        candidatos = elegiveis[elegiveis['estoque_medio_geral'] >= limiar_em]
        if len(candidatos) > 0:
            sku = str(candidatos.iloc[0]['sku'])
            em = float(candidatos.iloc[0]['estoque_medio_geral'])
            pz = float(candidatos.iloc[0]['pct_zeros'])
            return sku, em, limiar_em, pz
    return None


def _top_n_eligible(stats_sku, n=10, min_estoque_medio=1.0, max_pct_zeros=50.0, min_cv_mensal=1e-6):
    """
    Retorna os top N SKUs elegiveis (zeros <= max_pct_zeros, estoque >= limiar, cv_mensal >= min_cv_mensal),
    ordenados por diferenca_alta_outros (maior primeiro).
    Exclui SKUs com variacao mensal praticamente nula (evita series constantes).
    Usado para processar os N melhores SKUs nas figuras 5-7 e Tabela 2.
    """
    if stats_sku is None or len(stats_sku) == 0:
        return []
    df = stats_sku.copy()
    if 'pct_zeros' not in df.columns:
        df['pct_zeros'] = 100.0
    if 'cv_mensal' not in df.columns:
        df['cv_mensal'] = 0.0
    elegiveis = df[df['pct_zeros'] <= max_pct_zeros]
    elegiveis = elegiveis[elegiveis['cv_mensal'].fillna(0) >= min_cv_mensal]
    if len(elegiveis) == 0:
        return []
    candidatos = None
    for limiar_em in (min_estoque_medio, 0.5, 0.2):
        cand = elegiveis[elegiveis['estoque_medio_geral'] >= limiar_em]
        if len(cand) > 0:
            candidatos = cand
            break
    if candidatos is None or len(candidatos) == 0:
        return []
    top = candidatos.head(n)
    return [str(r['sku']) for _, r in top.iterrows()]


def visualizar_padroes_sazonais(df, agregados_mensais, stats_sku=None, sku_exemplo=None, dir_figuras_tcc=None):
    """
    PARTE 5: VISUALIZAÇÃO DOS PADRÕES SAZONAIS (Figuras TCC 1–4)
    
    Gera figuras separadas conforme TCC:
    1. Evolução temporal do estoque total agregado (Figura 1)
    2. Distribuição mensal do estoque – boxplots (Figura 2)
    3. Estoque médio por mês (Figura 3)
    4. Série temporal de SKU com maior variação sazonal (Figura 4)
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame com dados completos
    agregados_mensais : pd.DataFrame
        Agregados mensais calculados anteriormente
    stats_sku : pd.DataFrame, optional
        Estatísticas por SKU (ordenadas por diferenca_alta_outros). Usado para Fig 4.
    sku_exemplo : str, optional
        SKU específico para Fig 4. Se None, usa top-1 de stats_sku (maior variação sazonal).
    dir_figuras_tcc : pathlib.Path, optional
        Se informado, salva figura1.png ... figura4.png neste diretório (nomenclatura TCC).
    """
    _log("\n" + "=" * 80)
    _log("PARTE 5: VISUALIZACAO DOS PADROES SAZONAIS (FIGURAS TCC 1-4)")
    _log("=" * 80)
    
    meses_ordem = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                   'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    
    # SKU para Figura 4: maior variação sazonal (TCC); usa sku_exemplo se informado
    if sku_exemplo is not None:
        _log(f"  Fig 4: SKU representativo = {sku_exemplo}")
    elif stats_sku is not None and len(stats_sku) > 0:
        res = _escolher_sku_representativo(stats_sku)
        if res:
            sku_exemplo, em, limiar_em, pz = res
            _log(f"  Fig 4: SKU representativo = {sku_exemplo} (estoque medio = {em:.2f}, zeros = {pz:.1f}%)")
        else:
            sku_exemplo = str(stats_sku.iloc[0]['sku'])
            _log(f"  Fig 4: SKU com maior variacao sazonal = {sku_exemplo}")
    else:
        sku_exemplo = df.groupby('sku').size().idxmax()
        _log(f"  Fig 4: SKU com mais observacoes (fallback) = {sku_exemplo}")

    out_dir = Path(dir_figuras_tcc) if dir_figuras_tcc else DIR_FIGURAS
    if dir_figuras_tcc:
        out_dir.mkdir(parents=True, exist_ok=True)
    sufixo = ('figura1', 'figura2', 'figura3', 'figura4') if dir_figuras_tcc else (
        'figura_01_evolucao_estoque_total', 'figura_02_distribuicao_mensal',
        'figura_03_estoque_medio_mes', 'figura_04_serie_temporal_sku')

    # --- Figura 1: Evolução temporal do estoque total agregado ---
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    estoque_diario = df.groupby('data')['estoque_atual'].sum()
    ax1.plot(estoque_diario.index, estoque_diario.values, linewidth=1.5, alpha=0.7, color='steelblue')
    ax1.set_title('Figura 1 – Evolucao Temporal do Estoque Total Agregado', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Data')
    ax1.set_ylabel('Estoque Total (unidades)')
    ax1.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    p1 = out_dir / f'{sufixo[0]}.png'
    plt.savefig(p1, dpi=300, bbox_inches='tight')
    plt.close()
    _log(f"[OK] Figura 1: {p1}")

    # --- Figura 2: Distribuição mensal do estoque (boxplots) ---
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    df_plot = df.copy()
    df_plot['mes_nome_ord'] = pd.Categorical(df_plot['mes_nome'], categories=meses_ordem, ordered=True)
    df_plot = df_plot.sort_values('mes_nome_ord')
    df_boxplot = df_plot.sample(min(50000, len(df_plot))) if len(df_plot) > 50000 else df_plot
    sns.boxplot(data=df_boxplot, x='mes_nome_ord', y='estoque_atual', ax=ax2)
    ax2.set_title('Figura 2 – Distribuicao Mensal do Estoque', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Mes')
    ax2.set_ylabel('Estoque (unidades)')
    plt.xticks(rotation=45)
    for i, mes in enumerate(meses_ordem):
        if mes in ['Out', 'Dez']:
            ax2.axvline(x=i, color='red', linestyle='--', alpha=0.3, linewidth=2)
    plt.tight_layout()
    p2 = out_dir / f'{sufixo[1]}.png'
    plt.savefig(p2, dpi=300, bbox_inches='tight')
    plt.close()
    _log(f"[OK] Figura 2: {p2}")

    # --- Figura 3: Estoque médio por mês ---
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    agregados_ordenados = agregados_mensais.sort_values('mes')
    cores = ['red' if m in ['Out', 'Dez'] else 'steelblue' for m in agregados_ordenados['mes_nome']]
    ax3.bar(range(len(agregados_ordenados)), agregados_ordenados['estoque_medio'],
            color=cores, alpha=0.7, edgecolor='black')
    ax3.set_xticks(range(len(agregados_ordenados)))
    ax3.set_xticklabels(agregados_ordenados['mes_nome'], rotation=45)
    ax3.set_title('Figura 3 – Estoque Medio por Mes', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Mes')
    ax3.set_ylabel('Estoque Medio (unidades)')
    ax3.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    p3 = out_dir / f'{sufixo[2]}.png'
    plt.savefig(p3, dpi=300, bbox_inches='tight')
    plt.close()
    _log(f"[OK] Figura 3: {p3}")

    # --- Figura 4: Série temporal do SKU com maior variação sazonal ---
    df_sku = df[df['sku'] == sku_exemplo].sort_values('data')
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    if len(df_sku) > 0:
        ax4.plot(df_sku['data'], df_sku['estoque_atual'], linewidth=1.5, color='darkgreen', alpha=0.8)
        has_leg = False
        for mes in [10, 12]:
            mask = df_sku['mes'] == mes
            if mask.sum() > 0:
                ax4.scatter(df_sku.loc[mask, 'data'], df_sku.loc[mask, 'estoque_atual'],
                            color='red', s=30, alpha=0.6, zorder=5, label='Out/Dez' if not has_leg else '')
                has_leg = True
        if has_leg:
            ax4.legend()
    ax4.set_title(f'Figura 4 – Serie Temporal do Estoque de SKU Selecionado ({sku_exemplo})', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Data')
    ax4.set_ylabel('Estoque (unidades)')
    ax4.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    p4 = out_dir / f'{sufixo[3]}.png'
    plt.savefig(p4, dpi=300, bbox_inches='tight')
    plt.close()
    _log(f"[OK] Figura 4: {p4}")

    # Figura combinada (apenas se nao for modo TCC)
    if not dir_figuras_tcc:
        fig = plt.figure(figsize=(16, 12))
        ax1c = plt.subplot(2, 2, 1)
        estoque_diario = df.groupby('data')['estoque_atual'].sum()
        ax1c.plot(estoque_diario.index, estoque_diario.values, linewidth=1.5, alpha=0.7, color='steelblue')
        ax1c.set_title('Evolucao Temporal: Estoque Total Diario', fontsize=12, fontweight='bold')
        ax1c.set_xlabel('Data')
        ax1c.set_ylabel('Estoque Total (unidades)')
        ax1c.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        ax2c = plt.subplot(2, 2, 2)
        sns.boxplot(data=df_boxplot, x='mes_nome_ord', y='estoque_atual', ax=ax2c)
        ax2c.set_title('Distribuicao de Estoque por Mes (Boxplot)', fontsize=12, fontweight='bold')
        ax2c.set_xlabel('Mes')
        ax2c.set_ylabel('Estoque (unidades)')
        plt.xticks(rotation=45)
        ax3c = plt.subplot(2, 2, 3)
        ax3c.bar(range(len(agregados_ordenados)), agregados_ordenados['estoque_medio'],
                 color=cores, alpha=0.7, edgecolor='black')
        ax3c.set_xticks(range(len(agregados_ordenados)))
        ax3c.set_xticklabels(agregados_ordenados['mes_nome'], rotation=45)
        ax3c.set_title('Estoque Medio por Mes', fontsize=12, fontweight='bold')
        ax3c.set_xlabel('Mes')
        ax3c.set_ylabel('Estoque Medio (unidades)')
        ax3c.grid(True, alpha=0.3, axis='y')
        ax4c = plt.subplot(2, 2, 4)
        if len(df_sku) > 0:
            ax4c.plot(df_sku['data'], df_sku['estoque_atual'], linewidth=1.5, color='darkgreen', alpha=0.8)
        ax4c.set_title(f'Serie Temporal: SKU {sku_exemplo}', fontsize=12, fontweight='bold')
        ax4c.set_xlabel('Data')
        ax4c.set_ylabel('Estoque (unidades)')
        ax4c.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        p_comb = DIR_FIGURAS / 'analise_sazonalidade_padroes.png'
        plt.savefig(p_comb, dpi=300, bbox_inches='tight')
        plt.close()
        _log(f"[OK] Figura combinada: {p_comb}")


def gerar_relatorio_completo(df, agregados_mensais, stats_sku):
    """
    PARTE 6: GERAÇÃO DE RELATÓRIO COMPLETO
    
    Gera um relatório textual resumindo todas as análises realizadas.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame completo
    agregados_mensais : pd.DataFrame
        Agregados mensais
    stats_sku : pd.DataFrame
        Estatísticas por SKU
    """
    _log("\n" + "=" * 80)
    _log("PARTE 6: RELATORIO COMPLETO")
    _log("=" * 80)
    
    relatorio = []
    relatorio.append("=" * 80)
    relatorio.append("RELATORIO DE ANALISE EXPLORATORIA: PADROES SAZONAIS")
    relatorio.append("=" * 80)
    relatorio.append(f"\nData de analise: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    relatorio.append(f"\nPeriodo analisado: {df['data'].min().date()} ate {df['data'].max().date()}")
    relatorio.append(f"Total de registros: {len(df):,}")
    relatorio.append(f"SKUs analisados: {df['sku'].nunique()}")
    
    # Resumo mensal
    relatorio.append("\n" + "-" * 80)
    relatorio.append("RESUMO POR MES")
    relatorio.append("-" * 80)
    relatorio.append(agregados_mensais.to_string(index=False))
    
    # Comparação alta temporada
    estoque_alta = df[df['mes_alta_temporada']]['estoque_atual'].mean()
    estoque_outros = df[~df['mes_alta_temporada']]['estoque_atual'].mean()
    
    relatorio.append("\n" + "-" * 80)
    relatorio.append("COMPARACAO: ALTA TEMPORADA (OUT/DEZ) vs OUTROS MESES")
    relatorio.append("-" * 80)
    relatorio.append(f"Estoque medio (Out/Dez): {estoque_alta:.2f} unidades")
    relatorio.append(f"Estoque medio (outros meses): {estoque_outros:.2f} unidades")
    relatorio.append(f"Diferenca absoluta: {estoque_alta - estoque_outros:.2f} unidades")
    relatorio.append(f"Diferenca percentual: {((estoque_alta/estoque_outros - 1) * 100):.1f}%")
    
    if estoque_alta > estoque_outros:
        relatorio.append("\nCONCLUSAO: Ha mais estoque nos meses de alta temporada (esperado)")
    else:
        relatorio.append("\nCONCLUSAO: Estoque menor nos meses de alta temporada (pode indicar alta rotacao)")
    
    # Top produtos
    relatorio.append("\n" + "-" * 80)
    relatorio.append("TOP 10 PRODUTOS COM MAIOR VARIACAO SAZONAL")
    relatorio.append("-" * 80)
    top_10 = stats_sku.head(10)[['sku', 'estoque_medio_geral', 'estoque_medio_out_dez', 
                                  'estoque_medio_outros', 'diferenca_alta_outros']]
    relatorio.append(top_10.to_string(index=False))
    
    # Salva relatório
    texto_relatorio = "\n".join(relatorio)
    path_rel = DIR_RESULTADOS / 'relatorio_analise_sazonalidade.txt'
    with open(path_rel, 'w', encoding='utf-8') as f:
        f.write(texto_relatorio)
    
    _log(f"\n[OK] Relatorio salvo: {path_rel}")
    _log("\n" + texto_relatorio)


def main(usar_nomes_tcc=False, caminho_dados=None):
    """
    FUNÇÃO PRINCIPAL
    
    Orquestra todo o processo de análise exploratória executando cada parte sequencialmente.
    
    Parameters:
    -----------
    usar_nomes_tcc : bool
        Se True, salva figura1.png ... figura4.png em resultados/figuras_tcc/
    caminho_dados : str, optional
        Caminho do CSV processado (default: DB/historico_estoque_atual_processado.csv)
    """
    import sys
    if '--tcc' in sys.argv:
        usar_nomes_tcc = True
    
    _log("\n" + "=" * 80)
    _log("ANALISE EXPLORATORIA: PADROES SAZONAIS EM DADOS DE ESTOQUE")
    _log("=" * 80)
    
    # PARTE 1: Carregar dados
    df = carregar_dados_processados(caminho=caminho_dados or 'DB/historico_estoque_atual_processado.csv')
    
    # PARTE 2: Criar variáveis temporais
    df = adicionar_variaveis_temporais(df)
    
    # PARTE 3: Análise de agregados mensais
    agregados_mensais = analise_agregados_mensais(df)
    
    # PARTE 4: Análise por SKU individual
    stats_sku = analise_por_sku_individual(df, top_n=10)
    
    # SKU representativo: maior variacao sazonal entre os com estoque ok e POUCOS ZEROS (<= 30%)
    res = _escolher_sku_representativo(
        stats_sku,
        min_estoque_medio=MIN_ESTOQUE_MEDIO_REPRESENTATIVO,
        max_pct_zeros=MAX_PCT_ZEROS_REPRESENTATIVO,
    )
    if res is None:
        n_eleg = (stats_sku['pct_zeros'] <= MAX_PCT_ZEROS_REPRESENTATIVO).sum() if 'pct_zeros' in stats_sku.columns else 0
        raise RuntimeError(
            f"Nenhum SKU elegivel: exige estoque medio >= {MIN_ESTOQUE_MEDIO_REPRESENTATIVO} e "
            f"zeros <= {MAX_PCT_ZEROS_REPRESENTATIVO}%. SKUs elegiveis: {n_eleg}. "
            "Verifique os dados ou ajuste MAX_PCT_ZEROS_REPRESENTATIVO em analise_exploratoria_sazonalidade."
        )
    sku_representativo = res[0]
    em_sel, limiar_em, pz_sel = res[1], res[2], res[3]
    _log(f"\n[SKU REPRESENTATIVO] {sku_representativo} (estoque medio >= {limiar_em}, zeros = {pz_sel:.1f}%)")
    
    # Verificacao: garantir que o escolhido realmente atende aos criterios (zeros <= 30%)
    row = stats_sku[stats_sku['sku'].astype(str) == str(sku_representativo)]
    if len(row) > 0:
        pz = float(row.iloc[0]['pct_zeros'])
        if pz > MAX_PCT_ZEROS_REPRESENTATIVO:
            raise RuntimeError(f"SKU {sku_representativo} tem {pz:.1f}% zeros (max {MAX_PCT_ZEROS_REPRESENTATIVO}%). Nao deve ter sido escolhido.")
        _log(f"  [OK] Verificacao: SKU {sku_representativo} atende aos criterios (zeros <= {MAX_PCT_ZEROS_REPRESENTATIVO}%).")
    
    # PARTE 5: Visualização (Figuras TCC 1–4)
    dir_tcc = Path('resultados/figuras_tcc') if usar_nomes_tcc else None
    visualizar_padroes_sazonais(df, agregados_mensais, stats_sku=stats_sku, sku_exemplo=sku_representativo, dir_figuras_tcc=dir_tcc)
    
    # PARTE 6: Relatório completo
    gerar_relatorio_completo(df, agregados_mensais, stats_sku)
    
    _log("\n" + "=" * 80)
    _log("ANALISE EXPLORATORIA CONCLUIDA COM SUCESSO!")
    _log("=" * 80)
    top10_skus = _top_n_eligible(
        stats_sku,
        n=10,
        min_estoque_medio=MIN_ESTOQUE_MEDIO_REPRESENTATIVO,
        max_pct_zeros=MAX_PCT_ZEROS_REPRESENTATIVO,
    )
    top300_skus = _top_n_eligible(
        stats_sku,
        n=300,
        min_estoque_medio=MIN_ESTOQUE_MEDIO_REPRESENTATIVO,
        max_pct_zeros=MAX_PCT_ZEROS_REPRESENTATIVO,
    )
    _log(f"\n[TOP 10 SKUs] {top10_skus}")
    _log(f"\n[TOP 300 CANDIDATOS] {len(top300_skus)} SKUs (para rodada de metricas)")
    _log(f"\nSKU representativo (Fig 4): {sku_representativo}")
    if usar_nomes_tcc:
        _log("\nFiguras TCC (resultados/figuras_tcc/):")
        for i in range(1, 5):
            _log(f"  - figura{i}.png")
    else:
        _log("\nArquivos gerados:")
        _log(f"  - {DIR_FIGURAS}/figura_01_evolucao_estoque_total.png")
        _log(f"  - {DIR_FIGURAS}/figura_02_distribuicao_mensal.png")
        _log(f"  - {DIR_FIGURAS}/figura_03_estoque_medio_mes.png")
        _log(f"  - {DIR_FIGURAS}/figura_04_serie_temporal_sku.png")
        _log(f"  - {DIR_FIGURAS}/analise_sazonalidade_padroes.png (combinada)")
    _log(f"  - {DIR_RESULTADOS}/relatorio_analise_sazonalidade.txt")
    _log("=" * 80)
    return df, agregados_mensais, stats_sku, sku_representativo, top10_skus, top300_skus


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERRO] Erro durante execucao: {str(e)}")
        import traceback
        traceback.print_exc()

