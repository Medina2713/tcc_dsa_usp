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
import warnings
warnings.filterwarnings('ignore')

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
    print("=" * 80)
    print("PARTE 1: CARREGAMENTO DE DADOS")
    print("=" * 80)
    print(f"\nCarregando dados de: {caminho}")
    
    df = pd.read_csv(caminho)
    df['data'] = pd.to_datetime(df['data'])
    
    print(f"[OK] Dados carregados: {len(df):,} registros")
    print(f"[OK] Periodo: {df['data'].min().date()} ate {df['data'].max().date()}")
    print(f"[OK] SKUs unicos: {df['sku'].nunique()}")
    print(f"[OK] Total de dias: {(df['data'].max() - df['data'].min()).days}")
    
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
    print("\n" + "=" * 80)
    print("PARTE 2: CRIACAO DE VARIAVEIS TEMPORAIS")
    print("=" * 80)
    
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
    
    print("\n[OK] Variaveis temporais criadas:")
    print("     - ano, mes, dia, dia_semana, trimestre, semana_ano")
    print("     - mes_nome, mes_alta_temporada (flag para out/dez)")
    
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
    print("\n" + "=" * 80)
    print("PARTE 3: ANALISE DE AGREGADOS MENSAIS")
    print("=" * 80)
    
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
    
    print("\nEstatisticas agregadas por mes:")
    print(agregados.to_string(index=False))
    
    # Comparação específica: Outubro e Dezembro vs outros meses
    print("\n" + "-" * 80)
    print("COMPARACAO: Meses de alta temporada (Out/Dez) vs Outros")
    print("-" * 80)
    
    estoque_alta = df[df['mes_alta_temporada'] == True]['estoque_atual'].mean()
    estoque_outros = df[df['mes_alta_temporada'] == False]['estoque_atual'].mean()
    
    print(f"Estoque medio (Out/Dez): {estoque_alta:.2f}")
    print(f"Estoque medio (outros meses): {estoque_outros:.2f}")
    print(f"Diferenca: {estoque_alta - estoque_outros:.2f} ({((estoque_alta/estoque_outros - 1) * 100):.1f}%)")
    
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
    print("\n" + "=" * 80)
    print("PARTE 4: ANALISE DE PRODUTOS INDIVIDUAIS (SKUs)")
    print("=" * 80)
    
    # Para cada SKU, calcula estatísticas mensais
    stats_sku = []
    
    for sku in df['sku'].unique():
        df_sku = df[df['sku'] == sku].copy()
        
        # Estatísticas por mês
        estoque_por_mes = df_sku.groupby('mes')['estoque_atual'].mean()
        
        # Estatísticas gerais
        stats = {
            'sku': sku,
            'estoque_medio_geral': df_sku['estoque_atual'].mean(),
            'estoque_medio_out_dez': df_sku[df_sku['mes_alta_temporada']]['estoque_atual'].mean(),
            'estoque_medio_outros': df_sku[~df_sku['mes_alta_temporada']]['estoque_atual'].mean(),
            'cv_mensal': estoque_por_mes.std() / estoque_por_mes.mean() if estoque_por_mes.mean() > 0 else 0,
            'diferenca_alta_outros': 0
        }
        
        # Diferença entre alta temporada e outros meses
        if not pd.isna(stats['estoque_medio_out_dez']) and not pd.isna(stats['estoque_medio_outros']):
            if stats['estoque_medio_outros'] > 0:
                stats['diferenca_alta_outros'] = ((stats['estoque_medio_out_dez'] - stats['estoque_medio_outros']) 
                                                  / stats['estoque_medio_outros'] * 100)
        
        stats_sku.append(stats)
    
    df_stats_sku = pd.DataFrame(stats_sku)
    
    # Ordena por maior diferença entre alta temporada e outros meses
    df_stats_sku = df_stats_sku.sort_values('diferenca_alta_outros', ascending=False, 
                                            na_position='last')
    
    print(f"\nAnalisados {len(df_stats_sku)} SKUs")
    print(f"\nTop {top_n} produtos com maior variacao entre alta temporada e outros meses:")
    print(df_stats_sku.head(top_n)[['sku', 'estoque_medio_geral', 'estoque_medio_out_dez', 
                                    'estoque_medio_outros', 'diferenca_alta_outros']].to_string(index=False))
    
    return df_stats_sku


def visualizar_padroes_sazonais(df, agregados_mensais, sku_exemplo=None):
    """
    PARTE 5: VISUALIZAÇÃO DOS PADRÕES SAZONAIS
    
    Cria gráficos para visualizar padrões sazonais:
    1. Evolução temporal agregada (estoque total ao longo do tempo)
    2. Boxplot por mês (distribuição de estoque em cada mês)
    3. Média de estoque por mês (bar chart)
    4. Série temporal de um produto específico (se fornecido)
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame com dados completos
    agregados_mensais : pd.DataFrame
        Agregados mensais calculados anteriormente
    sku_exemplo : str, optional
        SKU específico para análise detalhada
    """
    print("\n" + "=" * 80)
    print("PARTE 5: VISUALIZACAO DOS PADROES SAZONAIS")
    print("=" * 80)
    
    fig = plt.figure(figsize=(16, 12))
    
    # Gráfico 1: Evolução temporal agregada (estoque total ao longo do tempo)
    ax1 = plt.subplot(2, 2, 1)
    estoque_diario = df.groupby('data')['estoque_atual'].sum()
    ax1.plot(estoque_diario.index, estoque_diario.values, linewidth=1.5, alpha=0.7, color='steelblue')
    ax1.set_title('Evolucao Temporal: Estoque Total Diario', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Data')
    ax1.set_ylabel('Estoque Total (unidades)')
    ax1.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Gráfico 2: Boxplot por mês (distribuição de estoque em cada mês)
    ax2 = plt.subplot(2, 2, 2)
    meses_ordem = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 
                   'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    df_plot = df.copy()
    df_plot['mes_nome_ord'] = pd.Categorical(df_plot['mes_nome'], categories=meses_ordem, ordered=True)
    df_plot = df_plot.sort_values('mes_nome_ord')
    
    # Dados para boxplot (amostragem para performance se muitos dados)
    df_boxplot = df_plot.sample(min(50000, len(df_plot))) if len(df_plot) > 50000 else df_plot
    
    sns.boxplot(data=df_boxplot, x='mes_nome_ord', y='estoque_atual', ax=ax2)
    ax2.set_title('Distribuicao de Estoque por Mes (Boxplot)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Mes')
    ax2.set_ylabel('Estoque (unidades)')
    plt.xticks(rotation=45)
    
    # Destacar Outubro e Dezembro
    for i, mes in enumerate(meses_ordem):
        if mes in ['Out', 'Dez']:
            ax2.axvline(x=i, color='red', linestyle='--', alpha=0.3, linewidth=2)
    
    # Gráfico 3: Média de estoque por mês (bar chart)
    ax3 = plt.subplot(2, 2, 3)
    agregados_ordenados = agregados_mensais.sort_values('mes')
    cores = ['red' if mes in ['Out', 'Dez'] else 'steelblue' for mes in agregados_ordenados['mes_nome']]
    ax3.bar(range(len(agregados_ordenados)), agregados_ordenados['estoque_medio'], 
           color=cores, alpha=0.7, edgecolor='black')
    ax3.set_xticks(range(len(agregados_ordenados)))
    ax3.set_xticklabels(agregados_ordenados['mes_nome'], rotation=45)
    ax3.set_title('Estoque Medio por Mes', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Mes')
    ax3.set_ylabel('Estoque Medio (unidades)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Gráfico 4: Série temporal de produto específico (se fornecido ou top 1)
    ax4 = plt.subplot(2, 2, 4)
    if sku_exemplo is None:
        # Pega o SKU com mais observações
        sku_exemplo = df.groupby('sku').size().idxmax()
    
    df_sku = df[df['sku'] == sku_exemplo].sort_values('data')
    
    if len(df_sku) > 0:
        ax4.plot(df_sku['data'], df_sku['estoque_atual'], linewidth=1.5, color='darkgreen', alpha=0.8)
        ax4.set_title(f'Serie Temporal: SKU {sku_exemplo}', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Data')
        ax4.set_ylabel('Estoque (unidades)')
        ax4.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Destacar meses de alta temporada
        for mes in [10, 12]:  # Outubro e Dezembro
            mask = df_sku['mes'] == mes
            if mask.sum() > 0:
                ax4.scatter(df_sku[mask]['data'], df_sku[mask]['estoque_atual'], 
                          color='red', s=30, alpha=0.6, zorder=5, label='Out/Dez' if mes == 10 else '')
        if mask.sum() > 0:
            ax4.legend()
    
    plt.tight_layout()
    
    # Salva figura
    nome_arquivo = 'analise_sazonalidade_padroes.png'
    plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Grafico salvo: {nome_arquivo}")
    plt.close()


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
    print("\n" + "=" * 80)
    print("PARTE 6: RELATORIO COMPLETO")
    print("=" * 80)
    
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
    with open('relatorio_analise_sazonalidade.txt', 'w', encoding='utf-8') as f:
        f.write(texto_relatorio)
    
    print("\n[OK] Relatorio salvo: relatorio_analise_sazonalidade.txt")
    print("\n" + texto_relatorio)


def main():
    """
    FUNÇÃO PRINCIPAL
    
    Orquestra todo o processo de análise exploratória executando cada parte sequencialmente.
    """
    print("\n" + "=" * 80)
    print("ANALISE EXPLORATORIA: PADROES SAZONAIS EM DADOS DE ESTOQUE")
    print("=" * 80)
    
    # PARTE 1: Carregar dados
    df = carregar_dados_processados()
    
    # PARTE 2: Criar variáveis temporais
    df = adicionar_variaveis_temporais(df)
    
    # PARTE 3: Análise de agregados mensais
    agregados_mensais = analise_agregados_mensais(df)
    
    # PARTE 4: Análise por SKU individual
    stats_sku = analise_por_sku_individual(df, top_n=10)
    
    # PARTE 5: Visualização
    visualizar_padroes_sazonais(df, agregados_mensais)
    
    # PARTE 6: Relatório completo
    gerar_relatorio_completo(df, agregados_mensais, stats_sku)
    
    print("\n" + "=" * 80)
    print("ANALISE EXPLORATORIA CONCLUIDA COM SUCESSO!")
    print("=" * 80)
    print("\nArquivos gerados:")
    print("  - analise_sazonalidade_padroes.png (graficos)")
    print("  - relatorio_analise_sazonalidade.txt (relatorio textual)")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERRO] Erro durante execucao: {str(e)}")
        import traceback
        traceback.print_exc()

