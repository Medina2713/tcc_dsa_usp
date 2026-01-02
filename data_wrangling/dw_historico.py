"""
Data Wrangling: Preparação de Série Histórica para Modelo SARIMA
TCC MBA Data Science & Analytics - E-commerce de Brinquedos

Este script prepara os dados do arquivo historico_estoque.csv para uso com
o modelo SARIMA, realizando limpeza, transformação e agregação necessárias.

Autor: Medina2713
Data: 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def carregar_dados(caminho_arquivo='DB/historico_estoque.csv'):
    """
    Carrega o arquivo CSV de histórico de estoque.
    
    Parameters:
    -----------
    caminho_arquivo : str
        Caminho para o arquivo CSV
        
    Returns:
    --------
    pd.DataFrame
        DataFrame com os dados carregados
    """
    print(f"📂 Carregando dados de: {caminho_arquivo}")
    df = pd.read_csv(caminho_arquivo)
    print(f"   ✓ {len(df):,} registros carregados")
    return df


def limpar_dados(df):
    """
    Remove registros inválidos e limpa dados faltantes.
    
    Steps:
    1. Remove registros com SKU nulo
    2. Remove registros com saldo negativo (se houver)
    3. Remove registros com data inválida
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame com dados brutos
        
    Returns:
    --------
    pd.DataFrame
        DataFrame limpo
    """
    print("\n🧹 Limpando dados...")
    n_inicial = len(df)
    
    # Remove SKUs nulos
    n_sku_null = df['sku'].isnull().sum()
    df = df[df['sku'].isnull() == False].copy()
    if n_sku_null > 0:
        print(f"   ✓ Removidos {n_sku_null:,} registros com SKU nulo")
    
    # Remove saldos negativos (estoque não pode ser negativo)
    n_negativos = (df['saldo'] < 0).sum()
    if n_negativos > 0:
        df = df[df['saldo'] >= 0].copy()
        print(f"   ✓ Removidos {n_negativos:,} registros com saldo negativo")
    else:
        print(f"   ✓ Nenhum saldo negativo encontrado")
    
    # Converte data (lida com formatos mistos)
    df['created_at'] = pd.to_datetime(df['created_at'], format='mixed', errors='coerce')
    
    # Remove registros com data inválida
    n_data_invalida = df['created_at'].isnull().sum()
    if n_data_invalida > 0:
        df = df[df['created_at'].notna()].copy()
        print(f"   ✓ Removidos {n_data_invalida:,} registros com data inválida")
    
    n_final = len(df)
    removidos = n_inicial - n_final
    print(f"   ✓ Total removido: {removidos:,} registros ({removidos/n_inicial*100:.1f}%)")
    print(f"   ✓ Registros restantes: {n_final:,}")
    
    return df


def agregar_por_dia(df):
    """
    Agrega múltiplos registros do mesmo SKU no mesmo dia.
    
    Como pode haver múltiplos registros do mesmo SKU no mesmo dia (diferentes horários),
    agrega usando a última observação do dia (último saldo conhecido).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame com dados limpos
        
    Returns:
    --------
    pd.DataFrame
        DataFrame agregado por SKU e data
    """
    print("\n📊 Agregando dados por dia...")
    
    # Extrai apenas a data (sem horário)
    df['data'] = df['created_at'].dt.date
    
    # Agrupa por SKU e data, pegando o último registro do dia (último saldo conhecido)
    # Ordena por created_at para garantir que pegamos o mais recente
    df_ordenado = df.sort_values('created_at')
    
    # Agrega mantendo o último saldo do dia
    df_agregado = df_ordenado.groupby(['sku', 'data']).agg({
        'saldo': 'last',  # Último saldo do dia
        'created_at': 'last'  # Última data/hora (para referência)
    }).reset_index()
    
    # Remove a coluna created_at (não é mais necessária, temos 'data')
    df_agregado = df_agregado[['sku', 'data', 'saldo']].copy()
    
    n_antes = len(df)
    n_depois = len(df_agregado)
    reducao = n_antes - n_depois
    
    print(f"   ✓ {n_antes:,} registros → {n_depois:,} registros únicos (SKU + Data)")
    if reducao > 0:
        print(f"   ✓ {reducao:,} registros agregados (múltiplos registros no mesmo dia)")
    
    return df_agregado


def criar_serie_temporal_completa(df, data_inicio=None, data_fim=None):
    """
    Cria séries temporais completas (sem gaps) para cada SKU.
    
    Para modelos SARIMA, é importante ter séries temporais completas (sem datas faltantes).
    Esta função preenche gaps com o último valor conhecido (forward fill) ou zero.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame agregado por SKU e data
    data_inicio : str/datetime, optional
        Data de início para a série (padrão: primeira data disponível)
    data_fim : str/datetime, optional
        Data de fim para a série (padrão: última data disponível)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame com séries temporais completas
    """
    print("\n📅 Criando séries temporais completas...")
    
    # Converte data para datetime se necessário
    df['data'] = pd.to_datetime(df['data'])
    
    # Define range de datas
    if data_inicio is None:
        data_inicio = df['data'].min()
    else:
        data_inicio = pd.to_datetime(data_inicio)
    
    if data_fim is None:
        data_fim = df['data'].max()
    else:
        data_fim = pd.to_datetime(data_fim)
    
    # Cria range completo de datas
    todas_datas = pd.date_range(start=data_inicio, end=data_fim, freq='D')
    print(f"   ✓ Range de datas: {data_inicio.date()} até {data_fim.date()}")
    print(f"   ✓ Total de dias: {len(todas_datas)}")
    
    # Para cada SKU, cria série completa
    resultados = []
    skus = df['sku'].unique()
    
    for sku in skus:
        df_sku = df[df['sku'] == sku].copy().set_index('data').sort_index()
        
        # Cria série completa para este SKU
        serie_completa = pd.DataFrame(index=todas_datas)
        serie_completa['sku'] = sku
        serie_completa['saldo'] = np.nan
        
        # Preenche com valores conhecidos
        serie_completa.loc[df_sku.index, 'saldo'] = df_sku['saldo']
        
        # Forward fill (preenche gaps com último valor conhecido)
        serie_completa['saldo'] = serie_completa['saldo'].ffill()
        
        # Se ainda houver NaN no início (antes do primeiro registro), preenche com zero
        serie_completa['saldo'] = serie_completa['saldo'].fillna(0)
        
        resultados.append(serie_completa.reset_index())
    
    df_completo = pd.concat(resultados, ignore_index=True)
    df_completo = df_completo.rename(columns={'index': 'data'})
    
    # Converte data para formato date (sem hora)
    df_completo['data'] = pd.to_datetime(df_completo['data']).dt.date
    
    print(f"   ✓ Séries completas criadas para {len(skus)} SKUs")
    print(f"   ✓ Total de registros: {len(df_completo):,}")
    
    return df_completo


def filtrar_skus_suficientes(df, min_observacoes=30):
    """
    Filtra apenas SKUs com número mínimo de observações.
    
    Para modelos SARIMA, é recomendado ter pelo menos 30 observações.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame com séries temporais
    min_observacoes : int
        Número mínimo de observações não-nulas por SKU
        
    Returns:
    --------
    pd.DataFrame
        DataFrame filtrado
    """
    print(f"\n🔍 Filtrando SKUs com pelo menos {min_observacoes} observações...")
    
    # Conta observações válidas por SKU (saldo > 0 ou não-nulo)
    contagem = df.groupby('sku')['saldo'].count()
    skus_validos = contagem[contagem >= min_observacoes].index.tolist()
    
    df_filtrado = df[df['sku'].isin(skus_validos)].copy()
    
    n_antes = df['sku'].nunique()
    n_depois = len(skus_validos)
    
    print(f"   ✓ SKUs antes: {n_antes}")
    print(f"   ✓ SKUs após filtro: {n_depois} (mínimo {min_observacoes} observações)")
    print(f"   ✓ Registros: {len(df_filtrado):,}")
    
    return df_filtrado


def formatar_para_sarima(df):
    """
    Formata o DataFrame para o formato esperado pelo módulo SARIMA.
    
    O módulo sarima_estoque.py espera:
    - Coluna 'data': datetime
    - Coluna 'sku': string
    - Coluna 'estoque_atual': numérico
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame processado
        
    Returns:
    --------
    pd.DataFrame
        DataFrame formatado para SARIMA
    """
    print("\n✨ Formatando para formato SARIMA...")
    
    df_formatado = df.copy()
    
    # Renomeia coluna 'saldo' para 'estoque_atual'
    df_formatado = df_formatado.rename(columns={'saldo': 'estoque_atual'})
    
    # Garante que 'data' é datetime
    df_formatado['data'] = pd.to_datetime(df_formatado['data'])
    
    # Garante que 'sku' é string
    df_formatado['sku'] = df_formatado['sku'].astype(str)
    
    # Garante que 'estoque_atual' é numérico
    df_formatado['estoque_atual'] = pd.to_numeric(df_formatado['estoque_atual'], errors='coerce')
    
    # Remove qualquer NaN que possa ter surgido
    df_formatado = df_formatado.dropna()
    
    # Ordena por SKU e data
    df_formatado = df_formatado.sort_values(['sku', 'data']).reset_index(drop=True)
    
    # Reordena colunas: data, sku, estoque_atual
    df_formatado = df_formatado[['data', 'sku', 'estoque_atual']].copy()
    
    print(f"   ✓ DataFrame formatado: {len(df_formatado):,} registros")
    print(f"   ✓ Colunas: {df_formatado.columns.tolist()}")
    print(f"   ✓ SKUs únicos: {df_formatado['sku'].nunique()}")
    
    return df_formatado


def processar_historico_estoque(
    caminho_entrada='DB/historico_estoque.csv',
    caminho_saida='DB/historico_estoque_processado.csv',
    min_observacoes=30,
    data_inicio=None,
    data_fim=None,
    criar_serie_completa=True
):
    """
    Pipeline completo de processamento de dados históricos para SARIMA.
    
    Parameters:
    -----------
    caminho_entrada : str
        Caminho do arquivo CSV de entrada
    caminho_saida : str
        Caminho do arquivo CSV de saída
    min_observacoes : int
        Número mínimo de observações por SKU
    data_inicio : str/datetime, optional
        Data de início para série completa
    data_fim : str/datetime, optional
        Data de fim para série completa
    criar_serie_completa : bool
        Se True, preenche gaps nas séries temporais
        
    Returns:
    --------
    pd.DataFrame
        DataFrame processado e pronto para SARIMA
    """
    print("=" * 70)
    print("DATA WRANGLING: Histórico de Estoque para SARIMA")
    print("=" * 70)
    
    # 1. Carregar dados
    df = carregar_dados(caminho_entrada)
    
    # 2. Limpar dados
    df = limpar_dados(df)
    
    # 3. Agregar por dia (último saldo do dia)
    df = agregar_por_dia(df)
    
    # 4. Criar séries temporais completas (opcional)
    if criar_serie_completa:
        df = criar_serie_temporal_completa(df, data_inicio, data_fim)
    else:
        # Se não criar série completa, pelo menos converte data para datetime
        df['data'] = pd.to_datetime(df['data'])
    
    # 5. Filtrar SKUs com observações suficientes
    df = filtrar_skus_suficientes(df, min_observacoes)
    
    # 6. Formatar para SARIMA
    df = formatar_para_sarima(df)
    
    # 7. Salvar resultado
    print(f"\n💾 Salvando resultado em: {caminho_saida}")
    df.to_csv(caminho_saida, index=False)
    print(f"   ✓ Arquivo salvo com sucesso!")
    
    print("\n" + "=" * 70)
    print("✅ PROCESSAMENTO CONCLUÍDO")
    print("=" * 70)
    print(f"\n📊 Resumo Final:")
    print(f"   - Total de registros: {len(df):,}")
    print(f"   - SKUs processados: {df['sku'].nunique()}")
    print(f"   - Período: {df['data'].min()} até {df['data'].max()}")
    print(f"   - Média de observações por SKU: {len(df) / df['sku'].nunique():.1f}")
    print("\n✅ Dados prontos para uso com modelo SARIMA!")
    print("=" * 70)
    
    return df


if __name__ == "__main__":
    # Executa o pipeline completo
    df_processado = processar_historico_estoque(
        caminho_entrada='DB/historico_estoque.csv',
        caminho_saida='DB/historico_estoque_processado.csv',
        min_observacoes=30,
        criar_serie_completa=True
    )
    
    # Mostra amostra do resultado
    print("\n📋 Amostra dos dados processados:")
    print(df_processado.head(20))
    
    print("\n📋 Estatísticas por SKU (exemplos):")
    stats_sku = df_processado.groupby('sku')['estoque_atual'].agg([
        'count', 'mean', 'min', 'max'
    ]).head(10)
    print(stats_sku)

