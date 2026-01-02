"""
Script para validar extração de dados de venda_produtos para elencação
Verifica se consegue extrair:
1. Quantidade vendida (soma)
2. Margem proporcional (média)
3. Rentabilidade (R(t)) = Média (Valor de Venda Unitário - Custo de Aquisição Unitário)
4. Nível de Urgência (U(t)) = Estoque Atual / Venda Média Diária Histórica
"""

import pandas as pd
import numpy as np
from pathlib import Path

def validar_estrutura_vendas():
    """Valida estrutura do arquivo venda_produtos"""
    print("=" * 80)
    print("VALIDACAO: Estrutura do arquivo venda_produtos_atual.csv")
    print("=" * 80)
    
    caminho_vendas = Path("DB/venda_produtos_atual.csv")
    
    if not caminho_vendas.exists():
        print(f"\n[ERRO] Arquivo nao encontrado: {caminho_vendas}")
        return False
    
    print(f"\n[OK] Arquivo encontrado: {caminho_vendas}")
    
    # Lê primeiras linhas para ver estrutura
    print("\n1. Verificando estrutura do CSV...")
    df_sample = pd.read_csv(caminho_vendas, nrows=10)
    print(f"\nColunas encontradas: {list(df_sample.columns)}")
    print(f"\nPrimeiras linhas:")
    print(df_sample.head(3).to_string())
    
    # Verifica colunas necessárias
    colunas_necessarias = ['sku', 'quantidade', 'margem_proporcional', 'valor_unitario', 'custo_unitario', 'created_at']
    colunas_faltando = [col for col in colunas_necessarias if col not in df_sample.columns]
    
    if colunas_faltando:
        print(f"\n[ERRO] Colunas faltando: {colunas_faltando}")
        return False
    
    print(f"\n[OK] Todas as colunas necessarias encontradas!")
    return True


def extrair_dados_vendas(caminho_vendas="DB/venda_produtos_atual.csv"):
    """
    Extrai dados agregados de venda_produtos por SKU.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame com colunas:
        - sku: identificador do produto
        - quantidade_vendida_total: soma da quantidade vendida
        - margem_proporcional_media: média da margem proporcional
        - valor_unitario_medio: valor unitário médio
        - custo_unitario_medio: custo unitário médio
        - rentabilidade_media: média (valor_unitario - custo_unitario)
        - num_vendas: número de transações
    """
    print("\n" + "=" * 80)
    print("EXTRAINDO: Dados agregados de venda_produtos")
    print("=" * 80)
    
    print(f"\nLendo arquivo: {caminho_vendas}...")
    df_vendas = pd.read_csv(caminho_vendas, low_memory=False)
    
    print(f"[OK] {len(df_vendas):,} registros carregados")
    
    # Converte created_at para datetime
    print("\nConvertendo datas...")
    df_vendas['created_at'] = pd.to_datetime(df_vendas['created_at'], errors='coerce')
    
    # Remove registros com SKU nulo
    print("Filtrando registros validos...")
    df_vendas = df_vendas[df_vendas['sku'].notna()]
    
    # Converte margem_proporcional para numérico (pode ter valores vazios)
    df_vendas['margem_proporcional'] = pd.to_numeric(df_vendas['margem_proporcional'], errors='coerce')
    
    # Converte valores para numérico
    df_vendas['quantidade'] = pd.to_numeric(df_vendas['quantidade'], errors='coerce')
    df_vendas['valor_unitario'] = pd.to_numeric(df_vendas['valor_unitario'], errors='coerce')
    df_vendas['custo_unitario'] = pd.to_numeric(df_vendas['custo_unitario'], errors='coerce')
    
    print(f"[OK] {len(df_vendas):,} registros validos apos filtros")
    
    # Agrega por SKU
    print("\nAgregando dados por SKU...")
    df_agregado = df_vendas.groupby('sku').agg({
        'quantidade': 'sum',  # Soma da quantidade vendida
        'margem_proporcional': 'mean',  # Média da margem proporcional
        'valor_unitario': 'mean',  # Valor unitário médio
        'custo_unitario': 'mean',  # Custo unitário médio
        'venda_id': 'count'  # Número de vendas
    }).reset_index()
    
    # Renomeia colunas
    df_agregado.columns = [
        'sku',
        'quantidade_vendida_total',
        'margem_proporcional_media',
        'valor_unitario_medio',
        'custo_unitario_medio',
        'num_vendas'
    ]
    
    # Calcula Rentabilidade: R(t) = Média (Valor de Venda Unitário - Custo de Aquisição Unitário)
    # Usamos os valores médios por SKU
    df_agregado['rentabilidade_media'] = (
        df_agregado['valor_unitario_medio'] - df_agregado['custo_unitario_medio']
    )
    
    print(f"[OK] {len(df_agregado):,} SKUs unicos encontrados")
    
    return df_agregado


def calcular_venda_media_diaria(df_vendas, periodo_dias=365):
    """
    Calcula venda média diária histórica por SKU.
    
    Parameters:
    -----------
    df_vendas : pd.DataFrame
        DataFrame com vendas (deve ter 'sku', 'created_at', 'quantidade')
    periodo_dias : int
        Período em dias para calcular média (padrão: 365)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame com 'sku' e 'venda_media_diaria'
    """
    print(f"\nCalculando venda media diaria (ultimos {periodo_dias} dias)...")
    
    # Filtra período
    data_limite = df_vendas['created_at'].max() - pd.Timedelta(days=periodo_dias)
    df_periodo = df_vendas[df_vendas['created_at'] >= data_limite].copy()
    
    # Agrupa por SKU e data, soma quantidade
    df_vendas_diarias = df_periodo.groupby(['sku', pd.Grouper(key='created_at', freq='D')])['quantidade'].sum().reset_index()
    
    # Calcula média diária por SKU
    venda_media = df_vendas_diarias.groupby('sku')['quantidade'].mean().reset_index()
    venda_media.columns = ['sku', 'venda_media_diaria']
    
    print(f"[OK] Venda media calculada para {len(venda_media):,} SKUs")
    
    return venda_media


def calcular_nivel_urgencia(df_estoque_atual, df_venda_media_diaria):
    """
    Calcula Nível de Urgência: U(t) = Estoque Atual / Venda Média Diária Histórica
    
    Parameters:
    -----------
    df_estoque_atual : pd.DataFrame
        DataFrame com 'sku' e 'saldo' (estoque atual)
    df_venda_media_diaria : pd.DataFrame
        DataFrame com 'sku' e 'venda_media_diaria'
        
    Returns:
    --------
    pd.DataFrame
        DataFrame com 'sku', 'estoque_atual', 'venda_media_diaria', 'nivel_urgencia'
    """
    print("\nCalculando Nivel de Urgencia...")
    
    # Merge
    df_merge = df_estoque_atual.merge(
        df_venda_media_diaria,
        on='sku',
        how='inner'
    )
    
    # Calcula U(t) = Estoque Atual / Venda Média Diária
    # Evita divisão por zero
    df_merge['nivel_urgencia'] = np.where(
        df_merge['venda_media_diaria'] > 0,
        df_merge['saldo'] / df_merge['venda_media_diaria'],
        np.inf  # Se não há vendas, urgência é infinita (estoque dura "para sempre")
    )
    
    print(f"[OK] Nivel de urgencia calculado para {len(df_merge):,} SKUs")
    
    return df_merge[['sku', 'saldo', 'venda_media_diaria', 'nivel_urgencia']]


def validar_extracao_completa():
    """Valida extração completa de todas as métricas"""
    print("\n" + "=" * 80)
    print("VALIDACAO COMPLETA: Todas as metricas para elencacao")
    print("=" * 80)
    
    # 1. Valida estrutura
    if not validar_estrutura_vendas():
        return
    
    # 2. Extrai dados de vendas
    df_vendas_agregado = extrair_dados_vendas()
    
    # 3. Calcula venda média diária
    df_vendas_raw = pd.read_csv("DB/venda_produtos_atual.csv", low_memory=False)
    df_vendas_raw['created_at'] = pd.to_datetime(df_vendas_raw['created_at'], errors='coerce')
    df_vendas_raw['quantidade'] = pd.to_numeric(df_vendas_raw['quantidade'], errors='coerce')
    df_vendas_raw = df_vendas_raw[df_vendas_raw['sku'].notna()]
    
    df_venda_media_diaria = calcular_venda_media_diaria(df_vendas_raw)
    
    # 4. Tenta calcular nível de urgência (precisa do estoque atual)
    caminho_estoque = Path("DB/historico_estoque_atual.csv")
    if caminho_estoque.exists():
        print(f"\nLendo estoque atual de: {caminho_estoque}")
        df_estoque = pd.read_csv(caminho_estoque)
        df_estoque['created_at'] = pd.to_datetime(df_estoque['created_at'], errors='coerce')
        
        # Pega último saldo por SKU (estoque atual)
        df_estoque_atual = df_estoque.sort_values('created_at').groupby('sku').last().reset_index()[['sku', 'saldo']]
        df_estoque_atual['saldo'] = pd.to_numeric(df_estoque_atual['saldo'], errors='coerce')
        
        df_urgencia = calcular_nivel_urgencia(df_estoque_atual, df_venda_media_diaria)
    else:
        print(f"\n[AVISO] Arquivo de estoque nao encontrado: {caminho_estoque}")
        df_urgencia = None
    
    # 5. Merge final
    print("\n" + "=" * 80)
    print("RESULTADO FINAL: Metricas agregadas")
    print("=" * 80)
    
    df_final = df_vendas_agregado.merge(
        df_venda_media_diaria,
        on='sku',
        how='left'
    )
    
    if df_urgencia is not None:
        df_final = df_final.merge(
            df_urgencia[['sku', 'nivel_urgencia', 'saldo']],
            on='sku',
            how='left'
        )
    
    # Mostra resumo
    print(f"\nTotal de SKUs: {len(df_final):,}")
    print(f"\nColunas disponiveis:")
    for col in df_final.columns:
        print(f"  - {col}")
    
    print(f"\nPrimeiros 10 SKUs (ordenados por quantidade vendida):")
    df_mostrar = df_final.nlargest(10, 'quantidade_vendida_total')
    print(df_mostrar[['sku', 'quantidade_vendida_total', 'margem_proporcional_media', 
                      'rentabilidade_media', 'venda_media_diaria']].to_string(index=False))
    
    if df_urgencia is not None:
        print(f"\nTop 10 SKUs com menor nivel de urgencia (maior risco):")
        df_urgencia_ordenado = df_urgencia.nsmallest(10, 'nivel_urgencia')
        print(df_urgencia_ordenado.to_string(index=False))
    
    # Validações
    print("\n" + "=" * 80)
    print("VALIDACOES:")
    print("=" * 80)
    
    print(f"\n1. Quantidade vendida (soma):")
    print(f"   [OK] Coluna 'quantidade_vendida_total' presente")
    print(f"   Total geral: {df_final['quantidade_vendida_total'].sum():,.0f} unidades")
    
    print(f"\n2. Margem proporcional (media):")
    print(f"   [OK] Coluna 'margem_proporcional_media' presente")
    print(f"   Media geral: {df_final['margem_proporcional_media'].mean():.2f}%")
    
    print(f"\n3. Rentabilidade R(t) = Media (Valor Unitario - Custo Unitario):")
    print(f"   [OK] Coluna 'rentabilidade_media' calculada")
    print(f"   Media geral: R$ {df_final['rentabilidade_media'].mean():.2f}")
    
    print(f"\n4. Venda media diaria historica:")
    print(f"   [OK] Coluna 'venda_media_diaria' calculada")
    print(f"   Media geral: {df_final['venda_media_diaria'].mean():.2f} unidades/dia")
    
    if df_urgencia is not None:
        print(f"\n5. Nivel de Urgencia U(t) = Estoque Atual / Venda Media Diaria:")
        print(f"   [OK] Coluna 'nivel_urgencia' calculada")
        print(f"   Media geral: {df_final['nivel_urgencia'].mean():.1f} dias")
    
    print("\n" + "=" * 80)
    print("[OK] Validacao concluida! Sistema consegue extrair todas as metricas.")
    print("=" * 80)
    
    return df_final


if __name__ == "__main__":
    df_resultado = validar_extracao_completa()
    
    if df_resultado is not None:
        print(f"\n[INFO] Resultado disponivel na variavel df_resultado")
        print(f"[INFO] Para salvar, execute: df_resultado.to_csv('metricas_vendas_para_elencacao.csv', index=False)")

