"""
Script para calcular métricas de elencação a partir de venda_produtos
Implementa as 3 métricas da tabela 2.2:

1. Giro Futuro Previsto (GP(t)) - Resultado do SARIMA (soma das previsões)
2. Rentabilidade (R(t)) - Média (Valor de Venda Unitário - Custo de Aquisição Unitário)
3. Nível de Urgência (U(t)) - Estoque Atual / Venda Média Diária Histórica
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def carregar_dados_vendas(caminho="DB/venda_produtos_atual.csv"):
    """Carrega e prepara dados de vendas"""
    print(f"Carregando dados de vendas: {caminho}")
    df = pd.read_csv(caminho, low_memory=False)
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df['quantidade'] = pd.to_numeric(df['quantidade'], errors='coerce')
    df['margem_proporcional'] = pd.to_numeric(df['margem_proporcional'], errors='coerce')
    df['valor_unitario'] = pd.to_numeric(df['valor_unitario'], errors='coerce')
    df['custo_unitario'] = pd.to_numeric(df['custo_unitario'], errors='coerce')
    df = df[df['sku'].notna()]
    print(f"[OK] {len(df):,} registros carregados")
    return df


def calcular_rentabilidade(df_vendas):
    """
    Calcula Rentabilidade R(t) = Média (Valor de Venda Unitário - Custo de Aquisição Unitário)
    
    Conforme tabela 2.2:
    - Propósito: Valor Financeiro. O retorno médio por unidade.
    - Dado de Origem: venda_produtos
    - Fórmula: R(t) = Média (Valor de Venda Unitário - Custo de Aquisição Unitário)
    """
    print("\nCalculando Rentabilidade R(t)...")
    
    # Agrega por SKU
    df_agregado = df_vendas.groupby('sku').agg({
        'valor_unitario': 'mean',
        'custo_unitario': 'mean',
        'margem_proporcional': 'mean',  # Também calcula média da margem proporcional
        'quantidade': 'sum'  # Soma da quantidade vendida
    }).reset_index()
    
    df_agregado.columns = ['sku', 'valor_unitario_medio', 'custo_unitario_medio', 
                          'margem_proporcional_media', 'quantidade_vendida_total']
    
    # R(t) = Média (Valor de Venda Unitário - Custo de Aquisição Unitário)
    df_agregado['rentabilidade'] = (
        df_agregado['valor_unitario_medio'] - df_agregado['custo_unitario_medio']
    )
    
    print(f"[OK] Rentabilidade calculada para {len(df_agregado):,} SKUs")
    print(f"     Média geral: R$ {df_agregado['rentabilidade'].mean():.2f}")
    
    return df_agregado[['sku', 'quantidade_vendida_total', 'margem_proporcional_media', 
                        'rentabilidade', 'valor_unitario_medio', 'custo_unitario_medio']]


def calcular_venda_media_diaria(df_vendas, periodo_dias=365):
    """
    Calcula venda média diária histórica por SKU.
    Usado para calcular Nível de Urgência U(t).
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
    print(f"     Média geral: {venda_media['venda_media_diaria'].mean():.2f} unidades/dia")
    
    return venda_media


def calcular_nivel_urgencia(df_estoque_atual, df_venda_media_diaria):
    """
    Calcula Nível de Urgência U(t) = Estoque Atual / Venda Média Diária Histórica
    
    Conforme tabela 2.2:
    - Propósito: Fator de Estoque. Quanto tempo o estoque atual dura.
    - Dado de Origem: historico_estoque e venda_produtos
    - Fórmula: U(t) = Estoque Atual / Venda Média Diária Histórica
    """
    print("\nCalculando Nivel de Urgencia U(t)...")
    
    # Merge
    df_merge = df_estoque_atual.merge(
        df_venda_media_diaria,
        on='sku',
        how='inner'
    )
    
    # U(t) = Estoque Atual / Venda Média Diária
    df_merge['nivel_urgencia'] = np.where(
        df_merge['venda_media_diaria'] > 0,
        df_merge['saldo'] / df_merge['venda_media_diaria'],
        np.inf  # Se não há vendas, estoque dura "para sempre"
    )
    
    print(f"[OK] Nivel de urgencia calculado para {len(df_merge):,} SKUs")
    print(f"     Média geral: {df_merge[df_merge['nivel_urgencia'] < np.inf]['nivel_urgencia'].mean():.1f} dias")
    
    return df_merge[['sku', 'saldo', 'venda_media_diaria', 'nivel_urgencia']]


def carregar_estoque_atual(caminho="DB/historico_estoque_atual.csv"):
    """Carrega estoque atual (último saldo por SKU)"""
    print(f"\nCarregando estoque atual: {caminho}")
    
    if not Path(caminho).exists():
        print(f"[AVISO] Arquivo nao encontrado: {caminho}")
        return None
    
    df = pd.read_csv(caminho, low_memory=False)
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df['saldo'] = pd.to_numeric(df['saldo'], errors='coerce')
    df = df[df['sku'].notna()]
    
    # Pega último saldo por SKU (estoque atual)
    df_estoque_atual = df.sort_values('created_at').groupby('sku').last().reset_index()[['sku', 'saldo']]
    
    print(f"[OK] Estoque atual para {len(df_estoque_atual):,} SKUs")
    
    return df_estoque_atual


def calcular_metricas_completas(salvar_resultado=True):
    """
    Calcula todas as métricas necessárias para elencação.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame com todas as métricas por SKU:
        - sku
        - quantidade_vendida_total (soma)
        - margem_proporcional_media (média)
        - rentabilidade (R(t))
        - venda_media_diaria
        - nivel_urgencia (U(t))
        - saldo (estoque atual)
    """
    print("=" * 80)
    print("CALCULO DE METRICAS PARA ELENCAO")
    print("=" * 80)
    print("\nConforme Tabela 2.2 - Normalizacao das Metricas:")
    print("1. Giro Futuro Previsto (GP(t)) - Resultado do SARIMA")
    print("2. Rentabilidade (R(t)) - Média (Valor Unitário - Custo Unitário)")
    print("3. Nível de Urgência (U(t)) - Estoque Atual / Venda Média Diária")
    print("=" * 80)
    
    # 1. Carrega dados de vendas
    df_vendas = carregar_dados_vendas()
    
    # 2. Calcula Rentabilidade R(t)
    df_rentabilidade = calcular_rentabilidade(df_vendas)
    
    # 3. Calcula venda média diária
    df_venda_media = calcular_venda_media_diaria(df_vendas)
    
    # 4. Carrega estoque atual
    df_estoque_atual = carregar_estoque_atual()
    
    # 5. Merge de todas as métricas
    print("\n" + "=" * 80)
    print("CONSOLIDANDO METRICAS")
    print("=" * 80)
    
    df_final = df_rentabilidade.merge(
        df_venda_media,
        on='sku',
        how='left'
    )
    
    if df_estoque_atual is not None:
        df_urgencia = calcular_nivel_urgencia(df_estoque_atual, df_venda_media)
        df_final = df_final.merge(
            df_urgencia,
            on='sku',
            how='left'
        )
    
    print(f"\n[OK] Metricas consolidadas para {len(df_final):,} SKUs")
    
    # Resumo
    print("\n" + "=" * 80)
    print("RESUMO DAS METRICAS")
    print("=" * 80)
    print(f"\nTotal de SKUs: {len(df_final):,}")
    print(f"\nMetricas disponiveis:")
    print(f"  - quantidade_vendida_total: Soma da quantidade vendida")
    print(f"  - margem_proporcional_media: Média da margem proporcional")
    print(f"  - rentabilidade: R(t) = Média (Valor Unitário - Custo Unitário)")
    print(f"  - venda_media_diaria: Venda média diária histórica")
    if df_estoque_atual is not None:
        print(f"  - nivel_urgencia: U(t) = Estoque Atual / Venda Média Diária")
        print(f"  - saldo: Estoque atual")
    
    print(f"\nTop 10 SKUs por Rentabilidade:")
    df_top_rent = df_final.nlargest(10, 'rentabilidade')
    print(df_top_rent[['sku', 'rentabilidade', 'margem_proporcional_media', 
                       'quantidade_vendida_total']].to_string(index=False))
    
    if df_estoque_atual is not None and 'nivel_urgencia' in df_final.columns:
        print(f"\nTop 10 SKUs com menor Nível de Urgência (maior risco):")
        df_urgencia_filtrado = df_final[df_final['nivel_urgencia'] < np.inf].copy()
        if len(df_urgencia_filtrado) > 0:
            df_urgencia_ordenado = df_urgencia_filtrado.nsmallest(10, 'nivel_urgencia')
            colunas_mostrar = ['sku', 'nivel_urgencia', 'saldo']
            if 'venda_media_diaria' in df_urgencia_ordenado.columns:
                colunas_mostrar.append('venda_media_diaria')
            print(df_urgencia_ordenado[colunas_mostrar].to_string(index=False))
    
    # Salva resultado
    if salvar_resultado:
        from pathlib import Path
        Path("resultados").mkdir(exist_ok=True)
        caminho_saida = "resultados/metricas_elencacao.csv"
        df_final.to_csv(caminho_saida, index=False)
        print(f"\n[OK] Resultado salvo em: {caminho_saida}")
    
    print("\n" + "=" * 80)
    print("CALCULO CONCLUIDO!")
    print("=" * 80)
    print("\nNOTA: Para Giro Futuro Previsto (GP(t)), use o resultado do SARIMA")
    print("      Este script calcula R(t) e U(t) conforme tabela 2.2")
    print("=" * 80)
    
    return df_final


if __name__ == "__main__":
    df_resultado = calcular_metricas_completas(salvar_resultado=True)

