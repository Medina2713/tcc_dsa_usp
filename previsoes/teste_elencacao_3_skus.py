"""
Script de Teste: Previsão de Demanda e Elencação para 3 SKUs com Melhor Movimentação
TCC MBA Data Science & Analytics

Este script:
1. Identifica os 3 SKUs com maior quantidade vendida (melhor movimentação)
2. Gera previsões SARIMA para esses SKUs
3. Calcula métricas de elencação (R(t), U(t), GP(t))
4. Gera ranking de priorização
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sarima_estoque import PrevisorEstoqueSARIMA

def identificar_top_skus_movimentacao(top_n=3, caminho_vendas="DB/venda_produtos_atual.csv"):
    """
    Identifica os N SKUs com maior quantidade vendida (melhor movimentação).
    
    Returns:
    --------
    list
        Lista de SKUs ordenados por quantidade vendida (maior para menor)
    """
    print("=" * 80)
    print("IDENTIFICANDO TOP SKUs POR MOVIMENTACAO")
    print("=" * 80)
    
    print(f"\nCarregando dados de vendas: {caminho_vendas}")
    df_vendas = pd.read_csv(caminho_vendas, low_memory=False)
    df_vendas['quantidade'] = pd.to_numeric(df_vendas['quantidade'], errors='coerce')
    df_vendas = df_vendas[df_vendas['sku'].notna()]
    
    # Agrega por SKU
    vendas_por_sku = df_vendas.groupby('sku')['quantidade'].sum().reset_index()
    vendas_por_sku.columns = ['sku', 'quantidade_vendida_total']
    vendas_por_sku = vendas_por_sku.sort_values('quantidade_vendida_total', ascending=False)
    
    top_skus = vendas_por_sku.head(top_n)['sku'].tolist()
    
    print(f"\n[OK] Top {top_n} SKUs identificados:")
    for i, (_, row) in enumerate(vendas_por_sku.head(top_n).iterrows(), 1):
        print(f"  {i}. SKU {row['sku']}: {row['quantidade_vendida_total']:,.0f} unidades vendidas")
    
    return top_skus, vendas_por_sku.head(top_n)


def calcular_metricas_vendas(skus, caminho_vendas="DB/venda_produtos_atual.csv"):
    """
    Calcula métricas de vendas (R(t)) para os SKUs selecionados.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame com métricas de vendas por SKU
    """
    print("\n" + "=" * 80)
    print("CALCULANDO METRICAS DE VENDAS (RENTABILIDADE)")
    print("=" * 80)
    
    df_vendas = pd.read_csv(caminho_vendas, low_memory=False)
    df_vendas['quantidade'] = pd.to_numeric(df_vendas['quantidade'], errors='coerce')
    df_vendas['valor_unitario'] = pd.to_numeric(df_vendas['valor_unitario'], errors='coerce')
    df_vendas['custo_unitario'] = pd.to_numeric(df_vendas['custo_unitario'], errors='coerce')
    df_vendas['margem_proporcional'] = pd.to_numeric(df_vendas['margem_proporcional'], errors='coerce')
    df_vendas = df_vendas[df_vendas['sku'].notna()]
    
    # Filtra apenas os SKUs selecionados
    df_vendas = df_vendas[df_vendas['sku'].isin(skus)]
    
    # Agrega por SKU
    df_agregado = df_vendas.groupby('sku').agg({
        'valor_unitario': 'mean',
        'custo_unitario': 'mean',
        'margem_proporcional': 'mean',
        'quantidade': 'sum'
    }).reset_index()
    
    df_agregado.columns = ['sku', 'valor_unitario_medio', 'custo_unitario_medio', 
                          'margem_proporcional_media', 'quantidade_vendida_total']
    
    # R(t) = Média (Valor Unitário - Custo Unitário)
    df_agregado['rentabilidade'] = (
        df_agregado['valor_unitario_medio'] - df_agregado['custo_unitario_medio']
    )
    
    print(f"\n[OK] Métricas calculadas para {len(df_agregado)} SKUs:")
    for _, row in df_agregado.iterrows():
        print(f"  SKU {row['sku']}:")
        print(f"    - Quantidade vendida: {row['quantidade_vendida_total']:,.0f} unidades")
        print(f"    - Rentabilidade R(t): R$ {row['rentabilidade']:.2f}")
        print(f"    - Margem proporcional média: {row['margem_proporcional_media']:.2f}%")
    
    return df_agregado


def calcular_venda_media_diaria(skus, caminho_vendas="DB/venda_produtos_atual.csv", periodo_dias=365):
    """Calcula venda média diária histórica por SKU"""
    print("\nCalculando venda média diária histórica...")
    
    df_vendas = pd.read_csv(caminho_vendas, low_memory=False)
    df_vendas['created_at'] = pd.to_datetime(df_vendas['created_at'], errors='coerce')
    df_vendas['quantidade'] = pd.to_numeric(df_vendas['quantidade'], errors='coerce')
    df_vendas = df_vendas[df_vendas['sku'].notna()]
    df_vendas = df_vendas[df_vendas['sku'].isin(skus)]
    
    # Filtra período
    data_limite = df_vendas['created_at'].max() - pd.Timedelta(days=periodo_dias)
    df_periodo = df_vendas[df_vendas['created_at'] >= data_limite].copy()
    
    # Agrupa por SKU e data, soma quantidade
    df_vendas_diarias = df_periodo.groupby(['sku', pd.Grouper(key='created_at', freq='D')])['quantidade'].sum().reset_index()
    
    # Calcula média diária por SKU
    venda_media = df_vendas_diarias.groupby('sku')['quantidade'].mean().reset_index()
    venda_media.columns = ['sku', 'venda_media_diaria']
    
    print(f"[OK] Venda média diária calculada")
    return venda_media


def calcular_nivel_urgencia(skus, caminho_estoque="DB/historico_estoque_atual.csv", df_venda_media=None):
    """
    Calcula Nível de Urgência U(t) = Estoque Atual / Venda Média Diária
    """
    print("\nCalculando Nível de Urgência U(t)...")
    
    if not Path(caminho_estoque).exists():
        print(f"[AVISO] Arquivo de estoque não encontrado: {caminho_estoque}")
        return None
    
    df_estoque = pd.read_csv(caminho_estoque, low_memory=False)
    df_estoque['created_at'] = pd.to_datetime(df_estoque['created_at'], errors='coerce')
    df_estoque['saldo'] = pd.to_numeric(df_estoque['saldo'], errors='coerce')
    df_estoque = df_estoque[df_estoque['sku'].notna()]
    df_estoque = df_estoque[df_estoque['sku'].isin(skus)]
    
    # Pega último saldo por SKU (estoque atual)
    df_estoque_atual = df_estoque.sort_values('created_at').groupby('sku').last().reset_index()[['sku', 'saldo']]
    
    if df_venda_media is not None:
        df_merge = df_estoque_atual.merge(df_venda_media, on='sku', how='left')
        df_merge['nivel_urgencia'] = np.where(
            df_merge['venda_media_diaria'] > 0,
            df_merge['saldo'] / df_merge['venda_media_diaria'],
            np.inf
        )
        print(f"[OK] Nível de urgência calculado")
        return df_merge[['sku', 'saldo', 'venda_media_diaria', 'nivel_urgencia']]
    else:
        print(f"[OK] Estoque atual obtido (venda média não disponível)")
        return df_estoque_atual


def gerar_previsoes_sarima(skus, caminho_estoque="DB/historico_estoque_atual.csv", horizonte=30):
    """
    Gera previsões SARIMA para os SKUs selecionados.
    
    Returns:
    --------
    dict
        Dicionário com previsões por SKU
    """
    print("\n" + "=" * 80)
    print("GERANDO PREVISOES SARIMA")
    print("=" * 80)
    
    if not Path(caminho_estoque).exists():
        print(f"[ERRO] Arquivo de estoque não encontrado: {caminho_estoque}")
        return {}
    
    print(f"\nCarregando dados de estoque: {caminho_estoque}")
    df_estoque = pd.read_csv(caminho_estoque, low_memory=False)
    df_estoque['created_at'] = pd.to_datetime(df_estoque['created_at'], errors='coerce')
    df_estoque['saldo'] = pd.to_numeric(df_estoque['saldo'], errors='coerce')
    df_estoque = df_estoque[df_estoque['sku'].notna()]
    
    # Filtra apenas os SKUs selecionados
    df_estoque = df_estoque[df_estoque['sku'].isin(skus)]
    
    # Converte para formato esperado pelo SARIMA (data e estoque_atual)
    df_estoque['data'] = df_estoque['created_at']
    df_estoque['estoque_atual'] = df_estoque['saldo']
    
    print(f"[OK] {len(df_estoque):,} registros carregados")
    
    # Inicializa previsor
    previsor = PrevisorEstoqueSARIMA(horizonte_previsao=horizonte, frequencia='D')
    
    previsoes = {}
    
    for sku in skus:
        print(f"\n--- Processando SKU {sku} ---")
        
        try:
            # Prepara série temporal
            serie = previsor.preparar_serie_temporal(df_estoque, sku)
            
            if len(serie) < 30:
                print(f"  [AVISO] Dados insuficientes ({len(serie)} observações). Mínimo: 30")
                continue
            
            print(f"  [OK] Série temporal preparada: {len(serie)} observações")
            
            # Treina modelo
            modelo = previsor.treinar_modelo(serie, sku)
            if modelo is None:
                print(f"  [ERRO] Falha ao treinar modelo")
                continue
            
            print(f"  [OK] Modelo treinado: {modelo.order} x {modelo.seasonal_order}")
            
            # Gera previsão
            previsao = previsor.prever(serie, modelo=modelo, sku=sku)
            if previsao is None:
                print(f"  [ERRO] Falha ao gerar previsão")
                continue
            
            # Calcula GP(t) = Soma das previsões (Giro Futuro Previsto)
            giro_futuro_previsto = previsao.sum()
            estoque_medio_previsto = previsao.mean()
            
            previsoes[sku] = {
                'previsao': previsao,
                'modelo': modelo,
                'giro_futuro_previsto': giro_futuro_previsto,
                'estoque_medio_previsto': estoque_medio_previsto,
                'estoque_atual': serie.iloc[-1]
            }
            
            print(f"  [OK] Previsão gerada:")
            print(f"      - Estoque atual: {serie.iloc[-1]:.1f}")
            print(f"      - Estoque médio previsto: {estoque_medio_previsto:.1f}")
            print(f"      - GP(t) (soma previsões): {giro_futuro_previsto:.1f}")
            
        except Exception as e:
            print(f"  [ERRO] Erro ao processar SKU {sku}: {str(e)}")
            continue
    
    print(f"\n[OK] Previsões geradas para {len(previsoes)} SKUs")
    return previsoes


def calcular_score_elencacao(rentabilidade, nivel_urgencia, giro_futuro_previsto, 
                             peso_rentabilidade=0.4, peso_urgencia=0.3, peso_giro=0.3):
    """
    Calcula score de elencação combinando as três métricas.
    
    Nota: As métricas são normalizadas para [0, 1] antes do cálculo.
    """
    # Normalização simples (pode ser ajustada conforme necessidade)
    # Para este teste, vamos usar valores normalizados diretamente
    
    # Rentabilidade normalizada (assumindo range baseado nos dados)
    rent_norm = min(1.0, rentabilidade / 100.0) if rentabilidade > 0 else 0.0
    
    # Urgência normalizada (menor urgência = maior score, inverso)
    # Se urgência é 0 (risco alto), score alto; se alto, score baixo
    urgencia_norm = 1.0 / (1.0 + nivel_urgencia) if nivel_urgencia > 0 else 1.0
    urgencia_norm = min(1.0, urgencia_norm)
    
    # Giro futuro normalizado (assumindo range baseado nos dados)
    giro_norm = min(1.0, giro_futuro_previsto / 1000.0) if giro_futuro_previsto > 0 else 0.0
    
    # Score final (pesos somam 1.0)
    score = (
        peso_rentabilidade * rent_norm +
        peso_urgencia * urgencia_norm +
        peso_giro * giro_norm
    )
    
    return score, {'rentabilidade_norm': rent_norm, 'urgencia_norm': urgencia_norm, 'giro_norm': giro_norm}


def gerar_elencacao_completa():
    """
    Função principal que gera previsões e elencação completa.
    """
    print("\n" + "=" * 80)
    print("TESTE: PREVISAO DE DEMANDA E ELENCAO - 3 SKUs")
    print("=" * 80)
    
    # 1. Identifica top 3 SKUs por movimentação
    top_skus, df_top_skus = identificar_top_skus_movimentacao(top_n=3)
    
    if len(top_skus) == 0:
        print("\n[ERRO] Nenhum SKU encontrado!")
        return
    
    # 2. Calcula métricas de vendas (R(t))
    df_metricas_vendas = calcular_metricas_vendas(top_skus)
    
    # 3. Calcula venda média diária
    df_venda_media = calcular_venda_media_diaria(top_skus)
    
    # 4. Calcula nível de urgência (U(t))
    df_urgencia = calcular_nivel_urgencia(top_skus, df_venda_media=df_venda_media)
    
    # 5. Gera previsões SARIMA (GP(t))
    previsoes_sarima = gerar_previsoes_sarima(top_skus, horizonte=30)
    
    # 6. Consolida todas as métricas
    print("\n" + "=" * 80)
    print("CONSOLIDANDO METRICAS E GERANDO ELENCAO")
    print("=" * 80)
    
    resultados = []
    
    for sku in top_skus:
        # Métricas de vendas
        vendas_sku = df_metricas_vendas[df_metricas_vendas['sku'] == sku]
        if len(vendas_sku) == 0:
            continue
        
        rentabilidade = vendas_sku.iloc[0]['rentabilidade']
        quantidade_vendida = vendas_sku.iloc[0]['quantidade_vendida_total']
        margem_prop = vendas_sku.iloc[0]['margem_proporcional_media']
        
        # Nível de urgência
        urgencia_sku = df_urgencia[df_urgencia['sku'] == sku] if df_urgencia is not None else None
        nivel_urgencia = urgencia_sku.iloc[0]['nivel_urgencia'] if urgencia_sku is not None and len(urgencia_sku) > 0 else np.nan
        estoque_atual = urgencia_sku.iloc[0]['saldo'] if urgencia_sku is not None and len(urgencia_sku) > 0 else np.nan
        
        # Previsão SARIMA (GP(t))
        previsao_sku = previsoes_sarima.get(sku, None)
        giro_futuro_previsto = previsao_sku['giro_futuro_previsto'] if previsao_sku else np.nan
        estoque_medio_previsto = previsao_sku['estoque_medio_previsto'] if previsao_sku else np.nan
        
        # Calcula score de elencação
        if not np.isnan(nivel_urgencia) and not np.isnan(giro_futuro_previsto):
            score, scores_norm = calcular_score_elencacao(
                rentabilidade, 
                nivel_urgencia, 
                giro_futuro_previsto
            )
        else:
            score = np.nan
            scores_norm = {}
        
        resultados.append({
            'sku': sku,
            'quantidade_vendida_total': quantidade_vendida,
            'rentabilidade_Rt': rentabilidade,
            'margem_proporcional_media': margem_prop,
            'estoque_atual': estoque_atual,
            'nivel_urgencia_Ut': nivel_urgencia,
            'giro_futuro_previsto_GPt': giro_futuro_previsto,
            'estoque_medio_previsto': estoque_medio_previsto,
            'score_elencacao': score
        })
    
    df_resultado = pd.DataFrame(resultados)
    
    # Remove linhas com valores faltantes para score
    df_resultado_completo = df_resultado.dropna(subset=['score_elencacao'])
    
    if len(df_resultado_completo) > 0:
        # Ordena por score (maior = maior prioridade)
        df_resultado_completo = df_resultado_completo.sort_values('score_elencacao', ascending=False)
        df_resultado_completo['ranking'] = range(1, len(df_resultado_completo) + 1)
        
        # 7. Exibe resultados
        print("\n" + "=" * 80)
        print("RANKING DE ELENCAO (Ordenado por Score)")
        print("=" * 80)
        print(f"\n{'Rank':<6} {'SKU':<20} {'R(t)':<12} {'U(t)':<12} {'GP(t)':<15} {'Score':<10}")
        print("-" * 80)
        
        for _, row in df_resultado_completo.iterrows():
            print(f"{int(row['ranking']):<6} "
                  f"{str(row['sku']):<20} "
                  f"R$ {row['rentabilidade_Rt']:<10.2f} "
                  f"{row['nivel_urgencia_Ut']:<12.1f} "
                  f"{row['giro_futuro_previsto_GPt']:<15.1f} "
                  f"{row['score_elencacao']:<10.3f}")
        
        print("-" * 80)
        
        print("\n" + "=" * 80)
        print("DETALHES POR SKU")
        print("=" * 80)
        
        for _, row in df_resultado_completo.iterrows():
            print(f"\nSKU {row['sku']} (Rank {int(row['ranking'])})")
            print(f"  Quantidade vendida total: {row['quantidade_vendida_total']:,.0f} unidades")
            print(f"  Rentabilidade R(t): R$ {row['rentabilidade_Rt']:.2f}")
            print(f"  Margem proporcional média: {row['margem_proporcional_media']:.2f}%")
            print(f"  Estoque atual: {row['estoque_atual']:.1f} unidades")
            print(f"  Nível de urgência U(t): {row['nivel_urgencia_Ut']:.1f} dias")
            print(f"  Giro futuro previsto GP(t): {row['giro_futuro_previsto_GPt']:.1f}")
            print(f"  Estoque médio previsto: {row['estoque_medio_previsto']:.1f} unidades")
            print(f"  Score de elencação: {row['score_elencacao']:.3f}")
        
        # Salva resultado
        from pathlib import Path
        Path("resultados").mkdir(exist_ok=True)
        caminho_saida = "resultados/resultado_elencacao_3_skus.csv"
        df_resultado_completo.to_csv(caminho_saida, index=False)
        print(f"\n[OK] Resultado salvo em: {caminho_saida}")
    
    else:
        print("\n[AVISO] Nenhum SKU com todas as métricas disponíveis para gerar elencação completa")
    
    print("\n" + "=" * 80)
    print("TESTE CONCLUIDO!")
    print("=" * 80)
    
    return df_resultado


if __name__ == "__main__":
    resultado = gerar_elencacao_completa()

