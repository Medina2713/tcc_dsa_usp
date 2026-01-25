"""
Script de Teste: Previsão de Demanda e Elencação para 3 SKUs com Melhor Movimentação
TCC MBA Data Science & Analytics

VERSÃO OTIMIZADA:
- Carregamento único de dados
- Cache de modelos treinados
- Processamento paralelo
- Logs de progresso com porcentagem
- Sistema de checkpoint para retomar processamento

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
import json
import sys
import time

# Importação local
sys.path.insert(0, str(Path(__file__).parent))
from sarima_estoque import PrevisorEstoqueSARIMA

# Configurações
DIR_CHECKPOINT = Path('cache_checkpoints')
DIR_CHECKPOINT.mkdir(exist_ok=True)
ARQUIVO_CHECKPOINT = DIR_CHECKPOINT / 'checkpoint_elencacao.json'


def carregar_dados(caminho_vendas="DB/venda_produtos_atual.csv", 
                   caminho_estoque="DB/historico_estoque_atual.csv"):
    """
    Carrega dados uma única vez e prepara para uso.
    
    Returns:
    --------
    tuple
        (df_vendas, df_estoque) - DataFrames preparados
    """
    print("=" * 80)
    print("CARREGANDO DADOS")
    print("=" * 80)
    
    print(f"\n[1/2] Carregando vendas: {caminho_vendas}")
    df_vendas = pd.read_csv(caminho_vendas, low_memory=False)
    df_vendas['created_at'] = pd.to_datetime(df_vendas['created_at'], errors='coerce')
    df_vendas['quantidade'] = pd.to_numeric(df_vendas['quantidade'], errors='coerce')
    df_vendas['valor_unitario'] = pd.to_numeric(df_vendas['valor_unitario'], errors='coerce')
    df_vendas['custo_unitario'] = pd.to_numeric(df_vendas['custo_unitario'], errors='coerce')
    df_vendas['margem_proporcional'] = pd.to_numeric(df_vendas['margem_proporcional'], errors='coerce')
    df_vendas = df_vendas[df_vendas['sku'].notna()]
    print(f"      [OK] {len(df_vendas):,} registros carregados")
    
    print(f"\n[2/2] Carregando estoque: {caminho_estoque}")
    if not Path(caminho_estoque).exists():
        print(f"      [ERRO] Arquivo não encontrado: {caminho_estoque}")
        return None, None
    
    df_estoque = pd.read_csv(caminho_estoque, low_memory=False)
    df_estoque['created_at'] = pd.to_datetime(df_estoque['created_at'], errors='coerce')
    df_estoque['saldo'] = pd.to_numeric(df_estoque['saldo'], errors='coerce')
    df_estoque = df_estoque[df_estoque['sku'].notna()]
    
    # Converte para formato esperado pelo SARIMA
    df_estoque['data'] = df_estoque['created_at']
    df_estoque['estoque_atual'] = df_estoque['saldo']
    
    print(f"      [OK] {len(df_estoque):,} registros carregados")
    print("\n[OK] Dados carregados com sucesso!")
    
    return df_vendas, df_estoque


def identificar_top_skus_movimentacao(df_vendas, top_n=3):
    """
    Identifica os N SKUs com maior quantidade vendida.
    
    Parameters:
    -----------
    df_vendas : pd.DataFrame
        DataFrame com dados de vendas (já carregado)
    top_n : int
        Número de SKUs a retornar
        
    Returns:
    --------
    list
        Lista de SKUs ordenados por quantidade vendida
    """
    print("\n" + "=" * 80)
    print("IDENTIFICANDO TOP SKUs POR MOVIMENTACAO")
    print("=" * 80)
    
    # Agrega por SKU
    vendas_por_sku = df_vendas.groupby('sku')['quantidade'].sum().reset_index()
    vendas_por_sku.columns = ['sku', 'quantidade_vendida_total']
    vendas_por_sku = vendas_por_sku.sort_values('quantidade_vendida_total', ascending=False)
    
    top_skus = vendas_por_sku.head(top_n)['sku'].tolist()
    
    print(f"\n[OK] Top {top_n} SKUs identificados:")
    for i, (_, row) in enumerate(vendas_por_sku.head(top_n).iterrows(), 1):
        print(f"  {i}. SKU {row['sku']}: {row['quantidade_vendida_total']:,.0f} unidades vendidas")
    
    return top_skus, vendas_por_sku.head(top_n)


def calcular_metricas_vendas(df_vendas, skus):
    """
    Calcula métricas de vendas (R(t)) para os SKUs selecionados.
    
    Parameters:
    -----------
    df_vendas : pd.DataFrame
        DataFrame com dados de vendas (já carregado)
    skus : list
        Lista de SKUs
        
    Returns:
    --------
    pd.DataFrame
        DataFrame com métricas de vendas por SKU
    """
    print("\n" + "=" * 80)
    print("CALCULANDO METRICAS DE VENDAS (RENTABILIDADE)")
    print("=" * 80)
    
    # Filtra apenas os SKUs selecionados
    df_vendas_filtrado = df_vendas[df_vendas['sku'].isin(skus)].copy()
    
    # Agrega por SKU
    df_agregado = df_vendas_filtrado.groupby('sku').agg({
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


def calcular_venda_media_diaria(df_vendas, skus, periodo_dias=365):
    """Calcula venda média diária histórica por SKU"""
    print("\nCalculando venda média diária histórica...")
    
    # Filtra SKUs
    df_vendas_filtrado = df_vendas[df_vendas['sku'].isin(skus)].copy()
    
    # Filtra período
    data_limite = df_vendas_filtrado['created_at'].max() - pd.Timedelta(days=periodo_dias)
    df_periodo = df_vendas_filtrado[df_vendas_filtrado['created_at'] >= data_limite].copy()
    
    # Agrupa por SKU e data, soma quantidade
    df_vendas_diarias = df_periodo.groupby(['sku', pd.Grouper(key='created_at', freq='D')])['quantidade'].sum().reset_index()
    
    # Calcula média diária por SKU
    venda_media = df_vendas_diarias.groupby('sku')['quantidade'].mean().reset_index()
    venda_media.columns = ['sku', 'venda_media_diaria']
    
    print(f"[OK] Venda média diária calculada")
    return venda_media


def calcular_nivel_urgencia(df_estoque, skus, df_venda_media):
    """
    Calcula Nível de Urgência U(t) = Estoque Atual / Venda Média Diária
    """
    print("\nCalculando Nível de Urgência U(t)...")
    
    # Filtra SKUs
    df_estoque_filtrado = df_estoque[df_estoque['sku'].isin(skus)].copy()
    
    # Pega último saldo por SKU (estoque atual)
    df_estoque_atual = df_estoque_filtrado.sort_values('created_at').groupby('sku').last().reset_index()[['sku', 'saldo']]
    
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


def gerar_previsoes_sarima_sequencial(df_estoque, skus, horizonte=30, callback_progresso=None):
    """
    Gera previsões SARIMA para múltiplos SKUs sequencialmente (com logs de progresso).
    Nota: auto_arima já usa n_jobs=-1 internamente, então paralelização adicional
    pode não trazer ganho significativo.
    
    Parameters:
    -----------
    df_estoque : pd.DataFrame
        DataFrame com dados de estoque
    skus : list
        Lista de SKUs para processar
    horizonte : int
        Horizonte de previsão
    callback_progresso : callable
        Função callback(progresso, total, sku_atual, tempo_estimado) para log de progresso
        
    Returns:
    --------
    dict
        Dicionário com previsões por SKU
    """
    print("\n" + "=" * 80)
    print("GERANDO PREVISOES SARIMA")
    print("=" * 80)
    
    print(f"\n[INFO] Processando {len(skus)} SKUs sequencialmente")
    print(f"       (auto_arima usa todos os cores disponíveis internamente)")
    
    # Inicializa previsor (uma vez)
    previsor = PrevisorEstoqueSARIMA(horizonte_previsao=horizonte, frequencia='D')
    
    previsoes = {}
    inicio = time.time()
    tempos_skus = []
    
    for i, sku in enumerate(skus, 1):
        inicio_sku = time.time()
        print(f"\n--- Processando SKU {sku} ({i}/{len(skus)}) ---")
        
        try:
            # Prepara série temporal (com cache)
            serie = previsor.preparar_serie_temporal(df_estoque, sku, usar_cache=True)
            
            if len(serie) < 30:
                print(f"  [AVISO] Dados insuficientes ({len(serie)} observações). Mínimo: 30")
                if callback_progresso:
                    callback_progresso(i, len(skus), sku, None)
                continue
            
            print(f"  [OK] Série temporal preparada: {len(serie)} observações")
            
            # Treina modelo (com cache)
            modelo = previsor.treinar_modelo(serie, sku, usar_cache=True)
            if modelo is None:
                print(f"  [ERRO] Falha ao treinar modelo")
                if callback_progresso:
                    callback_progresso(i, len(skus), sku, None)
                continue
            
            print(f"  [OK] Modelo treinado: {modelo.order} x {modelo.seasonal_order}")
            
            # Gera previsão
            previsao = previsor.prever(serie, modelo=modelo, sku=sku)
            if previsao is None:
                print(f"  [ERRO] Falha ao gerar previsão")
                if callback_progresso:
                    callback_progresso(i, len(skus), sku, None)
                continue
            
            # Calcula GP(t) = Soma das previsões
            giro_futuro_previsto = previsao.sum()
            estoque_medio_previsto = previsao.mean()
            
            previsoes[sku] = {
                'giro_futuro_previsto': giro_futuro_previsto,
                'estoque_medio_previsto': estoque_medio_previsto,
                'estoque_atual': float(serie.iloc[-1]),
                'modelo_order': modelo.order,
                'modelo_seasonal_order': modelo.seasonal_order
            }
            
            tempo_sku = time.time() - inicio_sku
            tempos_skus.append(tempo_sku)
            tempo_medio = np.mean(tempos_skus) if tempos_skus else 0
            tempo_restante = tempo_medio * (len(skus) - i) if tempo_medio > 0 else None
            
            print(f"  [OK] Previsão gerada:")
            print(f"      - Estoque atual: {serie.iloc[-1]:.1f}")
            print(f"      - Estoque médio previsto: {estoque_medio_previsto:.1f}")
            print(f"      - GP(t) (soma previsões): {giro_futuro_previsto:.1f}")
            print(f"      - Tempo: {tempo_sku:.1f}s")
            
            # Callback de progresso
            if callback_progresso:
                callback_progresso(i, len(skus), sku, tempo_restante)
            
        except Exception as e:
            print(f"  [ERRO] Erro ao processar SKU {sku}: {str(e)}")
            if callback_progresso:
                callback_progresso(i, len(skus), sku, None)
            continue
    
    tempo_total = time.time() - inicio
    print(f"\n[OK] Previsões geradas para {len(previsoes)}/{len(skus)} SKUs em {tempo_total:.1f}s")
    if tempos_skus:
        print(f"     Tempo médio por SKU: {np.mean(tempos_skus):.1f}s")
    
    return previsoes


def carregar_checkpoint():
    """Carrega checkpoint de processamento"""
    if ARQUIVO_CHECKPOINT.exists():
        try:
            with open(ARQUIVO_CHECKPOINT, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}


def salvar_checkpoint(dados):
    """Salva checkpoint de processamento"""
    dados['ultima_atualizacao'] = datetime.now().isoformat()
    with open(ARQUIVO_CHECKPOINT, 'w') as f:
        json.dump(dados, f, indent=2)


def calcular_score_elencacao(rentabilidade, nivel_urgencia, giro_futuro_previsto, 
                             peso_rentabilidade=0.4, peso_urgencia=0.3, peso_giro=0.3):
    """
    Calcula score de elencação combinando as três métricas.
    """
    # Normalização simples
    rent_norm = min(1.0, rentabilidade / 100.0) if rentabilidade > 0 else 0.0
    
    urgencia_norm = 1.0 / (1.0 + nivel_urgencia) if nivel_urgencia > 0 else 1.0
    urgencia_norm = min(1.0, urgencia_norm)
    
    giro_norm = min(1.0, giro_futuro_previsto / 1000.0) if giro_futuro_previsto > 0 else 0.0
    
    score = (
        peso_rentabilidade * rent_norm +
        peso_urgencia * urgencia_norm +
        peso_giro * giro_norm
    )
    
    return score, {'rentabilidade_norm': rent_norm, 'urgencia_norm': urgencia_norm, 'giro_norm': giro_norm}


def gerar_elencacao_completa(top_n=3, usar_checkpoint=True):
    """
    Função principal que gera previsões e elencação completa.
    
    Parameters:
    -----------
    top_n : int
        Número de SKUs para processar
    usar_checkpoint : bool
        Se True, tenta retomar de checkpoint
    n_workers : int
        Número de workers paralelos (None = auto)
    """
    print("\n" + "=" * 80)
    print("TESTE: PREVISAO DE DEMANDA E ELENCAO - VERSÃO OTIMIZADA")
    print("=" * 80)
    
    # Carrega checkpoint
    checkpoint = {}
    if usar_checkpoint:
        checkpoint = carregar_checkpoint()
        if checkpoint:
            print(f"\n[CHECKPOINT] Encontrado checkpoint de {checkpoint.get('ultima_atualizacao', 'data desconhecida')}")
    
    # 1. Carrega dados UMA VEZ
    df_vendas, df_estoque = carregar_dados()
    if df_vendas is None or df_estoque is None:
        print("\n[ERRO] Falha ao carregar dados!")
        return None
    
    # 2. Identifica top SKUs
    top_skus, df_top_skus = identificar_top_skus_movimentacao(df_vendas, top_n=top_n)
    
    if len(top_skus) == 0:
        print("\n[ERRO] Nenhum SKU encontrado!")
        return None
    
    # 3. Calcula métricas de vendas (R(t))
    df_metricas_vendas = calcular_metricas_vendas(df_vendas, top_skus)
    
    # 4. Calcula venda média diária
    df_venda_media = calcular_venda_media_diaria(df_vendas, top_skus)
    
    # 5. Calcula nível de urgência (U(t))
    df_urgencia = calcular_nivel_urgencia(df_estoque, top_skus, df_venda_media)
    
    # 6. Callback de progresso
    def callback_progresso(atual, total, sku_atual, tempo_restante):
        porcentagem = (atual / total) * 100
        if tempo_restante is not None:
            minutos_restantes = int(tempo_restante // 60)
            segundos_restantes = int(tempo_restante % 60)
            print(f"\n[PROGRESSO] {atual}/{total} SKUs processados ({porcentagem:.1f}%) - "
                  f"SKU atual: {sku_atual} - "
                  f"Tempo restante estimado: {minutos_restantes}m {segundos_restantes}s")
        else:
            print(f"\n[PROGRESSO] {atual}/{total} SKUs processados ({porcentagem:.1f}%) - SKU atual: {sku_atual}")
    
    # 7. Gera previsões SARIMA (GP(t)) - SEQUENCIAL COM LOGS
    previsoes_sarima = gerar_previsoes_sarima_sequencial(
        df_estoque, 
        top_skus, 
        horizonte=30,
        callback_progresso=callback_progresso
    )
    
    # Salva checkpoint
    checkpoint['previsoes_completas'] = len(previsoes_sarima) == len(top_skus)
    checkpoint['skus_processados'] = list(previsoes_sarima.keys())
    salvar_checkpoint(checkpoint)
    
    # 8. Consolida todas as métricas
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
        if previsao_sku:
            giro_futuro_previsto = previsao_sku['giro_futuro_previsto']
            estoque_medio_previsto = previsao_sku['estoque_medio_previsto']
        else:
            giro_futuro_previsto = np.nan
            estoque_medio_previsto = np.nan
        
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
        
        # 9. Exibe resultados
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
    resultado = gerar_elencacao_completa(top_n=3, usar_checkpoint=True)
