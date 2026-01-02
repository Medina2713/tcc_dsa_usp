"""
Teste de Tempo de Processamento para Previsão SARIMA
Estima tempo necessário para processar múltiplos SKUs

Autor: Medina2713
Data: 2024
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from sarima_estoque import PrevisorEstoqueSARIMA
import warnings
warnings.filterwarnings('ignore')


def estimar_tempo_processamento(n_skus=10, n_observacoes_por_sku=200, 
                                periodo_sazonal=30, horizonte_previsao=7):
    """
    Estima tempo de processamento para N SKUs.
    
    Parameters:
    -----------
    n_skus : int
        Número de SKUs a processar
    n_observacoes_por_sku : int
        Número médio de observações por SKU
    periodo_sazonal : int
        Período sazonal (30 para mensal)
    horizonte_previsao : int
        Período de previsão (7 dias)
    """
    print("=" * 80)
    print("ESTIMATIVA DE TEMPO DE PROCESSAMENTO - SARIMA")
    print("=" * 80)
    
    # Informações do hardware (já coletadas)
    print("\nESPECIFICACOES DO HARDWARE:")
    print("  CPU: Intel Core i5-9400F @ 2.90GHz")
    print("  Cores: 6 fisicos, 6 threads")
    print("  RAM: 16 GB total, ~9.5 GB disponivel")
    
    # Teste com 1 SKU para calibrar
    print("\n" + "=" * 80)
    print("TESTE DE CALIBRAÇÃO: Processando 1 SKU...")
    print("=" * 80)
    
    # Gera dados de teste
    np.random.seed(42)
    datas = pd.date_range(start='2023-01-01', periods=n_observacoes_por_sku, freq='D')
    
    # Simula série com tendência e sazonalidade
    tendencia = np.linspace(100, 150, n_observacoes_por_sku)
    sazonalidade = 20 * np.sin(2 * np.pi * np.arange(n_observacoes_por_sku) / periodo_sazonal)
    ruido = np.random.normal(0, 5, n_observacoes_por_sku)
    estoque = tendencia + sazonalidade + ruido
    estoque = np.maximum(estoque, 0)  # Não-negativo
    
    df_teste = pd.DataFrame({
        'data': datas,
        'sku': 'SKU_TESTE',
        'estoque_atual': estoque
    })
    
    # Mede tempo de processamento
    previsor = PrevisorEstoqueSARIMA(
        horizonte_previsao=horizonte_previsao,
        frequencia='D'
    )
    
    inicio = time.time()
    
    # Prepara série
    serie = previsor.preparar_serie_temporal(df_teste, sku='SKU_TESTE')
    tempo_preparacao = time.time() - inicio
    
    # Treina modelo (parte mais lenta)
    inicio = time.time()
    modelo = previsor.treinar_modelo(serie, sku='SKU_TESTE')
    tempo_treinamento = time.time() - inicio
    
    # Gera previsão
    inicio = time.time()
    if modelo:
        previsao = previsor.prever(serie, modelo=modelo)
    tempo_previsao = time.time() - inicio
    
    tempo_total_1_sku = tempo_preparacao + tempo_treinamento + tempo_previsao
    
    print(f"\nTEMPOS MEDIDOS (1 SKU com {n_observacoes_por_sku} observacoes):")
    print(f"  Preparação de dados: {tempo_preparacao:.2f} segundos")
    print(f"  Treinamento do modelo: {tempo_treinamento:.2f} segundos (PARTE MAIS LENTA)")
    print(f"  Geração de previsão: {tempo_previsao:.2f} segundos")
    print(f"  TOTAL: {tempo_total_1_sku:.2f} segundos ({tempo_total_1_sku/60:.2f} minutos)")
    
    # Estimativa para N SKUs
    print("\n" + "=" * 80)
    print(f"ESTIMATIVA PARA {n_skus} SKUs")
    print("=" * 80)
    
    # Tempo total estimado
    tempo_estimado_segundos = tempo_total_1_sku * n_skus
    tempo_estimado_minutos = tempo_estimado_segundos / 60
    
    print(f"\nTEMPO ESTIMADO:")
    print(f"  Por SKU: {tempo_total_1_sku:.2f} segundos")
    print(f"  Total ({n_skus} SKUs): {tempo_estimado_segundos:.2f} segundos")
    print(f"  Total ({n_skus} SKUs): {tempo_estimado_minutos:.2f} minutos")
    print(f"  Total ({n_skus} SKUs): {tempo_estimado_minutos/60:.2f} horas")
    
    # Análise de componentes
    print(f"\nBREAKDOWN DO TEMPO:")
    print(f"  Preparação: {tempo_preparacao * n_skus:.2f}s ({tempo_preparacao * n_skus / tempo_estimado_segundos * 100:.1f}%)")
    print(f"  Treinamento: {tempo_treinamento * n_skus:.2f}s ({tempo_treinamento * n_skus / tempo_estimado_segundos * 100:.1f}%)")
    print(f"  Previsão: {tempo_previsao * n_skus:.2f}s ({tempo_previsao * n_skus / tempo_estimado_segundos * 100:.1f}%)")
    
    # Fatores que afetam o tempo
    print("\n" + "=" * 80)
    print("FATORES QUE AFETAM O TEMPO")
    print("=" * 80)
    print("""
    FATORES QUE ACELERAM:
    - Série menor (< 200 observações): Mais rápido
    - Modelo mais simples (parâmetros menores): Mais rápido
    - Sem sazonalidade (m=None): Mais rápido
    - n_jobs=-1 (usa todos os 6 cores): Mais rápido
    
    FATORES QUE DESACELERAM:
    - Série maior (> 500 observações): Mais lento
    - Modelo mais complexo (parâmetros maiores): Mais lento
    - Sazonalidade alta (m=365): Muito mais lento
    - Muitos SKUs processados sequencialmente: Linear
    
    NOTA: O auto_arima usa busca stepwise, que e eficiente.
    Mas ainda precisa testar múltiplas combinações de parâmetros.
    """)
    
    # Recomendações
    print("\n" + "=" * 80)
    print("RECOMENDAÇÕES")
    print("=" * 80)
    
    if tempo_estimado_minutos < 5:
        print("OK: Tempo estimado e RAZOAVEL (< 5 minutos)")
        print("   Pode processar sem otimizacoes adicionais")
    elif tempo_estimado_minutos < 30:
        print("ATENCAO: Tempo estimado e MODERADO (5-30 minutos)")
        print("   Considere processar em lotes ou otimizar parametros")
    else:
        print("ALTO: Tempo estimado e ALTO (> 30 minutos)")
        print("   Recomendações:")
        print("   1. Reduzir max_p, max_q, max_P, max_Q")
        print("   2. Processar em lotes menores")
        print("   3. Usar n_jobs=6 explicitamente")
        print("   4. Considerar processamento paralelo")
    
    # Estimativa conservadora vs otimista
    print("\n" + "=" * 80)
    print("CENÁRIOS")
    print("=" * 80)
    
    # Cenário otimista (séries menores, modelos simples)
    tempo_otimista = tempo_total_1_sku * 0.5 * n_skus
    print(f"\nCENARIO OTIMISTA (series menores, modelos simples):")
    print(f"   {tempo_otimista/60:.2f} minutos")
    
    # Cenário realista (baseado no teste)
    print(f"\nCENARIO REALISTA (baseado no teste):")
    print(f"   {tempo_estimado_minutos:.2f} minutos")
    
    # Cenário pessimista (séries maiores, modelos complexos)
    tempo_pessimista = tempo_total_1_sku * 2.0 * n_skus
    print(f"\nCENARIO PESSIMISTA (series maiores, modelos complexos):")
    print(f"   {tempo_pessimista/60:.2f} minutos")
    
    return {
        'tempo_por_sku': tempo_total_1_sku,
        'tempo_total_estimado': tempo_estimado_segundos,
        'tempo_total_minutos': tempo_estimado_minutos,
        'tempo_treinamento': tempo_treinamento,
        'n_skus': n_skus,
        'n_observacoes': n_observacoes_por_sku
    }


def processar_10_skus_reais():
    """
    Processa 10 SKUs reais do banco de dados.
    """
    print("\n" + "=" * 80)
    print("PROCESSAMENTO REAL: 10 SKUs")
    print("=" * 80)
    
    # Carrega dados
    print("\nCarregando dados...")
    try:
        df = pd.read_csv('DB/historico_estoque_atual_processado.csv')
        df['data'] = pd.to_datetime(df['data'])
        print(f"✓ {len(df):,} registros carregados")
        print(f"✓ {df['sku'].nunique()} SKUs disponíveis")
    except FileNotFoundError:
        print("❌ Arquivo não encontrado. Execute primeiro o processamento dos dados.")
        return
    
    # Seleciona top 10 SKUs por número de observações
    stats = df.groupby('sku')['estoque_atual'].agg(['count', 'mean']).reset_index()
    stats = stats[stats['mean'] >= 1.0].copy()
    stats = stats.sort_values('count', ascending=False)
    
    top_10_skus = stats.head(10)['sku'].tolist()
    
    print(f"\nSKUs selecionados (top 10 por observacoes):")
    for i, sku in enumerate(top_10_skus, 1):
        n_obs = stats[stats['sku'] == sku]['count'].iloc[0]
        print(f"  {i}. {sku}: {int(n_obs)} observações")
    
    # Processa
    print("\n" + "=" * 80)
    print("INICIANDO PROCESSAMENTO...")
    print("=" * 80)
    
    previsor = PrevisorEstoqueSARIMA(horizonte_previsao=7, frequencia='D')
    
    inicio_total = time.time()
    resultados = []
    tempos_por_sku = []
    
    for i, sku in enumerate(top_10_skus, 1):
        print(f"\n[{i}/10] Processando {sku}...")
        inicio_sku = time.time()
        
        try:
            # Prepara série
            serie = previsor.preparar_serie_temporal(df, sku=sku)
            
            if len(serie) < 30:
                print(f"  AVISO: Dados insuficientes ({len(serie)} obs). Pulando...")
                continue
            
            # Treina modelo
            print(f"  Treinando modelo... (pode levar alguns segundos)")
            modelo = previsor.treinar_modelo(serie, sku=sku)
            
            if modelo is None:
                print(f"  ERRO: Erro ao treinar modelo. Pulando...")
                continue
            
            # Gera previsão
            previsao = previsor.prever(serie, modelo=modelo)
            
            if previsao is not None:
                resultado = pd.DataFrame({
                    'sku': sku,
                    'data': previsao.index,
                    'estoque_previsto': previsao.values,
                    'estoque_atual': serie.iloc[-1]
                })
                resultados.append(resultado)
                
                tempo_sku = time.time() - inicio_sku
                tempos_por_sku.append(tempo_sku)
                print(f"  OK: Concluido em {tempo_sku:.2f} segundos")
            else:
                print(f"  ERRO: Erro ao gerar previsao. Pulando...")
                
        except Exception as e:
            print(f"  ERRO: {str(e)}")
            continue
    
    tempo_total = time.time() - inicio_total
    
    # Resultados
    print("\n" + "=" * 80)
    print("RESULTADOS")
    print("=" * 80)
    
    if resultados:
        df_resultado = pd.concat(resultados, ignore_index=True)
        
        print(f"\nOK: Processamento concluido!")
        print(f"   SKUs processados com sucesso: {len(resultados)}/10")
        print(f"   Tempo total: {tempo_total:.2f} segundos ({tempo_total/60:.2f} minutos)")
        
        if tempos_por_sku:
            print(f"   Tempo medio por SKU: {np.mean(tempos_por_sku):.2f} segundos")
            print(f"   Tempo minimo: {np.min(tempos_por_sku):.2f} segundos")
            print(f"   Tempo maximo: {np.max(tempos_por_sku):.2f} segundos")
        
        # Salva resultados
        nome_arquivo = f'previsoes_10_skus_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df_resultado.to_csv(nome_arquivo, index=False)
        print(f"\nResultados salvos em: {nome_arquivo}")
        
        # Estatísticas
        print(f"\nEstatisticas das Previsoes:")
        print(f"   Total de previsoes: {len(df_resultado)}")
        print(f"   Periodo previsto: {df_resultado['data'].min()} a {df_resultado['data'].max()}")
        print(f"   Estoque medio previsto: {df_resultado['estoque_previsto'].mean():.2f} unidades")
    else:
        print("\nERRO: Nenhum SKU foi processado com sucesso.")
    
    return tempo_total


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TESTE DE TEMPO DE PROCESSAMENTO - SARIMA")
    print("=" * 80)
    
    # Opção 1: Estimativa baseada em teste
    print("\n[OPÇÃO 1] Estimativa baseada em teste de calibração...")
    resultado = estimar_tempo_processamento(n_skus=10, n_observacoes_por_sku=200)
    
    # Opção 2: Processamento real (descomente para executar)
    print("\n" + "=" * 80)
    print("[OPÇÃO 2] Para processar 10 SKUs reais, descomente a linha abaixo:")
    print("=" * 80)
    print("# processar_10_skus_reais()")
    
    # Descomente a linha abaixo para processar SKUs reais
    # processar_10_skus_reais()

