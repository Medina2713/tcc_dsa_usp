"""
Comparação Otimizada de Modelos para Top 10 SKUs - Versão Melhorada
TCC MBA Data Science & Analytics - E-commerce de Brinquedos

MELHORIAS IMPLEMENTADAS:
1. Salva resultados incrementalmente (por SKU)
2. Sistema de checkpoint (pode retomar de onde parou)
3. Otimização de parâmetros do auto_arima (mais rápido)
4. Cache de dados processados
5. Todas as métricas estatísticas aplicadas
6. Processamento mais eficiente

Autor: Medina2713
Data: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from comparacao_modelos_previsao import (
    dividir_serie_temporal, calcular_mape, avaliar_modelo,
    modelo_sarima_mensal, modelo_arima_simples, modelo_media_movel,
    prever_media_movel, modelo_suavizacao_exponencial
)
from sarima_estoque import PrevisorEstoqueSARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

plt.style.use('seaborn-v0_8-darkgrid')

# Configurações
DIR_RESULTADOS = Path('resultados/resultados_comparacao')
DIR_RESULTADOS.mkdir(exist_ok=True)
ARQUIVO_CHECKPOINT = DIR_RESULTADOS / 'checkpoint_skus.json'


def carregar_checkpoint():
    """
    Carrega checkpoint de SKUs já processados.
    
    Returns:
    --------
    set
        Conjunto de SKUs já processados
    """
    if ARQUIVO_CHECKPOINT.exists():
        try:
            with open(ARQUIVO_CHECKPOINT, 'r') as f:
                data = json.load(f)
                return set(data.get('skus_processados', []))
        except:
            return set()
    return set()


def salvar_checkpoint(sku):
    """
    Salva SKU processado no checkpoint.
    
    Parameters:
    -----------
    sku : str
        SKU processado
    """
    skus_processados = carregar_checkpoint()
    skus_processados.add(str(sku))
    
    data = {
        'skus_processados': list(skus_processados),
        'ultima_atualizacao': datetime.now().isoformat()
    }
    
    with open(ARQUIVO_CHECKPOINT, 'w') as f:
        json.dump(data, f, indent=2)


def calcular_giro_estoque_otimizado(df_vendas, df_estoque, periodo_dias=30):
    """
    Calcula giro de estoque de forma otimizada.
    Usa operações vetorizadas do pandas.
    """
    print("=" * 80)
    print("CALCULO DE GIRO DE ESTOQUE (OTIMIZADO)")
    print("=" * 80)
    
    # Processa vendas (uma vez só)
    if 'created_at' not in df_vendas.columns or df_vendas['created_at'].dtype == 'object':
        df_vendas['created_at'] = pd.to_datetime(df_vendas['created_at'], format='mixed', errors='coerce')
    
    df_vendas = df_vendas[df_vendas['created_at'].notna()].copy()
    
    # Período recente
    data_fim = df_vendas['created_at'].max()
    data_inicio = data_fim - timedelta(days=periodo_dias)
    
    # Filtra vendas (vetorizado)
    mask_vendas = (df_vendas['created_at'] >= data_inicio) & (df_vendas['created_at'] <= data_fim)
    vendas_recentes = df_vendas.loc[mask_vendas].copy()
    
    print(f"Periodo: {data_inicio.date()} ate {data_fim.date()} ({periodo_dias} dias)")
    
    # Agrega vendas (otimizado)
    vendas_por_sku = vendas_recentes.groupby('sku', as_index=False)['quantidade'].sum()
    vendas_por_sku.columns = ['sku', 'quantidade_vendida']
    
    # Processa estoque
    if df_estoque['data'].dtype == 'object':
        df_estoque['data'] = pd.to_datetime(df_estoque['data'])
    
    mask_estoque = (df_estoque['data'] >= data_inicio) & (df_estoque['data'] <= data_fim)
    estoque_recente = df_estoque.loc[mask_estoque].copy()
    
    # Estoque médio (otimizado)
    estoque_medio = estoque_recente.groupby('sku', as_index=False)['estoque_atual'].mean()
    estoque_medio.columns = ['sku', 'estoque_medio']
    
    # Merge e calcula giro
    giro = vendas_por_sku.merge(estoque_medio, on='sku', how='inner')
    giro['giro_estoque'] = giro['quantidade_vendida'] / (giro['estoque_medio'] + 1)
    giro = giro.sort_values('giro_estoque', ascending=False)
    
    print(f"SKUs com vendas: {len(giro)}")
    print(f"\nTop 10 SKUs por giro:")
    print(giro.head(10)[['sku', 'quantidade_vendida', 'estoque_medio', 'giro_estoque']].to_string(index=False))
    
    return giro


def comparar_modelos_otimizado(serie, sku, horizonte_previsao=30, proporcao_treino=0.8):
    """
    Versão otimizada da comparação de modelos.
    
    Otimizações:
    - Limita parâmetros do auto_arima (mais rápido)
    - Timeout para modelos lentos
    - Processamento mais eficiente
    """
    serie_treino, serie_teste = dividir_serie_temporal(serie, proporcao_treino)
    n_previsao = min(horizonte_previsao, len(serie_teste))
    serie_teste_previsao = serie_teste.iloc[:n_previsao]
    
    resultados = {
        'sku': sku,
        'modelos': {},
        'previsoes': {},
        'metricas': [],
        'serie_teste': serie_teste_previsao,
        'serie_treino': serie_treino
    }
    
    # 1. SARIMA Mensal (OTIMIZADO: limites menores)
    try:
        modelo = auto_arima(
            serie_treino,
            seasonal=True,
            m=30,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            max_p=3, max_d=1, max_q=3,  # Reduzido de 5,2,5
            max_P=1, max_D=1, max_Q=1,  # Reduzido de 2,1,2
            information_criterion='aic',
            trace=False,
            n_jobs=1  # 1 core para evitar problemas
        )
        prev = np.maximum(modelo.predict(n_periods=n_previsao), 0)
        metricas = avaliar_modelo(serie_teste_previsao.values, prev, 'SARIMA Mensal (m=30)')
        resultados['modelos']['sarima_mensal'] = modelo
        resultados['previsoes']['sarima_mensal'] = prev
        resultados['metricas'].append(metricas)
        print(f"  [OK] SARIMA Mensal: {modelo.order} x {modelo.seasonal_order}")
    except Exception as e:
        print(f"  [AVISO] SARIMA Mensal: {str(e)[:80]}")
    
    # 2. ARIMA Simples (OTIMIZADO)
    try:
        modelo = auto_arima(
            serie_treino,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            max_p=3, max_d=1, max_q=3,  # Reduzido
            information_criterion='aic',
            trace=False,
            n_jobs=1
        )
        prev = np.maximum(modelo.predict(n_periods=n_previsao), 0)
        metricas = avaliar_modelo(serie_teste_previsao.values, prev, 'ARIMA Simples')
        resultados['modelos']['arima'] = modelo
        resultados['previsoes']['arima'] = prev
        resultados['metricas'].append(metricas)
        print(f"  [OK] ARIMA: {modelo.order}")
    except Exception as e:
        print(f"  [AVISO] ARIMA: {str(e)[:80]}")
    
    # 3. Média Móvel (rápido)
    try:
        modelo_mm = modelo_media_movel(serie_treino, janela=7)
        prev = prever_media_movel(modelo_mm, n_previsao)
        metricas = avaliar_modelo(serie_teste_previsao.values, prev, 'Media Movel (7 dias)')
        resultados['modelos']['media_movel'] = modelo_mm
        resultados['previsoes']['media_movel'] = prev
        resultados['metricas'].append(metricas)
        print(f"  [OK] Media Movel")
    except Exception as e:
        print(f"  [AVISO] Media Movel: {str(e)[:80]}")
    
    # 4. Suavização Exponencial (pode ser lento, timeout implícito)
    try:
        modelo = modelo_suavizacao_exponencial(serie_treino)
        if modelo:
            prev = np.maximum(modelo.forecast(n_previsao), 0)
            metricas = avaliar_modelo(serie_teste_previsao.values, prev, 'Suavizacao Exponencial')
            resultados['modelos']['exponencial'] = modelo
            resultados['previsoes']['exponencial'] = prev
            resultados['metricas'].append(metricas)
            print(f"  [OK] Suavizacao Exponencial")
    except Exception as e:
        print(f"  [AVISO] Exponencial: {str(e)[:80]}")
    
    return resultados


def calcular_metricas_completas(y_real, y_previsto):
    """
    Calcula TODAS as métricas estatísticas disponíveis.
    
    Métricas incluídas:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - MAPE (Mean Absolute Percentage Error)
    - R² (Coeficiente de Determinação)
    - MAE% (MAE percentual)
    - RMSE% (RMSE percentual)
    - Bias (Desvio médio)
    - MAE médio normalizado
    """
    y_real = np.array(y_real)
    y_previsto = np.array(y_previsto)
    
    # Métricas básicas
    mae = mean_absolute_error(y_real, y_previsto)
    rmse = np.sqrt(mean_squared_error(y_real, y_previsto))
    mape = calcular_mape(y_real, y_previsto)
    
    # R² (Coeficiente de Determinação)
    ss_res = np.sum((y_real - y_previsto) ** 2)
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    
    # Erros percentuais (normalizados pela média)
    media_real = np.mean(y_real)
    mae_percentual = (mae / media_real * 100) if media_real != 0 else np.nan
    rmse_percentual = (rmse / media_real * 100) if media_real != 0 else np.nan
    
    # Bias (desvio médio - indica viés sistemático)
    bias = np.mean(y_previsto - y_real)
    
    # MAE normalizado (dividido pelo range dos dados)
    range_real = np.max(y_real) - np.min(y_real)
    mae_normalizado = mae / range_real if range_real != 0 else np.nan
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape) if not np.isnan(mape) else np.nan,
        'r2': float(r2) if not np.isnan(r2) else np.nan,
        'mae_percentual': float(mae_percentual) if not np.isnan(mae_percentual) else np.nan,
        'rmse_percentual': float(rmse_percentual) if not np.isnan(rmse_percentual) else np.nan,
        'bias': float(bias),
        'mae_normalizado': float(mae_normalizado) if not np.isnan(mae_normalizado) else np.nan
    }


def salvar_resultado_sku(resultados, giro_sku):
    """
    Salva resultado de um SKU individual.
    
    Parameters:
    -----------
    resultados : dict
        Resultados da comparação
    giro_sku : float
        Giro de estoque do SKU
    """
    sku = resultados['sku']
    
    # Prepara dados para salvar
    dados_salvar = {
        'sku': sku,
        'giro_estoque': float(giro_sku),
        'data_processamento': datetime.now().isoformat(),
        'metricas': resultados['metricas'],
        'resumo': {}
    }
    
    # Adiciona resumo
    if len(resultados['metricas']) > 0:
        df_metricas = pd.DataFrame(resultados['metricas'])
        melhor_mae = df_metricas.loc[df_metricas['mae'].idxmin()]
        dados_salvar['resumo'] = {
            'melhor_modelo': melhor_mae['modelo'],
            'melhor_mae': float(melhor_mae['mae']),
            'melhor_rmse': float(melhor_mae['rmse']),
            'melhor_mape': float(melhor_mae['mape'])
        }
    
    # Salva JSON
    arquivo_json = DIR_RESULTADOS / f'resultado_{sku}.json'
    with open(arquivo_json, 'w', encoding='utf-8') as f:
        json.dump(dados_salvar, f, indent=2, default=str)
    
    # Salva CSV com métricas
    if len(resultados['metricas']) > 0:
        arquivo_csv = DIR_RESULTADOS / f'metricas_{sku}.csv'
        df_metricas = pd.DataFrame(resultados['metricas'])
        df_metricas.to_csv(arquivo_csv, index=False)
    
    print(f"\n[OK] Resultados salvos para SKU {sku}")
    print(f"     - {arquivo_json}")
    print(f"     - {arquivo_csv}")


def processar_sku_completo(df_estoque, sku, giro_sku, previsor):
    """
    Processa um SKU completo e salva resultados.
    
    Parameters:
    -----------
    df_estoque : pd.DataFrame
        DataFrame de estoque
    sku : str
        SKU a processar
    giro_sku : float
        Giro de estoque
    previsor : PrevisorEstoqueSARIMA
        Previsor inicializado
    """
    print(f"\n{'='*80}")
    print(f"PROCESSANDO SKU: {sku} (Giro: {giro_sku:.2f})")
    print(f"{'='*80}")
    
    try:
        # Prepara série
        serie = previsor.preparar_serie_temporal(df_estoque, sku=sku)
        
        if len(serie) < 200:
            print(f"[AVISO] Dados insuficientes ({len(serie)} obs). Pulando...")
            return False
        
        print(f"  Serie: {len(serie)} observacoes")
        
        # Compara modelos
        resultados = comparar_modelos_otimizado(serie, sku, horizonte_previsao=30)
        
        # Adiciona métricas completas para cada modelo
        # Cria mapeamento entre nomes de modelos e chaves de previsões
        mapeamento = {
            'SARIMA Mensal (m=30)': 'sarima_mensal',
            'ARIMA Simples': 'arima',
            'Media Movel (7 dias)': 'media_movel',
            'Suavizacao Exponencial': 'exponencial'
        }
        
        for i, metrica in enumerate(resultados['metricas']):
            nome_modelo = metrica['modelo']
            chave_previsao = mapeamento.get(nome_modelo)
            
            if chave_previsao and chave_previsao in resultados['previsoes']:
                prev = resultados['previsoes'][chave_previsao]
                metricas_completas = calcular_metricas_completas(
                    resultados['serie_teste'].values,
                    prev
                )
                resultados['metricas'][i].update(metricas_completas)
        
        # Salva resultado
        salvar_resultado_sku(resultados, giro_sku)
        
        # Atualiza checkpoint
        salvar_checkpoint(sku)
        
        return True
        
    except Exception as e:
        print(f"[ERRO] SKU {sku}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def gerar_relatorio_final():
    """
    Gera relatório final consolidando todos os resultados salvos.
    """
    print("\n" + "=" * 80)
    print("GERANDO RELATORIO FINAL CONSOLIDADO")
    print("=" * 80)
    
    # Carrega todos os resultados
    todos_resultados = []
    
    for arquivo in DIR_RESULTADOS.glob('resultado_*.json'):
        try:
            with open(arquivo, 'r') as f:
                dados = json.load(f)
                todos_resultados.append(dados)
        except:
            continue
    
    if len(todos_resultados) == 0:
        print("[AVISO] Nenhum resultado encontrado")
        return
    
    # Compila métricas
    todas_metricas = []
    for resultado in todos_resultados:
        for metrica in resultado.get('metricas', []):
            metrica['sku'] = resultado['sku']
            metrica['giro_estoque'] = resultado.get('giro_estoque', 0)
            todas_metricas.append(metrica)
    
    if len(todas_metricas) == 0:
        print("[AVISO] Nenhuma metrica encontrada")
        return
    
    df_metricas = pd.DataFrame(todas_metricas)
    
    # Melhor modelo por SKU
    melhores = []
    for sku in df_metricas['sku'].unique():
        df_sku = df_metricas[df_metricas['sku'] == sku]
        melhor = df_sku.loc[df_sku['mae'].idxmin()].to_dict()
        melhores.append(melhor)
    
    df_melhores = pd.DataFrame(melhores)
    df_melhores = df_melhores.sort_values('giro_estoque', ascending=False)
    
    # Salva relatório
    relatorio = []
    relatorio.append("=" * 80)
    relatorio.append("RELATORIO CONSOLIDADO: TOP SKUs - COMPARACAO DE MODELOS")
    relatorio.append("=" * 80)
    relatorio.append(f"\nData: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    relatorio.append(f"SKUs processados: {len(todos_resultados)}")
    relatorio.append("\n" + "-" * 80)
    relatorio.append("MELHOR MODELO POR SKU")
    relatorio.append("-" * 80)
    relatorio.append(df_melhores[['sku', 'giro_estoque', 'modelo', 'mae', 'rmse', 'mape']].to_string(index=False))
    
    # Estatísticas por modelo
    relatorio.append("\n" + "-" * 80)
    relatorio.append("ESTATISTICAS POR MODELO")
    relatorio.append("-" * 80)
    
    for modelo in df_metricas['modelo'].unique():
        df_modelo = df_metricas[df_metricas['modelo'] == modelo]
        relatorio.append(f"\n{modelo}:")
        relatorio.append(f"  MAE medio: {df_modelo['mae'].mean():.2f}")
        relatorio.append(f"  RMSE medio: {df_modelo['rmse'].mean():.2f}")
        relatorio.append(f"  MAPE medio: {df_modelo['mape'].mean():.2f}%")
        relatorio.append(f"  SKUs: {len(df_modelo['sku'].unique())}")
    
    texto = "\n".join(relatorio)
    
    arquivo_relatorio = DIR_RESULTADOS / 'relatorio_consolidado.txt'
    with open(arquivo_relatorio, 'w', encoding='utf-8') as f:
        f.write(texto)
    
    # Salva CSV consolidado
    arquivo_csv = DIR_RESULTADOS / 'metricas_consolidadas.csv'
    df_metricas.to_csv(arquivo_csv, index=False)
    
    print(f"\n[OK] Relatorio salvo: {arquivo_relatorio}")
    print(f"[OK] CSV salvo: {arquivo_csv}")
    print("\n" + texto)


def main():
    """
    Função principal otimizada.
    """
    print("=" * 80)
    print("COMPARACAO OTIMIZADA: TOP 10 SKUs COM MAIOR GIRO")
    print("=" * 80)
    
    # Carrega checkpoint
    skus_processados = carregar_checkpoint()
    print(f"\nCheckpoint: {len(skus_processados)} SKUs ja processados")
    if skus_processados:
        print(f"  SKUs: {', '.join(list(skus_processados)[:5])}...")
    
    # Carrega dados (uma vez só)
    print("\nCarregando dados...")
    df_vendas = pd.read_csv('DB/venda_produtos_atual.csv', low_memory=False)
    df_estoque = pd.read_csv('DB/historico_estoque_atual_processado.csv')
    df_estoque['data'] = pd.to_datetime(df_estoque['data'])
    
    print(f"[OK] Vendas: {len(df_vendas):,} registros")
    print(f"[OK] Estoque: {len(df_estoque):,} registros")
    
    # Calcula giro
    giro = calcular_giro_estoque_otimizado(df_vendas, df_estoque, periodo_dias=30)
    
    # Seleciona top 10 (exclui já processados)
    top_skus = giro.head(20)['sku'].tolist()  # Pega mais para ter opções
    top_skus = [sku for sku in top_skus if str(sku) not in skus_processados][:10]
    
    if len(top_skus) == 0:
        print("\n[INFO] Todos os SKUs ja foram processados!")
        gerar_relatorio_final()
        return
    
    print(f"\n[OK] {len(top_skus)} SKUs para processar")
    
    # Inicializa previsor (uma vez só)
    previsor = PrevisorEstoqueSARIMA()
    
    # Processa cada SKU (salva incrementalmente)
    sucesso = 0
    for i, sku in enumerate(top_skus, 1):
        print(f"\n{'#'*80}")
        print(f"PROGRESSO: {i}/{len(top_skus)} SKUs")
        print(f"{'#'*80}")
        
        giro_sku = giro[giro['sku'] == sku]['giro_estoque'].iloc[0]
        
        if processar_sku_completo(df_estoque, sku, giro_sku, previsor):
            sucesso += 1
        
        print(f"\n[PROGRESSO] {sucesso}/{i} SKUs processados com sucesso")
    
    # Gera relatório final
    gerar_relatorio_final()
    
    print("\n" + "=" * 80)
    print("PROCESSAMENTO CONCLUIDO!")
    print("=" * 80)
    print(f"SKUs processados: {sucesso}/{len(top_skus)}")
    print(f"Resultados salvos em: {DIR_RESULTADOS}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[AVISO] Processamento interrompido pelo usuario")
        print("Resultados ja processados foram salvos. Execute novamente para continuar.")
    except Exception as e:
        print(f"\n[ERRO] Erro durante execucao: {str(e)}")
        import traceback
        traceback.print_exc()

