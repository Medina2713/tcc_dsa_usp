"""
Comparação de Modelos de Previsão de Demanda
TCC MBA Data Science & Analytics - E-commerce de Brinquedos

Este script compara diferentes modelos de previsão temporal:
1. SARIMA com sazonalidade anual (m=365)
2. SARIMA com sazonalidade mensal (m=30)
3. ARIMA (sem sazonalidade)
4. Média Móvel Simples
5. Suavização Exponencial (Holt-Winters)

Para cada modelo, calcula métricas de avaliação:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

Autor: Medina2713
Data: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sarima_estoque import PrevisorEstoqueSARIMA

# Configuração
plt.style.use('seaborn-v0_8-darkgrid')


def calcular_mape(y_real, y_previsto):
    """
    PARTE 1: CÁLCULO DE MAPE
    
    Calcula Mean Absolute Percentage Error (MAPE).
    
    MAPE = (1/n) * Σ |y_real - y_previsto| / |y_real| * 100
    
    Útil para comparar modelos em diferentes escalas.
    Valores próximos a 0 são melhores.
    
    Parameters:
    -----------
    y_real : array-like
        Valores reais
    y_previsto : array-like
        Valores previstos
        
    Returns:
    --------
    float
        MAPE em percentual
    """
    y_real = np.array(y_real)
    y_previsto = np.array(y_previsto)
    
    # Remove valores zero para evitar divisão por zero
    mask = y_real != 0
    if mask.sum() == 0:
        return np.nan
    
    mape = np.mean(np.abs((y_real[mask] - y_previsto[mask]) / y_real[mask])) * 100
    return mape


def dividir_serie_temporal(serie, proporcao_treino=0.8):
    """
    PARTE 2: DIVISÃO TREINO/TESTE
    
    Divide série temporal em conjunto de treino e teste.
    Mantém ordem temporal (não embaralha).
    
    Parameters:
    -----------
    serie : pd.Series
        Série temporal completa
    proporcao_treino : float
        Proporção para treino (padrão: 0.8 = 80%)
        
    Returns:
    --------
    tuple
        (serie_treino, serie_teste)
    """
    n = len(serie)
    n_treino = int(n * proporcao_treino)
    
    serie_treino = serie.iloc[:n_treino]
    serie_teste = serie.iloc[n_treino:]
    
    return serie_treino, serie_teste


def modelo_sarima_anual(serie_treino):
    """
    PARTE 3A: SARIMA COM SAZONALIDADE ANUAL
    
    Treina modelo SARIMA com período sazonal de 365 dias (anual).
    Captura padrões que se repetem anualmente (ex: outubro e dezembro).
    
    NOTA: Sazonalidade anual requer muitos dados e muita memória.
    Usamos parâmetros mais conservadores para evitar problemas de memória.
    
    Parameters:
    -----------
    serie_treino : pd.Series
        Série temporal de treino
        
    Returns:
    --------
    modelo_fit
        Modelo SARIMA treinado ou None se falhar
    """
    try:
        # Para sazonalidade anual, reduzimos limites para economizar memória
        modelo = auto_arima(
            serie_treino,
            seasonal=True,
            m=365,  # Sazonalidade anual
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            max_p=3, max_d=1, max_q=3,  # Reduzido para economizar memória
            max_P=1, max_D=1, max_Q=1,  # Reduzido para economizar memória
            information_criterion='aic',
            trace=False,
            n_jobs=1  # Usa apenas 1 core para economizar memória
        )
        return modelo
    except Exception as e:
        print(f"  [AVISO] SARIMA Anual falhou (pode requerer mais memoria): {str(e)[:100]}")
        return None


def modelo_sarima_mensal(serie_treino):
    """
    PARTE 3B: SARIMA COM SAZONALIDADE MENSAL
    
    Treina modelo SARIMA com período sazonal de 30 dias (mensal).
    
    Parameters:
    -----------
    serie_treino : pd.Series
        Série temporal de treino
        
    Returns:
    --------
    modelo_fit
        Modelo SARIMA treinado ou None se falhar
    """
    try:
        modelo = auto_arima(
            serie_treino,
            seasonal=True,
            m=30,  # Sazonalidade mensal
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            max_p=5, max_d=2, max_q=5,
            max_P=2, max_D=1, max_Q=2,
            information_criterion='aic',
            trace=False,
            n_jobs=-1
        )
        return modelo
    except Exception as e:
        print(f"  [ERRO] SARIMA Mensal: {str(e)}")
        return None


def modelo_arima_simples(serie_treino):
    """
    PARTE 3C: ARIMA SIMPLES (SEM SAZONALIDADE)
    
    Treina modelo ARIMA sem componente sazonal.
    Modelo mais simples, útil como baseline.
    
    Parameters:
    -----------
    serie_treino : pd.Series
        Série temporal de treino
        
    Returns:
    --------
    modelo_fit
        Modelo ARIMA treinado ou None se falhar
    """
    try:
        modelo = auto_arima(
            serie_treino,
            seasonal=False,  # Sem sazonalidade
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            max_p=5, max_d=2, max_q=5,
            information_criterion='aic',
            trace=False,
            n_jobs=-1
        )
        return modelo
    except Exception as e:
        print(f"  [ERRO] ARIMA Simples: {str(e)}")
        return None


def modelo_media_movel(serie_treino, janela=7):
    """
    PARTE 3D: MÉDIA MÓVEL SIMPLES
    
    Modelo baseline simples que prevê a média dos últimos N valores.
    Útil como referência de desempenho mínimo esperado.
    
    Parameters:
    -----------
    serie_treino : pd.Series
        Série temporal de treino
    janela : int
        Janela de média móvel (padrão: 7 dias)
        
    Returns:
    --------
    dict
        Dicionário com informações do modelo (para compatibilidade)
    """
    return {'tipo': 'media_movel', 'janela': janela, 'ultimos_valores': serie_treino.tail(janela).values}


def prever_media_movel(modelo_info, n_periodos):
    """
    Previsão usando média móvel.
    
    Parameters:
    -----------
    modelo_info : dict
        Informações do modelo de média móvel
    n_periodos : int
        Número de períodos a prever
        
    Returns:
    --------
    np.array
        Previsões
    """
    media = np.mean(modelo_info['ultimos_valores'])
    return np.full(n_periodos, media)


def modelo_suavizacao_exponencial(serie_treino):
    """
    PARTE 3E: SUAVIZAÇÃO EXPONENCIAL (HOLT-WINTERS)
    
    Modelo de suavização exponencial que captura:
    - Tendência
    - Sazonalidade (se presente)
    
    Útil quando há padrões suaves e previsíveis.
    
    Parameters:
    -----------
    serie_treino : pd.Series
        Série temporal de treino
        
    Returns:
    --------
    modelo_fit
        Modelo treinado ou None se falhar
    """
    try:
        # Tenta com sazonalidade primeiro
        if len(serie_treino) > 365:
            modelo = ExponentialSmoothing(
                serie_treino,
                seasonal_periods=365,  # Sazonalidade anual
                trend='add',
                seasonal='add'
            ).fit()
        else:
            # Sem sazonalidade se dados insuficientes
            modelo = ExponentialSmoothing(
                serie_treino,
                trend='add'
            ).fit()
        return modelo
    except Exception as e:
        try:
            # Fallback: sem sazonalidade
            modelo = ExponentialSmoothing(serie_treino, trend='add').fit()
            return modelo
        except:
            print(f"  [ERRO] Suavizacao Exponencial: {str(e)}")
            return None


def avaliar_modelo(y_real, y_previsto, nome_modelo):
    """
    PARTE 4: AVALIAÇÃO DE MODELOS
    
    Calcula métricas de avaliação para um modelo.
    
    Métricas:
    - MAE (Mean Absolute Error): Erro médio absoluto
    - RMSE (Root Mean Squared Error): Raiz do erro quadrático médio
    - MAPE (Mean Absolute Percentage Error): Erro percentual médio absoluto
    
    Parameters:
    -----------
    y_real : array-like
        Valores reais
    y_previsto : array-like
        Valores previstos
    nome_modelo : str
        Nome do modelo (para exibição)
        
    Returns:
    --------
    dict
        Dicionário com métricas
    """
    y_real = np.array(y_real)
    y_previsto = np.array(y_previsto)
    
    mae = mean_absolute_error(y_real, y_previsto)
    rmse = np.sqrt(mean_squared_error(y_real, y_previsto))
    mape = calcular_mape(y_real, y_previsto)
    
    return {
        'modelo': nome_modelo,
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }


def comparar_modelos(serie, sku, horizonte_previsao=30, proporcao_treino=0.8):
    """
    PARTE 5: COMPARAÇÃO COMPLETA DE MODELOS
    
    Treina e compara múltiplos modelos de previsão.
    
    Fluxo:
    1. Divide série em treino/teste
    2. Treina cada modelo
    3. Gera previsões
    4. Calcula métricas
    5. Retorna resultados comparativos
    
    Parameters:
    -----------
    serie : pd.Series
        Série temporal completa
    sku : str
        Código do SKU (para identificação)
    horizonte_previsao : int
        Número de períodos a prever
    proporcao_treino : float
        Proporção para treino
        
    Returns:
    --------
    dict
        Resultados completos com métricas e previsões
    """
    print("\n" + "=" * 80)
    print(f"COMPARACAO DE MODELOS - SKU: {sku}")
    print("=" * 80)
    
    # Divide série
    serie_treino, serie_teste = dividir_serie_temporal(serie, proporcao_treino)
    print(f"\nDivisao treino/teste:")
    print(f"  Treino: {len(serie_treino)} observacoes ({proporcao_treino*100:.0f}%)")
    print(f"  Teste: {len(serie_teste)} observacoes ({(1-proporcao_treino)*100:.0f}%)")
    
    n_previsao = min(horizonte_previsao, len(serie_teste))
    serie_teste_previsao = serie_teste.iloc[:n_previsao]
    
    resultados = {
        'sku': sku,
        'modelos': {},
        'previsoes': {},
        'metricas': []
    }
    
    # 1. SARIMA Anual (m=365)
    print("\n[1/5] Treinando SARIMA com sazonalidade ANUAL (m=365)...")
    modelo_sarima_a = modelo_sarima_anual(serie_treino)
    if modelo_sarima_a:
        try:
            prev_sarima_a = modelo_sarima_a.predict(n_periods=n_previsao)
            prev_sarima_a = np.maximum(prev_sarima_a, 0)  # Garante não-negativo
            metricas = avaliar_modelo(serie_teste_previsao.values, prev_sarima_a, 'SARIMA Anual (m=365)')
            resultados['modelos']['sarima_anual'] = modelo_sarima_a
            resultados['previsoes']['sarima_anual'] = prev_sarima_a
            resultados['metricas'].append(metricas)
            aic_a = getattr(modelo_sarima_a, 'aic', None)
            aic_a_val = aic_a() if callable(aic_a) else (aic_a if aic_a is not None else 'N/A')
            print(f"  [OK] Modelo: {modelo_sarima_a.order} x {modelo_sarima_a.seasonal_order}, AIC: {aic_a_val}")
        except Exception as e:
            print(f"  [ERRO] Previsao falhou: {str(e)}")
    
    # 2. SARIMA Mensal (m=30)
    print("\n[2/5] Treinando SARIMA com sazonalidade MENSAL (m=30)...")
    modelo_sarima_m = modelo_sarima_mensal(serie_treino)
    if modelo_sarima_m:
        try:
            prev_sarima_m = modelo_sarima_m.predict(n_periods=n_previsao)
            prev_sarima_m = np.maximum(prev_sarima_m, 0)
            metricas = avaliar_modelo(serie_teste_previsao.values, prev_sarima_m, 'SARIMA Mensal (m=30)')
            resultados['modelos']['sarima_mensal'] = modelo_sarima_m
            resultados['previsoes']['sarima_mensal'] = prev_sarima_m
            resultados['metricas'].append(metricas)
            aic_m = getattr(modelo_sarima_m, 'aic', None)
            aic_m_val = aic_m() if callable(aic_m) else (aic_m if aic_m is not None else 'N/A')
            print(f"  [OK] Modelo: {modelo_sarima_m.order} x {modelo_sarima_m.seasonal_order}, AIC: {aic_m_val}")
        except Exception as e:
            print(f"  [ERRO] Previsao falhou: {str(e)}")
    
    # 3. ARIMA Simples
    print("\n[3/5] Treinando ARIMA Simples (sem sazonalidade)...")
    modelo_arima = modelo_arima_simples(serie_treino)
    if modelo_arima:
        try:
            prev_arima = modelo_arima.predict(n_periods=n_previsao)
            prev_arima = np.maximum(prev_arima, 0)
            metricas = avaliar_modelo(serie_teste_previsao.values, prev_arima, 'ARIMA Simples')
            resultados['modelos']['arima'] = modelo_arima
            resultados['previsoes']['arima'] = prev_arima
            resultados['metricas'].append(metricas)
            aic_a = getattr(modelo_arima, 'aic', None)
            aic_a_val = aic_a() if callable(aic_a) else (aic_a if aic_a is not None else 'N/A')
            print(f"  [OK] Modelo: {modelo_arima.order}, AIC: {aic_a_val}")
        except Exception as e:
            print(f"  [ERRO] Previsao falhou: {str(e)}")
    
    # 4. Média Móvel
    print("\n[4/5] Treinando Media Movel Simples...")
    modelo_mm = modelo_media_movel(serie_treino, janela=7)
    prev_mm = prever_media_movel(modelo_mm, n_previsao)
    metricas = avaliar_modelo(serie_teste_previsao.values, prev_mm, 'Media Movel (7 dias)')
    resultados['modelos']['media_movel'] = modelo_mm
    resultados['previsoes']['media_movel'] = prev_mm
    resultados['metricas'].append(metricas)
    print(f"  [OK] Media dos ultimos 7 dias: {np.mean(modelo_mm['ultimos_valores']):.2f}")
    
    # 5. Suavização Exponencial
    print("\n[5/5] Treinando Suavizacao Exponencial (Holt-Winters)...")
    modelo_exp = modelo_suavizacao_exponencial(serie_treino)
    if modelo_exp:
        try:
            prev_exp = modelo_exp.forecast(n_previsao)
            prev_exp = np.maximum(prev_exp, 0)
            metricas = avaliar_modelo(serie_teste_previsao.values, prev_exp, 'Suavizacao Exponencial')
            resultados['modelos']['exponencial'] = modelo_exp
            resultados['previsoes']['exponencial'] = prev_exp
            resultados['metricas'].append(metricas)
            print(f"  [OK] Modelo treinado com sucesso")
        except Exception as e:
            print(f"  [ERRO] Previsao falhou: {str(e)}")
    
    # Adiciona dados de teste para visualização
    resultados['serie_teste'] = serie_teste_previsao
    resultados['serie_treino'] = serie_treino
    
    return resultados


def visualizar_comparacao(resultados):
    """
    PARTE 6: VISUALIZAÇÃO COMPARATIVA
    
    Cria gráficos comparando previsões dos diferentes modelos.
    
    Parameters:
    -----------
    resultados : dict
        Resultados da comparação de modelos
    """
    print("\n" + "=" * 80)
    print("GERANDO VISUALIZACOES...")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    serie_treino = resultados['serie_treino']
    serie_teste = resultados['serie_teste']
    previsoes = resultados['previsoes']
    
    # Índices para plot
    idx_treino = range(len(serie_treino))
    idx_teste = range(len(serie_treino), len(serie_treino) + len(serie_teste))
    idx_previsao = idx_teste
    
    # Gráfico 1: Visão geral (treino + teste + previsões)
    ax1 = axes[0]
    
    # Histórico de treino
    ax1.plot(idx_treino, serie_treino.values, label='Treino', color='blue', linewidth=2, alpha=0.7)
    
    # Valores reais de teste
    ax1.plot(idx_teste, serie_teste.values, label='Real (Teste)', color='green', 
             linewidth=2, marker='o', markersize=4)
    
    # Previsões de cada modelo
    cores = {'sarima_anual': 'red', 'sarima_mensal': 'orange', 'arima': 'purple', 
             'media_movel': 'brown', 'exponencial': 'pink'}
    nomes = {'sarima_anual': 'SARIMA Anual', 'sarima_mensal': 'SARIMA Mensal', 
             'arima': 'ARIMA', 'media_movel': 'Media Movel', 'exponencial': 'Exponencial'}
    
    for chave, prev in previsoes.items():
        if len(prev) == len(serie_teste):
            ax1.plot(idx_previsao, prev, label=nomes.get(chave, chave), 
                    linestyle='--', linewidth=1.5, color=cores.get(chave, 'gray'), alpha=0.8)
    
    ax1.axvline(x=len(serie_treino), color='black', linestyle=':', linewidth=2, alpha=0.5)
    ax1.set_title(f'Comparacao de Modelos - SKU: {resultados["sku"]}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Periodo')
    ax1.set_ylabel('Estoque (unidades)')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Zoom no período de teste
    ax2 = axes[1]
    ax2.plot(serie_teste.values, label='Real', color='green', linewidth=2, marker='o', markersize=5)
    
    for chave, prev in previsoes.items():
        if len(prev) == len(serie_teste):
            ax2.plot(prev, label=nomes.get(chave, chave), linestyle='--', 
                    linewidth=2, color=cores.get(chave, 'gray'), marker='s', markersize=3, alpha=0.8)
    
    ax2.set_title('Zoom: Periodo de Teste (Comparacao Detalhada)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Dias de Teste')
    ax2.set_ylabel('Estoque (unidades)')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    nome_arquivo = f'comparacao_modelos_{resultados["sku"]}.png'
    plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Grafico salvo: {nome_arquivo}")
    plt.close()


def gerar_relatorio_comparacao(resultados):
    """
    PARTE 7: RELATÓRIO DE COMPARAÇÃO
    
    Gera relatório textual com métricas comparativas.
    
    Parameters:
    -----------
    resultados : dict
        Resultados da comparação
    """
    print("\n" + "=" * 80)
    print("RELATORIO DE COMPARACAO")
    print("=" * 80)
    
    metricas = resultados['metricas']
    
    if len(metricas) == 0:
        print("\n[AVISO] Nenhuma metrica disponivel")
        return
    
    df_metricas = pd.DataFrame(metricas)
    df_metricas = df_metricas.sort_values('mae')  # Ordena por MAE (menor é melhor)
    
    print("\nMETRICAS DE AVALIACAO (ordenadas por MAE - menor e melhor):")
    print("-" * 80)
    print(df_metricas.to_string(index=False))
    
    # Melhor modelo por métrica
    print("\n" + "-" * 80)
    print("MELHOR MODELO POR METRICA:")
    print("-" * 80)
    
    melhor_mae = df_metricas.loc[df_metricas['mae'].idxmin()]
    melhor_rmse = df_metricas.loc[df_metricas['rmse'].idxmin()]
    melhor_mape = df_metricas.loc[df_metricas['mape'].idxmin()]
    
    print(f"Menor MAE: {melhor_mae['modelo']} (MAE = {melhor_mae['mae']:.2f})")
    print(f"Menor RMSE: {melhor_rmse['modelo']} (RMSE = {melhor_rmse['rmse']:.2f})")
    print(f"Menor MAPE: {melhor_mape['modelo']} (MAPE = {melhor_mape['mape']:.2f}%)")
    
    # Salva relatório
    relatorio = []
    relatorio.append("=" * 80)
    relatorio.append("RELATORIO DE COMPARACAO DE MODELOS DE PREVISAO")
    relatorio.append("=" * 80)
    relatorio.append(f"\nSKU: {resultados['sku']}")
    relatorio.append(f"Data: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    relatorio.append("\n" + "-" * 80)
    relatorio.append("METRICAS DE AVALIACAO")
    relatorio.append("-" * 80)
    relatorio.append(df_metricas.to_string(index=False))
    relatorio.append("\n" + "-" * 80)
    relatorio.append("INTERPRETACAO DAS METRICAS:")
    relatorio.append("-" * 80)
    relatorio.append("MAE (Mean Absolute Error): Erro medio absoluto. Menor e melhor.")
    relatorio.append("RMSE (Root Mean Squared Error): Penaliza erros grandes. Menor e melhor.")
    relatorio.append("MAPE (Mean Absolute Percentage Error): Erro percentual. Menor e melhor.")
    
    texto = "\n".join(relatorio)
    
    nome_arquivo = f'relatorio_comparacao_{resultados["sku"]}.txt'
    with open(nome_arquivo, 'w', encoding='utf-8') as f:
        f.write(texto)
    
    print(f"\n[OK] Relatorio salvo: {nome_arquivo}")
    
    return df_metricas


def main():
    """
    FUNÇÃO PRINCIPAL
    
    Executa comparação completa de modelos.
    """
    print("=" * 80)
    print("COMPARACAO DE MODELOS DE PREVISAO DE DEMANDA")
    print("=" * 80)
    
    # Carrega dados
    print("\nCarregando dados processados...")
    df = pd.read_csv('DB/historico_estoque_atual_processado.csv')
    df['data'] = pd.to_datetime(df['data'])
    
    # Seleciona SKU para teste (pode ser o mesmo do teste anterior ou outro)
    # Usa o mesmo critério: maior variabilidade
    stats = df.groupby('sku')['estoque_atual'].agg(['count', 'mean', 'std']).reset_index()
    stats['cv'] = stats['std'] / stats['mean']
    stats = stats[(stats['count'] >= 200) & (stats['mean'] >= 1.0)].copy()
    stats['score'] = stats['count'] * stats['cv'] * stats['mean']
    stats = stats.sort_values('score', ascending=False)
    
    sku_selecionado = stats.iloc[0]['sku']
    print(f"\nSKU selecionado: {sku_selecionado}")
    print(f"  Observacoes: {int(stats.iloc[0]['count'])}")
    print(f"  Media: {stats.iloc[0]['mean']:.2f}")
    
    # Prepara série temporal
    previsor = PrevisorEstoqueSARIMA()
    serie = previsor.preparar_serie_temporal(df, sku=sku_selecionado)
    
    print(f"\nSerie temporal preparada:")
    print(f"  Periodo: {serie.index[0].date()} ate {serie.index[-1].date()}")
    print(f"  Observacoes: {len(serie)}")
    
    # Compara modelos
    resultados = comparar_modelos(serie, sku_selecionado, horizonte_previsao=30)
    
    # Visualiza
    visualizar_comparacao(resultados)
    
    # Relatório
    metricas = gerar_relatorio_comparacao(resultados)
    
    print("\n" + "=" * 80)
    print("COMPARACAO CONCLUIDA!")
    print("=" * 80)
    print("\nArquivos gerados:")
    print(f"  - comparacao_modelos_{sku_selecionado}.png")
    print(f"  - relatorio_comparacao_{sku_selecionado}.txt")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERRO] Erro durante execucao: {str(e)}")
        import traceback
        traceback.print_exc()

