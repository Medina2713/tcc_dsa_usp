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
from pathlib import Path
import sys
import time
import os
import warnings
warnings.filterwarnings('ignore')

# Configuração de limite de CPU (80%)
CPU_LIMIT_PERCENT = 80.0

# Tentar importar psutil para monitoramento de CPU (opcional)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def _log(msg, flush=True):
    """Imprime mensagem e garante flush para atualizacao imediata do log."""
    print(msg, flush=flush)


def _calcular_n_jobs_limite_cpu(limite_percent=80.0):
    """
    Calcula n_jobs baseado no limite de CPU desejado.
    Se limite=80% e temos 8 cores, retorna ~6 cores.
    """
    try:
        n_cores = os.cpu_count() or 4
        n_jobs = max(1, int(n_cores * limite_percent / 100.0))
        return n_jobs
    except:
        return 2  # Fallback conservador


def _aguardar_cpu_abaixo_limite(limite_percent=80.0, intervalo=0.5):
    """
    Aguarda até que o uso de CPU fique abaixo do limite.
    Usa psutil se disponível, caso contrário retorna imediatamente.
    """
    if not PSUTIL_AVAILABLE:
        return
    try:
        while True:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent < limite_percent:
                break
            time.sleep(intervalo)
    except:
        pass


# Line buffering para que trace do auto_arima atualize o log a cada linha
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sarima_estoque import PrevisorEstoqueSARIMA

# Configuração e pastas de saída (TCC)
plt.style.use('seaborn-v0_8-darkgrid')
DIR_FIGURAS_MODELOS = Path('resultados/figuras_modelos')
DIR_TABELAS_TCC = Path('resultados/tabelas_tcc')
DIR_RESULTADOS = Path('resultados')
for d in (DIR_FIGURAS_MODELOS, DIR_TABELAS_TCC, DIR_RESULTADOS):
    d.mkdir(parents=True, exist_ok=True)


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
        _log("  [busca] Iniciando stepwise; avaliando (p,d,q) x (P,D,Q,s)...", flush=True)
        t0 = time.time()
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
            trace=True,
            n_jobs=1  # Usa apenas 1 core para economizar memória
        )
        _log(f"  [busca] Stepwise concluido em {time.time() - t0:.1f}s", flush=True)
        return modelo
    except Exception as e:
        _log(f"  [AVISO] SARIMA Anual falhou (pode requerer mais memoria): {str(e)[:100]}", flush=True)
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
        _log("  [busca] Iniciando stepwise; avaliando (p,d,q) x (P,D,Q,s)...", flush=True)
        t0 = time.time()
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
            trace=True,
            n_jobs=_calcular_n_jobs_limite_cpu(CPU_LIMIT_PERCENT)
        )
        _log(f"  [busca] Stepwise concluido em {time.time() - t0:.1f}s", flush=True)
        return modelo
    except Exception as e:
        _log(f"  [ERRO] SARIMA Mensal: {str(e)}", flush=True)
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
        _log("  [busca] Iniciando stepwise; avaliando (p,d,q)...", flush=True)
        t0 = time.time()
        modelo = auto_arima(
            serie_treino,
            seasonal=False,  # Sem sazonalidade
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            max_p=5, max_d=2, max_q=5,
            information_criterion='aic',
            trace=True,
            n_jobs=_calcular_n_jobs_limite_cpu(CPU_LIMIT_PERCENT)
        )
        _log(f"  [busca] Stepwise concluido em {time.time() - t0:.1f}s", flush=True)
        return modelo
    except Exception as e:
        _log(f"  [ERRO] ARIMA Simples: {str(e)}", flush=True)
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
    - Tendência (quando presente; usa damped para evitar extrapolação linear explosiva)
    - Sazonalidade (se dados suficientes)
    
    Para séries quase constantes (CV < 1% ou std < 0.01): usa SES (sem tendência)
    para evitar tendência espúria e previsões lineares irreais.
    
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
        m_ = float(serie_treino.mean())
        s_ = float(serie_treino.std())
        cv = (s_ / m_ * 100) if m_ > 0 else 0.0
        
        # Série quase constante: usar SES (sem tendência) para evitar tendência espúria
        usar_trend = s_ >= 0.01 and cv >= 1.0
        
        if not usar_trend:
            # Simple Exponential Smoothing (sem tendência)
            modelo = ExponentialSmoothing(serie_treino, trend=None).fit()
            return modelo
        
        # Com tendência: usa damped_trend para evitar extrapolação linear explosiva
        if len(serie_treino) > 365:
            modelo = ExponentialSmoothing(
                serie_treino,
                seasonal_periods=365,
                trend='add',
                damped_trend=True,
                seasonal='add'
            ).fit()
        else:
            modelo = ExponentialSmoothing(
                serie_treino,
                trend='add',
                damped_trend=True
            ).fit()
        return modelo
    except Exception as e:
        try:
            modelo = ExponentialSmoothing(serie_treino, trend=None).fit()
            return modelo
        except:
            _log(f"  [ERRO] Suavizacao Exponencial: {str(e)}")
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
    _log("\n" + "=" * 80)
    _log(f"COMPARACAO DE MODELOS - SKU: {sku}")
    _log("=" * 80)
    
    # Divide série
    serie_treino, serie_teste = dividir_serie_temporal(serie, proporcao_treino)
    _log("\nDivisao treino/teste:")
    _log(f"  Treino: {len(serie_treino)} observacoes ({proporcao_treino*100:.0f}%)")
    _log(f"  Teste: {len(serie_teste)} observacoes ({(1-proporcao_treino)*100:.0f}%)")
    
    # Diagnostico da serie de treino (evita surpresas com modelos triviais)
    m, s = float(serie_treino.mean()), float(serie_treino.std())
    nz = (serie_treino == 0).sum()
    pct_z = 100.0 * nz / len(serie_treino)
    dias_treino = len(serie_treino)
    meses_treino = dias_treino / 30.0
    _log(f"  Serie treino: media={m:.2f}, desvio={s:.2f}, min={serie_treino.min():.0f}, max={serie_treino.max():.0f}, zeros={pct_z:.1f}%")
    _log(f"  Periodo de treino: {dias_treino} dias (~{meses_treino:.1f} meses)")
    if s < 0.01:
        _log("  [AVISO] Serie com variancia quase nula; modelos podem ser triviais (0,0,0).")
    
    n_previsao = min(horizonte_previsao, len(serie_teste))
    serie_teste_previsao = serie_teste.iloc[:n_previsao]
    
    # Diagnostico da serie de teste
    m_teste, s_teste = float(serie_teste_previsao.mean()), float(serie_teste_previsao.std())
    min_teste, max_teste = float(serie_teste_previsao.min()), float(serie_teste_previsao.max())
    range_teste = max_teste - min_teste
    cv_teste = (s_teste / m_teste * 100) if m_teste > 0 else 0.0
    _log(f"  Serie teste: media={m_teste:.2f}, desvio={s_teste:.2f}, min={min_teste:.0f}, max={max_teste:.0f}, range={range_teste:.2f}, CV={cv_teste:.2f}%")
    
    # Verifica se a serie de teste e constante ou quase constante
    teste_constante = (s_teste < 0.01 or range_teste < 0.01)
    if teste_constante:
        _log("  [AVISO CRITICO] Serie de teste e CONSTANTE ou quase constante!")
        _log("    -> Todos os modelos terao metricas IDENTICAS (previsoes serao iguais)")
        _log("    -> Isso e esperado: se o teste e constante, qualquer modelo preve o mesmo valor")
        _log("    -> MAE/RMSE/MAPE zerados nao sao comparaveis; relatorio usara N/A.")
    elif cv_teste < 1.0:
        _log("  [AVISO] Serie de teste tem variacao muito baixa (CV < 1%)")
        _log("    -> Modelos podem convergir para previsoes similares")
    
    resultados = {
        'sku': sku,
        'modelos': {},
        'previsoes': {},
        'metricas': [],
        'serie_treino': serie_treino,
        'serie_teste': serie_teste_previsao,
        'teste_constante': teste_constante,
    }
    
    # Verifica se há dados suficientes para SARIMA anual (m=365)
    # Requer pelo menos 730 dias (2 anos) para estimar sazonalidade anual adequadamente
    MIN_DIAS_SARIMA_ANUAL = 730
    usar_sarima_anual = dias_treino >= MIN_DIAS_SARIMA_ANUAL
    
    if not usar_sarima_anual:
        _log(f"\n[INFO] SARIMA Anual (m=365) DESABILITADO: serie tem {dias_treino} dias (~{meses_treino:.1f} meses)")
        _log(f"  [INFO] SARIMA anual requer pelo menos {MIN_DIAS_SARIMA_ANUAL} dias (2 anos) para estimar sazonalidade anual.")
        _log(f"  [INFO] Usando apenas SARIMA mensal (m=30), que e adequado para series mais curtas.")
    
    # Contador de modelos (ajusta se SARIMA anual for pulado)
    modelo_idx = 0
    total_modelos = 5 if usar_sarima_anual else 4
    
    # 1. SARIMA Anual (m=365) - apenas se houver dados suficientes
    modelo_sarima_a = None
    if usar_sarima_anual:
        modelo_idx += 1
        _log(f"\n[{modelo_idx}/{total_modelos}] Treinando SARIMA com sazonalidade ANUAL (m=365)...")
        _log("  (busca de parametros pode levar 1-2 min; saida do stepwise abaixo)")
        t1 = time.time()
        modelo_sarima_a = modelo_sarima_anual(serie_treino)
        dt1 = time.time() - t1
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
                _log(f"  [OK] Modelo: {modelo_sarima_a.order} x {modelo_sarima_a.seasonal_order}, AIC: {aic_a_val}")
            except Exception as e:
                _log(f"  [ERRO] Previsao falhou: {str(e)}")
        _log(f"  [{modelo_idx}/{total_modelos}] SARIMA Anual concluido em {dt1:.1f}s")
    
    # 2. SARIMA Mensal (m=30)
    modelo_idx += 1
    _log(f"\n[{modelo_idx}/{total_modelos}] Treinando SARIMA com sazonalidade MENSAL (m=30)...")
    _log("  (busca de parametros pode levar 1-2 min; saida do stepwise abaixo)")
    t2 = time.time()
    modelo_sarima_m = modelo_sarima_mensal(serie_treino)
    dt2 = time.time() - t2
    if modelo_sarima_m:
        try:
            prev_sarima_m = modelo_sarima_m.predict(n_periods=n_previsao)
            prev_sarima_m = np.maximum(prev_sarima_m, 0)
            # Log das previsoes (primeiros 5 valores)
            _log(f"  [DEBUG] Previsoes SARIMA Mensal (primeiros 5): {prev_sarima_m[:5]}")
            _log(f"  [DEBUG] Valores reais teste (primeiros 5): {serie_teste_previsao.values[:5]}")
            metricas = avaliar_modelo(serie_teste_previsao.values, prev_sarima_m, 'SARIMA Mensal (m=30)')
            resultados['modelos']['sarima_mensal'] = modelo_sarima_m
            resultados['previsoes']['sarima_mensal'] = prev_sarima_m
            resultados['metricas'].append(metricas)
            aic_m = getattr(modelo_sarima_m, 'aic', None)
            aic_m_val = aic_m() if callable(aic_m) else (aic_m if aic_m is not None else 'N/A')
            ordem_m = modelo_sarima_m.order
            sazonal_m = modelo_sarima_m.seasonal_order
            _log(f"  [OK] Modelo: {ordem_m} x {sazonal_m}, AIC: {aic_m_val}")
            if ordem_m == (0, 1, 0) and (sazonal_m is None or sazonal_m == (0, 0, 0, 0)):
                _log("  [INFO] Modelo convergiu para Random Walk (0,1,0) - previsao sera constante (ultimo valor)")
        except Exception as e:
            _log(f"  [ERRO] Previsao falhou: {str(e)}")
    _log(f"  [{modelo_idx}/{total_modelos}] SARIMA Mensal concluido em {dt2:.1f}s")
    
    # 3. ARIMA Simples
    modelo_idx += 1
    _log(f"\n[{modelo_idx}/{total_modelos}] Treinando ARIMA Simples (sem sazonalidade)...")
    _log("  (busca de parametros pode levar 1-2 min; saida do stepwise abaixo)")
    t3 = time.time()
    modelo_arima = modelo_arima_simples(serie_treino)
    dt3 = time.time() - t3
    if modelo_arima:
        try:
            prev_arima = modelo_arima.predict(n_periods=n_previsao)
            prev_arima = np.maximum(prev_arima, 0)
            # Log das previsoes (primeiros 5 valores)
            _log(f"  [DEBUG] Previsoes ARIMA (primeiros 5): {prev_arima[:5]}")
            metricas = avaliar_modelo(serie_teste_previsao.values, prev_arima, 'ARIMA Simples')
            resultados['modelos']['arima'] = modelo_arima
            resultados['previsoes']['arima'] = prev_arima
            resultados['metricas'].append(metricas)
            aic_a = getattr(modelo_arima, 'aic', None)
            aic_a_val = aic_a() if callable(aic_a) else (aic_a if aic_a is not None else 'N/A')
            ordem_a = modelo_arima.order
            _log(f"  [OK] Modelo: {ordem_a}, AIC: {aic_a_val}")
            if ordem_a == (0, 1, 0):
                _log("  [INFO] Modelo convergiu para Random Walk (0,1,0) - previsao sera constante (ultimo valor)")
        except Exception as e:
            _log(f"  [ERRO] Previsao falhou: {str(e)}")
    _log(f"  [{modelo_idx}/{total_modelos}] ARIMA Simples concluido em {dt3:.1f}s")
    
    # 4. Média Móvel
    modelo_idx += 1
    _log(f"\n[{modelo_idx}/{total_modelos}] Treinando Media Movel Simples...")
    t4 = time.time()
    modelo_mm = modelo_media_movel(serie_treino, janela=7)
    prev_mm = prever_media_movel(modelo_mm, n_previsao)
    # Log das previsoes (primeiros 5 valores)
    _log(f"  [DEBUG] Previsoes Media Movel (primeiros 5): {prev_mm[:5]}")
    metricas = avaliar_modelo(serie_teste_previsao.values, prev_mm, 'Media Movel (7 dias)')
    resultados['modelos']['media_movel'] = modelo_mm
    resultados['previsoes']['media_movel'] = prev_mm
    resultados['metricas'].append(metricas)
    _log(f"  [OK] Media dos ultimos 7 dias: {np.mean(modelo_mm['ultimos_valores']):.2f}")
    _log(f"  [{modelo_idx}/{total_modelos}] Media Movel concluida em {time.time() - t4:.1f}s")
    
    # 5. Suavização Exponencial
    modelo_idx += 1
    _log(f"\n[{modelo_idx}/{total_modelos}] Treinando Suavizacao Exponencial (Holt-Winters)...")
    t5 = time.time()
    modelo_exp = modelo_suavizacao_exponencial(serie_treino)
    if modelo_exp:
        try:
            prev_exp = modelo_exp.forecast(n_previsao)
            prev_exp = np.maximum(prev_exp, 0)
            metricas = avaliar_modelo(serie_teste_previsao.values, prev_exp, 'Suavizacao Exponencial')
            resultados['modelos']['exponencial'] = modelo_exp
            resultados['previsoes']['exponencial'] = prev_exp
            resultados['metricas'].append(metricas)
            _log("  [OK] Modelo treinado com sucesso")
        except Exception as e:
            _log(f"  [ERRO] Previsao falhou: {str(e)}")
    _log(f"  [{modelo_idx}/{total_modelos}] Holt-Winters concluido em {time.time() - t5:.1f}s")
    
    # Adiciona dados de teste para visualização
    resultados['serie_teste'] = serie_teste_previsao
    resultados['serie_treino'] = serie_treino
    
    # Compara previsoes para detectar se sao identicas
    _log("\n" + "=" * 80)
    _log("VERIFICACAO: Previsoes identicas entre modelos?")
    _log("=" * 80)
    
    # Verifica se as previsoes sao identicas entre modelos
    prevs_comparar = {}
    modelos_info = {}
    if 'sarima_mensal' in resultados['previsoes']:
        prevs_comparar['SARIMA Mensal'] = resultados['previsoes']['sarima_mensal']
        if 'sarima_mensal' in resultados['modelos']:
            mod = resultados['modelos']['sarima_mensal']
            ordem = getattr(mod, 'order', 'N/A')
            sazonal = getattr(mod, 'seasonal_order', 'N/A')
            modelos_info['SARIMA Mensal'] = f"{ordem} x {sazonal}"
    if 'arima' in resultados['previsoes']:
        prevs_comparar['ARIMA'] = resultados['previsoes']['arima']
        if 'arima' in resultados['modelos']:
            mod = resultados['modelos']['arima']
            ordem = getattr(mod, 'order', 'N/A')
            modelos_info['ARIMA'] = f"{ordem}"
    if 'media_movel' in resultados['previsoes']:
        prevs_comparar['Media Movel'] = resultados['previsoes']['media_movel']
        modelos_info['Media Movel'] = "Media dos ultimos 7 dias"
    
    if len(prevs_comparar) >= 2:
        nomes = list(prevs_comparar.keys())
        for i in range(len(nomes) - 1):
            for j in range(i + 1, len(nomes)):
                p1 = prevs_comparar[nomes[i]]
                p2 = prevs_comparar[nomes[j]]
                if len(p1) == len(p2):
                    sao_identicas = np.allclose(p1, p2, rtol=1e-10, atol=1e-10)
                    diff_max = np.max(np.abs(p1 - p2)) if not sao_identicas else 0.0
                    if sao_identicas:
                        _log(f"  [AVISO CRITICO] {nomes[i]} e {nomes[j]} tem previsoes IDENTICAS!")
                        _log(f"    Previsoes (primeiros 5): {nomes[i]}={p1[:5]}, {nomes[j]}={p2[:5]}")
                        _log(f"    Modelos: {nomes[i]}={modelos_info.get(nomes[i], 'N/A')}, {nomes[j]}={modelos_info.get(nomes[j], 'N/A')}")
                        _log(f"    POSSIVEL CAUSA: Modelos convergiram para Random Walk (0,1,0) ou serie de teste e constante")
                        _log(f"    IMPACTO: Metricas (MAE, RMSE, MAPE) serao identicas entre esses modelos")
                    else:
                        _log(f"  [OK] {nomes[i]} vs {nomes[j]}: diferenca maxima = {diff_max:.6f}")
                        if diff_max < 0.01:
                            _log(f"    [AVISO] Diferenca muito pequena - metricas podem ser quase identicas")
    
    # Aviso se resultados triviais (modelos 0,0,0 ou metricas zeradas)
    trivials = []
    for k, v in resultados.get('modelos', {}).items():
        o = getattr(v, 'order', None)
        so = getattr(v, 'seasonal_order', None)
        if o == (0, 0, 0) and (so is None or so == (0, 0, 0, 0)):
            trivials.append(k)
    all_mae_zero = all(m.get('mae', 1) == 0 for m in resultados.get('metricas', []))
    if trivials or all_mae_zero:
        _log("\n  [AVISO] Modelos triviais ou metricas zeradas detectados.")
        if trivials:
            _log(f"    Modelos (0,0,0): {', '.join(trivials)}")
        if all_mae_zero:
            _log("    MAE=0 para todos (serie de teste pode ser constante ou previsoes coincidirem).")
        _log("    Verifique o diagnostico da serie de treino acima. SKU com mais movimento pode ajudar.")
    
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
    _log("\n" + "=" * 80)
    _log("GERANDO VISUALIZACOES...")
    _log("=" * 80)
    
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
    
    nome_arquivo = DIR_FIGURAS_MODELOS / f'comparacao_modelos_{resultados["sku"]}.png'
    plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Grafico salvo: {nome_arquivo}")
    plt.close()


def _plotar_figura_modelo_unico(resultados, chave_previsao, titulo, nome_arquivo, dir_saida=None):
    """Plota histórico + previsão para um único modelo (Figuras TCC 5, 6, 7)."""
    if chave_previsao not in resultados.get('previsoes', {}):
        return
    prev = resultados['previsoes'][chave_previsao]
    serie_treino = resultados['serie_treino']
    serie_teste = resultados['serie_teste']
    if len(prev) != len(serie_teste):
        return
    plt.figure(figsize=(14, 6))
    hist = serie_treino.iloc[-90:] if len(serie_treino) > 90 else serie_treino
    plt.plot(hist.index, hist.values, label='Historico Real', color='#2E86AB', linewidth=2, alpha=0.8)
    plt.plot(serie_teste.index, prev, label='Previsao (Teste)', color='#A23B72', linewidth=2.5,
             linestyle='--', marker='o', markersize=4)
    plt.axvline(x=serie_treino.index[-1], color='gray', linestyle=':', alpha=0.7, linewidth=2)
    plt.title(titulo, fontsize=14, fontweight='bold')
    plt.xlabel('Data')
    plt.ylabel('Estoque (unidades)')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    base = Path(dir_saida) if dir_saida else DIR_FIGURAS_MODELOS
    path = base / nome_arquivo
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] {path}")


def run_comparison_for_sku(sku, path_csv, horizonte_previsao=30):
    """
    Executa comparacao de modelos para um unico SKU (carrega dados, prepara serie, treina).
    Retorna o dict resultados para uso em figuras/relatorios consolidados.
    """
    df = pd.read_csv(path_csv)
    df['data'] = pd.to_datetime(df['data'])
    df['sku'] = df['sku'].astype(str)
    sku_str = str(sku).strip()
    if sku_str not in df['sku'].values:
        return None
    previsor = PrevisorEstoqueSARIMA()
    serie = previsor.preparar_serie_temporal(df, sku=sku_str)
    if serie is None or len(serie) < 60:
        return None
    return comparar_modelos(serie, sku_str, horizonte_previsao=horizonte_previsao)


def _subplot_modelo_sku(ax, resultados, chave_previsao, sku_label):
    """Plota historico + previsao em um subplot (para figura multi-SKU)."""
    if chave_previsao not in resultados.get('previsoes', {}):
        ax.text(0.5, 0.5, f'SKU {sku_label}\n(sem previsao)', ha='center', va='center', transform=ax.transAxes)
        return
    prev = resultados['previsoes'][chave_previsao]
    serie_treino = resultados['serie_treino']
    serie_teste = resultados['serie_teste']
    if len(prev) != len(serie_teste):
        ax.text(0.5, 0.5, f'SKU {sku_label}\n(tamanho incompativel)', ha='center', va='center', transform=ax.transAxes)
        return
    hist = serie_treino.iloc[-90:] if len(serie_treino) > 90 else serie_treino
    ax.plot(hist.index, hist.values, label='Historico', color='#2E86AB', linewidth=1.5, alpha=0.8)
    ax.plot(serie_teste.index, prev, label='Previsao', color='#A23B72', linewidth=2, linestyle='--', marker='o', markersize=3)
    ax.axvline(x=serie_treino.index[-1], color='gray', linestyle=':', alpha=0.7)
    ax.set_title(f'SKU {sku_label}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Data')
    ax.set_ylabel('Estoque')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)


def salvar_figuras_tcc_multiplos_skus(lista_resultados, dir_figuras_tcc, sku_figura4=None):
    """
    Gera Figuras TCC 5, 6 e 7 com o SKU da Figura 4 (mesmo da análise exploratória).
    Cada figura mostra apenas um modelo (Holt-Winters, ARIMA, SARIMA) para esse SKU.
    
    Segundo o TCC: figuras 5-7 usam o mesmo SKU representativo da Figura 4.
    Se sku_figura4 não estiver em lista_resultados, usa fallback (maior variabilidade MAE).
    """
    dir_figuras_tcc = Path(dir_figuras_tcc)
    dir_figuras_tcc.mkdir(parents=True, exist_ok=True)
    
    if len(lista_resultados) == 0:
        _log("[ERRO] Nenhum resultado para gerar figuras TCC")
        return
    
    # Prioridade: usar SKU da Figura 4 (maior variação sazonal)
    sku_representativo = None
    sku_codigo = None
    
    if sku_figura4 is not None:
        sku_fig4_str = str(sku_figura4).strip()
        for res in lista_resultados:
            if str(res.get('sku', '')).strip() == sku_fig4_str:
                sku_representativo = res
                sku_codigo = res['sku']
                _log(f"\n[FIGURAS TCC] Usando SKU da Figura 4: {sku_codigo}")
                break
    
    # Fallback: SKU com maior variabilidade nas métricas entre modelos
    if sku_representativo is None:
        melhor_sku_idx = 0
        maior_variabilidade = 0.0
        for idx, res in enumerate(lista_resultados):
            metricas = res.get('metricas', [])
            if len(metricas) < 2:
                continue
            maes = [m.get('mae', 0) for m in metricas if not np.isnan(m.get('mae', np.nan))]
            if len(maes) >= 2:
                std_mae = np.std(maes) if len(maes) > 1 else 0.0
                mean_mae = np.mean(maes)
                cv_mae = (std_mae / mean_mae * 100) if mean_mae > 0 else 0.0
                if cv_mae > maior_variabilidade:
                    maior_variabilidade = cv_mae
                    melhor_sku_idx = idx
        sku_representativo = lista_resultados[melhor_sku_idx]
        sku_codigo = sku_representativo['sku']
        _log(f"\n[FIGURAS TCC] SKU Fig 4 nao encontrado. Fallback: {sku_codigo} (variabilidade MAE: {maior_variabilidade:.2f}%)")
    
    _log(f"  [INFO] Gerando figura 5 (Holt-Winters), 6 (ARIMA), 7 (SARIMA) para SKU {sku_codigo}...")
    
    configs = [
        ('exponencial', 'figura5.png', f'Figura 5 – Previsao do Estoque com o Modelo Holt-Winters (SKU {sku_codigo})'),
        ('arima', 'figura6.png', f'Figura 6 – Previsao do Estoque com o Modelo ARIMA (SKU {sku_codigo})'),
        ('sarima_mensal', 'figura7.png', f'Figura 7 – Previsao do Estoque com o Modelo SARIMA (SKU {sku_codigo})'),
    ]
    
    for chave, nome_arq, titulo in configs:
        if chave not in sku_representativo.get('previsoes', {}):
            _log(f"  [AVISO] Previsao '{chave}' nao encontrada para SKU {sku_codigo}. Pulando {nome_arq}")
            continue
        
        # Gera figura única (não subplot) para o SKU representativo
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        prev = sku_representativo['previsoes'][chave]
        serie_treino = sku_representativo['serie_treino']
        serie_teste = sku_representativo['serie_teste']
        
        if len(prev) != len(serie_teste):
            _log(f"  [AVISO] Tamanho incompativel para {chave}. Pulando {nome_arq}")
            plt.close(fig)
            continue
        
        # Plota histórico (últimos 90 dias ou todos se menor)
        hist = serie_treino.iloc[-90:] if len(serie_treino) > 90 else serie_treino
        ax.plot(hist.index, hist.values, label='Historico (Treino)', color='#2E86AB', linewidth=2, alpha=0.8)
        
        # Plota VALORES REAIS do teste (ground truth) — essencial para comparar previsão vs realidade
        ax.plot(serie_teste.index, serie_teste.values, label='Real (Teste)', color='#2D8B57', linewidth=2,
                alpha=0.9, linestyle='-', marker='s', markersize=3)
        
        # Plota previsão do modelo
        ax.plot(serie_teste.index, prev, label='Previsao', color='#A23B72', linewidth=2.5,
                linestyle='--', marker='o', markersize=4)
        
        # Linha vertical separando treino e teste
        ax.axvline(x=serie_treino.index[-1], color='gray', linestyle=':', alpha=0.7, linewidth=2, label='Inicio da Previsao')
        
        ax.set_title(titulo, fontsize=14, fontweight='bold')
        ax.set_xlabel('Data', fontsize=12)
        ax.set_ylabel('Estoque (unidades)', fontsize=12)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        path = dir_figuras_tcc / nome_arq
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        _log(f"  [OK] {path}")


def salvar_figuras_individuais_tcc(resultados, dir_figuras_tcc=None):
    """
    Gera Figuras TCC 5, 6 e 7: previsão com Holt-Winters, ARIMA e SARIMA (separadas).
    Se dir_figuras_tcc for informado, salva figura5.png, figura6.png, figura7.png nele.
    """
    _log("\n[FIGURAS TCC] Gerando figura 5 (Holt-Winters), 6 (ARIMA), 7 (SARIMA)...")
    sku = resultados['sku']
    if dir_figuras_tcc:
        nomes = ('figura5.png', 'figura6.png', 'figura7.png')
        _plotar_figura_modelo_unico(
            resultados, 'exponencial',
            f'Figura 5 – Previsao do Estoque com o Modelo Holt-Winters (SKU: {sku})',
            nomes[0], dir_saida=dir_figuras_tcc
        )
        _plotar_figura_modelo_unico(
            resultados, 'arima',
            f'Figura 6 – Previsao do Estoque com o Modelo ARIMA (SKU: {sku})',
            nomes[1], dir_saida=dir_figuras_tcc
        )
        _plotar_figura_modelo_unico(
            resultados, 'sarima_mensal',
            f'Figura 7 – Previsao do Estoque com o Modelo SARIMA (SKU: {sku})',
            nomes[2], dir_saida=dir_figuras_tcc
        )
    else:
        _plotar_figura_modelo_unico(
            resultados, 'exponencial',
            f'Figura 5 – Previsao do Estoque com o Modelo Holt-Winters (SKU: {sku})',
            f'figura_05_holt_winters_{sku}.png'
        )
        _plotar_figura_modelo_unico(
            resultados, 'arima',
            f'Figura 6 – Previsao do Estoque com o Modelo ARIMA (SKU: {sku})',
            f'figura_06_arima_{sku}.png'
        )
        _plotar_figura_modelo_unico(
            resultados, 'sarima_mensal',
            f'Figura 7 – Previsao do Estoque com o Modelo SARIMA (SKU: {sku})',
            f'figura_07_sarima_{sku}.png'
        )


def gerar_relatorio_comparacao(resultados, salvar_tabela=True):
    """
    PARTE 7: RELATÓRIO DE COMPARAÇÃO
    
    Gera relatório textual com métricas comparativas.
    Quando série de teste é constante, MAE/RMSE/MAPE zerados não são comparáveis;
    exibe N/A e evita afirmar "melhor modelo".
    
    Parameters:
    -----------
    resultados : dict
        Resultados da comparação
    salvar_tabela : bool
        Se True, salva tabela_02_desempenho_modelos.csv. Use False em modo multi-SKU.
    """
    print("\n" + "=" * 80)
    print("RELATORIO DE COMPARACAO")
    print("=" * 80)
    
    metricas = resultados['metricas']
    teste_constante = resultados.get('teste_constante', False)
    all_mae_zero = all(m.get('mae', 1) == 0 for m in metricas) if metricas else False
    usar_na = teste_constante and all_mae_zero
    
    if len(metricas) == 0:
        print("\n[AVISO] Nenhuma metrica disponivel")
        return None
    
    df_metricas = pd.DataFrame(metricas)
    df_metricas = df_metricas.sort_values('mae', na_position='last')
    
    # Para exibição: substituir 0/0/0 por N/A quando teste constante
    df_exibir = df_metricas.copy()
    if usar_na:
        df_exibir['mae'] = 'N/A'
        df_exibir['rmse'] = 'N/A'
        df_exibir['mape'] = 'N/A'
    
    print("\nMETRICAS DE AVALIACAO (ordenadas por MAE - menor e melhor):")
    print("-" * 80)
    print(df_exibir.to_string(index=False))
    if usar_na:
        print("\n  [N/A] Serie de teste constante. MAE, RMSE e MAPE zerados nao sao comparaveis.")
    
    print("\n" + "-" * 80)
    print("MELHOR MODELO POR METRICA:")
    print("-" * 80)
    
    if usar_na:
        print("N/A (serie de teste constante; metricas nao comparaveis)")
    else:
        def _idxmin_safe(series):
            idx = series.idxmin()
            return None if pd.isna(idx) else idx
        
        idx_mae = _idxmin_safe(df_metricas['mae'])
        idx_rmse = _idxmin_safe(df_metricas['rmse'])
        idx_mape = _idxmin_safe(df_metricas['mape'])
        
        if idx_mae is not None:
            r = df_metricas.loc[idx_mae]
            print(f"Menor MAE: {r['modelo']} (MAE = {r['mae']:.2f})")
        else:
            print("Menor MAE: (todas metricas invalidas/NaN)")
        if idx_rmse is not None:
            r = df_metricas.loc[idx_rmse]
            print(f"Menor RMSE: {r['modelo']} (RMSE = {r['rmse']:.2f})")
        else:
            print("Menor RMSE: (todas metricas invalidas/NaN)")
        if idx_mape is not None:
            r = df_metricas.loc[idx_mape]
            v = r['mape']
            print(f"Menor MAPE: {r['modelo']} (MAPE = {v:.2f}%)" if pd.notna(v) else "Menor MAPE: (MAPE NaN)")
        else:
            print("Menor MAPE: (todas metricas invalidas/NaN)")
    
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
    relatorio.append(df_exibir.to_string(index=False))
    if usar_na:
        relatorio.append("\n  [N/A] Serie de teste constante. MAE, RMSE e MAPE zerados nao sao comparaveis.")
    relatorio.append("\n" + "-" * 80)
    relatorio.append("INTERPRETACAO DAS METRICAS:")
    relatorio.append("-" * 80)
    relatorio.append("MAE (Mean Absolute Error): Erro medio absoluto. Menor e melhor.")
    relatorio.append("RMSE (Root Mean Squared Error): Penaliza erros grandes. Menor e melhor.")
    relatorio.append("MAPE (Mean Absolute Percentage Error): Erro percentual. Menor e melhor.")
    
    texto = "\n".join(relatorio)
    
    path_rel = DIR_RESULTADOS / f'relatorio_comparacao_{resultados["sku"]}.txt'
    with open(path_rel, 'w', encoding='utf-8') as f:
        f.write(texto)
    
    print(f"\n[OK] Relatorio salvo: {path_rel}")

    # Tabela 2 (TCC): só salva se não for teste constante (evita 0/0/0 como comparação)
    if salvar_tabela and not usar_na:
        tab2 = df_metricas[['modelo', 'mae', 'rmse', 'mape']].copy()
        tab2.columns = ['Modelo', 'MAE', 'RMSE', 'MAPE']
        tab2['MAPE'] = tab2['MAPE'].apply(lambda x: f'{x:.2f}%' if pd.notna(x) else 'NaN')
        path_tab2 = DIR_TABELAS_TCC / 'tabela_02_desempenho_modelos.csv'
        tab2.to_csv(path_tab2, index=False, encoding='utf-8-sig', sep=';')
        print(f"[OK] Tabela 2 (TCC) salva: {path_tab2}")

    return df_metricas


def gerar_tabela_02_multiplos_skus(lista_resultados, path_tab2=None):
    """
    Concatena metricas de todos os SKUs e salva Tabela 2 (TCC) consolidada.
    Colunas: SKU, Modelo, MAE, RMSE, MAPE.
    SKUs com serie de teste constante (MAE/RMSE/MAPE zerados) sao excluidos da tabela.
    """
    path_tab2 = Path(path_tab2) if path_tab2 else DIR_TABELAS_TCC / 'tabela_02_desempenho_modelos.csv'
    path_tab2.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for res in lista_resultados:
        if res.get('teste_constante', False):
            _log(f"  [TABELA 2] SKU {res.get('sku', '')} excluido (serie de teste constante; metricas nao comparaveis)")
            continue
        sku = res.get('sku', '')
        for m in res.get('metricas', []):
            rows.append({
                'SKU': sku,
                'Modelo': m.get('modelo', ''),
                'MAE': m.get('mae'),
                'RMSE': m.get('rmse'),
                'MAPE': m.get('mape'),
            })
    if not rows:
        _log("[AVISO] Tabela 2 vazia (nenhum SKU com metricas comparaveis)")
        return
    df = pd.DataFrame(rows)
    df['MAPE_str'] = df['MAPE'].apply(lambda x: f'{x:.2f}%' if pd.notna(x) else 'NaN')
    out = df[['SKU', 'Modelo', 'MAE', 'RMSE', 'MAPE_str']].copy()
    out.columns = ['SKU', 'Modelo', 'MAE', 'RMSE', 'MAPE']
    out.to_csv(path_tab2, index=False, encoding='utf-8-sig', sep=';')
    _log(f"[OK] Tabela 2 (TCC) consolidada salva: {path_tab2}")


def main(usar_nomes_tcc=False, sku_forcado=None, caminho_dados=None):
    """
    FUNÇÃO PRINCIPAL
    
    Executa comparação completa de modelos.
    
    Parameters:
    -----------
    usar_nomes_tcc : bool
        Se True, salva figura5.png, figura6.png, figura7.png em resultados/figuras_tcc/
    sku_forcado : str, optional
        SKU a usar (ex.: representativo da figura4). Se None, seleciona por variabilidade.
    caminho_dados : str, optional
        Caminho do CSV processado.
    """
    import sys
    if '--tcc' in sys.argv:
        usar_nomes_tcc = True
    sku_arg = [a for a in sys.argv if a.startswith('--sku=')]
    if sku_arg:
        sku_forcado = sku_arg[0].split('=', 1)[1].strip()
    
    path_csv = caminho_dados or 'DB/historico_estoque_atual_processado.csv'
    print("=" * 80)
    print("COMPARACAO DE MODELOS DE PREVISAO DE DEMANDA")
    print("=" * 80)
    
    print("\nCarregando dados processados...")
    df = pd.read_csv(path_csv)
    df['data'] = pd.to_datetime(df['data'])
    
    if sku_forcado:
        sku_selecionado = str(sku_forcado)
        if sku_selecionado not in df['sku'].values:
            print(f"[ERRO] SKU {sku_selecionado} nao encontrado no CSV.")
            return
        print(f"\nSKU (forcado): {sku_selecionado}")
    else:
        stats = df.groupby('sku')['estoque_atual'].agg(['count', 'mean', 'std']).reset_index()
        stats['cv'] = stats['std'] / stats['mean']
        stats = stats[(stats['count'] >= 200) & (stats['mean'] >= 1.0)].copy()
        stats['score'] = stats['count'] * stats['cv'] * stats['mean']
        stats = stats.sort_values('score', ascending=False)
        sku_selecionado = stats.iloc[0]['sku']
        print(f"\nSKU selecionado: {sku_selecionado}")
        print(f"  Observacoes: {int(stats.iloc[0]['count'])}")
        print(f"  Media: {stats.iloc[0]['mean']:.2f}")
    
    previsor = PrevisorEstoqueSARIMA()
    serie = previsor.preparar_serie_temporal(df, sku=sku_selecionado)
    
    print(f"\nSerie temporal preparada:")
    print(f"  Periodo: {serie.index[0].date()} ate {serie.index[-1].date()}")
    print(f"  Observacoes: {len(serie)}")
    
    resultados = comparar_modelos(serie, sku_selecionado, horizonte_previsao=30)
    
    visualizar_comparacao(resultados)
    dir_tcc = Path('resultados/figuras_tcc') if usar_nomes_tcc else None
    dir_tcc.mkdir(parents=True, exist_ok=True) if dir_tcc else None
    salvar_figuras_individuais_tcc(resultados, dir_figuras_tcc=dir_tcc)
    
    metricas = gerar_relatorio_comparacao(resultados)
    
    print("\n" + "=" * 80)
    print("COMPARACAO CONCLUIDA!")
    print("=" * 80)
    if usar_nomes_tcc and dir_tcc:
        print("\nFiguras TCC (resultados/figuras_tcc/):")
        for i in range(5, 8):
            print(f"  - figura{i}.png")
    else:
        print("\nArquivos gerados:")
        print(f"  - {DIR_FIGURAS_MODELOS}/comparacao_modelos_{sku_selecionado}.png")
        print(f"  - {DIR_FIGURAS_MODELOS}/figura_05_holt_winters_{sku_selecionado}.png")
        print(f"  - {DIR_FIGURAS_MODELOS}/figura_06_arima_{sku_selecionado}.png")
        print(f"  - {DIR_FIGURAS_MODELOS}/figura_07_sarima_{sku_selecionado}.png")
    print(f"  - {DIR_RESULTADOS}/relatorio_comparacao_{sku_selecionado}.txt")
    print(f"  - {DIR_TABELAS_TCC}/tabela_02_desempenho_modelos.csv")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERRO] Erro durante execucao: {str(e)}")
        import traceback
        traceback.print_exc()

