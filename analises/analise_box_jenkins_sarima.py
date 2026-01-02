"""
Análise Completa Box-Jenkins para Modelo SARIMA
TCC MBA Data Science & Analytics - E-commerce de Brinquedos

Este script implementa todas as etapas do método Box-Jenkins:
1. IDENTIFICAÇÃO: Análise de estacionariedade, ACF/PACF, sazonalidade
2. ESTIMAÇÃO: Ajuste de parâmetros (via auto_arima)
3. DIAGNÓSTICO: Testes de resíduos (Ljung-Box, normalidade, heterocedasticidade)
4. PREVISÃO: Geração de previsões com intervalos de confiança

Autor: Medina2713
Data: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from pmdarima.arima import ADFTest
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')


class AnaliseBoxJenkins:
    """
    Classe para realizar análise completa Box-Jenkins de séries temporais.
    Implementa todas as etapas do método para validação de modelos SARIMA.
    """
    
    def __init__(self, serie, sku=None):
        """
        Inicializa a análise Box-Jenkins.
        
        Parameters:
        -----------
        serie : pd.Series
            Série temporal a ser analisada
        sku : str, optional
            Identificador do produto (para relatórios)
        """
        self.serie = serie.copy()
        self.sku = sku or "N/A"
        self.modelo = None
        self.residuos = None
        self.resultados = {}
        
    # ============================================================================
    # ETAPA 1: IDENTIFICAÇÃO
    # ============================================================================
    
    def teste_estacionariedade_adf(self, verbose=True):
        """
        ETAPA 1.1: Teste de Estacionariedade (Augmented Dickey-Fuller)
        
        Testa se a série é estacionária. Se não for, indica necessidade de diferenciação.
        
        H0: A série possui raiz unitária (não estacionária)
        H1: A série é estacionária
        
        Returns:
        --------
        dict
            Resultados do teste ADF
        """
        if verbose:
            print("\n" + "=" * 80)
            print("ETAPA 1.1: TESTE DE ESTACIONARIEDADE (ADF)")
            print("=" * 80)
        
        resultado = adfuller(self.serie.dropna(), autolag='AIC')
        
        adf_statistic = resultado[0]
        p_value = resultado[1]
        critical_values = resultado[4]
        n_lags = resultado[2]
        n_obs = resultado[3]
        
        is_stationary = p_value < 0.05
        
        resultado_dict = {
            'adf_statistic': adf_statistic,
            'p_value': p_value,
            'is_stationary': is_stationary,
            'critical_values': critical_values,
            'n_lags': n_lags,
            'n_obs': n_obs
        }
        
        self.resultados['estacionariedade'] = resultado_dict
        
        if verbose:
            print(f"\nEstatística ADF: {adf_statistic:.4f}")
            print(f"p-value: {p_value:.4f}")
            print(f"\nValores Críticos:")
            for key, value in critical_values.items():
                print(f"  {key}: {value:.4f}")
            
            print(f"\nConclusão: A série é {'ESTACIONÁRIA' if is_stationary else 'NÃO ESTACIONÁRIA'}")
            if not is_stationary:
                print("  → Necessário aplicar diferenciação (parâmetro 'd' no ARIMA)")
        
        return resultado_dict
    
    def analise_acf_pacf(self, lags=40, verbose=True):
        """
        ETAPA 1.2: Análise de Autocorrelação (ACF) e Autocorrelação Parcial (PACF)
        
        ACF e PACF ajudam a identificar:
        - Ordem do componente AR (p): baseado no corte do PACF
        - Ordem do componente MA (q): baseado no corte do ACF
        - Padrões sazonais: picos em lags múltiplos do período sazonal
        
        Parameters:
        -----------
        lags : int
            Número de lags a analisar
        verbose : bool
            Se True, imprime resultados
            
        Returns:
        --------
        dict
            Valores de ACF e PACF
        """
        if verbose:
            print("\n" + "=" * 80)
            print("ETAPA 1.2: ANÁLISE DE AUTOCORRELAÇÃO (ACF/PACF)")
            print("=" * 80)
        
        # Calcula ACF e PACF
        acf_values, acf_confint = acf(self.serie.dropna(), nlags=lags, alpha=0.05, fft=True)
        pacf_values, pacf_confint = pacf(self.serie.dropna(), nlags=lags, alpha=0.05)
        
        # Identifica lags significativos (fora do intervalo de confiança)
        acf_significativos = []
        pacf_significativos = []
        
        for i in range(1, min(lags + 1, len(acf_values))):
            if abs(acf_values[i]) > abs(acf_confint[i][0] - acf_values[i]):
                acf_significativos.append(i)
            if abs(pacf_values[i]) > abs(pacf_confint[i][0] - pacf_values[i]):
                pacf_significativos.append(i)
        
        resultado_dict = {
            'acf_values': acf_values,
            'pacf_values': pacf_values,
            'acf_confint': acf_confint,
            'pacf_confint': pacf_confint,
            'acf_significativos': acf_significativos[:10],  # Primeiros 10
            'pacf_significativos': pacf_significativos[:10]
        }
        
        self.resultados['acf_pacf'] = resultado_dict
        
        if verbose:
            print(f"\nLags com ACF significativo (primeiros 10): {acf_significativos[:10]}")
            print(f"Lags com PACF significativo (primeiros 10): {pacf_significativos[:10]}")
            print("\nInterpretação:")
            print("  - ACF: Identifica ordem MA (q) - corte abrupto indica ordem")
            print("  - PACF: Identifica ordem AR (p) - corte abrupto indica ordem")
            print("  - Padrões sazonais: picos em lags 7, 14, 21 (semanal) ou 30, 60 (mensal)")
        
        return resultado_dict
    
    def decomposicao_sazonal(self, periodo=30, verbose=True):
        """
        ETAPA 1.3: Decomposição Sazonal
        
        Decompõe a série em componentes:
        - Tendência
        - Sazonalidade
        - Resíduo
        
        Parameters:
        -----------
        periodo : int
            Período sazonal (30 para mensal, 7 para semanal, 365 para anual)
        verbose : bool
            Se True, imprime resultados
            
        Returns:
        --------
        dict
            Componentes da decomposição
        """
        if verbose:
            print("\n" + "=" * 80)
            print("ETAPA 1.3: DECOMPOSIÇÃO SAZONAL")
            print("=" * 80)
        
        try:
            # Requer pelo menos 2 períodos completos
            if len(self.serie) < 2 * periodo:
                if verbose:
                    print(f"[AVISO] Série muito curta ({len(self.serie)} obs) para período {periodo}")
                    print(f"  Tentando com período reduzido...")
                periodo = len(self.serie) // 2
            
            decomposicao = seasonal_decompose(
                self.serie.dropna(),
                model='additive',
                period=periodo,
                extrapolate_trend='freq'
            )
            
            resultado_dict = {
                'tendencia': decomposicao.trend,
                'sazonalidade': decomposicao.seasonal,
                'residuo': decomposicao.resid,
                'observado': decomposicao.observed,
                'periodo': periodo
            }
            
            self.resultados['decomposicao'] = resultado_dict
            
            if verbose:
                # Calcula força da sazonalidade
                var_sazonal = np.var(decomposicao.seasonal.dropna())
                var_residuo = np.var(decomposicao.resid.dropna())
                forca_sazonal = var_sazonal / (var_sazonal + var_residuo) if (var_sazonal + var_residuo) > 0 else 0
                
                print(f"\nPeríodo sazonal usado: {periodo}")
                print(f"Força da sazonalidade: {forca_sazonal:.3f} (0=fraca, 1=forte)")
                print(f"  → Valores > 0.5 indicam sazonalidade forte")
            
            return resultado_dict
            
        except Exception as e:
            if verbose:
                print(f"[ERRO] Erro na decomposição: {str(e)}")
            return None
    
    # ============================================================================
    # ETAPA 2: ESTIMAÇÃO
    # ============================================================================
    
    def estimar_modelo(self, seasonal=True, m=30, verbose=True):
        """
        ETAPA 2: Estimação do Modelo SARIMA
        
        Usa auto_arima para encontrar automaticamente os melhores parâmetros.
        
        Parameters:
        -----------
        seasonal : bool
            Se True, usa SARIMA (com sazonalidade)
        m : int
            Período sazonal (30 para mensal)
        verbose : bool
            Se True, imprime resultados
            
        Returns:
        --------
        modelo
            Modelo SARIMA treinado
        """
        if verbose:
            print("\n" + "=" * 80)
            print("ETAPA 2: ESTIMAÇÃO DO MODELO SARIMA")
            print("=" * 80)
        
        try:
            modelo = auto_arima(
                self.serie,
                seasonal=seasonal,
                m=m if seasonal else None,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                max_p=5, max_d=2, max_q=5,
                max_P=2, max_D=1, max_Q=2,
                information_criterion='aic',
                trace=False,
                n_jobs=1
            )
            
            self.modelo = modelo
            
            # Extrai resíduos
            self.residuos = modelo.resid()
            
            resultado_dict = {
                'order': modelo.order,
                'seasonal_order': modelo.seasonal_order if seasonal else None,
                'aic': modelo.aic(),
                'bic': modelo.bic() if hasattr(modelo, 'bic') else None,
                'aicc': modelo.aicc() if hasattr(modelo, 'aicc') else None
            }
            
            self.resultados['modelo'] = resultado_dict
            
            if verbose:
                print(f"\n[OK] Modelo estimado com sucesso!")
                print(f"Parâmetros ARIMA: {modelo.order}")
                if seasonal:
                    print(f"Parâmetros Sazonais: {modelo.seasonal_order}")
                print(f"AIC: {modelo.aic():.2f}")
                if hasattr(modelo, 'bic'):
                    print(f"BIC: {modelo.bic():.2f}")
            
            return modelo
            
        except Exception as e:
            if verbose:
                print(f"[ERRO] Erro ao estimar modelo: {str(e)}")
            return None
    
    # ============================================================================
    # ETAPA 3: DIAGNÓSTICO
    # ============================================================================
    
    def teste_ljung_box(self, lags=10, verbose=True):
        """
        ETAPA 3.1: Teste de Ljung-Box para Resíduos
        
        Testa se os resíduos são não correlacionados (ruído branco).
        
        H0: Os resíduos são não correlacionados (modelo adequado)
        H1: Os resíduos são correlacionados (modelo inadequado)
        
        Parameters:
        -----------
        lags : int
            Número de lags a testar
        verbose : bool
            Se True, imprime resultados
            
        Returns:
        --------
        dict
            Resultados do teste Ljung-Box
        """
        if self.residuos is None:
            if verbose:
                print("[ERRO] Modelo não foi estimado ainda. Execute estimar_modelo() primeiro.")
            return None
        
        if verbose:
            print("\n" + "=" * 80)
            print("ETAPA 3.1: TESTE DE LJUNG-BOX (RESÍDUOS)")
            print("=" * 80)
        
        # Remove NaN dos resíduos
        residuos_clean = self.residuos.dropna()
        
        if len(residuos_clean) < lags + 1:
            lags = len(residuos_clean) - 1
        
        resultado = acorr_ljungbox(residuos_clean, lags=lags, return_df=True)
        
        # Pega o último lag (teste geral)
        p_value_final = resultado['lb_pvalue'].iloc[-1]
        estatistica_final = resultado['lb_stat'].iloc[-1]
        
        # Resíduos são não correlacionados se p-value > 0.05
        residuos_ok = p_value_final > 0.05
        
        resultado_dict = {
            'estatistica': estatistica_final,
            'p_value': p_value_final,
            'residuos_ok': residuos_ok,
            'tabela_completa': resultado
        }
        
        self.resultados['ljung_box'] = resultado_dict
        
        if verbose:
            print(f"\nEstatística Ljung-Box (lag {lags}): {estatistica_final:.4f}")
            print(f"p-value: {p_value_final:.4f}")
            print(f"\nConclusão: Resíduos são {'NÃO CORRELACIONADOS (modelo adequado)' if residuos_ok else 'CORRELACIONADOS (modelo pode ser melhorado)'}")
            if not residuos_ok:
                print("  → Considere aumentar a ordem do modelo ou verificar outros problemas")
        
        return resultado_dict
    
    def teste_normalidade_residuos(self, verbose=True):
        """
        ETAPA 3.2: Teste de Normalidade dos Resíduos
        
        Testa se os resíduos seguem distribuição normal.
        Usa múltiplos testes: Shapiro-Wilk, Jarque-Bera, Anderson-Darling.
        
        Parameters:
        -----------
        verbose : bool
            Se True, imprime resultados
            
        Returns:
        --------
        dict
            Resultados dos testes de normalidade
        """
        if self.residuos is None:
            if verbose:
                print("[ERRO] Modelo não foi estimado ainda.")
            return None
        
        if verbose:
            print("\n" + "=" * 80)
            print("ETAPA 3.2: TESTE DE NORMALIDADE DOS RESÍDUOS")
            print("=" * 80)
        
        residuos_clean = self.residuos.dropna()
        
        # Teste Shapiro-Wilk (para amostras pequenas/médias)
        if len(residuos_clean) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(residuos_clean)
        else:
            # Para amostras grandes, usa amostra
            shapiro_stat, shapiro_p = stats.shapiro(residuos_clean.sample(5000))
        
        # Teste Jarque-Bera (testa assimetria e curtose)
        jb_stat, jb_p = stats.jarque_bera(residuos_clean)
        
        # Teste Anderson-Darling
        anderson_result = stats.anderson(residuos_clean, dist='norm')
        anderson_stat = anderson_result.statistic
        # Pega o valor crítico para 5% de significância
        anderson_critical = anderson_result.critical_values[2]  # 5%
        anderson_ok = anderson_stat < anderson_critical
        
        resultado_dict = {
            'shapiro_wilk': {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'is_normal': shapiro_p > 0.05
            },
            'jarque_bera': {
                'statistic': jb_stat,
                'p_value': jb_p,
                'is_normal': jb_p > 0.05
            },
            'anderson_darling': {
                'statistic': anderson_stat,
                'critical_5pct': anderson_critical,
                'is_normal': anderson_ok
            }
        }
        
        self.resultados['normalidade'] = resultado_dict
        
        if verbose:
            print("\nTeste Shapiro-Wilk:")
            print(f"  Estatística: {shapiro_stat:.4f}")
            print(f"  p-value: {shapiro_p:.4f}")
            print(f"  Normal? {'SIM' if shapiro_p > 0.05 else 'NÃO'}")
            
            print("\nTeste Jarque-Bera:")
            print(f"  Estatística: {jb_stat:.4f}")
            print(f"  p-value: {jb_p:.4f}")
            print(f"  Normal? {'SIM' if jb_p > 0.05 else 'NÃO'}")
            
            print("\nTeste Anderson-Darling:")
            print(f"  Estatística: {anderson_stat:.4f}")
            print(f"  Valor crítico (5%): {anderson_critical:.4f}")
            print(f"  Normal? {'SIM' if anderson_ok else 'NÃO'}")
            
            # Conclusão geral
            todos_ok = (shapiro_p > 0.05) and (jb_p > 0.05) and anderson_ok
            print(f"\nConclusão Geral: Resíduos são {'NORMALMENTE DISTRIBUÍDOS' if todos_ok else 'NÃO NORMALMENTE DISTRIBUÍDOS'}")
            if not todos_ok:
                print("  → Resíduos não normais podem indicar problemas no modelo")
                print("  → Mas isso não invalida o modelo se outros testes estiverem OK")
        
        return resultado_dict
    
    def teste_heterocedasticidade(self, verbose=True):
        """
        ETAPA 3.3: Teste de Heterocedasticidade (ARCH)
        
        Testa se a variância dos resíduos é constante (homocedasticidade).
        
        H0: Resíduos são homocedásticos (variância constante)
        H1: Resíduos são heterocedásticos (variância não constante)
        
        Parameters:
        -----------
        verbose : bool
            Se True, imprime resultados
            
        Returns:
        --------
        dict
            Resultados do teste ARCH
        """
        if self.residuos is None:
            if verbose:
                print("[ERRO] Modelo não foi estimado ainda.")
            return None
        
        if verbose:
            print("\n" + "=" * 80)
            print("ETAPA 3.3: TESTE DE HETEROCEDASTICIDADE (ARCH)")
            print("=" * 80)
        
        residuos_clean = self.residuos.dropna()
        
        try:
            # Teste ARCH (Engle)
            resultado = het_arch(residuos_clean, maxlag=5)
            
            lm_stat = resultado[0]
            lm_pvalue = resultado[1]
            f_stat = resultado[2]
            f_pvalue = resultado[3]
            
            # Homocedástico se p-value > 0.05
            is_homocedastico = lm_pvalue > 0.05
            
            resultado_dict = {
                'lm_statistic': lm_stat,
                'lm_pvalue': lm_pvalue,
                'f_statistic': f_stat,
                'f_pvalue': f_pvalue,
                'is_homocedastico': is_homocedastico
            }
            
            self.resultados['heterocedasticidade'] = resultado_dict
            
            if verbose:
                print(f"\nTeste LM (Lagrange Multiplier):")
                print(f"  Estatística: {lm_stat:.4f}")
                print(f"  p-value: {lm_pvalue:.4f}")
                print(f"\nTeste F:")
                print(f"  Estatística: {f_stat:.4f}")
                print(f"  p-value: {f_pvalue:.4f}")
                print(f"\nConclusão: Resíduos são {'HOMOCEDÁSTICOS (variância constante)' if is_homocedastico else 'HETEROCEDÁSTICOS (variância não constante)'}")
                if not is_homocedastico:
                    print("  → Considere usar modelos GARCH ou transformações na série")
            
            return resultado_dict
            
        except Exception as e:
            if verbose:
                print(f"[AVISO] Erro no teste ARCH: {str(e)}")
                print("  → Teste pode não ser aplicável para esta série")
            return None
    
    def analise_residuos_completa(self, verbose=True):
        """
        ETAPA 3.4: Análise Visual dos Resíduos
        
        Cria gráficos para análise visual dos resíduos:
        - Resíduos ao longo do tempo
        - Histograma dos resíduos
        - Q-Q plot (normalidade)
        - ACF dos resíduos
        
        Parameters:
        -----------
        verbose : bool
            Se True, imprime informações
            
        Returns:
        --------
        dict
            Informações sobre os resíduos
        """
        if self.residuos is None:
            if verbose:
                print("[ERRO] Modelo não foi estimado ainda.")
            return None
        
        if verbose:
            print("\n" + "=" * 80)
            print("ETAPA 3.4: ANÁLISE VISUAL DOS RESÍDUOS")
            print("=" * 80)
        
        residuos_clean = self.residuos.dropna()
        
        # Estatísticas descritivas
        stats_dict = {
            'media': residuos_clean.mean(),
            'desvio_padrao': residuos_clean.std(),
            'assimetria': stats.skew(residuos_clean),
            'curtose': stats.kurtosis(residuos_clean),
            'min': residuos_clean.min(),
            'max': residuos_clean.max()
        }
        
        self.resultados['residuos_stats'] = stats_dict
        
        if verbose:
            print("\nEstatísticas dos Resíduos:")
            print(f"  Média: {stats_dict['media']:.4f} (deve ser próxima de 0)")
            print(f"  Desvio padrão: {stats_dict['desvio_padrao']:.4f}")
            print(f"  Assimetria: {stats_dict['assimetria']:.4f} (0 = simétrico)")
            print(f"  Curtose: {stats_dict['curtose']:.4f} (3 = normal)")
            print(f"  Min: {stats_dict['min']:.4f}")
            print(f"  Max: {stats_dict['max']:.4f}")
        
        return stats_dict
    
    # ============================================================================
    # ETAPA 4: PREVISÃO
    # ============================================================================
    
    def gerar_previsao(self, n_periodos=30, intervalo_confianca=0.95, verbose=True):
        """
        ETAPA 4: Geração de Previsões
        
        Gera previsões futuras com intervalos de confiança.
        
        Parameters:
        -----------
        n_periodos : int
            Número de períodos a prever
        intervalo_confianca : float
            Nível de confiança (0.95 = 95%)
        verbose : bool
            Se True, imprime resultados
            
        Returns:
        --------
        dict
            Previsões e intervalos de confiança
        """
        if self.modelo is None:
            if verbose:
                print("[ERRO] Modelo não foi estimado ainda.")
            return None
        
        if verbose:
            print("\n" + "=" * 80)
            print("ETAPA 4: GERAÇÃO DE PREVISÕES")
            print("=" * 80)
        
        try:
            previsao, conf_int = self.modelo.predict(
                n_periods=n_periodos,
                return_conf_int=True,
                alpha=1 - intervalo_confianca
            )
            
            # Cria índice de datas futuras
            ultima_data = self.serie.index[-1]
            if isinstance(ultima_data, pd.Timestamp):
                datas_futuras = pd.date_range(
                    start=ultima_data + pd.Timedelta(days=1),
                    periods=n_periodos,
                    freq='D'
                )
            else:
                datas_futuras = range(len(self.serie), len(self.serie) + n_periodos)
            
            resultado_dict = {
                'previsao': previsao,
                'intervalo_inferior': conf_int[:, 0],
                'intervalo_superior': conf_int[:, 1],
                'datas': datas_futuras,
                'n_periodos': n_periodos,
                'intervalo_confianca': intervalo_confianca
            }
            
            self.resultados['previsao'] = resultado_dict
            
            if verbose:
                print(f"\n[OK] Previsão gerada para {n_periodos} períodos")
                print(f"Intervalo de confiança: {intervalo_confianca*100:.0f}%")
                print(f"\nEstatísticas da Previsão:")
                print(f"  Média: {previsao.mean():.2f}")
                print(f"  Mínimo: {previsao.min():.2f}")
                print(f"  Máximo: {previsao.max():.2f}")
            
            return resultado_dict
            
        except Exception as e:
            if verbose:
                print(f"[ERRO] Erro ao gerar previsão: {str(e)}")
            return None
    
    # ============================================================================
    # VISUALIZAÇÕES
    # ============================================================================
    
    def plotar_analise_completa(self, caminho_saida=None):
        """
        Cria visualizações completas da análise Box-Jenkins.
        
        Parameters:
        -----------
        caminho_saida : str, optional
            Caminho para salvar a figura
        """
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Série Original
        ax1 = plt.subplot(4, 3, 1)
        ax1.plot(self.serie.index, self.serie.values, linewidth=1.5, color='steelblue')
        ax1.set_title('1. Série Temporal Original', fontweight='bold')
        ax1.set_xlabel('Data')
        ax1.set_ylabel('Valor')
        ax1.grid(True, alpha=0.3)
        
        # 2. ACF
        ax2 = plt.subplot(4, 3, 2)
        if 'acf_pacf' in self.resultados:
            acf_vals = self.resultados['acf_pacf']['acf_values']
            plot_acf(self.serie.dropna(), lags=min(40, len(self.serie)-1), ax=ax2, alpha=0.05)
        ax2.set_title('2. ACF (Autocorrelação)', fontweight='bold')
        
        # 3. PACF
        ax3 = plt.subplot(4, 3, 3)
        if 'acf_pacf' in self.resultados:
            plot_pacf(self.serie.dropna(), lags=min(40, len(self.serie)-1), ax=ax3, alpha=0.05)
        ax3.set_title('3. PACF (Autocorrelação Parcial)', fontweight='bold')
        
        # 4. Decomposição - Tendência
        ax4 = plt.subplot(4, 3, 4)
        if 'decomposicao' in self.resultados:
            ax4.plot(self.resultados['decomposicao']['tendencia'], color='green')
            ax4.set_title('4. Decomposição - Tendência', fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        # 5. Decomposição - Sazonalidade
        ax5 = plt.subplot(4, 3, 5)
        if 'decomposicao' in self.resultados:
            ax5.plot(self.resultados['decomposicao']['sazonalidade'], color='orange')
            ax5.set_title('5. Decomposição - Sazonalidade', fontweight='bold')
            ax5.grid(True, alpha=0.3)
        
        # 6. Decomposição - Resíduo
        ax6 = plt.subplot(4, 3, 6)
        if 'decomposicao' in self.resultados:
            ax6.plot(self.resultados['decomposicao']['residuo'], color='red', alpha=0.7)
            ax6.set_title('6. Decomposição - Resíduo', fontweight='bold')
            ax6.grid(True, alpha=0.3)
        
        # 7. Resíduos do Modelo
        ax7 = plt.subplot(4, 3, 7)
        if self.residuos is not None:
            residuos_clean = self.residuos.dropna()
            ax7.plot(residuos_clean.index, residuos_clean.values, color='purple', alpha=0.7, linewidth=1)
            ax7.axhline(y=0, color='black', linestyle='--', linewidth=1)
            ax7.set_title('7. Resíduos do Modelo', fontweight='bold')
            ax7.set_xlabel('Data')
            ax7.set_ylabel('Resíduo')
            ax7.grid(True, alpha=0.3)
        
        # 8. Histograma dos Resíduos
        ax8 = plt.subplot(4, 3, 8)
        if self.residuos is not None:
            residuos_clean = self.residuos.dropna()
            ax8.hist(residuos_clean, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            ax8.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax8.set_title('8. Distribuição dos Resíduos', fontweight='bold')
            ax8.set_xlabel('Resíduo')
            ax8.set_ylabel('Frequência')
            ax8.grid(True, alpha=0.3, axis='y')
        
        # 9. Q-Q Plot (Normalidade)
        ax9 = plt.subplot(4, 3, 9)
        if self.residuos is not None:
            residuos_clean = self.residuos.dropna()
            stats.probplot(residuos_clean, dist="norm", plot=ax9)
            ax9.set_title('9. Q-Q Plot (Normalidade)', fontweight='bold')
            ax9.grid(True, alpha=0.3)
        
        # 10. ACF dos Resíduos
        ax10 = plt.subplot(4, 3, 10)
        if self.residuos is not None:
            residuos_clean = self.residuos.dropna()
            plot_acf(residuos_clean, lags=min(20, len(residuos_clean)-1), ax=ax10, alpha=0.05)
            ax10.set_title('10. ACF dos Resíduos', fontweight='bold')
        
        # 11. Previsão
        ax11 = plt.subplot(4, 3, 11)
        if 'previsao' in self.resultados:
            prev = self.resultados['previsao']
            # Últimos 90 dias da série original
            serie_recente = self.serie.iloc[-90:] if len(self.serie) > 90 else self.serie
            ax11.plot(serie_recente.index, serie_recente.values, label='Histórico', 
                     color='steelblue', linewidth=2)
            ax11.plot(prev['datas'], prev['previsao'], label='Previsão', 
                     color='red', linewidth=2, linestyle='--')
            ax11.fill_between(prev['datas'], prev['intervalo_inferior'], 
                             prev['intervalo_superior'], alpha=0.3, color='red', label='IC 95%')
            ax11.axvline(x=serie_recente.index[-1], color='black', linestyle=':', linewidth=2)
            ax11.set_title('11. Previsão com Intervalos de Confiança', fontweight='bold')
            ax11.set_xlabel('Data')
            ax11.set_ylabel('Valor')
            ax11.legend()
            ax11.grid(True, alpha=0.3)
        
        # 12. Resumo Estatístico (texto)
        ax12 = plt.subplot(4, 3, 12)
        ax12.axis('off')
        texto_resumo = self._gerar_texto_resumo()
        ax12.text(0.1, 0.5, texto_resumo, fontsize=10, verticalalignment='center',
                 family='monospace', transform=ax12.transAxes)
        ax12.set_title('12. Resumo Estatístico', fontweight='bold')
        
        plt.suptitle(f'Análise Box-Jenkins Completa - SKU: {self.sku}', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        if caminho_saida:
            plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
            print(f"\n[OK] Gráfico salvo: {caminho_saida}")
        else:
            nome_arquivo = f'analise_box_jenkins_{self.sku}.png'
            plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
            print(f"\n[OK] Gráfico salvo: {nome_arquivo}")
        
        plt.close()
    
    def _gerar_texto_resumo(self):
        """Gera texto resumo para o gráfico."""
        linhas = []
        linhas.append("RESUMO ESTATÍSTICO")
        linhas.append("=" * 40)
        
        if 'estacionariedade' in self.resultados:
            est = self.resultados['estacionariedade']
            linhas.append(f"\nEstacionariedade:")
            linhas.append(f"  ADF p-value: {est['p_value']:.4f}")
            linhas.append(f"  Estacionária: {'SIM' if est['is_stationary'] else 'NÃO'}")
        
        if 'modelo' in self.resultados:
            mod = self.resultados['modelo']
            linhas.append(f"\nModelo:")
            linhas.append(f"  ARIMA: {mod['order']}")
            if mod['seasonal_order']:
                linhas.append(f"  Sazonal: {mod['seasonal_order']}")
            linhas.append(f"  AIC: {mod['aic']:.2f}")
        
        if 'ljung_box' in self.resultados:
            lb = self.resultados['ljung_box']
            linhas.append(f"\nLjung-Box:")
            linhas.append(f"  p-value: {lb['p_value']:.4f}")
            linhas.append(f"  OK: {'SIM' if lb['residuos_ok'] else 'NÃO'}")
        
        if 'normalidade' in self.resultados:
            norm = self.resultados['normalidade']
            shapiro_ok = norm['shapiro_wilk']['is_normal']
            linhas.append(f"\nNormalidade:")
            linhas.append(f"  Shapiro: {'SIM' if shapiro_ok else 'NÃO'}")
        
        return "\n".join(linhas)
    
    # ============================================================================
    # EXECUÇÃO COMPLETA
    # ============================================================================
    
    def executar_analise_completa(self, periodo_sazonal=30, n_previsao=30, salvar_graficos=True):
        """
        Executa análise Box-Jenkins completa.
        
        Parameters:
        -----------
        periodo_sazonal : int
            Período sazonal para decomposição
        n_previsao : int
            Número de períodos a prever
        salvar_graficos : bool
            Se True, salva gráficos
            
        Returns:
        --------
        dict
            Todos os resultados da análise
        """
        print("\n" + "=" * 80)
        print("ANÁLISE BOX-JENKINS COMPLETA")
        print("=" * 80)
        print(f"\nSKU: {self.sku}")
        print(f"Observações: {len(self.serie)}")
        print(f"Período: {self.serie.index[0]} a {self.serie.index[-1]}")
        
        # ETAPA 1: IDENTIFICAÇÃO
        print("\n" + ">" * 80)
        print("ETAPA 1: IDENTIFICAÇÃO")
        print(">" * 80)
        self.teste_estacionariedade_adf()
        self.analise_acf_pacf()
        self.decomposicao_sazonal(periodo=periodo_sazonal)
        
        # ETAPA 2: ESTIMAÇÃO
        print("\n" + ">" * 80)
        print("ETAPA 2: ESTIMAÇÃO")
        print(">" * 80)
        self.estimar_modelo(seasonal=True, m=periodo_sazonal)
        
        if self.modelo is None:
            print("\n[ERRO] Não foi possível estimar o modelo. Análise interrompida.")
            return self.resultados
        
        # ETAPA 3: DIAGNÓSTICO
        print("\n" + ">" * 80)
        print("ETAPA 3: DIAGNÓSTICO")
        print(">" * 80)
        self.teste_ljung_box()
        self.teste_normalidade_residuos()
        self.teste_heterocedasticidade()
        self.analise_residuos_completa()
        
        # ETAPA 4: PREVISÃO
        print("\n" + ">" * 80)
        print("ETAPA 4: PREVISÃO")
        print(">" * 80)
        self.gerar_previsao(n_periodos=n_previsao)
        
        # Visualizações
        if salvar_graficos:
            print("\n" + ">" * 80)
            print("GERANDO VISUALIZAÇÕES")
            print(">" * 80)
            self.plotar_analise_completa()
        
        print("\n" + "=" * 80)
        print("ANÁLISE BOX-JENKINS CONCLUÍDA!")
        print("=" * 80)
        
        return self.resultados
    
    def gerar_relatorio_completo(self, caminho_saida=None):
        """
        Gera relatório textual completo com todos os resultados.
        
        Parameters:
        -----------
        caminho_saida : str, optional
            Caminho para salvar o relatório
        """
        linhas = []
        linhas.append("=" * 80)
        linhas.append("RELATÓRIO COMPLETO: ANÁLISE BOX-JENKINS")
        linhas.append("=" * 80)
        linhas.append(f"\nSKU: {self.sku}")
        linhas.append(f"Data da análise: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        linhas.append(f"Observações: {len(self.serie)}")
        linhas.append(f"Período: {self.serie.index[0]} a {self.serie.index[-1]}")
        
        # ETAPA 1
        linhas.append("\n" + "-" * 80)
        linhas.append("ETAPA 1: IDENTIFICAÇÃO")
        linhas.append("-" * 80)
        
        if 'estacionariedade' in self.resultados:
            est = self.resultados['estacionariedade']
            linhas.append("\n1.1 Teste de Estacionariedade (ADF):")
            linhas.append(f"   Estatística ADF: {est['adf_statistic']:.4f}")
            linhas.append(f"   p-value: {est['p_value']:.4f}")
            linhas.append(f"   Conclusão: {'Série é ESTACIONÁRIA' if est['is_stationary'] else 'Série NÃO é estacionária (necessita diferenciação)'}")
        
        if 'acf_pacf' in self.resultados:
            acf_pacf = self.resultados['acf_pacf']
            linhas.append("\n1.2 Análise ACF/PACF:")
            linhas.append(f"   Lags ACF significativos: {acf_pacf['acf_significativos']}")
            linhas.append(f"   Lags PACF significativos: {acf_pacf['pacf_significativos']}")
        
        if 'decomposicao' in self.resultados:
            dec = self.resultados['decomposicao']
            linhas.append(f"\n1.3 Decomposição Sazonal:")
            linhas.append(f"   Período usado: {dec['periodo']}")
        
        # ETAPA 2
        linhas.append("\n" + "-" * 80)
        linhas.append("ETAPA 2: ESTIMAÇÃO")
        linhas.append("-" * 80)
        
        if 'modelo' in self.resultados:
            mod = self.resultados['modelo']
            linhas.append(f"\nModelo SARIMA estimado:")
            linhas.append(f"   Ordem ARIMA: {mod['order']}")
            if mod['seasonal_order']:
                linhas.append(f"   Ordem Sazonal: {mod['seasonal_order']}")
            linhas.append(f"   AIC: {mod['aic']:.2f}")
            if mod['bic']:
                linhas.append(f"   BIC: {mod['bic']:.2f}")
        
        # ETAPA 3
        linhas.append("\n" + "-" * 80)
        linhas.append("ETAPA 3: DIAGNÓSTICO")
        linhas.append("-" * 80)
        
        if 'ljung_box' in self.resultados:
            lb = self.resultados['ljung_box']
            linhas.append("\n3.1 Teste de Ljung-Box:")
            linhas.append(f"   Estatística: {lb['estatistica']:.4f}")
            linhas.append(f"   p-value: {lb['p_value']:.4f}")
            linhas.append(f"   Conclusão: Resíduos são {'NÃO CORRELACIONADOS (modelo adequado)' if lb['residuos_ok'] else 'CORRELACIONADOS (modelo pode ser melhorado)'}")
        
        if 'normalidade' in self.resultados:
            norm = self.resultados['normalidade']
            linhas.append("\n3.2 Teste de Normalidade:")
            linhas.append(f"   Shapiro-Wilk p-value: {norm['shapiro_wilk']['p_value']:.4f}")
            linhas.append(f"   Jarque-Bera p-value: {norm['jarque_bera']['p_value']:.4f}")
            linhas.append(f"   Conclusão: Resíduos são {'NORMALMENTE DISTRIBUÍDOS' if norm['shapiro_wilk']['is_normal'] else 'NÃO NORMALMENTE DISTRIBUÍDOS'}")
        
        if 'heterocedasticidade' in self.resultados:
            het = self.resultados['heterocedasticidade']
            linhas.append("\n3.3 Teste de Heterocedasticidade:")
            linhas.append(f"   LM p-value: {het['lm_pvalue']:.4f}")
            linhas.append(f"   Conclusão: Resíduos são {'HOMOCEDÁSTICOS' if het['is_homocedastico'] else 'HETEROCEDÁSTICOS'}")
        
        if 'residuos_stats' in self.resultados:
            res = self.resultados['residuos_stats']
            linhas.append("\n3.4 Estatísticas dos Resíduos:")
            linhas.append(f"   Média: {res['media']:.4f}")
            linhas.append(f"   Desvio padrão: {res['desvio_padrao']:.4f}")
            linhas.append(f"   Assimetria: {res['assimetria']:.4f}")
            linhas.append(f"   Curtose: {res['curtose']:.4f}")
        
        # ETAPA 4
        linhas.append("\n" + "-" * 80)
        linhas.append("ETAPA 4: PREVISÃO")
        linhas.append("-" * 80)
        
        if 'previsao' in self.resultados:
            prev = self.resultados['previsao']
            linhas.append(f"\nPrevisão gerada:")
            linhas.append(f"   Períodos: {prev['n_periodos']}")
            linhas.append(f"   Intervalo de confiança: {prev['intervalo_confianca']*100:.0f}%")
            linhas.append(f"   Média prevista: {prev['previsao'].mean():.2f}")
            linhas.append(f"   Mínimo previsto: {prev['previsao'].min():.2f}")
            linhas.append(f"   Máximo previsto: {prev['previsao'].max():.2f}")
        
        # CONCLUSÃO
        linhas.append("\n" + "=" * 80)
        linhas.append("CONCLUSÃO GERAL")
        linhas.append("=" * 80)
        
        # Avalia qualidade do modelo
        qualidade = "BOM"
        problemas = []
        
        if 'ljung_box' in self.resultados:
            if not self.resultados['ljung_box']['residuos_ok']:
                qualidade = "MODERADO"
                problemas.append("Resíduos correlacionados")
        
        if 'normalidade' in self.resultados:
            if not self.resultados['normalidade']['shapiro_wilk']['is_normal']:
                problemas.append("Resíduos não normais")
        
        if 'heterocedasticidade' in self.resultados:
            if not self.resultados['heterocedasticidade']['is_homocedastico']:
                qualidade = "MODERADO"
                problemas.append("Heterocedasticidade detectada")
        
        if len(problemas) == 0:
            linhas.append("\n✓ Modelo apresenta qualidade BOM")
            linhas.append("  Todos os testes estatísticos indicam adequação do modelo.")
        else:
            linhas.append(f"\n⚠ Modelo apresenta qualidade {qualidade}")
            linhas.append("  Problemas detectados:")
            for problema in problemas:
                linhas.append(f"    - {problema}")
            linhas.append("\n  Recomendações:")
            linhas.append("    - Considere aumentar a ordem do modelo")
            linhas.append("    - Verifique se há outliers ou eventos especiais")
            linhas.append("    - Considere transformações na série (log, diferença)")
        
        texto = "\n".join(linhas)
        
        if caminho_saida:
            with open(caminho_saida, 'w', encoding='utf-8') as f:
                f.write(texto)
            print(f"\n[OK] Relatório salvo: {caminho_saida}")
        else:
            nome_arquivo = f'relatorio_box_jenkins_{self.sku}.txt'
            with open(nome_arquivo, 'w', encoding='utf-8') as f:
                f.write(texto)
            print(f"\n[OK] Relatório salvo: {nome_arquivo}")
        
        return texto


def main():
    """
    Exemplo de uso da análise Box-Jenkins completa.
    """
    print("=" * 80)
    print("ANÁLISE BOX-JENKINS PARA MODELO SARIMA")
    print("=" * 80)
    
    # Carrega dados
    print("\nCarregando dados...")
    try:
        df = pd.read_csv('DB/historico_estoque_atual_processado.csv')
        df['data'] = pd.to_datetime(df['data'])
    except FileNotFoundError:
        print("[ERRO] Arquivo de dados não encontrado.")
        print("  Execute primeiro o processamento dos dados.")
        return
    
    # Seleciona SKU com mais observações
    stats = df.groupby('sku')['estoque_atual'].agg(['count', 'mean', 'std']).reset_index()
    stats = stats[stats['mean'] >= 1.0].copy()
    stats['cv'] = stats['std'] / stats['mean']
    stats['score'] = stats['count'] * stats['cv'] * stats['mean']
    stats = stats.sort_values('score', ascending=False)
    
    sku_selecionado = stats.iloc[0]['sku']
    print(f"\nSKU selecionado: {sku_selecionado}")
    
    # Prepara série temporal
    df_sku = df[df['sku'] == sku_selecionado].copy()
    df_sku = df_sku.sort_values('data')
    df_sku = df_sku.set_index('data')
    serie = df_sku['estoque_atual'].asfreq('D', method='ffill').dropna()
    
    print(f"Observações: {len(serie)}")
    print(f"Período: {serie.index[0].date()} a {serie.index[-1].date()}")
    
    # Executa análise completa
    analise = AnaliseBoxJenkins(serie, sku=sku_selecionado)
    resultados = analise.executar_analise_completa(
        periodo_sazonal=30,
        n_previsao=30,
        salvar_graficos=True
    )
    
    # Gera relatório
    analise.gerar_relatorio_completo()
    
    print("\n" + "=" * 80)
    print("ANÁLISE CONCLUÍDA!")
    print("=" * 80)
    print("\nArquivos gerados:")
    print(f"  - analise_box_jenkins_{sku_selecionado}.png")
    print(f"  - relatorio_box_jenkins_{sku_selecionado}.txt")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERRO] Erro durante execução: {str(e)}")
        import traceback
        traceback.print_exc()

