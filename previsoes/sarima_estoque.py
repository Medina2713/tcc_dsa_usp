"""
Script de Previsão de Estoque usando SARIMA (Auto-ARIMA)
TCC MBA Data Science & Analytics - E-commerce de Brinquedos

Este módulo implementa previsões de estoque futuro usando o modelo SARIMA
com busca automática de parâmetros via pmdarima.auto_arima.

Autor: [Seu Nome]
Data: 2024
"""

import pandas as pd
import numpy as np
from pmdarima import auto_arima
from pmdarima.arima import ADFTest
import warnings
warnings.filterwarnings('ignore')


class PrevisorEstoqueSARIMA:
    """
    Classe para gerar previsões de estoque usando modelo SARIMA.
    
    Utiliza auto_arima para encontrar automaticamente os melhores parâmetros
    (p, d, q) x (P, D, Q, s) para cada produto/SKU.
    """
    
    def __init__(self, horizonte_previsao=7, frequencia='D'):
        """
        Inicializa o previsor.
        
        Parameters:
        -----------
        horizonte_previsao : int
            Número de dias à frente para prever (padrão: 7)
        frequencia : str
            Frequência da série temporal ('D' para diária, 'W' para semanal)
        """
        self.horizonte_previsao = horizonte_previsao
        self.frequencia = frequencia
        self.modelos = {}  # Armazena modelos treinados por SKU
        
    
    def preparar_serie_temporal(self, df_estoque, sku):
        """
        Prepara a série temporal de estoque para um SKU específico.
        
        Parameters:
        -----------
        df_estoque : pd.DataFrame
            DataFrame com colunas: 'data' (datetime), 'sku', 'estoque_atual'
        sku : str
            Código do produto (SKU)
            
        Returns:
        --------
        pd.Series
            Série temporal de estoque com índice datetime
        """
        # Filtra dados do SKU
        df_sku = df_estoque[df_estoque['sku'] == sku].copy()
        
        # Garante que a coluna de data é datetime
        df_sku['data'] = pd.to_datetime(df_sku['data'])
        
        # Ordena por data
        df_sku = df_sku.sort_values('data')
        
        # Remove duplicatas mantendo a última ocorrência
        df_sku = df_sku.drop_duplicates(subset='data', keep='last')
        
        # Define data como índice
        df_sku = df_sku.set_index('data')
        
        # Cria série temporal
        serie = df_sku['estoque_atual'].asfreq(self.frequencia, method='ffill')
        
        # Remove valores NaN no início (se houver)
        serie = serie.dropna()
        
        return serie
    
    
    def verificar_estacionariedade(self, serie):
        """
        Verifica se a série temporal é estacionária usando o Teste ADF.
        
        Parameters:
        -----------
        serie : pd.Series
            Série temporal a ser testada
            
        Returns:
        --------
        bool
            True se estacionária, False caso contrário
        """
        if len(serie) < 10:
            return False
        
        adf_test = ADFTest(alpha=0.05)
        is_stationary, _ = adf_test.should_diff(serie)
        
        return is_stationary
    
    
    def treinar_modelo(self, serie, sku):
        """
        Treina modelo SARIMA usando auto_arima para encontrar melhores parâmetros.
        
        Parameters:
        -----------
        serie : pd.Series
            Série temporal de estoque
        sku : str
            Código do produto (SKU)
            
        Returns:
        --------
        modelo_fit
            Modelo SARIMA treinado
        """
        # Verifica se há dados suficientes (mínimo de 30 observações recomendado)
        if len(serie) < 30:
            print(f"[AVISO] SKU {sku}: Dados insuficientes ({len(serie)} observacoes). Minimo: 30")
            return None
        
        try:
            # AUTO-ARIMA: Busca automática dos melhores parâmetros
            # 
            # Parâmetros explicados:
            # - seasonal=True: Habilita componente sazonal (importante para estoque)
            # - m=7: Período sazonal de 7 dias (semanal) - ajuste conforme seu caso
            # - stepwise=True: Busca eficiente (mais rápido)
            # - suppress_warnings=True: Suprime warnings durante a busca
            # - error_action='ignore': Ignora erros durante busca
            # - max_p, max_d, max_q: Limites superiores para parâmetros ARIMA
            # - max_P, max_D, max_Q: Limites superiores para parâmetros sazonais
            # - trace=True: Mostra progresso da busca (útil para debug)
            
            # Para sazonalidade mensal em dados diários: m=30 (aproximadamente 30 dias = 1 mês)
            # Isso captura padrões que se repetem mensalmente (ex: outubro e dezembro)
            modelo = auto_arima(
                serie,
                seasonal=True,           # Ativa componente sazonal (SARIMA)
                m=30,                    # Período sazonal: 30 dias (mensal) - captura padrões de out/dez
                stepwise=True,           # Busca eficiente (stepwise selection)
                suppress_warnings=True,  # Suprime warnings
                error_action='ignore',   # Ignora erros na busca
                
                # Limites para parâmetros não-sazonais (p, d, q)
                max_p=5,                 # Ordem máxima do componente AR
                max_d=2,                 # Máximo de diferenciações
                max_q=5,                 # Ordem máxima do componente MA
                
                # Limites para parâmetros sazonais (P, D, Q)
                max_P=2,                 # Ordem máxima do AR sazonal
                max_D=1,                 # Máximo de diferenciações sazonais
                max_Q=2,                 # Ordem máxima do MA sazonal
                
                # Critério de seleção
                information_criterion='aic',  # AIC, AICc, ou BIC
                
                # Outros
                trace=False,             # Mude para True para ver progresso
                n_jobs=-1                # Usa todos os cores disponíveis
            )
            
            print(f"[OK] SKU {sku}: Modelo encontrado - {modelo.order} x {modelo.seasonal_order}")
            
            return modelo
            
        except Exception as e:
            print(f"[ERRO] SKU {sku}: Erro ao treinar modelo - {str(e)}")
            return None
    
    
    def prever(self, serie, modelo=None, sku=None):
        """
        Gera previsão de estoque futuro.
        
        Parameters:
        -----------
        serie : pd.Series
            Série temporal de estoque (usada se modelo não fornecido)
        modelo : modelo_fit, optional
            Modelo SARIMA já treinado
        sku : str, optional
            Código do produto (para usar modelo já treinado)
            
        Returns:
        --------
        pd.Series
            Previsões com índice de datas futuras
        """
        # Se SKU fornecido, tenta usar modelo já treinado
        if sku and sku in self.modelos:
            modelo = self.modelos[sku]
        
        # Se não há modelo, treina um novo
        if modelo is None:
            if sku:
                modelo = self.treinar_modelo(serie, sku)
                if modelo:
                    self.modelos[sku] = modelo
            else:
                raise ValueError("Forneça um modelo treinado ou um SKU válido")
        
        if modelo is None:
            return None
        
        try:
            # Gera previsão
            previsao, intervalo_confianca = modelo.predict(
                n_periods=self.horizonte_previsao,
                return_conf_int=True
            )
            
            # Cria índice de datas futuras
            ultima_data = serie.index[-1]
            datas_futuras = pd.date_range(
                start=ultima_data + pd.Timedelta(days=1),
                periods=self.horizonte_previsao,
                freq=self.frequencia
            )
            
            # Cria série com previsões
            previsao_serie = pd.Series(
                previsao,
                index=datas_futuras,
                name='estoque_previsto'
            )
            
            # Garante valores não-negativos (estoque não pode ser negativo)
            previsao_serie = previsao_serie.clip(lower=0)
            
            return previsao_serie
            
        except Exception as e:
            print(f"[ERRO] Erro ao gerar previsao: {str(e)}")
            return None
    
    
    def processar_lote(self, df_estoque, lista_skus=None):
        """
        Processa previsões para múltiplos SKUs.
        
        Parameters:
        -----------
        df_estoque : pd.DataFrame
            DataFrame com dados de estoque
        lista_skus : list, optional
            Lista de SKUs para processar. Se None, processa todos.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame com previsões para todos os SKUs
        """
        # Se lista não fornecida, pega todos os SKUs únicos
        if lista_skus is None:
            lista_skus = df_estoque['sku'].unique().tolist()
        
        resultados = []
        
        for sku in lista_skus:
            # Prepara série temporal
            serie = self.preparar_serie_temporal(df_estoque, sku)
            
            if len(serie) < 30:
                print(f"[AVISO] SKU {sku}: Dados insuficientes. Pulando...")
                continue
            
            # Treina modelo
            modelo = self.treinar_modelo(serie, sku)
            
            if modelo is None:
                continue
            
            # Armazena modelo
            self.modelos[sku] = modelo
            
            # Gera previsão
            previsao = self.prever(serie, modelo=modelo)
            
            if previsao is not None:
                # Prepara resultado
                resultado = pd.DataFrame({
                    'sku': sku,
                    'data': previsao.index,
                    'estoque_previsto': previsao.values,
                    'estoque_atual': serie.iloc[-1]  # Último valor conhecido
                })
                
                resultados.append(resultado)
        
        # Concatena todos os resultados
        if resultados:
            df_resultado = pd.concat(resultados, ignore_index=True)
            return df_resultado
        else:
            return pd.DataFrame()
    
    
    def calcular_risco_ruptura(self, previsao, estoque_minimo):
        """
        Calcula risco de ruptura baseado na previsão.
        
        Parameters:
        -----------
        previsao : pd.Series
            Série com previsões de estoque
        estoque_minimo : float
            Estoque mínimo desejado
            
        Returns:
        --------
        float
            Score de risco (0 a 1, onde 1 é alto risco)
        """
        if previsao is None or len(previsao) == 0:
            return 1.0  # Alto risco se não há previsão
        
        # Média das previsões
        estoque_medio_previsto = previsao.mean()
        
        # Se estoque médio previsto está abaixo do mínimo, há risco
        if estoque_medio_previsto < estoque_minimo:
            # Calcula score de risco (normalizado)
            risco = min(1.0, (estoque_minimo - estoque_medio_previsto) / estoque_minimo)
            return risco
        else:
            return 0.0


if __name__ == "__main__":
    # Exemplo de uso básico
    print("=" * 60)
    print("Módulo de Previsão de Estoque - SARIMA (Auto-ARIMA)")
    print("=" * 60)
    print("\nEste módulo deve ser importado e utilizado conforme exemplo:")
    print("\n  from sarima_estoque import PrevisorEstoqueSARIMA")
    print("  previsor = PrevisorEstoqueSARIMA(horizonte_previsao=7)")
    print("  resultados = previsor.processar_lote(df_estoque)")
    print("\nVeja o arquivo 'exemplo_uso_sarima.py' para exemplos completos.")
    print("=" * 60)


