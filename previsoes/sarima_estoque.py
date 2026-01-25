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
import pickle
import hashlib
from pathlib import Path
warnings.filterwarnings('ignore')


class PrevisorEstoqueSARIMA:
    """
    Classe para gerar previsões de estoque usando modelo SARIMA.
    
    Utiliza auto_arima para encontrar automaticamente os melhores parâmetros
    (p, d, q) x (P, D, Q, s) para cada produto/SKU.
    """
    
    def __init__(self, horizonte_previsao=7, frequencia='D', cache_dir='cache_modelos'):
        """
        Inicializa o previsor.
        
        Parameters:
        -----------
        horizonte_previsao : int
            Número de dias à frente para prever (padrão: 7)
        frequencia : str
            Frequência da série temporal ('D' para diária, 'W' para semanal)
        cache_dir : str
            Diretório para armazenar modelos em cache
        """
        self.horizonte_previsao = horizonte_previsao
        self.frequencia = frequencia
        self.modelos = {}  # Armazena modelos treinados por SKU
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.series_cache = {}  # Cache de séries temporais preparadas
        
    
    def preparar_serie_temporal(self, df_estoque, sku, usar_cache=True):
        """
        Prepara a série temporal de estoque para um SKU específico.
        Usa cache para evitar reprocessamento.
        
        Parameters:
        -----------
        df_estoque : pd.DataFrame
            DataFrame com colunas: 'data' (datetime), 'sku', 'estoque_atual'
        sku : str
            Código do produto (SKU)
        usar_cache : bool
            Se True, usa cache de séries preparadas
            
        Returns:
        --------
        pd.Series
            Série temporal de estoque com índice datetime
        """
        # Verifica cache
        if usar_cache and sku in self.series_cache:
            return self.series_cache[sku]
        
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
        
        # Armazena no cache
        if usar_cache:
            self.series_cache[sku] = serie
        
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
    
    
    def _calcular_hash_serie(self, serie):
        """Calcula hash da série para validação de cache"""
        # Usa primeiros e últimos valores + tamanho para hash rápido
        hash_data = f"{len(serie)}_{serie.iloc[0]}_{serie.iloc[-1]}_{serie.sum()}"
        return hashlib.md5(hash_data.encode()).hexdigest()
    
    def _caminho_cache_modelo(self, sku):
        """Retorna caminho do arquivo de cache do modelo"""
        return self.cache_dir / f"modelo_{sku}.pkl"
    
    def _caminho_cache_metadata(self, sku):
        """Retorna caminho do arquivo de metadata do cache"""
        return self.cache_dir / f"metadata_{sku}.pkl"
    
    def carregar_modelo_cache(self, sku, serie):
        """
        Carrega modelo do cache se existir e for válido.
        
        Parameters:
        -----------
        sku : str
            Código do produto
        serie : pd.Series
            Série temporal atual (para validar cache)
            
        Returns:
        --------
        modelo ou None
            Modelo carregado do cache ou None se não existir/inválido
        """
        cache_path = self._caminho_cache_modelo(sku)
        metadata_path = self._caminho_cache_metadata(sku)
        
        if not cache_path.exists() or not metadata_path.exists():
            return None
        
        try:
            # Carrega metadata
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # Valida hash da série
            hash_atual = self._calcular_hash_serie(serie)
            if metadata.get('hash_serie') != hash_atual:
                # Série mudou, cache inválido
                return None
            
            # Carrega modelo
            with open(cache_path, 'rb') as f:
                modelo = pickle.load(f)
            
            return modelo
        except Exception as e:
            print(f"[AVISO] Erro ao carregar cache para SKU {sku}: {str(e)}")
            return None
    
    def salvar_modelo_cache(self, sku, modelo, serie):
        """
        Salva modelo no cache.
        
        Parameters:
        -----------
        sku : str
            Código do produto
        modelo : modelo_fit
            Modelo treinado
        serie : pd.Series
            Série temporal usada (para validar cache depois)
        """
        try:
            cache_path = self._caminho_cache_modelo(sku)
            metadata_path = self._caminho_cache_metadata(sku)
            
            # Salva modelo
            with open(cache_path, 'wb') as f:
                pickle.dump(modelo, f)
            
            # Salva metadata
            metadata = {
                'hash_serie': self._calcular_hash_serie(serie),
                'len_serie': len(serie),
                'order': modelo.order,
                'seasonal_order': modelo.seasonal_order
            }
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
        except Exception as e:
            print(f"[AVISO] Erro ao salvar cache para SKU {sku}: {str(e)}")
    
    def treinar_modelo(self, serie, sku, usar_cache=True):
        """
        Treina modelo SARIMA usando auto_arima para encontrar melhores parâmetros.
        Usa cache para evitar retreinamento.
        
        Parameters:
        -----------
        serie : pd.Series
            Série temporal de estoque
        sku : str
            Código do produto (SKU)
        usar_cache : bool
            Se True, tenta carregar do cache antes de treinar
            
        Returns:
        --------
        modelo_fit
            Modelo SARIMA treinado
        """
        # Verifica se há dados suficientes (mínimo de 30 observações recomendado)
        if len(serie) < 30:
            print(f"[AVISO] SKU {sku}: Dados insuficientes ({len(serie)} observacoes). Minimo: 30")
            return None
        
        # Tenta carregar do cache
        if usar_cache:
            modelo_cache = self.carregar_modelo_cache(sku, serie)
            if modelo_cache is not None:
                print(f"[CACHE] SKU {sku}: Modelo carregado do cache - {modelo_cache.order} x {modelo_cache.seasonal_order}")
                self.modelos[sku] = modelo_cache
                return modelo_cache
        
        try:
            # AUTO-ARIMA: Busca automática dos melhores parâmetros
            modelo = auto_arima(
                serie,
                seasonal=True,           # Ativa componente sazonal (SARIMA)
                m=30,                    # Período sazonal: 30 dias (mensal)
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
            
            # Salva no cache
            if usar_cache:
                self.salvar_modelo_cache(sku, modelo, serie)
            
            # Armazena em memória
            self.modelos[sku] = modelo
            
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


