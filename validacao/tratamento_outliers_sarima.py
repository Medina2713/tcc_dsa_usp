"""
Tratamento de Outliers para Séries Temporais SARIMA
TCC MBA Data Science & Analytics - E-commerce de Brinquedos

Identifica e trata outliers em séries temporais antes de aplicar modelos SARIMA.
Outliers podem distorcer previsões, especialmente em eventos especiais como
Dia das Crianças ou Black Friday.

Métodos implementados:
1. Método IQR (Interquartile Range)
2. Método Z-Score
3. Método de Suavização (para não perder dados)

Autor: Medina2713
Data: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')


class TratamentoOutliers:
    """
    Classe para identificar e tratar outliers em séries temporais.
    """
    
    def __init__(self, serie):
        """
        Inicializa o tratamento de outliers.
        
        Parameters:
        -----------
        serie : pd.Series
            Série temporal a ser analisada
        """
        self.serie_original = serie.copy()
        self.serie = serie.copy()
        self.outliers_detectados = None
        self.metodo_usado = None
        
    def identificar_outliers_iqr(self, fator=1.5):
        """
        Identifica outliers usando o método IQR (Interquartile Range).
        
        Outlier = valor < Q1 - fator*IQR ou valor > Q3 + fator*IQR
        
        Parameters:
        -----------
        fator : float
            Fator multiplicador do IQR (padrão: 1.5)
            
        Returns:
        --------
        pd.Series
            Série booleana indicando outliers (True = outlier)
        """
        Q1 = self.serie.quantile(0.25)
        Q3 = self.serie.quantile(0.75)
        IQR = Q3 - Q1
        
        limite_inferior = Q1 - fator * IQR
        limite_superior = Q3 + fator * IQR
        
        outliers = (self.serie < limite_inferior) | (self.serie > limite_superior)
        
        self.outliers_detectados = outliers
        self.metodo_usado = 'IQR'
        
        return outliers
    
    def identificar_outliers_zscore(self, limite=3.0):
        """
        Identifica outliers usando Z-Score.
        
        Outlier = |z-score| > limite
        
        Parameters:
        -----------
        limite : float
            Limite do z-score (padrão: 3.0 = 3 desvios padrão)
            
        Returns:
        --------
        pd.Series
            Série booleana indicando outliers (True = outlier)
        """
        z_scores = np.abs(stats.zscore(self.serie.dropna()))
        
        # Cria série com mesmo índice
        outliers = pd.Series(False, index=self.serie.index)
        outliers.loc[self.serie.dropna().index] = z_scores > limite
        
        self.outliers_detectados = outliers
        self.metodo_usado = 'Z-Score'
        
        return outliers
    
    def remover_outliers(self, metodo='iqr', fator=1.5, limite_zscore=3.0):
        """
        Remove outliers da série.
        
        Parameters:
        -----------
        metodo : str
            Método a usar ('iqr' ou 'zscore')
        fator : float
            Fator para IQR (se método='iqr')
        limite_zscore : float
            Limite para Z-Score (se método='zscore')
            
        Returns:
        --------
        pd.Series
            Série sem outliers (valores substituídos por NaN)
        """
        if metodo == 'iqr':
            outliers = self.identificar_outliers_iqr(fator=fator)
        elif metodo == 'zscore':
            outliers = self.identificar_outliers_zscore(limite=limite_zscore)
        else:
            raise ValueError("Método deve ser 'iqr' ou 'zscore'")
        
        serie_sem_outliers = self.serie.copy()
        serie_sem_outliers[outliers] = np.nan
        
        return serie_sem_outliers
    
    def substituir_outliers_mediana(self, metodo='iqr', fator=1.5, limite_zscore=3.0):
        """
        Substitui outliers pela mediana da série.
        
        Parameters:
        -----------
        metodo : str
            Método a usar ('iqr' ou 'zscore')
        fator : float
            Fator para IQR (se método='iqr')
        limite_zscore : float
            Limite para Z-Score (se método='zscore')
            
        Returns:
        --------
        pd.Series
            Série com outliers substituídos pela mediana
        """
        if metodo == 'iqr':
            outliers = self.identificar_outliers_iqr(fator=fator)
        elif metodo == 'zscore':
            outliers = self.identificar_outliers_zscore(limite=limite_zscore)
        else:
            raise ValueError("Método deve ser 'iqr' ou 'zscore'")
        
        mediana = self.serie.median()
        serie_tratada = self.serie.copy()
        serie_tratada[outliers] = mediana
        
        self.serie = serie_tratada
        return serie_tratada
    
    def substituir_outliers_suavizacao(self, metodo='iqr', fator=1.5, limite_zscore=3.0, 
                                      janela=5):
        """
        Substitui outliers por valores suavizados (média móvel).
        
        Útil quando não queremos perder informação, apenas suavizar picos.
        
        Parameters:
        -----------
        metodo : str
            Método a usar ('iqr' ou 'zscore')
        fator : float
            Fator para IQR (se método='iqr')
        limite_zscore : float
            Limite para Z-Score (se método='zscore')
        janela : int
            Janela para média móvel
            
        Returns:
        --------
        pd.Series
            Série com outliers substituídos por valores suavizados
        """
        if metodo == 'iqr':
            outliers = self.identificar_outliers_iqr(fator=fator)
        elif metodo == 'zscore':
            outliers = self.identificar_outliers_zscore(limite=limite_zscore)
        else:
            raise ValueError("Método deve ser 'iqr' ou 'zscore'")
        
        serie_tratada = self.serie.copy()
        
        # Calcula média móvel
        media_movel = self.serie.rolling(window=janela, center=True, min_periods=1).mean()
        
        # Substitui outliers
        serie_tratada[outliers] = media_movel[outliers]
        
        self.serie = serie_tratada
        return serie_tratada
    
    def estatisticas_outliers(self):
        """
        Retorna estatísticas sobre outliers detectados.
        
        Returns:
        --------
        dict
            Estatísticas sobre outliers
        """
        if self.outliers_detectados is None:
            return None
        
        n_outliers = self.outliers_detectados.sum()
        n_total = len(self.serie)
        percentual = (n_outliers / n_total) * 100
        
        if n_outliers > 0:
            valores_outliers = self.serie_original[self.outliers_detectados]
            stats_dict = {
                'total_outliers': n_outliers,
                'percentual': percentual,
                'metodo': self.metodo_usado,
                'valor_min_outlier': valores_outliers.min(),
                'valor_max_outlier': valores_outliers.max(),
                'valor_medio_outlier': valores_outliers.mean(),
                'indices_outliers': valores_outliers.index.tolist()
            }
        else:
            stats_dict = {
                'total_outliers': 0,
                'percentual': 0,
                'metodo': self.metodo_usado
            }
        
        return stats_dict
    
    def plotar_comparacao(self, caminho_saida=None):
        """
        Cria gráfico comparando série original vs tratada.
        
        Parameters:
        -----------
        caminho_saida : str, optional
            Caminho para salvar a figura
        """
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # Gráfico 1: Série original com outliers destacados
        ax1 = axes[0]
        ax1.plot(self.serie_original.index, self.serie_original.values, 
                label='Série Original', color='steelblue', linewidth=1.5, alpha=0.7)
        
        if self.outliers_detectados is not None and self.outliers_detectados.sum() > 0:
            outliers_vals = self.serie_original[self.outliers_detectados]
            ax1.scatter(outliers_vals.index, outliers_vals.values, 
                       color='red', s=50, zorder=5, label='Outliers Detectados', 
                       marker='x', linewidths=2)
        
        ax1.set_title('Série Original com Outliers Destacados', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Data')
        ax1.set_ylabel('Valor')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Comparação original vs tratada
        ax2 = axes[1]
        ax2.plot(self.serie_original.index, self.serie_original.values, 
                label='Original', color='steelblue', linewidth=1.5, alpha=0.5)
        ax2.plot(self.serie.index, self.serie.values, 
                label='Tratada', color='green', linewidth=2, alpha=0.8)
        ax2.set_title('Comparação: Original vs Tratada', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Data')
        ax2.set_ylabel('Valor')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Tratamento de Outliers - Método: {self.metodo_usado or "N/A"}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if caminho_saida:
            plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
            print(f"\n[OK] Gráfico salvo: {caminho_saida}")
        else:
            nome_arquivo = 'tratamento_outliers.png'
            plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
            print(f"\n[OK] Gráfico salvo: {nome_arquivo}")
        
        plt.close()


def main():
    """
    Exemplo de uso do tratamento de outliers.
    """
    print("=" * 80)
    print("TRATAMENTO DE OUTLIERS PARA SÉRIES TEMPORAIS")
    print("=" * 80)
    
    # Carrega dados
    print("\nCarregando dados...")
    try:
        df = pd.read_csv('DB/historico_estoque_atual_processado.csv')
        df['data'] = pd.to_datetime(df['data'])
    except FileNotFoundError:
        print("[ERRO] Arquivo de dados não encontrado.")
        return
    
    # Seleciona SKU
    stats = df.groupby('sku')['estoque_atual'].agg(['count', 'mean', 'std']).reset_index()
    stats = stats[stats['mean'] >= 1.0].copy()
    stats['cv'] = stats['std'] / stats['mean']
    stats['score'] = stats['count'] * stats['cv'] * stats['mean']
    stats = stats.sort_values('score', ascending=False)
    
    sku_selecionado = stats.iloc[0]['sku']
    print(f"\nSKU selecionado: {sku_selecionado}")
    
    # Prepara série temporal
    df_sku = df[df['sku'] == sku_selecionado].copy()
    df_sku = df_sku.sort_values('data').set_index('data')
    serie = df_sku['estoque_atual'].asfreq('D', method='ffill').dropna()
    
    print(f"Observações: {len(serie)}")
    
    # Trata outliers
    tratamento = TratamentoOutliers(serie)
    
    # Identifica outliers
    print("\n" + "=" * 80)
    print("IDENTIFICAÇÃO DE OUTLIERS")
    print("=" * 80)
    
    outliers_iqr = tratamento.identificar_outliers_iqr(fator=1.5)
    stats_iqr = tratamento.estatisticas_outliers()
    
    print(f"\nMétodo: IQR (fator=1.5)")
    print(f"Outliers detectados: {stats_iqr['total_outliers']} ({stats_iqr['percentual']:.2f}%)")
    
    if stats_iqr['total_outliers'] > 0:
        print(f"Valor mínimo outlier: {stats_iqr['valor_min_outlier']:.2f}")
        print(f"Valor máximo outlier: {stats_iqr['valor_max_outlier']:.2f}")
        print(f"Valor médio dos outliers: {stats_iqr['valor_medio_outlier']:.2f}")
    
    # Trata usando suavização (preserva dados)
    print("\n" + "=" * 80)
    print("TRATAMENTO DE OUTLIERS (SUAVIZAÇÃO)")
    print("=" * 80)
    
    serie_tratada = tratamento.substituir_outliers_suavizacao(
        metodo='iqr', 
        fator=1.5, 
        janela=5
    )
    
    print(f"[OK] Outliers substituídos por valores suavizados")
    print(f"Observações tratadas: {stats_iqr['total_outliers']}")
    
    # Visualizações
    tratamento.plotar_comparacao()
    
    print("\n" + "=" * 80)
    print("TRATAMENTO CONCLUÍDO!")
    print("=" * 80)
    print("\nArquivos gerados:")
    print("  - tratamento_outliers.png")
    print("=" * 80)
    
    return serie_tratada


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERRO] Erro durante execução: {str(e)}")
        import traceback
        traceback.print_exc()

