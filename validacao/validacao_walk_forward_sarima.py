"""
Validação Cruzada Walk-Forward para Modelo SARIMA
TCC MBA Data Science & Analytics - E-commerce de Brinquedos

Implementa validação cruzada de janela expandida (Walk-Forward) para séries temporais.
Diferente de modelos comuns, em séries temporais a ordem importa:
- Treina com meses 1-6, testa no mês 7
- Treina com meses 1-7, testa no mês 8
- E assim por diante...

Isso garante que o modelo seja estável ao longo do tempo antes de ser usado no ranking.

Autor: Medina2713
Data: 2024
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')


def calcular_mape(y_real, y_previsto):
    """
    Calcula Mean Absolute Percentage Error (MAPE).
    
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


class ValidacaoWalkForward:
    """
    Classe para realizar validação cruzada walk-forward em séries temporais.
    
    Walk-Forward é o método correto para validar modelos de séries temporais,
    pois respeita a ordem temporal dos dados.
    """
    
    def __init__(self, serie, tamanho_treino_inicial=0.7, tamanho_teste=0.1, 
                 passo=1, periodo_sazonal=30):
        """
        Inicializa a validação walk-forward.
        
        Parameters:
        -----------
        serie : pd.Series
            Série temporal completa
        tamanho_treino_inicial : float
            Proporção inicial para treino (ex: 0.7 = 70%)
        tamanho_teste : float
            Proporção para cada fold de teste (ex: 0.1 = 10%)
        passo : int
            Número de períodos a avançar em cada fold (1 = um período por vez)
        periodo_sazonal : int
            Período sazonal para o modelo SARIMA
        """
        self.serie = serie.copy()
        self.tamanho_treino_inicial = tamanho_treino_inicial
        self.tamanho_teste = tamanho_teste
        self.passo = passo
        self.periodo_sazonal = periodo_sazonal
        self.resultados = []
        
    def executar_validacao(self, verbose=True):
        """
        Executa validação walk-forward completa.
        
        Parameters:
        -----------
        verbose : bool
            Se True, imprime progresso
            
        Returns:
        --------
        pd.DataFrame
            Resultados de cada fold com métricas
        """
        if verbose:
            print("\n" + "=" * 80)
            print("VALIDAÇÃO CRUZADA WALK-FORWARD")
            print("=" * 80)
            print(f"\nSérie temporal: {len(self.serie)} observações")
            print(f"Tamanho inicial de treino: {self.tamanho_treino_inicial*100:.0f}%")
            print(f"Tamanho de teste por fold: {self.tamanho_teste*100:.0f}%")
            print(f"Passo entre folds: {self.passo} períodos")
        
        n_total = len(self.serie)
        n_treino_inicial = int(n_total * self.tamanho_treino_inicial)
        n_teste = max(1, int(n_total * self.tamanho_teste))
        
        # Posição inicial
        pos_treino_fim = n_treino_inicial
        fold = 1
        
        while pos_treino_fim + n_teste <= n_total:
            # Define janelas de treino e teste
            serie_treino = self.serie.iloc[:pos_treino_fim]
            serie_teste = self.serie.iloc[pos_treino_fim:pos_treino_fim + n_teste]
            
            if verbose:
                print(f"\n{'='*80}")
                print(f"FOLD {fold}")
                print(f"{'='*80}")
                print(f"Treino: {len(serie_treino)} observações (até índice {pos_treino_fim-1})")
                print(f"Teste: {len(serie_teste)} observações (índices {pos_treino_fim} a {pos_treino_fim+n_teste-1})")
            
            # Treina modelo
            try:
                modelo = auto_arima(
                    serie_treino,
                    seasonal=True,
                    m=self.periodo_sazonal,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore',
                    max_p=5, max_d=2, max_q=5,
                    max_P=2, max_D=1, max_Q=2,
                    information_criterion='aic',
                    trace=False,
                    n_jobs=1
                )
                
                # Gera previsão
                previsao = modelo.predict(n_periods=len(serie_teste))
                previsao = np.maximum(previsao, 0)  # Garante não-negativo
                
                # Calcula métricas
                mae = mean_absolute_error(serie_teste.values, previsao)
                rmse = np.sqrt(mean_squared_error(serie_teste.values, previsao))
                mape = calcular_mape(serie_teste.values, previsao)
                
                # Armazena resultados
                resultado_fold = {
                    'fold': fold,
                    'tamanho_treino': len(serie_treino),
                    'tamanho_teste': len(serie_teste),
                    'indice_treino_fim': pos_treino_fim - 1,
                    'indice_teste_inicio': pos_treino_fim,
                    'indice_teste_fim': pos_treino_fim + n_teste - 1,
                    'modelo_order': modelo.order,
                    'modelo_seasonal_order': modelo.seasonal_order,
                    'aic': modelo.aic(),
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'erro_medio': np.mean(np.abs(serie_teste.values - previsao)),
                    'erro_std': np.std(np.abs(serie_teste.values - previsao))
                }
                
                self.resultados.append(resultado_fold)
                
                if verbose:
                    print(f"  Modelo: {modelo.order} x {modelo.seasonal_order}")
                    print(f"  AIC: {modelo.aic():.2f}")
                    print(f"  MAE: {mae:.2f}")
                    print(f"  RMSE: {rmse:.2f}")
                    print(f"  MAPE: {mape:.2f}%")
                
            except Exception as e:
                if verbose:
                    print(f"  [ERRO] Falha no fold {fold}: {str(e)}")
                resultado_fold = {
                    'fold': fold,
                    'tamanho_treino': len(serie_treino),
                    'tamanho_teste': len(serie_teste),
                    'erro': str(e)
                }
                self.resultados.append(resultado_fold)
            
            # Avança para próximo fold
            pos_treino_fim += self.passo
            fold += 1
            
            # Limite de segurança (evita loops infinitos)
            if fold > 100:
                if verbose:
                    print("\n[AVISO] Limite de 100 folds atingido. Interrompendo...")
                break
        
        # Cria DataFrame com resultados
        df_resultados = pd.DataFrame(self.resultados)
        
        if verbose:
            print("\n" + "=" * 80)
            print("RESUMO DA VALIDAÇÃO")
            print("=" * 80)
            print(f"\nTotal de folds executados: {len(df_resultados)}")
            
            if 'mae' in df_resultados.columns:
                folds_validos = df_resultados[df_resultados['mae'].notna()]
                if len(folds_validos) > 0:
                    print(f"\nMétricas Médias (apenas folds válidos):")
                    print(f"  MAE médio: {folds_validos['mae'].mean():.2f}")
                    print(f"  RMSE médio: {folds_validos['rmse'].mean():.2f}")
                    print(f"  MAPE médio: {folds_validos['mape'].mean():.2f}%")
                    print(f"\nDesvio Padrão das Métricas:")
                    print(f"  MAE std: {folds_validos['mae'].std():.2f}")
                    print(f"  RMSE std: {folds_validos['rmse'].std():.2f}")
                    print(f"  MAPE std: {folds_validos['mape'].std():.2f}%")
                    
                    # Estabilidade do modelo
                    if 'modelo_order' in folds_validos.columns:
                        modelos_unicos = folds_validos['modelo_order'].nunique()
                        print(f"\nEstabilidade do Modelo:")
                        print(f"  Modelos únicos encontrados: {modelos_unicos}")
                        if modelos_unicos == 1:
                            print("  ✓ Modelo estável (mesma ordem em todos os folds)")
                        else:
                            print(f"  ⚠ Modelo variou entre folds (pode indicar instabilidade)")
        
        return df_resultados
    
    def plotar_resultados(self, caminho_saida=None):
        """
        Cria visualização dos resultados da validação.
        
        Parameters:
        -----------
        caminho_saida : str, optional
            Caminho para salvar a figura
        """
        import matplotlib.pyplot as plt
        
        if len(self.resultados) == 0:
            print("[AVISO] Nenhum resultado para plotar. Execute executar_validacao() primeiro.")
            return
        
        df_resultados = pd.DataFrame(self.resultados)
        
        if 'mae' not in df_resultados.columns:
            print("[AVISO] Não há métricas para plotar.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        folds_validos = df_resultados[df_resultados['mae'].notna()]
        
        # 1. Evolução do MAE ao longo dos folds
        ax1 = axes[0, 0]
        ax1.plot(folds_validos['fold'], folds_validos['mae'], 
                marker='o', linewidth=2, markersize=6, color='steelblue')
        ax1.axhline(y=folds_validos['mae'].mean(), color='red', 
                   linestyle='--', linewidth=2, label=f'Média: {folds_validos["mae"].mean():.2f}')
        ax1.set_title('Evolução do MAE por Fold', fontweight='bold', fontsize=12)
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('MAE')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Evolução do RMSE
        ax2 = axes[0, 1]
        ax2.plot(folds_validos['fold'], folds_validos['rmse'], 
                marker='s', linewidth=2, markersize=6, color='orange')
        ax2.axhline(y=folds_validos['rmse'].mean(), color='red', 
                   linestyle='--', linewidth=2, label=f'Média: {folds_validos["rmse"].mean():.2f}')
        ax2.set_title('Evolução do RMSE por Fold', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Fold')
        ax2.set_ylabel('RMSE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Evolução do MAPE
        ax3 = axes[1, 0]
        ax3.plot(folds_validos['fold'], folds_validos['mape'], 
                marker='^', linewidth=2, markersize=6, color='green')
        ax3.axhline(y=folds_validos['mape'].mean(), color='red', 
                   linestyle='--', linewidth=2, label=f'Média: {folds_validos["mape"].mean():.2f}%')
        ax3.set_title('Evolução do MAPE por Fold', fontweight='bold', fontsize=12)
        ax3.set_xlabel('Fold')
        ax3.set_ylabel('MAPE (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Distribuição dos erros
        ax4 = axes[1, 1]
        ax4.hist(folds_validos['mae'], bins=min(20, len(folds_validos)), 
                color='skyblue', edgecolor='black', alpha=0.7)
        ax4.axvline(x=folds_validos['mae'].mean(), color='red', 
                   linestyle='--', linewidth=2, label=f'Média: {folds_validos["mae"].mean():.2f}')
        ax4.set_title('Distribuição do MAE', fontweight='bold', fontsize=12)
        ax4.set_xlabel('MAE')
        ax4.set_ylabel('Frequência')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Validação Walk-Forward - Resultados', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if caminho_saida:
            plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
            print(f"\n[OK] Gráfico salvo: {caminho_saida}")
        else:
            nome_arquivo = 'validacao_walk_forward.png'
            plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
            print(f"\n[OK] Gráfico salvo: {nome_arquivo}")
        
        plt.close()
    
    def gerar_relatorio(self, caminho_saida=None):
        """
        Gera relatório textual com resultados da validação.
        
        Parameters:
        -----------
        caminho_saida : str, optional
            Caminho para salvar o relatório
        """
        linhas = []
        linhas.append("=" * 80)
        linhas.append("RELATÓRIO: VALIDAÇÃO CRUZADA WALK-FORWARD")
        linhas.append("=" * 80)
        linhas.append(f"\nData: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        linhas.append(f"Total de observações: {len(self.serie)}")
        linhas.append(f"Tamanho inicial de treino: {self.tamanho_treino_inicial*100:.0f}%")
        linhas.append(f"Tamanho de teste por fold: {self.tamanho_teste*100:.0f}%")
        linhas.append(f"Passo entre folds: {self.passo}")
        
        df_resultados = pd.DataFrame(self.resultados)
        linhas.append(f"\nTotal de folds executados: {len(df_resultados)}")
        
        if 'mae' in df_resultados.columns:
            folds_validos = df_resultados[df_resultados['mae'].notna()]
            if len(folds_validos) > 0:
                linhas.append(f"Folds válidos: {len(folds_validos)}")
                
                linhas.append("\n" + "-" * 80)
                linhas.append("MÉTRICAS MÉDIAS")
                linhas.append("-" * 80)
                linhas.append(f"MAE médio: {folds_validos['mae'].mean():.2f}")
                linhas.append(f"RMSE médio: {folds_validos['rmse'].mean():.2f}")
                linhas.append(f"MAPE médio: {folds_validos['mape'].mean():.2f}%")
                
                linhas.append("\n" + "-" * 80)
                linhas.append("VARIABILIDADE DAS MÉTRICAS")
                linhas.append("-" * 80)
                linhas.append(f"MAE - Desvio padrão: {folds_validos['mae'].std():.2f}")
                linhas.append(f"RMSE - Desvio padrão: {folds_validos['rmse'].std():.2f}")
                linhas.append(f"MAPE - Desvio padrão: {folds_validos['mape'].std():.2f}%")
                
                # Estabilidade
                if 'modelo_order' in folds_validos.columns:
                    modelos_unicos = folds_validos['modelo_order'].nunique()
                    linhas.append("\n" + "-" * 80)
                    linhas.append("ESTABILIDADE DO MODELO")
                    linhas.append("-" * 80)
                    linhas.append(f"Modelos únicos encontrados: {modelos_unicos}")
                    
                    if modelos_unicos == 1:
                        modelo_mais_comum = folds_validos['modelo_order'].mode()[0]
                        linhas.append(f"✓ Modelo estável: {modelo_mais_comum}")
                    else:
                        linhas.append("⚠ Modelo variou entre folds")
                        linhas.append("\nDistribuição de modelos:")
                        dist_modelos = folds_validos['modelo_order'].value_counts()
                        for modelo, count in dist_modelos.items():
                            linhas.append(f"  {modelo}: {count} folds ({count/len(folds_validos)*100:.1f}%)")
                
                # Conclusão
                linhas.append("\n" + "=" * 80)
                linhas.append("CONCLUSÃO")
                linhas.append("=" * 80)
                
                cv_mae = (folds_validos['mae'].std() / folds_validos['mae'].mean()) * 100
                if cv_mae < 20:
                    linhas.append("✓ Modelo apresenta boa estabilidade (CV < 20%)")
                elif cv_mae < 50:
                    linhas.append("⚠ Modelo apresenta estabilidade moderada (20% < CV < 50%)")
                else:
                    linhas.append("✗ Modelo apresenta alta variabilidade (CV > 50%)")
                    linhas.append("  Recomendação: Revisar parâmetros ou considerar outros modelos")
        
        texto = "\n".join(linhas)
        
        if caminho_saida:
            with open(caminho_saida, 'w', encoding='utf-8') as f:
                f.write(texto)
            print(f"\n[OK] Relatório salvo: {caminho_saida}")
        else:
            nome_arquivo = 'relatorio_validacao_walk_forward.txt'
            with open(nome_arquivo, 'w', encoding='utf-8') as f:
                f.write(texto)
            print(f"\n[OK] Relatório salvo: {nome_arquivo}")
        
        return texto


def main():
    """
    Exemplo de uso da validação walk-forward.
    """
    print("=" * 80)
    print("VALIDAÇÃO CRUZADA WALK-FORWARD PARA SARIMA")
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
    print(f"Período: {serie.index[0].date()} a {serie.index[-1].date()}")
    
    # Executa validação walk-forward
    validacao = ValidacaoWalkForward(
        serie=serie,
        tamanho_treino_inicial=0.7,  # 70% inicial para treino
        tamanho_teste=0.1,            # 10% para cada fold de teste
        passo=7,                      # Avança 7 dias por fold
        periodo_sazonal=30
    )
    
    resultados = validacao.executar_validacao(verbose=True)
    
    # Visualizações
    validacao.plotar_resultados()
    
    # Relatório
    validacao.gerar_relatorio()
    
    print("\n" + "=" * 80)
    print("VALIDAÇÃO CONCLUÍDA!")
    print("=" * 80)
    print("\nArquivos gerados:")
    print("  - validacao_walk_forward.png")
    print("  - relatorio_validacao_walk_forward.txt")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERRO] Erro durante execução: {str(e)}")
        import traceback
        traceback.print_exc()

