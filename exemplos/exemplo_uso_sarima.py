"""
Exemplo Prático de Uso do SARIMA para Previsão de Estoque
TCC MBA Data Science & Analytics

Este script demonstra como usar o módulo sarima_estoque.py com dados simulados
e como adaptar para seus dados reais via API.

Autor: [Seu Nome]
Data: 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sarima_estoque import PrevisorEstoqueSARIMA


def gerar_dados_simulados(sku, n_dias=90, estoque_inicial=100):
    """
    Gera dados simulados de estoque para demonstração.
    
    Em produção, você substituirá esta função pela leitura via API.
    """
    # Gera datas dos últimos n_dias
    datas = pd.date_range(end=datetime.now(), periods=n_dias, freq='D')
    
    # Simula padrão de estoque com:
    # - Tendência decrescente leve (vendas constantes)
    # - Sazonalidade semanal (menos estoque nos fins de semana)
    # - Ruído aleatório
    
    np.random.seed(42)  # Para reprodutibilidade
    
    tendencia = np.linspace(estoque_inicial, estoque_inicial * 0.7, n_dias)
    
    # Sazonalidade semanal (dia da semana: 0=Segunda, 6=Domingo)
    dias_semana = [d.weekday() for d in datas]
    sazonalidade = [10 * np.sin(2 * np.pi * d / 7) for d in dias_semana]
    
    # Ruído
    ruido = np.random.normal(0, 5, n_dias)
    
    # Combina tudo
    estoque = tendencia + sazonalidade + ruido
    
    # Garante valores positivos
    estoque = np.maximum(estoque, 0)
    
    # Cria DataFrame
    df = pd.DataFrame({
        'data': datas,
        'sku': sku,
        'estoque_atual': estoque.astype(int)
    })
    
    return df


def exemplo_basico_um_produto():
    """
    Exemplo 1: Previsão para um único produto (SKU)
    """
    print("\n" + "=" * 60)
    print("EXEMPLO 1: Previsão para um único produto")
    print("=" * 60)
    
    # 1. Prepara dados (em produção, vem da API)
    print("\n1. Preparando dados simulados...")
    df_estoque = gerar_dados_simulados(sku='BRINQUEDO_001', n_dias=90)
    print(f"   ✓ {len(df_estoque)} registros gerados")
    print(f"   Período: {df_estoque['data'].min()} a {df_estoque['data'].max()}")
    
    # 2. Inicializa previsor
    print("\n2. Inicializando previsor SARIMA...")
    previsor = PrevisorEstoqueSARIMA(horizonte_previsao=7, frequencia='D')
    
    # 3. Prepara série temporal
    print("\n3. Preparando série temporal...")
    serie = previsor.preparar_serie_temporal(df_estoque, sku='BRINQUEDO_001')
    print(f"   ✓ Série com {len(serie)} observações")
    print(f"   Estoque atual: {serie.iloc[-1]:.0f} unidades")
    
    # 4. Treina modelo (auto_arima busca parâmetros automaticamente)
    print("\n4. Treinando modelo SARIMA (auto_arima)...")
    print("   Isso pode levar alguns segundos...")
    modelo = previsor.treinar_modelo(serie, sku='BRINQUEDO_001')
    
    if modelo is None:
        print("   ✗ Falha ao treinar modelo")
        return
    
    # 5. Gera previsão
    print("\n5. Gerando previsão para próximos 7 dias...")
    previsao = previsor.prever(serie, modelo=modelo)
    
    if previsao is not None:
        print("\n   PREVISÕES:")
        print(previsao.to_frame().T.to_string())
        
        # 6. Calcula risco de ruptura (exemplo)
        print("\n6. Calculando risco de ruptura...")
        estoque_minimo = 20
        risco = previsor.calcular_risco_ruptura(previsao, estoque_minimo)
        print(f"   Estoque mínimo desejado: {estoque_minimo} unidades")
        print(f"   Score de risco de ruptura: {risco:.2%}")
        
        # 7. Visualização (opcional)
        try:
            plotar_resultado(serie, previsao, 'BRINQUEDO_001')
        except:
            print("\n   (Visualização não disponível - instale matplotlib)")
    
    print("\n" + "=" * 60)


def exemplo_lote_multiplos_produtos():
    """
    Exemplo 2: Previsão para múltiplos produtos (processamento em lote)
    """
    print("\n" + "=" * 60)
    print("EXEMPLO 2: Previsão para múltiplos produtos (lote)")
    print("=" * 60)
    
    # 1. Gera dados para múltiplos SKUs
    print("\n1. Preparando dados para múltiplos produtos...")
    skus = ['BRINQUEDO_001', 'BRINQUEDO_002', 'BRINQUEDO_003']
    df_lista = []
    
    for sku in skus:
        df_sku = gerar_dados_simulados(sku=sku, n_dias=90, estoque_inicial=np.random.randint(50, 150))
        df_lista.append(df_sku)
    
    df_estoque = pd.concat(df_lista, ignore_index=True)
    print(f"   ✓ Dados para {len(skus)} produtos")
    print(f"   Total de registros: {len(df_estoque)}")
    
    # 2. Processa lote
    print("\n2. Processando previsões em lote...")
    previsor = PrevisorEstoqueSARIMA(horizonte_previsao=7, frequencia='D')
    resultados = previsor.processar_lote(df_estoque, lista_skus=skus)
    
    # 3. Exibe resultados
    if not resultados.empty:
        print("\n3. Resultados consolidados:")
        print("\n   Previsões por produto:")
        for sku in resultados['sku'].unique():
            df_sku = resultados[resultados['sku'] == sku]
            estoque_medio = df_sku['estoque_previsto'].mean()
            print(f"   - {sku}: Estoque médio previsto = {estoque_medio:.1f} unidades")
        
        print("\n   Tabela completa:")
        print(resultados.to_string(index=False))
        
        # 4. Exporta resultados (opcional)
        # resultados.to_csv('previsoes_estoque.csv', index=False)
        # print("\n   ✓ Resultados exportados para 'previsoes_estoque.csv'")
    
    print("\n" + "=" * 60)


def exemplo_com_dados_reais_api():
    """
    Exemplo 3: Como adaptar para seus dados reais da API
    
    Esta função mostra a estrutura que você deve seguir para integrar
    com sua API real. Substitua a função gerar_dados_simulados() pela
    chamada à sua API.
    """
    print("\n" + "=" * 60)
    print("EXEMPLO 3: Estrutura para integração com API real")
    print("=" * 60)
    
    print("""
    # PASSO 1: Conectar à API e obter dados
    # 
    # import requests  # ou sua biblioteca de API
    # 
    # def obter_dados_estoque_api(data_inicio, data_fim):
    #     url = "sua_api_endpoint/historico_estoque"
    #     params = {
    #         'data_inicio': data_inicio,
    #         'data_fim': data_fim
    #     }
    #     response = requests.get(url, params=params)
    #     dados = response.json()
    #     
    #     # Converte para DataFrame
    #     df = pd.DataFrame(dados)
    #     df['data'] = pd.to_datetime(df['data'])
    #     
    #     return df
    # 
    # # PASSO 2: Preparar dados no formato esperado
    # # O DataFrame deve ter as colunas: 'data', 'sku', 'estoque_atual'
    # 
    # data_fim = datetime.now()
    # data_inicio = data_fim - timedelta(days=180)  # 6 meses de histórico
    # 
    # df_estoque = obter_dados_estoque_api(data_inicio, data_fim)
    # 
    # # PASSO 3: Usar o previsor normalmente
    # previsor = PrevisorEstoqueSARIMA(horizonte_previsao=15)
    # resultados = previsor.processar_lote(df_estoque)
    # 
    # # PASSO 4: Integrar com sua fórmula de elencação
    # # (exemplo de como você pode usar as previsões)
    # 
    # for sku in resultados['sku'].unique():
    #     df_sku = resultados[resultados['sku'] == sku]
    #     estoque_medio_previsto = df_sku['estoque_previsto'].mean()
    #     
    #     # Sua fórmula de elencação (exemplo)
    #     score_estoque = 1 / (1 + estoque_medio_previsto)  # Normaliza para [0,1]
    #     # ... continue com margem de contribuição, giro, etc.
    """)
    
    print("\n" + "=" * 60)


def plotar_resultado(serie_historica, previsao, sku):
    """
    Plota gráfico comparando histórico e previsão.
    """
    plt.figure(figsize=(12, 6))
    
    # Histórico
    plt.plot(serie_historica.index, serie_historica.values, 
             label='Histórico', color='blue', linewidth=2)
    
    # Previsão
    plt.plot(previsao.index, previsao.values, 
             label='Previsão', color='red', linewidth=2, linestyle='--')
    
    # Linha divisória
    ultima_data = serie_historica.index[-1]
    plt.axvline(x=ultima_data, color='gray', linestyle=':', alpha=0.7)
    
    plt.title(f'Previsão de Estoque - {sku}', fontsize=14, fontweight='bold')
    plt.xlabel('Data')
    plt.ylabel('Estoque (unidades)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Salva gráfico
    plt.savefig(f'previsao_{sku}.png', dpi=150)
    print(f"\n   ✓ Gráfico salvo em 'previsao_{sku}.png'")
    plt.close()


def exemplo_parametros_avancados():
    """
    Exemplo 4: Ajuste de parâmetros avançados do auto_arima
    
    Mostra como você pode customizar a busca de parâmetros conforme
    necessário para seu caso de uso específico.
    """
    print("\n" + "=" * 60)
    print("EXEMPLO 4: Ajuste de parâmetros avançados")
    print("=" * 60)
    
    print("""
    O auto_arima pode ser customizado conforme seu conhecimento do domínio:
    
    1. SE VOCÊ SABE que há sazonalidade mensal (30 dias):
       modelo = auto_arima(serie, seasonal=True, m=30, ...)
    
    2. SE VOCÊ SABE que a série não precisa de diferenciação:
       modelo = auto_arima(serie, d=0, ...)
    
    3. SE VOCÊ QUER buscar apenas modelos simples (mais rápidos):
       modelo = auto_arima(serie, max_p=2, max_q=2, ...)
    
    4. SE VOCÊ PREFERE BIC ao invés de AIC (modelos mais simples):
       modelo = auto_arima(serie, information_criterion='bic', ...)
    
    Para a maioria dos casos, os parâmetros padrão do módulo funcionam bem.
    """)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "EXEMPLOS DE USO - SARIMA PARA ESTOQUE" + " " * 10 + "║")
    print("╚" + "═" * 58 + "╝")
    
    # Exemplo 1: Um produto
    exemplo_basico_um_produto()
    
    # Exemplo 2: Lote de produtos
    exemplo_lote_multiplos_produtos()
    
    # Exemplo 3: Estrutura para API
    exemplo_com_dados_reais_api()
    
    # Exemplo 4: Parâmetros avançados
    exemplo_parametros_avancados()
    
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "PRÓXIMOS PASSOS" + " " * 27 + "║")
    print("╚" + "═" * 58 + "╝")
    print("""
    1. Instale as dependências necessárias:
       pip install pmdarima pandas numpy matplotlib
    
    2. Execute este script para ver os exemplos:
       python exemplo_uso_sarima.py
    
    3. Adapte para seus dados reais:
       - Substitua gerar_dados_simulados() pela sua função de API
       - Ajuste horizonte_previsao conforme necessário (7-15 dias)
       - Integre as previsões com sua fórmula de elencação
    
    4. Para seu TCC, considere:
       - Métricas de avaliação (MAE, RMSE, MAPE)
       - Validação cruzada temporal
       - Comparação com métodos baseline (média móvel, etc.)
    """)


