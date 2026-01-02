"""
Exemplo de Integração: Fórmula de Elencação com Previsão SARIMA
TCC MBA Data Science & Analytics

Este script demonstra como integrar as previsões SARIMA com os outros componentes
da fórmula de elencação: Margem de Contribuição e Giro de Estoque.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sarima_estoque import PrevisorEstoqueSARIMA


def calcular_margem_contribuicao(preco_venda, custo_venda):
    """
    Calcula margem de contribuição normalizada.
    
    Returns:
    --------
    float
        Margem de contribuição normalizada (0 a 1)
    """
    margem_absoluta = preco_venda - custo_venda
    margem_percentual = margem_absoluta / preco_venda if preco_venda > 0 else 0
    
    # Normaliza para [0, 1] (assumindo margem máxima de 50%)
    margem_normalizada = min(1.0, margem_percentual / 0.5)
    
    return margem_normalizada


def calcular_giro_estoque(qtd_vendida_periodo, estoque_medio):
    """
    Calcula giro de estoque (quantas vezes o estoque foi renovado no período).
    
    Returns:
    --------
    float
        Giro de estoque normalizado (0 a 1)
    """
    if estoque_medio <= 0:
        return 0.0
    
    giro = qtd_vendida_periodo / estoque_medio
    
    # Normaliza para [0, 1] (assumindo giro máximo de 12x por período)
    giro_normalizado = min(1.0, giro / 12.0)
    
    return giro_normalizado


def calcular_score_risco_ruptura(estoque_previsto, estoque_minimo):
    """
    Calcula score de risco de ruptura baseado na previsão SARIMA.
    
    Parameters:
    -----------
    estoque_previsto : float
        Estoque médio previsto para os próximos dias
    estoque_minimo : float
        Estoque mínimo desejado
        
    Returns:
    --------
    float
        Score de risco (0 a 1, onde 1 = alto risco)
    """
    if estoque_previsto < estoque_minimo:
        # Risco aumenta conforme estoque previsto fica menor
        deficit = estoque_minimo - estoque_previsto
        risco = min(1.0, deficit / estoque_minimo)
        return risco
    else:
        return 0.0


def calcular_score_elencacao(margem_contrib, giro_estoque, risco_ruptura, 
                             peso_margem=0.4, peso_giro=0.3, peso_risco=0.3):
    """
    Calcula score final de elencação combinando os três pilares.
    
    Parameters:
    -----------
    margem_contrib : float
        Margem de contribuição normalizada (0 a 1)
    giro_estoque : float
        Giro de estoque normalizado (0 a 1)
    risco_ruptura : float
        Score de risco de ruptura (0 a 1, onde 1 = alto risco = alta prioridade)
    peso_margem : float
        Peso da margem de contribuição (padrão: 0.4)
    peso_giro : float
        Peso do giro de estoque (padrão: 0.3)
    peso_risco : float
        Peso do risco de ruptura (padrão: 0.3)
        
    Returns:
    --------
    float
        Score final de elencação (0 a 1, onde maior = maior prioridade)
    """
    # Garante que pesos somam 1.0
    total_pesos = peso_margem + peso_giro + peso_risco
    if total_pesos != 1.0:
        peso_margem /= total_pesos
        peso_giro /= total_pesos
        peso_risco /= total_pesos
    
    # Fórmula de elencação
    score = (
        peso_margem * margem_contrib +
        peso_giro * giro_estoque +
        peso_risco * risco_ruptura
    )
    
    return score


def exemplo_elencacao_completa():
    """
    Exemplo completo de elencação com os três pilares integrados.
    """
    print("\n" + "=" * 70)
    print("EXEMPLO: Fórmula de Elencação com Previsão SARIMA")
    print("=" * 70)
    
    # ============================================================
    # 1. DADOS DE ENTRADA (em produção, vem das suas APIs/tabelas)
    # ============================================================
    
    print("\n1. Preparando dados de entrada...")
    
    # Dados de produtos (exemplo)
    produtos = [
        {
            'sku': 'BRINQUEDO_001',
            'preco_venda': 50.00,
            'custo_venda': 30.00,
            'qtd_vendida_30dias': 120,
            'estoque_medio_30dias': 80,
            'estoque_minimo': 30
        },
        {
            'sku': 'BRINQUEDO_002',
            'preco_venda': 80.00,
            'custo_venda': 50.00,
            'qtd_vendida_30dias': 60,
            'estoque_medio_30dias': 100,
            'estoque_minimo': 40
        },
        {
            'sku': 'BRINQUEDO_003',
            'preco_venda': 30.00,
            'custo_venda': 20.00,
            'qtd_vendida_30dias': 200,
            'estoque_medio_30dias': 50,
            'estoque_minimo': 25
        }
    ]
    
    # Dados de estoque histórico (simulado - em produção vem da API)
    # Nota: Aqui você usaria seus dados reais
    print("   ✓ Dados de produtos preparados")
    print("   ✓ Dados de estoque histórico (simulado)")
    
    # ============================================================
    # 2. PREVISÃO SARIMA (Pilar 3)
    # ============================================================
    
    print("\n2. Gerando previsões SARIMA...")
    
    # Simula previsões SARIMA (em produção, use o módulo real)
    previsoes_sarima = {
        'BRINQUEDO_001': 35.0,  # Estoque médio previsto para próximos 7-15 dias
        'BRINQUEDO_002': 25.0,
        'BRINQUEDO_003': 15.0
    }
    
    print("   Previsões geradas:")
    for sku, estoque_previsto in previsoes_sarima.items():
        print(f"   - {sku}: {estoque_previsto:.1f} unidades (média prevista)")
    
    # ============================================================
    # 3. CÁLCULO DOS TRÊS PILARES
    # ============================================================
    
    print("\n3. Calculando componentes da fórmula de elencação...")
    
    resultados = []
    
    for produto in produtos:
        sku = produto['sku']
        
        # PILAR 1: Margem de Contribuição
        margem = calcular_margem_contribuicao(
            produto['preco_venda'],
            produto['custo_venda']
        )
        
        # PILAR 2: Giro de Estoque
        giro = calcular_giro_estoque(
            produto['qtd_vendida_30dias'],
            produto['estoque_medio_30dias']
        )
        
        # PILAR 3: Risco de Ruptura (baseado na previsão SARIMA)
        estoque_previsto = previsoes_sarima.get(sku, 0)
        risco = calcular_score_risco_ruptura(
            estoque_previsto,
            produto['estoque_minimo']
        )
        
        # SCORE FINAL de elencação
        score_final = calcular_score_elencacao(margem, giro, risco)
        
        resultados.append({
            'sku': sku,
            'margem_contribuicao': margem,
            'giro_estoque': giro,
            'risco_ruptura': risco,
            'estoque_previsto': estoque_previsto,
            'score_elencacao': score_final
        })
    
    # ============================================================
    # 4. RESULTADO: Ranking ordenado por prioridade
    # ============================================================
    
    df_resultado = pd.DataFrame(resultados)
    df_resultado = df_resultado.sort_values('score_elencacao', ascending=False)
    df_resultado['ranking'] = range(1, len(df_resultado) + 1)
    
    print("\n4. Ranking de Priorização (ordenado por score de elencação):")
    print("\n" + "-" * 70)
    print(f"{'Rank':<6} {'SKU':<20} {'Margem':<10} {'Giro':<10} {'Risco':<10} {'Score':<10}")
    print("-" * 70)
    
    for _, row in df_resultado.iterrows():
        print(f"{int(row['ranking']):<6} "
              f"{row['sku']:<20} "
              f"{row['margem_contribuicao']:.2f}     "
              f"{row['giro_estoque']:.2f}     "
              f"{row['risco_ruptura']:.2f}     "
              f"{row['score_elencacao']:.2f}")
    
    print("-" * 70)
    
    # ============================================================
    # 5. INTERPRETAÇÃO
    # ============================================================
    
    print("\n5. Interpretação:")
    print(f"\n   Produto de MAIOR prioridade: {df_resultado.iloc[0]['sku']}")
    print(f"   - Score: {df_resultado.iloc[0]['score_elencacao']:.2f}")
    print(f"   - Estoque previsto: {df_resultado.iloc[0]['estoque_previsto']:.1f} unidades")
    print(f"   - Risco de ruptura: {df_resultado.iloc[0]['risco_ruptura']:.2%}")
    
    print(f"\n   Produto de MENOR prioridade: {df_resultado.iloc[-1]['sku']}")
    print(f"   - Score: {df_resultado.iloc[-1]['score_elencacao']:.2f}")
    
    print("\n" + "=" * 70)


def exemplo_com_previsor_real():
    """
    Exemplo mostrando como integrar com o previsor SARIMA real.
    """
    print("\n" + "=" * 70)
    print("EXEMPLO: Integração com Previsor SARIMA Real")
    print("=" * 70)
    
    print("""
    # CÓDIGO PARA INTEGRAR COM SEUS DADOS REAIS:
    
    from sarima_estoque import PrevisorEstoqueSARIMA
    
    # 1. Obter dados de estoque histórico da API
    df_estoque = obter_dados_estoque_api()
    
    # 2. Obter dados de produtos (vendas, preços, custos) da API
    df_produtos = obter_dados_produtos_api()
    
    # 3. Gerar previsões SARIMA para todos os produtos
    previsor = PrevisorEstoqueSARIMA(horizonte_previsao=15)
    df_previsoes = previsor.processar_lote(df_estoque)
    
    # 4. Calcular estoque médio previsto por SKU
    estoque_previsto_por_sku = df_previsoes.groupby('sku')['estoque_previsto'].mean()
    
    # 5. Calcular componentes da fórmula para cada produto
    resultados = []
    
    for sku in df_produtos['sku'].unique():
        # Dados do produto
        produto = df_produtos[df_produtos['sku'] == sku].iloc[0]
        
        # Pilar 1: Margem
        margem = calcular_margem_contribuicao(
            produto['preco_venda'],
            produto['custo_venda']
        )
        
        # Pilar 2: Giro
        giro = calcular_giro_estoque(
            produto['qtd_vendida'],
            produto['estoque_medio']
        )
        
        # Pilar 3: Risco (usando previsão SARIMA)
        estoque_previsto = estoque_previsto_por_sku.get(sku, 0)
        risco = calcular_score_risco_ruptura(
            estoque_previsto,
            produto['estoque_minimo']
        )
        
        # Score final
        score = calcular_score_elencacao(margem, giro, risco)
        
        resultados.append({
            'sku': sku,
            'score_elencacao': score,
            # ... outros campos
        })
    
    # 6. Ordenar por score e usar para priorização
    df_ranking = pd.DataFrame(resultados)
    df_ranking = df_ranking.sort_values('score_elencacao', ascending=False)
    
    # Pronto! Use df_ranking para sua ferramenta de elencação.
    """)
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Exemplo completo
    exemplo_elencacao_completa()
    
    # Exemplo de integração
    exemplo_com_previsor_real()
    
    print("\n")
    print("💡 DICA: Adapte os pesos da fórmula conforme seu negócio:")
    print("   - Se margem é mais importante: aumente peso_margem")
    print("   - Se giro é mais importante: aumente peso_giro")
    print("   - Se evitar ruptura é crítico: aumente peso_risco")
    print("\n")


