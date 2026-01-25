# Tabela 1 - Explicacao da Base de Dados Utilizada

Fonte: Dados originais da pesquisa.

| Fonte | Variavel | Descricao da Variavel | Codigo e Rotulo da Variavel |
|-------|----------|------------------------|-----------------------------|
| historico_estoque | sku | Identificador unico do produto (Stock Keeping Unit) | sku (str) |
| historico_estoque | created_at | Data e hora do registro do saldo de estoque | created_at (datetime) |
| historico_estoque | saldo | Quantidade em estoque no momento do registro | saldo (int) |
| venda_produtos | sku | Identificador unico do produto (mesmo que historico_estoque) | sku (str) |
| venda_produtos | created_at | Data e hora da transacao de venda | created_at (datetime) |
| venda_produtos | quantidade | Quantidade de unidades vendidas na transacao | quantidade (int) |
| venda_produtos | valor_unitario | Preco de venda unitario do produto | valor_unitario (float) |
| venda_produtos | custo_unitario | Custo de aquisicao unitario do produto | custo_unitario (float) |
| venda_produtos | margem_proporcional | Margem de contribuicao em percentual ((valor - custo) / valor * 100) | margem_proporcional (float) |
