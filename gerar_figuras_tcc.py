#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gera todas as figuras do TCC (figura1 ... figura7) e o valor final da ferramenta de elencacao.

- Analise exploratoria -> figura1, figura2, figura3, figura4 (agregados + SKU representativo).
- Seleciona ate 300 candidatos (zeros <= 30%, estoque ok, cv_mensal) e roda metricas (ARIMA, SARIMA m=30, etc.) sem figuras.
- Filtra teste constante e resultados insatisfatorios; ranqueia por melhor MAE e escolhe os 10 melhores.
- Gera figura5, figura6, figura7 (SKU com menor MAE dos 10), Tabela 2, relatorios e graficos por SKU apenas para os 10.
- Gera elencacao final (R(t), U(t), GP(t)) para os 10 melhores e retorna o DataFrame do ranking.

Os modelos preveem ESTOQUE (saldo), nao vendas. GP(t) = soma das previsoes de estoque.

Salva em resultados/figuras_tcc/, resultados/tabelas_tcc/, resultados/candidatos_300_metricas.csv,
resultados/elencacao_final.csv.

Retorno: DataFrame com ranking de elencacao (ranking, sku, estoque_atual, R(t), U(t), GP(t), score_elencacao).

Execute a partir da raiz do repositorio:
  python gerar_figuras_tcc.py
"""

from __future__ import annotations

import importlib.util
import sys
import time
import warnings
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Tentar importar psutil para monitoramento de CPU (opcional)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def _aguardar_cpu_abaixo_limite(limite_percent=80.0, intervalo=0.5):
    """
    Aguarda até que o uso de CPU fique abaixo do limite.
    Usa psutil se disponível, caso contrário retorna imediatamente.
    """
    if not PSUTIL_AVAILABLE:
        return
    try:
        import psutil
        cpu_atual = psutil.cpu_percent(interval=0.1)
        if cpu_atual >= limite_percent:
            _log(f"  [CPU] Uso atual: {cpu_atual:.1f}% (limite: {limite_percent}%) - aguardando...")
            while True:
                cpu_atual = psutil.cpu_percent(interval=0.1)
                if cpu_atual < limite_percent:
                    _log(f"  [CPU] Uso atual: {cpu_atual:.1f}% - prosseguindo")
                    break
                time.sleep(intervalo)
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'previsoes'))

DIR_FIGURAS_TCC = ROOT / 'resultados' / 'figuras_tcc'
DIR_TABELAS_TCC = ROOT / 'resultados' / 'tabelas_tcc'
DIR_FIGURAS_MODELOS = ROOT / 'resultados' / 'figuras_modelos'
DIR_FIGURAS_EXPLORATORIA = ROOT / 'resultados' / 'figuras_exploratoria'
DIR_RESULTADOS = ROOT / 'resultados'
DIR_LOGS = ROOT / 'resultados' / 'logs'
CAMINHO_CSV = ROOT / 'DB' / 'historico_estoque_atual_processado.csv'

# Global logger instance
_logger = None


def _limpar_diretorios_saida():
    """
    Limpa os diretorios de saida antes de gerar novos arquivos.
    Remove arquivos de execucoes anteriores para evitar confusao.
    Mantem apenas os logs (que tem timestamp).
    """
    diretorios_limpar = [
        DIR_FIGURAS_TCC,
        DIR_TABELAS_TCC,
        DIR_FIGURAS_MODELOS,
        DIR_FIGURAS_EXPLORATORIA,
    ]
    
    # Arquivos especificos em resultados/ (relatorios de comparacao)
    padroes_arquivos = [
        (DIR_RESULTADOS, 'relatorio_comparacao_*.txt'),
        (DIR_RESULTADOS, 'comparacao_modelos_*.png'),
    ]
    
    _log("\n[LIMPEZA] Limpando diretorios de saida de execucoes anteriores...")
    
    # Limpa diretorios completos (remove todos os arquivos dentro)
    for dir_path in diretorios_limpar:
        if dir_path.exists():
            try:
                # Remove todos os arquivos dentro do diretorio
                for arquivo in dir_path.iterdir():
                    if arquivo.is_file():
                        arquivo.unlink()
                        _log(f"  [REMOVIDO] {arquivo.name}")
                    elif arquivo.is_dir():
                        shutil.rmtree(arquivo)
                        _log(f"  [REMOVIDO DIR] {arquivo.name}/")
                _log(f"  [OK] Diretorio limpo: {dir_path}")
            except Exception as e:
                _log(f"  [AVISO] Erro ao limpar {dir_path}: {e}")
        else:
            _log(f"  [INFO] Diretorio nao existe (sera criado): {dir_path}")
    
    # Remove arquivos especificos por padrao
    for dir_path, padrao in padroes_arquivos:
        if dir_path.exists():
            try:
                arquivos_encontrados = list(dir_path.glob(padrao))
                for arquivo in arquivos_encontrados:
                    arquivo.unlink()
                    _log(f"  [REMOVIDO] {arquivo.name}")
                if arquivos_encontrados:
                    _log(f"  [OK] {len(arquivos_encontrados)} arquivo(s) removido(s) de {dir_path}")
            except Exception as e:
                _log(f"  [AVISO] Erro ao limpar {padrao} em {dir_path}: {e}")
    
    _log("[LIMPEZA] Concluida.\n")


def _carregar_modulo(nome, path):
    spec = importlib.util.spec_from_file_location(nome, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nome] = mod
    spec.loader.exec_module(mod)
    return mod


def _rodar_data_wrangling():
    """Executa data wrangling se o CSV processado nao existir."""
    if CAMINHO_CSV.exists():
        return
    entradas = [
        ROOT / 'DB' / 'historico_estoque_atual.csv',
        ROOT / 'DB' / 'historico_estoque.csv',
    ]
    entrada = next((p for p in entradas if p.exists()), None)
    if not entrada:
        print(f"[ERRO] Nenhum CSV de historico encontrado em DB/. Coloque historico_estoque_atual.csv ou historico_estoque.csv")
        sys.exit(1)
    dw = _carregar_modulo('dw_historico', ROOT / 'data_wrangling' / 'dw_historico.py')
    print("\n[0/2] Executando data wrangling (CSV processado inexistente)...")
    dw.processar_historico_estoque(
        caminho_entrada=str(entrada),
        caminho_saida=str(CAMINHO_CSV),
        min_observacoes=30,
        criar_serie_completa=True,
    )
    if not CAMINHO_CSV.exists():
        print("[ERRO] Data wrangling nao gerou o arquivo esperado.")
        sys.exit(1)


def _rodar_analise_exploratoria():
    """Executa analise exploratoria e retorna (df, agregados, stats_sku, sku_representativo, top10_skus, top300_skus). Gera figura1-4."""
    ae = _carregar_modulo('ae', ROOT / 'analises' / 'analise_exploratoria_sazonalidade.py')
    out = ae.main(usar_nomes_tcc=True, caminho_dados=str(CAMINHO_CSV))
    df = out[0]
    agregados = out[1]
    stats_sku = out[2]
    sku_rep = out[3]
    top10_skus = out[4]
    top300_skus = out[5] if len(out) > 5 else top10_skus
    return df, agregados, stats_sku, sku_rep, top10_skus, top300_skus


N_CANDIDATOS = 300
N_MELHORES = 10
EPSILON_MAE_IGUAL = 0.01  # diff MAE < isso = metricas identicas (insatisfatorio)


def _rodar_comparacao_300_selecionar_10(top300_skus):
    """
    Fase 1: Roda comparacao (metricas apenas) para ate 300 candidatos.
    Fase 2: Filtra (sem teste constante, sem resultados insatisfatorios), ranqueia por melhor MAE, escolhe 10.
    Fase 3: Gera figuras, relatorios, Fig 5-7 e Tabela 2 apenas para os 10 melhores.
    Fig 5-7 usam o melhor dos 10 (menor MAE).
    """
    import csv
    try:
        import psutil
        _psutil_ok = True
    except ImportError:
        _psutil_ok = False

    cmp = _carregar_modulo('cmp', ROOT / 'modelos' / 'comparacao_modelos_previsao.py')
    candidatos = top300_skus[:N_CANDIDATOS] if top300_skus else []
    lista_300 = []
    tempos_sku = []
    t0 = time.time()
    CPU_LIMIT = 80.0

    _log(f"[CPU] Limite: {CPU_LIMIT}% | Monitoramento: {'Ativo' if _psutil_ok else 'Desabilitado'}")
    if _psutil_ok:
        try:
            n_cores = psutil.cpu_count()
            _log(f"[CPU] Cores: {n_cores} | n_jobs ~{max(1, int(n_cores * CPU_LIMIT / 100))}")
        except Exception:
            pass

    _log("\n[INFO] Os modelos preveem ESTOQUE (saldo), nao vendas. GP(t) na elencacao = soma das previsoes de estoque.")

    # --- Fase 1: metricas apenas para N_CANDIDATOS ---
    _log(f"\n[FASE 1/3] Rodando metricas para {len(candidatos)} candidatos (sem figuras/relatorios)...")
    for k, sku in enumerate(candidatos, 1):
        if _psutil_ok and k > 1:
            _aguardar_cpu_abaixo_limite(CPU_LIMIT)
        t_ini = time.time()
        _log(f"\n[CANDIDATO {k}/{len(candidatos)}] {sku}...")
        if k > 1 and tempos_sku:
            tm = sum(tempos_sku) / len(tempos_sku)
            rest = len(candidatos) - k + 1
            _log(f"  [ESTIMATIVA] Media: {tm:.1f}s/SKU | Restantes: {rest} | ETA: {tm * rest / 60:.1f} min")
        res = cmp.run_comparison_for_sku(str(sku), str(CAMINHO_CSV), horizonte_previsao=30)
        if res is None:
            _log(f"  [AVISO] SKU {sku} ignorado (serie invalida ou insuficiente).")
            continue
        lista_300.append(res)
        dt = time.time() - t_ini
        tempos_sku.append(dt)
        _log(f"  [CANDIDATO {k}/{len(candidatos)}] Concluido em {dt:.1f}s")
        if k < len(candidatos):
            time.sleep(0.3)

    if not lista_300:
        _log("[ERRO] Nenhum candidato processado com sucesso.")
        return

    _log(f"\n[FASE 1] Concluida em {(time.time()-t0)/60:.1f} min. {len(lista_300)} candidatos com metricas.")

    # Salva CSV dos 300 (para auditoria e doc)
    path_300 = DIR_RESULTADOS / 'candidatos_300_metricas.csv'
    path_300.parent.mkdir(parents=True, exist_ok=True)
    with open(path_300, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f, delimiter=';')
        w.writerow(['SKU', 'Modelo', 'MAE', 'RMSE', 'MAPE', 'teste_constante'])
        for r in lista_300:
            tc = r.get('teste_constante', False)
            for m in r.get('metricas', []):
                w.writerow([
                    r.get('sku', ''),
                    m.get('modelo', ''),
                    m.get('mae'),
                    m.get('rmse'),
                    m.get('mape'),
                    'sim' if tc else 'nao',
                ])
    _log(f"[FASE 1] CSV salvo: {path_300}")

    # --- Fase 2: filtrar, ranquear, escolher 10 ---
    _log("\n[FASE 2/3] Filtrando e selecionando os 10 melhores...")
    elegiveis = []
    for r in lista_300:
        if r.get('teste_constante', False):
            continue
        ms = r.get('metricas', [])
        if not ms:
            continue
        maes = [x.get('mae') for x in ms if x.get('mae') is not None]
        if not maes:
            continue
        best_mae = min(maes)
        diff_mae = max(maes) - min(maes)
        if diff_mae < EPSILON_MAE_IGUAL:
            continue  # insatisfatorio: todos modelos iguais
        elegiveis.append((r, best_mae))

    elegiveis.sort(key=lambda x: x[1])
    top10_resultados = [x[0] for x in elegiveis[:N_MELHORES]]

    n_const = sum(1 for r in lista_300 if r.get('teste_constante', False))
    n_insat = len(lista_300) - n_const - len(elegiveis)
    _log(f"  Excluidos: {n_const} (teste constante), {n_insat} (metricas insatisfatorias).")
    _log(f"  Elegiveis: {len(elegiveis)}. Top {N_MELHORES}: {[r['sku'] for r in top10_resultados]}.")

    if len(top10_resultados) < N_MELHORES:
        _log(f"  [AVISO] Apenas {len(top10_resultados)} SKUs elegiveis (esperados {N_MELHORES}).")

    if not top10_resultados:
        _log("[ERRO] Nenhum SKU elegivel apos filtros.")
        return

    # Melhor dos 10 para Fig 5-7 (menor MAE)
    best_of_10 = top10_resultados[0]['sku']

    # --- Fase 3: figuras, relatorios, Fig 5-7, Tabela 2 ---
    _log(f"\n[FASE 3/3] Gerando figuras e relatorios para os {len(top10_resultados)} melhores (Fig 5-7: {best_of_10})...")
    for r in top10_resultados:
        cmp.visualizar_comparacao(r)
        cmp.gerar_relatorio_comparacao(r, salvar_tabela=False)
    cmp.salvar_figuras_tcc_multiplos_skus(top10_resultados, DIR_FIGURAS_TCC, sku_figura4=str(best_of_10))
    cmp.gerar_tabela_02_multiplos_skus(top10_resultados, str(ROOT / 'resultados' / 'tabelas_tcc' / 'tabela_02_desempenho_modelos.csv'))

    _log(f"\n[OK] Comparacao concluida. {len(lista_300)} candidatos processados; {len(top10_resultados)} melhores com figuras e Tabela 2.")

    # --- Elencacao final: ranking por R(t), U(t), GP(t) -> retorno do script ---
    df_elencacao = _gerar_elencacao_final(top10_resultados)
    return df_elencacao


def _score_elencacao(rentabilidade, nivel_urgencia, giro_futuro_previsto,
                     peso_r=0.4, peso_u=0.3, peso_g=0.3):
    """Score de elencacao = peso_r*R_norm + peso_u*U_norm + peso_g*GP_norm (mesmo teste_elencacao)."""
    def _safe(x, default=0.0):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return float(x)
    rent_val = _safe(rentabilidade)
    rent_norm = min(1.0, rent_val / 100.0) if rent_val > 0 else 0.0
    u_val = _safe(nivel_urgencia)
    urgencia_norm = (1.0 / (1.0 + u_val)) if u_val >= 0 else 0.0
    urgencia_norm = min(1.0, urgencia_norm)
    gp_val = _safe(giro_futuro_previsto)
    giro_norm = min(1.0, gp_val / 1000.0) if gp_val > 0 else 0.0
    return peso_r * rent_norm + peso_u * urgencia_norm + peso_g * giro_norm


def _gerar_elencacao_final(top10_resultados):
    """
    Gera ranking de elencacao (R(t), U(t), GP(t)) para os 10 melhores.
    Salva resultados/elencacao_final.csv, imprime tabela e retorna o DataFrame.
    """
    _log("\n" + "=" * 80)
    _log("ELENCAO FINAL (valor da ferramenta de elencacao)")
    _log("=" * 80)
    _log("  Metricas: R(t)=Rentabilidade, U(t)=Nivel de Urgencia, GP(t)=Giro Futuro Previsto (soma previsoes de ESTOQUE)")

    path_vendas = ROOT / 'DB' / 'venda_produtos_atual.csv'
    df_vendas = None
    if path_vendas.exists():
        try:
            df_v = pd.read_csv(path_vendas, low_memory=False)
            df_v['created_at'] = pd.to_datetime(df_v['created_at'], errors='coerce')
            df_v['quantidade'] = pd.to_numeric(df_v['quantidade'], errors='coerce')
            df_v['valor_unitario'] = pd.to_numeric(df_v['valor_unitario'], errors='coerce')
            df_v['custo_unitario'] = pd.to_numeric(df_v['custo_unitario'], errors='coerce')
            df_v = df_v[df_v['sku'].notna()]
            df_vendas = df_v
        except Exception as e:
            _log(f"  [AVISO] Falha ao carregar vendas: {e}")
    else:
        _log(f"  [AVISO] {path_vendas} nao encontrado. R(t) e U(t) serao 0; ranking usa apenas GP(t).")

    skus_10 = [str(r['sku']) for r in top10_resultados]
    agg_vendas = None
    venda_media = None
    if df_vendas is not None and len(df_vendas) > 0:
        f = df_vendas[df_vendas['sku'].astype(str).isin(skus_10)]
        if len(f) > 0:
            agg = f.groupby(f['sku'].astype(str)).agg(
                valor_unitario=('valor_unitario', 'mean'),
                custo_unitario=('custo_unitario', 'mean'),
                quantidade=('quantidade', 'sum')
            ).reset_index()
            agg['rentabilidade'] = agg['valor_unitario'] - agg['custo_unitario']
            agg_vendas = agg.set_index('sku')
            d = f.copy()
            d['dia'] = d['created_at'].dt.date
            vm = d.groupby([d['sku'].astype(str), 'dia'])['quantidade'].sum().groupby(level=0).mean().reset_index()
            vm.columns = ['sku', 'venda_media_diaria']
            venda_media = vm.set_index('sku')

    linhas = []
    for r in top10_resultados:
        sku = str(r['sku'])
        st = r.get('serie_treino')
        estoque_atual = float(st.iloc[-1]) if st is not None and len(st) > 0 else np.nan
        prev = r.get('previsoes', {}).get('sarima_mensal')
        if prev is None:
            for k in ('arima', 'exponencial', 'media_movel'):
                p = r.get('previsoes', {}).get(k)
                if p is not None:
                    prev = p
                    break
        gp = float(np.sum(prev)) if prev is not None and len(prev) > 0 else np.nan
        rent = None
        vmd = None
        if agg_vendas is not None and sku in agg_vendas.index:
            rent = float(agg_vendas.loc[sku, 'rentabilidade'])
        if venda_media is not None and sku in venda_media.index:
            vmd = float(venda_media.loc[sku, 'venda_media_diaria'])
        nivel_urgencia = (estoque_atual / vmd) if vmd and vmd > 0 else np.nan
        score = _score_elencacao(rent, nivel_urgencia, gp)
        linhas.append({
            'sku': sku,
            'estoque_atual': estoque_atual,
            'R(t)_rentabilidade': rent if rent is not None else np.nan,
            'U(t)_nivel_urgencia': nivel_urgencia,
            'GP(t)_giro_futuro_previsto': gp,
            'score_elencacao': score,
        })

    df = pd.DataFrame(linhas)
    df = df.sort_values('score_elencacao', ascending=False).reset_index(drop=True)
    df['ranking'] = range(1, len(df) + 1)
    df = df[['ranking', 'sku', 'estoque_atual', 'R(t)_rentabilidade', 'U(t)_nivel_urgencia', 'GP(t)_giro_futuro_previsto', 'score_elencacao']]

    path_out = DIR_RESULTADOS / 'elencacao_final.csv'
    path_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_out, index=False, encoding='utf-8-sig', sep=';')
    _log(f"\n[OK] Elencacao salva: {path_out}")

    _log("\nRANKING DE ELENCAO (ordenado por score_elencacao):")
    _log("-" * 80)
    _log(df.to_string(index=False))
    _log("-" * 80)
    _log("  (Valor final da ferramenta de elencacao = este ranking + CSV acima)")
    return df


class Logger:
    """Logger que escreve no console e em arquivo TXT."""
    def __init__(self, log_file):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.file_handle = open(self.log_file, 'w', encoding='utf-8')
        self.start_time = time.time()
    
    def log(self, msg, flush=True):
        """Escreve mensagem no console e no arquivo."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        msg_with_time = f"[{timestamp}] {msg}"
        print(msg, flush=flush)
        self.file_handle.write(msg_with_time + '\n')
        self.file_handle.flush()
    
    def close(self):
        """Fecha o arquivo de log."""
        if self.file_handle:
            self.file_handle.close()
    
    def get_elapsed(self):
        """Retorna tempo decorrido desde o inicio em segundos."""
        return time.time() - self.start_time


def _log(msg, flush=True):
    """Wrapper para o logger global."""
    if _logger:
        _logger.log(msg, flush=flush)
    else:
        print(msg, flush=flush)


def main():
    global _logger
    
    # Inicializa logger com arquivo de log
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = DIR_LOGS / f'log_execucao_{timestamp}.txt'
    _logger = Logger(log_file)
    
    t0 = time.time()
    inicio_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    _log("=" * 80)
    _log("GERADOR DE FIGURAS TCC (figura1 ... figura7)")
    _log("=" * 80)
    _log(f"Inicio da execucao: {inicio_str}")
    _log(f"Arquivo de log: {log_file}")
    
    # Informações sobre limite de CPU
    try:
        import os
        n_cores = os.cpu_count() or 4
        n_jobs_limitado = max(1, int(n_cores * 0.8))
        _log(f"[CPU] Limite configurado: 80% | Cores disponiveis: {n_cores} | n_jobs limitado: ~{n_jobs_limitado}")
        if not PSUTIL_AVAILABLE:
            _log(f"[CPU] Monitoramento em tempo real: DESABILITADO (psutil nao instalado)")
            _log(f"[CPU] Para habilitar: pip install psutil")
        else:
            _log(f"[CPU] Monitoramento em tempo real: ATIVO")
    except:
        pass
    
    _log(f"\nSaida: {DIR_FIGURAS_TCC}")
    
    # Limpa diretorios de saida antes de começar
    _limpar_diretorios_saida()
    
    # Cria diretorios necessarios
    DIR_FIGURAS_TCC.mkdir(parents=True, exist_ok=True)
    DIR_TABELAS_TCC.mkdir(parents=True, exist_ok=True)
    DIR_FIGURAS_MODELOS.mkdir(parents=True, exist_ok=True)

    _log("\n[PASSO 0] Verificando data wrangling...")
    _rodar_data_wrangling()
    _log("[PASSO 0] OK.")

    _log("\n" + "=" * 80)
    _log("[1/2] Analise exploratoria -> figura1, figura2, figura3, figura4")
    _log("=" * 80)
    t1 = time.time()
    df, agregados, stats_sku, sku_rep, top10_skus, top300_skus = _rodar_analise_exploratoria()
    dt1 = time.time() - t1
    _log(f"\n[1/2] Concluido em {dt1:.1f}s")
    if df is None:
        _log("[ERRO] Analise exploratoria falhou.")
        sys.exit(1)
    if not sku_rep:
        _log("[ERRO] Nenhum SKU representativo. Abortando.")
        sys.exit(1)
    sku_rep = str(sku_rep)
    _log(f"[OK] SKU representativo (figura4): {sku_rep}")
    _log(f"[OK] Top 10 SKUs (exploratoria): {top10_skus}")
    _log(f"[OK] Top {N_CANDIDATOS} candidatos (comparacao): {len(top300_skus or [])} SKUs")
    LIMITE_ZEROS = 30.0
    if stats_sku is not None and len(stats_sku) > 0 and 'pct_zeros' in stats_sku.columns:
        row = stats_sku[stats_sku['sku'].astype(str) == sku_rep]
        if len(row) > 0:
            pz = float(row.iloc[0]['pct_zeros'])
            if pz <= LIMITE_ZEROS:
                _log(f"[VERIFICACAO] SKU {sku_rep} tem {pz:.1f}% zeros (limite {LIMITE_ZEROS:.0f}%). OK.")
            else:
                _log(f"[ERRO] SKU {sku_rep} tem {pz:.1f}% zeros (limite {LIMITE_ZEROS:.0f}%). Nao deveria ter sido escolhido.")
                sys.exit(1)
    if not top300_skus:
        _log("[ERRO] Nenhum candidato no top 300. Abortando.")
        sys.exit(1)

    _log("\n" + "=" * 80)
    _log(f"[2/2] Comparacao: {N_CANDIDATOS} candidatos -> selecao dos {N_MELHORES} melhores -> figura5, figura6, figura7 + Tabela 2")
    _log("=" * 80)
    _log("  (Fase 1: metricas para candidatos; Fase 2: filtrar constante/insatisfatorio, ranquear; Fase 3: figuras para os 10 melhores)")
    t2 = time.time()
    df_elencacao = _rodar_comparacao_300_selecionar_10(top300_skus)
    dt2 = time.time() - t2
    _log(f"\n[2/2] Concluido em {dt2:.1f}s")

    dt_total = time.time() - t0
    fim_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    _log("\n" + "=" * 80)
    _log("CONCLUIDO")
    _log("=" * 80)
    _log(f"Inicio: {inicio_str}")
    _log(f"Fim: {fim_str}")
    _log(f"Tempo total: {dt_total/60:.1f} min ({dt_total:.1f}s)")
    _log(f"\nFiguras em {DIR_FIGURAS_TCC}:")
    for i in range(1, 8):
        p = DIR_FIGURAS_TCC / f'figura{i}.png'
        if p.exists():
            _log(f"  [OK] figura{i}.png")
        else:
            _log(f"  [--] figura{i}.png (nao gerado)")
    _log(f"\nValor final da ferramenta de elencacao: {DIR_RESULTADOS / 'elencacao_final.csv'}")
    _log(f"Arquivo de log: {log_file}")
    _log("\nDocumentacao: documentacao/COMO_GERAR_FIGURAS_TCC.md")
    _log("=" * 80)
    
    if _logger:
        _logger.close()
    
    return df_elencacao


if __name__ == '__main__':
    try:
        resultado = main()
        if resultado is not None:
            print(f"\n[RETORNO] Valor final da ferramenta de elencacao: DataFrame com {len(resultado)} linhas (ranking). Salvo em resultados/elencacao_final.csv")
    except Exception as e:
        erro_msg = f"\n[ERRO] {e}"
        print(erro_msg)
        if _logger:
            _logger.log(erro_msg)
            import traceback
            _logger.log(traceback.format_exc())
            _logger.close()
        else:
            import traceback
            traceback.print_exc()
        sys.exit(1)
