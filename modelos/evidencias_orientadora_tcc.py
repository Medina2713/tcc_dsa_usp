"""
Gera CSVs de evidencia para a discussao do TCC (questoes da orientadora).
Chamado por gerar_figuras_tcc.py apos a Fase 1 (lista_300) e apos selecao do SKU Fig 5-7.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _melhor_modelo_unico(metricas: list):
    """Desempate: MAE, depois RMSE, depois MAPE."""
    rows = []
    for m in metricas:
        mae = m.get("mae")
        if mae is None or (isinstance(mae, float) and np.isnan(mae)):
            continue
        rows.append(m)
    if not rows:
        return None
    rows.sort(key=lambda x: (x["mae"], x.get("rmse", np.inf), x.get("mape", np.inf)))
    return rows[0].get("modelo")


def salvar_evidencias_orientadora(
    lista_300: list,
    top10_resultados: list,
    best_of_10_sku: str,
    sku_representativo_fig4,
    path_tabelas: Path,
    constantes: dict | None = None,
    log_fn=print,
):
    path_tabelas = Path(path_tabelas)
    path_tabelas.mkdir(parents=True, exist_ok=True)
    constantes = constantes or {}

    rows_a = []
    for r in lista_300:
        aud = r.get("auditoria_arima_sarima")
        if aud:
            rows_a.append(aud)
    if rows_a:
        df_a = pd.DataFrame(rows_a)
        p_a = path_tabelas / "evidencia_arima_sarima_por_sku.csv"
        df_a.to_csv(p_a, index=False, encoding="utf-8-sig", sep=";")
        log_fn(f"  [EVIDENCIA] {p_a.name} ({len(df_a)} SKUs)")

    vit_rows = []
    for r in lista_300:
        if r.get("teste_constante"):
            continue
        ms = r.get("metricas", [])
        w = _melhor_modelo_unico(ms)
        if w:
            vit_rows.append({"sku": r.get("sku"), "melhor_modelo_mae": w})
    if vit_rows:
        df_v = pd.DataFrame(vit_rows)
        df_v.to_csv(path_tabelas / "vitoria_modelo_por_sku.csv", index=False, encoding="utf-8-sig", sep=";")
        cnt = df_v["melhor_modelo_mae"].value_counts()
        df_cnt = cnt.reset_index()
        df_cnt.columns = ["modelo", "n_skus_vencedor"]
        df_cnt["pct"] = (100.0 * df_cnt["n_skus_vencedor"] / df_cnt["n_skus_vencedor"].sum()).round(2)
        p_cnt = path_tabelas / "taxa_vitoria_modelos_resumo.csv"
        df_cnt.to_csv(p_cnt, index=False, encoding="utf-8-sig", sep=";")
        log_fn(f"  [EVIDENCIA] vitoria_modelo_por_sku.csv, {p_cnt.name}")

    rows_m = []
    for r in lista_300:
        if r.get("teste_constante"):
            continue
        for m in r.get("metricas", []):
            rows_m.append({
                "sku": r.get("sku"),
                "modelo": m.get("modelo"),
                "mae": m.get("mae"),
                "rmse": m.get("rmse"),
                "mape": m.get("mape"),
            })
    if rows_m:
        df_m = pd.DataFrame(rows_m)
        agg = df_m.groupby("modelo", as_index=False).agg(
            n=("sku", "count"),
            MAE_medio=("mae", "mean"),
            RMSE_medio=("rmse", "mean"),
            MAPE_medio=("mape", "mean"),
        )
        p_m = path_tabelas / "medias_por_modelo_todos_candidatos_validos.csv"
        agg.to_csv(p_m, index=False, encoding="utf-8-sig", sep=";")
        log_fn(f"  [EVIDENCIA] {p_m.name}")

    rows_t = []
    for r in top10_resultados:
        if r.get("teste_constante"):
            continue
        for m in r.get("metricas", []):
            rows_t.append({
                "sku": r.get("sku"),
                "modelo": m.get("modelo"),
                "mae": m.get("mae"),
                "rmse": m.get("rmse"),
                "mape": m.get("mape"),
            })
    if rows_t:
        df_t = pd.DataFrame(rows_t)
        agg_t = df_t.groupby("modelo", as_index=False).agg(
            n=("sku", "count"),
            MAE_medio=("mae", "mean"),
            RMSE_medio=("rmse", "mean"),
            MAPE_medio=("mape", "mean"),
        )
        p_t = path_tabelas / "medias_por_modelo_apenas_top10_tcc.csv"
        agg_t.to_csv(p_t, index=False, encoding="utf-8-sig", sep=";")
        log_fn(f"  [EVIDENCIA] {p_t.name}")

    meta = {
        "sku_figuras_5_a_7": str(best_of_10_sku),
        "sku_representativo_figura_4": str(sku_representativo_fig4) if sku_representativo_fig4 is not None else None,
        "justificativa": (
            "Pool: ate 300 SKUs com sazonalidade/criterios da analise exploratoria. "
            "Excluem-se teste constante, CV_teste abaixo do minimo, range de teste baixo e SKUs onde todos os modelos tem MAE quase igual. "
            "Entre os elegiveis, o top 10 segue o menor melhor MAE. "
            "O SKU das Figuras 5-7 e escolhido entre os 30 melhores MAE: prioriza-se maior diferenca de MAE entre Holt-Winters, ARIMA e SARIMA (diff_mae_top3); "
            "se o SKU representativo da Figura 4 estiver no conjunto e seu MAE for no maximo 10% pior que o melhor do pool, usa-se para coerencia narrativa."
        ),
        "parametros_script": constantes,
    }
    p_j = path_tabelas / "criterio_selecao_figuras_5_7.json"
    with open(p_j, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log_fn(f"  [EVIDENCIA] {p_j.name}")
