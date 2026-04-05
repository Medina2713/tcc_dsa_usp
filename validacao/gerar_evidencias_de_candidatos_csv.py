#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regenera CSVs de evidencia para o TCC a partir de arquivos ja existentes,
sem rodar de novo os 300 SKUs no auto_arima.

Uso: python validacao/gerar_evidencias_de_candidatos_csv.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIR_TAB = ROOT / "resultados" / "tabelas_tcc"


def _melhor_modelo_por_sku(df_sku: pd.DataFrame) -> str | None:
    """
    Mesmo desempate que modelos/evidencias_orientadora_tcc._melhor_modelo_unico:
    menor MAE, depois RMSE, depois MAPE (nao usar idxmin no pivot — empates em MAE
    ficariam arbitrarios pela ordem alfabetica das colunas).
    """
    rows = []
    for _, r in df_sku.iterrows():
        mae = r["MAE"]
        if mae is None or (isinstance(mae, float) and np.isnan(mae)):
            continue
        rmse = r["RMSE"]
        mape = r["MAPE"]
        rows.append(
            (
                float(mae),
                float(rmse) if pd.notna(rmse) else np.inf,
                float(mape) if pd.notna(mape) else np.inf,
                str(r["Modelo"]),
            )
        )
    if not rows:
        return None
    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    return rows[0][3]


def main() -> int:
    DIR_TAB.mkdir(parents=True, exist_ok=True)
    cand = ROOT / "resultados" / "candidatos_300_metricas.csv"
    if not cand.exists():
        print(f"[ERRO] Falta {cand}")
        return 1
    df = pd.read_csv(cand, sep=";", encoding="utf-8-sig")
    df_ok = df[df["teste_constante"].astype(str).str.lower() == "nao"].copy()

    win_list = []
    for sku, g in df_ok.groupby("SKU"):
        w = _melhor_modelo_por_sku(g)
        if w:
            win_list.append({"sku": sku, "melhor_modelo_mae": w})
    winners = pd.DataFrame(win_list)
    winners.to_csv(DIR_TAB / "vitoria_modelo_por_sku.csv", index=False, sep=";", encoding="utf-8-sig")
    vc = winners["melhor_modelo_mae"].value_counts().reset_index()
    vc.columns = ["modelo", "n_skus_vencedor"]
    vc["pct"] = (100.0 * vc["n_skus_vencedor"] / vc["n_skus_vencedor"].sum()).round(2)
    vc.to_csv(DIR_TAB / "taxa_vitoria_modelos_resumo.csv", index=False, sep=";", encoding="utf-8-sig")

    agg = df_ok.groupby("Modelo", as_index=False).agg(
        n=("SKU", "count"),
        MAE_medio=("MAE", "mean"),
        RMSE_medio=("RMSE", "mean"),
        MAPE_medio=("MAPE", "mean"),
    )
    agg.to_csv(DIR_TAB / "medias_por_modelo_todos_candidatos_validos.csv", index=False, sep=";", encoding="utf-8-sig")

    t2 = DIR_TAB / "tabela_02_desempenho_modelos.csv"
    if t2.exists():
        t2df = pd.read_csv(t2, sep=";", encoding="utf-8-sig")
        skus = t2df["SKU"].unique()
        sub = df_ok[df_ok["SKU"].isin(skus)]
        agg10 = sub.groupby("Modelo", as_index=False).agg(
            n=("SKU", "count"),
            MAE_medio=("MAE", "mean"),
            RMSE_medio=("RMSE", "mean"),
            MAPE_medio=("MAPE", "mean"),
        )
        agg10.to_csv(DIR_TAB / "medias_por_modelo_apenas_top10_tcc.csv", index=False, sep=";", encoding="utf-8-sig")

    piv = df_ok.pivot_table(index="SKU", columns="Modelo", values="MAE", aggfunc="first")
    if "ARIMA Simples" in piv.columns and "SARIMA Mensal (m=30)" in piv.columns:
        diff = (piv["ARIMA Simples"] - piv["SARIMA Mensal (m=30)"]).abs()
        both = piv["ARIMA Simples"].notna() & piv["SARIMA Mensal (m=30)"].notna()
        out = pd.DataFrame(
            {
                "sku": piv.index[both],
                "mae_arima": piv.loc[both, "ARIMA Simples"].values,
                "mae_sarima": piv.loc[both, "SARIMA Mensal (m=30)"].values,
                "mae_numericamente_iguais": (diff[both] < 1e-9).values,
            }
        )
        out.to_csv(DIR_TAB / "heuristica_mae_arima_vs_sarima_por_sku.csv", index=False, sep=";", encoding="utf-8-sig")

    dn = DIR_TAB / "dados_numericos_figuras_5_7.csv"
    if dn.exists():
        dfn = pd.read_csv(dn, sep=";", encoding="utf-8-sig")
        sku = str(dfn["sku"].iloc[0])
        rows = []
        for modelo, g in dfn.groupby("modelo"):
            g = g.sort_values("indice_teste")
            reals = g["valor_real"].astype(float).values
            preds = g["valor_previsto"].astype(float).values
            x = g["indice_teste"].astype(float).values
            mae_h = float(np.mean(np.abs(reals - preds)))
            rmse_h = float(np.sqrt(np.mean((reals - preds) ** 2)))
            slope = float(np.polyfit(x, preds, 1)[0]) if len(x) >= 2 else float("nan")
            std_pred = float(np.std(preds))
            n = len(preds)
            k = min(7, n)
            mf = float(np.mean(preds[:k]))
            ml = float(np.mean(preds[n - k :]))
            rows.append(
                {
                    "sku": sku,
                    "modelo": modelo,
                    "mae_horizonte_teste": round(mae_h, 6),
                    "rmse_horizonte_teste": round(rmse_h, 6),
                    "inclinacao_previsao_unid_por_dia": round(slope, 6),
                    "desvio_padrao_previsao": round(std_pred, 6),
                    "media_previsao_primeiros_7d": round(mf, 6),
                    "media_previsao_ultimos_7d": round(ml, 6),
                    "amplitude_media_ultimos_7_menos_primeiros_7": round(ml - mf, 6),
                }
            )
        pd.DataFrame(rows).to_csv(
            DIR_TAB / "resumo_quantitativo_figuras_5_7.csv", index=False, sep=";", encoding="utf-8-sig"
        )

    print(f"[OK] Evidencias em {DIR_TAB}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
