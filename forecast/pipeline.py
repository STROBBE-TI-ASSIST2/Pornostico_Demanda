import pandas as pd
from .segmentation import segmentar
from .models import ses_forecast, tsb_forecast
from .models import (
    ses_forecast, tsb_forecast,
    holt_damped_forecast, arima_forecast, xgb_forecast,
    croston_forecast, sba_forecast
)

"""def build_monthly_series(g: pd.DataFrame) -> pd.Series:
    y = (
        g.set_index("mes")["ventas"]
        .asfreq("MS")
        .fillna(0)
        .astype(float)
    )
    return y
"""

def build_monthly_series(g: pd.DataFrame) -> pd.Series:
    """
    Construye serie mensual continua por código.
    Si hay duplicados (mismo mes repetido), los consolida (sum).
    """
    gg = g.copy()

    # Asegurar datetime
    gg["mes"] = pd.to_datetime(gg["mes"])

    # ✅ Consolidar duplicados por mes (regla: sumar ventas)
    gg = (gg.groupby("mes", as_index=True)["ventas"].sum().sort_index())

    # ✅ Forzar frecuencia mensual inicio de mes
    y = gg.asfreq("MS").fillna(0).astype(float)
    return y

def forecast_by_segment1(df_long: pd.DataFrame, origen: str, h_future=3, min_len=24) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df_long.sort_values(["codigo", "mes"]).copy()

    seg = segmentar(df)
    df = df.merge(seg, on="codigo", how="left")
    df["origen"] = origen

    out = []
    for codigo, g in df.groupby("codigo"):
        y = build_monthly_series(g)
        if len(y) < min_len:
            continue

        segmento = g["segmento"].iloc[0]

        future_dates = pd.date_range(
            start=y.index.max() + pd.offsets.MonthBegin(1),
            periods=h_future,
            freq="MS"
        )

        # ✅ Smooth / Erratic: 3 modelos
        if segmento in ("Smooth", "Erratic"):
            fc_holt = holt_damped_forecast(y, h=h_future).to_numpy()
            fc_arima = arima_forecast(y, h=h_future).to_numpy()
            fc_xgb  = xgb_forecast(y, h=h_future).to_numpy()

            for d, v in zip(future_dates, fc_holt):
                out.append({"codigo": str(codigo), "origen": origen, "mes": d, "yhat": float(v),
                            "modelo": "HOLT_DAMPED", "segmento": segmento})

            for d, v in zip(future_dates, fc_arima):
                out.append({"codigo": str(codigo), "origen": origen, "mes": d, "yhat": float(v),
                            "modelo": "ARIMA", "segmento": segmento})

            for d, v in zip(future_dates, fc_xgb):
                out.append({"codigo": str(codigo), "origen": origen, "mes": d, "yhat": float(v),
                            "modelo": "XGBOOST", "segmento": segmento})

        # ✅ Intermittent / Lumpy: 2 modelos
        else:
            vals_cros = croston_forecast(y, alpha=0.1, h=h_future)
            vals_sba  = sba_forecast(y, alpha=0.1, h=h_future)

            for d, v in zip(future_dates, vals_cros):
                out.append({"codigo": str(codigo), "origen": origen, "mes": d, "yhat": float(v),
                            "modelo": "CROSTON", "segmento": segmento})

            for d, v in zip(future_dates, vals_sba):
                out.append({"codigo": str(codigo), "origen": origen, "mes": d, "yhat": float(v),
                            "modelo": "SBA", "segmento": segmento})

    df_forecast = pd.DataFrame(out).sort_values(["origen", "codigo", "modelo", "mes"])
    df_segmentos = df[["codigo", "origen", "segmento"]].drop_duplicates().sort_values(["origen", "codigo"])
    return df_segmentos, df_forecast

def forecast_by_segment(df_long: pd.DataFrame, origen: str, h_future=3, min_len=24) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Devuelve:
      - df_segmentos: codigo, segmento, origen
      - df_forecast:  codigo, origen, mes, yhat, modelo, segmento
        (incluye múltiples modelos por segmento)
    """
    df = df_long.sort_values(["codigo", "mes"]).copy()

    seg = segmentar(df)
    df = df.merge(seg, on="codigo", how="left")
    df["origen"] = origen

    out = []
    for codigo, g in df.groupby("codigo"):
        y = build_monthly_series(g)
        if len(y) < min_len:
            continue

        segmento = g["segmento"].iloc[0]

        future_dates = pd.date_range(
            start=y.index.max() + pd.offsets.MonthBegin(1),
            periods=h_future,
            freq="MS"
        )

        # =========================
        # Smooth / Erratic: 4 modelos
        # =========================
        if segmento in ("Smooth", "Erratic"):
            fc_ses   = ses_forecast(y, h=h_future).to_numpy()
            fc_holt  = holt_damped_forecast(y, h=h_future).to_numpy()
            fc_arima = arima_forecast(y, h=h_future).to_numpy()
            fc_xgb   = xgb_forecast(y, h=h_future).to_numpy()

            for d, v in zip(future_dates, fc_ses):
                out.append({"codigo": str(codigo), "origen": origen, "mes": d, "yhat": float(v),
                            "modelo": "SES", "segmento": segmento})

            for d, v in zip(future_dates, fc_holt):
                out.append({"codigo": str(codigo), "origen": origen, "mes": d, "yhat": float(v),
                            "modelo": "HOLT_DAMPED", "segmento": segmento})

            for d, v in zip(future_dates, fc_arima):
                out.append({"codigo": str(codigo), "origen": origen, "mes": d, "yhat": float(v),
                            "modelo": "ARIMA", "segmento": segmento})

            for d, v in zip(future_dates, fc_xgb):
                out.append({"codigo": str(codigo), "origen": origen, "mes": d, "yhat": float(v),
                            "modelo": "XGBOOST", "segmento": segmento})

        # =========================
        # Intermittent / Lumpy: 3 modelos
        # =========================
        else:
            vals_tsb  = tsb_forecast(y, alpha=0.3, beta=0.3, h=h_future)
            vals_cros = croston_forecast(y, alpha=0.1, h=h_future)
            vals_sba  = sba_forecast(y, alpha=0.1, h=h_future)

            for d, v in zip(future_dates, vals_tsb):
                out.append({"codigo": str(codigo), "origen": origen, "mes": d, "yhat": float(v),
                            "modelo": "TSB", "segmento": segmento})

            for d, v in zip(future_dates, vals_cros):
                out.append({"codigo": str(codigo), "origen": origen, "mes": d, "yhat": float(v),
                            "modelo": "CROSTON", "segmento": segmento})

            for d, v in zip(future_dates, vals_sba):
                out.append({"codigo": str(codigo), "origen": origen, "mes": d, "yhat": float(v),
                            "modelo": "SBA", "segmento": segmento})

    df_forecast = pd.DataFrame(out).sort_values(["origen", "codigo", "modelo", "mes"])
    df_segmentos = df[["codigo", "origen", "segmento"]].drop_duplicates().sort_values(["origen", "codigo"])
    return df_segmentos, df_forecast

def run_pipeline(df_local: pd.DataFrame, df_expo: pd.DataFrame, codigos_linea: pd.Series, h_future=3) -> dict:

    df_local = df_local[df_local["codigo"].isin(codigos_linea)].copy()
    df_expo  = df_expo[df_expo["codigo"].isin(codigos_linea)].copy()

    seg_local, fc_local = forecast_by_segment(df_local, "LOCAL", h_future=h_future)
    seg_expo,  fc_expo  = forecast_by_segment(df_expo,  "EXPO",  h_future=h_future)

    fc_all = pd.concat([fc_local, fc_expo], ignore_index=True)

    # ✅ TOTAL por modelo (para consistencia)
    fc_total = (
        fc_all.groupby(["codigo", "mes", "modelo"], as_index=False)["yhat"].sum()
        .rename(columns={"yhat": "yhat_total"})
        .sort_values(["codigo", "modelo", "mes"])
    )

    segmentos_all = pd.concat([seg_local, seg_expo], ignore_index=True)

    return {
        "segmentos": segmentos_all,
        "forecast_origen": fc_all,
        "forecast_total": fc_total
    }