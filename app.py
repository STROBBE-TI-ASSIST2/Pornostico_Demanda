from flask import Flask, render_template, request, redirect, url_for, send_file
from io import BytesIO
import pandas as pd
from werkzeug import Response

from forecast.io import load_inputs, load_codigos_linea

from forecast.pipeline import run_pipeline

app = Flask(__name__)

# ====== CONFIG (ajusta rutas) ======
PATH_VENTAS = r"S:\Strobbe Sistemas\procedimientos\13 Administracion de TI\Proyectos Anuales\1. Pronostico de ventas PYTHON\TOTAL VENTAS 2022-2025.xls"

PATH_CODIGOS_LINEA = r"S:\Strobbe Sistemas\procedimientos\13 Administracion de TI\Proyectos Anuales\1. Pronostico de ventas PYTHON\DE OCTUBRE NOVIEMBRE DICIEMBRE 2025.xlsx"

# Cache simple en memoria (para empezar)
CACHE = {
    "result": None
}

@app.get("/")
def index():
    codigo = request.args.get("codigo", "").strip()
    result = CACHE["result"]
    pred_local, pred_expo, pred_total = [], [], []

    if result is not None and codigo:
        df_origen = result["forecast_origen"]
        df_total = result["forecast_total"]

        pred_local = (
            df_origen[(df_origen["codigo"] == codigo) & (df_origen["origen"] == "LOCAL")]
            .sort_values("mes")
            .to_dict("records")
        )
        pred_expo = (
            df_origen[(df_origen["codigo"] == codigo) & (df_origen["origen"] == "EXPO")]
            .sort_values("mes")
            .to_dict("records")
        )
        pred_total = (
            df_total[df_total["codigo"] == codigo]
            .sort_values("mes")
            .to_dict("records")
        )

    return render_template(
        "index.html",
        codigo=codigo,
        pred_local=pred_local,
        pred_expo=pred_expo,
        pred_total=pred_total,
        has_result=(result is not None),
    )

@app.post("/run")
def run_all() -> Response:
    h_future = int(request.form.get("h_future", 3))

    df_local, df_expo = load_inputs(PATH_VENTAS, start="2022-01-01", end="2025-09-01")
    codigos_linea = load_codigos_linea(PATH_CODIGOS_LINEA)
    result = run_pipeline(df_local, df_expo,codigos_linea, h_future=h_future)

    CACHE["result"] = result
    return redirect(url_for("index"))

@app.get("/download.xlsx")
def download_xlsx():
    if CACHE["result"] is None:
        return redirect(url_for("index"))

    result = CACHE["result"]
    output = BytesIO()

    df_origen = result["forecast_origen"].copy()
    df_total = result["forecast_total"].copy()

    local = df_origen[df_origen["origen"] == "LOCAL"].sort_values(["codigo", "mes"])
    expo = df_origen[df_origen["origen"] == "EXPO"].sort_values(["codigo", "mes"])

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        local.to_excel(writer, index=False, sheet_name="LOCAL")
        expo.to_excel(writer, index=False, sheet_name="EXPO")
        df_total.to_excel(writer, index=False, sheet_name="TOTAL")
        result["segmentos"].to_excel(writer, index=False, sheet_name="SEGMENTOS")

    output.seek(0)
    return send_file(
        output,
        as_attachment=True,
        download_name="pronostico_3_meses.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)
