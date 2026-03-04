# Pronóstico de Demanda (LOCAL / EXPO) — Flask + Segmentación ADI/CV²

Aplicación web (Flask) para ejecutar un **pipeline de pronóstico de ventas** por **código de producto**, separando **LOCAL** y **EXPO**, con **segmentación de demanda** (Smooth / Erratic / Intermittent / Lumpy) y exportación a **Excel**.

> Repo: `STROBBE-TI-ASSIST2/Pornostico_Demanda`

---

## ✨ Qué hace

- Lee ventas históricas desde un **Excel** con hojas `local` y `expo` (formato “ancho”: columnas por mes).
- Convierte ventas a formato “largo” (`codigo`, `mes`, `ventas`). :contentReference[oaicite:0]{index=0}
- Filtra **solo códigos activos de línea** desde otro Excel (`sheet=DE`) usando la columna **“PP 2025”** con el valor **`1SCHS`**. :contentReference[oaicite:1]{index=1}
- Segmenta cada código usando **ADI** y **CV²**:
  - Smooth: `ADI <= 1.32` y `CV² <= 0.49`
  - Erratic: `ADI <= 1.32` y `CV² > 0.49`
  - Intermittent: `ADI > 1.32` y `CV² <= 0.49`
  - Lumpy: `ADI > 1.32` y `CV² > 0.49` :contentReference[oaicite:2]{index=2}
- Genera pronósticos a **3 meses** (configurable) por segmento y por origen. :contentReference[oaicite:3]{index=3}
- UI web para:
  - ejecutar el pipeline (`PREDICCIÓN`)
  - consultar por código
  - descargar resultados en `XLSX` (LOCAL / EXPO / TOTAL / SEGMENTOS). :contentReference[oaicite:4]{index=4}

---

## 🧠 Modelos usados (por segmento)

Según el pipeline:

**Smooth / Erratic**
- SES
- Holt (damped trend)
- ARIMA (auto_arima; fallback a Holt si falta dependencia)
- XGBoost (forecast recursivo; fallback a Holt si falta dependencia)

**Intermittent / Lumpy**
- TSB
- Croston
- SBA :contentReference[oaicite:5]{index=5}

---

## 🧰 Stack

- Python
- Flask :contentReference[oaicite:6]{index=6}
- pandas / numpy :contentReference[oaicite:7]{index=7}
- statsmodels (SES/Holt) :contentReference[oaicite:8]{index=8}
- Export a Excel con `openpyxl` (engine) :contentReference[oaicite:9]{index=9}

---

## 📁 Estructura del proyecto

- `app.py` — servidor Flask, rutas y exportación XLSX :contentReference[oaicite:10]{index=10}  
- `forecast/`
  - `io.py` — lectura y transformación de Excels (wide→long) + filtro de códigos :contentReference[oaicite:11]{index=11}
  - `segmentation.py` — segmentación ADI/CV² :contentReference[oaicite:12]{index=12}
  - `models.py` — modelos (SES, Holt, ARIMA auto, XGB, Croston/SBA/TSB) :contentReference[oaicite:13]{index=13}
  - `pipeline.py` — ejecución completa y armado de salidas (origen + total) :contentReference[oaicite:14]{index=14}
- `templates/index.html` — interfaz web :contentReference[oaicite:15]{index=15}
- `static/styles.css` — estilos
- `requirements.txt` — dependencias base :contentReference[oaicite:16]{index=16}

---

## 🚀 Instalación y ejecución

### 1) Crear entorno virtual
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
