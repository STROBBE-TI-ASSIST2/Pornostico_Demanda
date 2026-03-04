
# 📈 Pronóstico de Demanda

Aplicación web desarrollada en **Python + Flask** para generar **pronósticos de demanda por producto** utilizando series de tiempo y segmentación de patrones de demanda.

El sistema permite analizar ventas históricas, clasificar el tipo de demanda de cada SKU y aplicar el modelo de pronóstico más adecuado automáticamente.

---

# 🚀 Características

- Lectura automática de ventas desde **Excel**
- Separación de ventas por:
  - LOCAL
  - EXPO
- Conversión de datos a formato de **series de tiempo**
- **Segmentación de demanda** mediante:
  - ADI
  - CV²
- Aplicación automática de modelos de pronóstico
- Interfaz web para ejecutar el proceso
- Exportación de resultados a **Excel**
- Consulta de pronóstico por **código de producto**

---

# 🧠 Segmentación de demanda

El sistema clasifica cada producto en uno de estos patrones:

| Tipo de demanda | Condición |
|---|---|
| Smooth | ADI ≤ 1.32 y CV² ≤ 0.49 |
| Erratic | ADI ≤ 1.32 y CV² > 0.49 |
| Intermittent | ADI > 1.32 y CV² ≤ 0.49 |
| Lumpy | ADI > 1.32 y CV² > 0.49 |

Esto permite aplicar el **modelo estadístico más adecuado** según el comportamiento del SKU.

---

# 📊 Modelos de pronóstico utilizados

## Demanda continua
- SES (Simple Exponential Smoothing)
- Holt (tendencia amortiguada)
- ARIMA
- XGBoost (opcional)

## Demanda intermitente
- Croston
- SBA
- TSB

---

# 🛠️ Tecnologías

- Python
- Flask
- Pandas
- Numpy
- Statsmodels
- Openpyxl
- XGBoost (opcional)
- Pmdarima (opcional)

---

# 📁 Estructura del proyecto

```bash

Pronostico_Demanda
│
├── app.py
├── requirements.txt
│
├── forecast
│ ├── io.py
│ ├── segmentation.py
│ ├── models.py
│ └── pipeline.py
│
├── templates
│ └── index.html
│
└── static
└── styles.css
