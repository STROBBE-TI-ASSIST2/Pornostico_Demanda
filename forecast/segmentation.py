import numpy as np
import pandas as pd

def compute_adi_cv2(df: pd.DataFrame, codigo_col="codigo", ventas_col="ventas") -> pd.DataFrame:
    records = []
    for codigo, g in df.groupby(codigo_col):
        y = g[ventas_col].to_numpy()
        total_periods = len(y)
        non_zero = y[y > 0]

        if len(non_zero) == 0:
            adi = np.inf
            cv2 = np.inf
        else:
            adi = total_periods / len(non_zero)
            mean_nz = non_zero.mean()
            cv2 = (non_zero.std() / mean_nz) ** 2 if mean_nz > 0 else np.inf

        records.append({"codigo": str(codigo), "ADI": adi, "CV2": cv2})
    return pd.DataFrame(records)

def classify_segment(adi, cv2) -> str:
    if adi <= 1.32 and cv2 <= 0.49:
        return "Smooth"
    elif adi <= 1.32 and cv2 > 0.49:
        return "Erratic"
    elif adi > 1.32 and cv2 <= 0.49:
        return "Intermittent"
    else:
        return "Lumpy"

def segmentar(df_long: pd.DataFrame) -> pd.DataFrame:
    seg = compute_adi_cv2(df_long, "codigo", "ventas")
    seg["segmento"] = seg.apply(lambda r: classify_segment(r["ADI"], r["CV2"]), axis=1)
    return seg[["codigo", "segmento"]]
