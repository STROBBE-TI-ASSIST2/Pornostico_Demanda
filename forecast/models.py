import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt

def ses_forecast(y: pd.Series, h: int = 3) -> pd.Series:
    model = SimpleExpSmoothing(y).fit(optimized=True)
    fc = model.forecast(h)
    fc.name = "forecast"
    return fc

def tsb_forecast(y: pd.Series, alpha=0.3, beta=0.3, h: int = 3) -> np.ndarray:
    """
    Implementación TSB (Teunter-Syntetos-Babai)
    """
    arr = y.to_numpy()
    p = np.zeros(len(arr))
    z = np.zeros(len(arr))

    if not np.any(arr > 0):
        return np.zeros(h)

    first_nonzero = int(np.argmax(arr > 0))
    p[first_nonzero] = 1
    z[first_nonzero] = arr[first_nonzero]

    for t in range(first_nonzero + 1, len(arr)):
        demand = arr[t]
        p[t] = p[t-1] + beta * ((demand > 0) - p[t-1])
        if demand > 0:
            z[t] = z[t-1] + alpha * (demand - z[t-1])
        else:
            z[t] = z[t-1]

    forecast = p[-1] * z[-1]
    return np.repeat(forecast, h)


#ARIMA auto (pmdarima)
try:
    from pmdarima import auto_arima
    _HAS_PMDARIMA = True
except Exception:
    _HAS_PMDARIMA = False

# XGBoost
try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False1


def holt_damped_forecast(y: pd.Series, h: int = 3) -> pd.Series:
    model = Holt(y, damped_trend=True, initialization_method="estimated").fit(optimized=True)
    fc = model.forecast(h)
    fc.name = "forecast"
    return fc


def arima_forecast(y: pd.Series, h: int = 3) -> pd.Series:
    """
    ARIMA con auto_arima. Si no está instalado pmdarima, cae a Holt Damped.
    """
    if not _HAS_PMDARIMA:
        return holt_damped_forecast(y, h=h)

    m = auto_arima(
        y,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        max_p=3, max_q=3, max_d=2
    )
    vals = m.predict(n_periods=h)
    idx = pd.date_range(start=y.index.max() + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    return pd.Series(vals, index=idx, name="forecast")


def _make_features(history: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"y": history})
    df["lag1"] = df["y"].shift(1)
    df["lag2"] = df["y"].shift(2)
    df["lag3"] = df["y"].shift(3)
    df["roll_mean_3"] = df["y"].shift(1).rolling(3).mean()
    df["roll_std_3"] = df["y"].shift(1).rolling(3).std()
    df["roll_mean_6"] = df["y"].shift(1).rolling(6).mean()
    df["month"] = df.index.month
    return df.dropna()


def xgb_forecast(y: pd.Series, h: int = 3) -> pd.Series:
    """
    Forecast recursivo con XGBoost usando lags + rolling stats (sin fuga).
    Si no hay xgboost, cae a Holt Damped.
    """
    if not _HAS_XGB:
        return holt_damped_forecast(y, h=h)

    df_feat = _make_features(y)
    if len(df_feat) < 12:
        return holt_damped_forecast(y, h=h)

    X = df_feat.drop(columns=["y"])
    y_target = df_feat["y"]

    model = xgb.XGBRegressor(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X, y_target)

    history = y.copy()
    preds = []
    for _ in range(h):
        feat_i = _make_features(history).iloc[[-1]].drop(columns=["y"])
        yhat = float(model.predict(feat_i)[0])
        next_date = history.index.max() + pd.offsets.MonthBegin(1)
        history.loc[next_date] = yhat
        preds.append((next_date, yhat))

    idx = [d for d, _ in preds]
    vals = [v for _, v in preds]
    return pd.Series(vals, index=idx, name="forecast")


# -------- Croston / SBA --------
def _croston_core(arr: np.ndarray, alpha: float = 0.1):
    if np.all(arr == 0):
        return 0.0, np.inf

    first = int(np.argmax(arr > 0))
    z = float(arr[first])
    p = 1.0
    q = 1

    for t in range(first + 1, len(arr)):
        if arr[t] > 0:
            z = z + alpha * (arr[t] - z)
            p = p + alpha * (q - p)
            q = 1
        else:
            q += 1
    return z, p


def croston_forecast(y: pd.Series, alpha: float = 0.1, h: int = 3) -> np.ndarray:
    arr = y.to_numpy(dtype=float)
    z, p = _croston_core(arr, alpha)
    if not np.isfinite(p) or p <= 0:
        return np.zeros(h)
    return np.repeat(z / p, h)


def sba_forecast(y: pd.Series, alpha: float = 0.1, h: int = 3) -> np.ndarray:
    arr = y.to_numpy(dtype=float)
    z, p = _croston_core(arr, alpha)
    if not np.isfinite(p) or p <= 0:
        return np.zeros(h)
    return np.repeat((1.0 - alpha / 2.0) * (z / p), h)