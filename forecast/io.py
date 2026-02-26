import pandas as pd

def wide_to_long_sales(df_in: pd.DataFrame, start=None, end=None, max_header_rows=8) -> pd.DataFrame:
    df = df_in.copy()

    header_row = None
    for r in range(min(max_header_rows, len(df))):
        if (df.iloc[r].astype(str).str.strip() == "Código").any():
            header_row = r
            break

    if header_row is not None:
        df.columns = df.iloc[header_row]
        df = df.iloc[header_row + 1:].reset_index(drop=True)

    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed")].copy()

    if "Código" in df.columns:
        df = df.rename(columns={"Código": "codigo"})
    elif "codigo" not in df.columns:
        raise ValueError("No encontré la columna 'Código' para usar como 'codigo'.")

    df["codigo"] = (
        df["codigo"].astype(str).str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        .ffill()
    )
    df = df.dropna(subset=["codigo"])

    mes_cols = []
    for c in df.columns:
        s = str(c).strip().replace(".0", "")
        if s.isdigit() and len(s) == 6:
            mes_cols.append(c)

    if not mes_cols:
        raise ValueError("No detecté columnas de meses tipo 202201/202201.0.")

    out = df.melt(id_vars="codigo", value_vars=mes_cols, var_name="mes", value_name="ventas")
    out["mes"] = out["mes"].astype(str).str.replace(".0", "", regex=False)
    out["mes"] = pd.to_datetime(out["mes"] + "01", format="%Y%m%d")
    out["ventas"] = pd.to_numeric(out["ventas"], errors="coerce").fillna(0)

    if start is not None:
        out = out[out["mes"] >= pd.to_datetime(start)]
    if end is not None:
        out = out[out["mes"] <= pd.to_datetime(end)]

    out["codigo"] = out["codigo"].astype(str).str.replace(".", "", regex=False)
    out = out.sort_values(["codigo", "mes"]).reset_index(drop=True)
    return out

#Filtrar solo codigos de linea
def load_codigos_linea(path_codigos: str, sheet="DE", flag_col="PP 2025", flag_value="1SCHS") -> pd.Series:
    """
    Devuelve una Serie con los códigos activos de línea.
    """
    df0 = pd.read_excel(path_codigos, sheet_name=sheet)

    # usar fila 1 como header real
    df0.columns = df0.iloc[1]
    df0 = df0.iloc[2:].reset_index(drop=True)
    df0 = df0.loc[:, ~df0.columns.astype(str).str.contains(r"^Unnamed")]

    df0 = df0.rename(columns={
        "Código": "codigo",
        flag_col: "pp_flag"
    })

    df_filtrado = df0[df0["pp_flag"].astype(str).str.strip() == flag_value]

    codigos = (
        df_filtrado["codigo"]
        .astype(str)
        .str.strip()
        .dropna()
        .drop_duplicates()
    )

    return codigos

def load_inputs(
    path_ventas: str,
    sheet_local="local",
    sheet_expo="expo",
    start="2022-01-01",
    end="2025-09-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ventas_local = pd.read_excel(path_ventas, sheet_name=sheet_local)
    ventas_expo  = pd.read_excel(path_ventas, sheet_name=sheet_expo)

    df0_long = wide_to_long_sales(ventas_local, start=start, end=end)
    df1_long = wide_to_long_sales(ventas_expo,  start=start, end=end)

    return df0_long, df1_long
