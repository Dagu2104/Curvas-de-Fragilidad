
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy.stats import norm

G = 9.80665

# ============================================================
# NUMBA OPCIONAL
# ============================================================
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper

    def prange(*args, **kwargs):
        return range(*args)


# ============================================================
# FUNCIONES DE LECTURA
# ============================================================
def read_accel_file(uploaded_file):
    """
    Lee un acelerograma .txt/.csv.

    Acepta:
    - Una columna: aceleración
    - Dos columnas: tiempo + aceleración

    Retorna:
    time, acc_original_unit, dt_detected, formato
    """
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    text = raw.decode("utf-8", errors="ignore")

    text = re.sub(r"(?<=\d),(?=\d)", ".", text)

    df = pd.read_csv(
        io.StringIO(text),
        sep=r"[\s,;]+",
        engine="python",
        header=None,
        comment="#"
    )

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")

    if df.empty:
        raise ValueError("No se encontraron datos numéricos.")

    numeric_cols = []
    for col in df.columns:
        values = df[col].dropna().to_numpy(dtype=float)
        if len(values) > 10:
            numeric_cols.append(col)

    if len(numeric_cols) == 0:
        raise ValueError("No hay columnas numéricas suficientes.")

    if len(numeric_cols) == 1:
        acc = df[numeric_cols[0]].dropna().to_numpy(dtype=float)
        return None, acc, None, "Una columna: aceleración"

    col0 = numeric_cols[0]
    col1 = numeric_cols[1]

    x0 = df[col0].dropna().to_numpy(dtype=float)
    x1 = df[col1].dropna().to_numpy(dtype=float)

    n = min(len(x0), len(x1))
    x0 = x0[:n]
    x1 = x1[:n]

    diffs = np.diff(x0)
    positive_ratio = np.mean(diffs > 0)
    dt_median = np.median(diffs)

    if positive_ratio > 0.95 and dt_median > 0:
        return x0, x1, float(dt_median), "Dos columnas: tiempo + aceleración"

    return None, x1, None, "Varias columnas: se usó la segunda columna como aceleración"


def read_objective_spectrum(uploaded_file):
    """
    Lee espectro objetivo con dos columnas:
    T, Sa
    """
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    text = raw.decode("utf-8", errors="ignore")
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)

    df = pd.read_csv(
        io.StringIO(text),
        sep=r"[\s,;]+",
        engine="python",
        header=None,
        comment="#"
    )

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")

    if df.shape[1] < 2:
        raise ValueError("El espectro objetivo debe tener dos columnas: T y Sa.")

    t = df.iloc[:, 0].to_numpy(dtype=float)
    sa = df.iloc[:, 1].to_numpy(dtype=float)

    mask = np.isfinite(t) & np.isfinite(sa) & (t >= 0) & (sa > 0)
    t = t[mask]
    sa = sa[mask]

    if len(t) < 3:
        raise ValueError("El espectro objetivo tiene pocos puntos válidos.")

    order = np.argsort(t)
    t = t[order]
    sa = sa[order]

    out = pd.DataFrame({"T (s)": t, "Sa objetivo (g)": sa})
    out = out.groupby("T (s)", as_index=False).mean()

    return out


def read_drift_file(uploaded_file):
    """
    Lee archivo de derivas.

    Formato esperado:
    Par,Deriva_maxima_%

    También acepta dos columnas sin encabezado:
    columna 1 = Par
    columna 2 = deriva máxima en %
    """
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    text = raw.decode("utf-8", errors="ignore")
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)

    # Primero intenta con encabezado
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except Exception:
        df = pd.read_csv(io.StringIO(text), sep=r"[\s,;]+", engine="python", header=None)

    if df.shape[1] < 2:
        raise ValueError("El archivo de derivas debe tener al menos dos columnas: Par y Deriva_maxima_%.")

    # Detectar columnas
    cols_lower = [str(c).strip().lower() for c in df.columns]

    par_col = None
    drift_col = None

    for c, cl in zip(df.columns, cols_lower):
        if cl in ["par", "registro", "id", "nombre"] or "par" in cl or "registro" in cl:
            par_col = c
        if "deriva" in cl or "drift" in cl or "idr" in cl:
            drift_col = c

    if par_col is None:
        par_col = df.columns[0]

    if drift_col is None:
        drift_col = df.columns[1]

    out = pd.DataFrame({
        "Par": df[par_col].astype(str).str.strip(),
        "Deriva_maxima_%": pd.to_numeric(df[drift_col], errors="coerce")
    })

    out = out.dropna(subset=["Par", "Deriva_maxima_%"])
    out = out[out["Par"] != ""]

    if out.empty:
        raise ValueError("No se encontraron derivas válidas.")

    return out


def convert_to_m_s2(accel, unit):
    if unit == "g":
        return accel * G
    if unit == "m/s²":
        return accel
    if unit == "cm/s²":
        return accel / 100.0
    raise ValueError("Unidad no reconocida.")


def make_scaled_txt(time, acc_scaled_original_unit):
    if time is not None:
        lines = [f"{t:.6f}\t{a:.8e}" for t, a in zip(time, acc_scaled_original_unit)]
    else:
        lines = [f"{a:.8e}" for a in acc_scaled_original_unit]
    return "\n".join(lines)


# ============================================================
# DETECCIÓN DE PARES POR NOMBRE
# ============================================================
def detect_direction_and_base(filename):
    """
    Detecta la dirección N/E y agrupa los pares por número RSN.

    Ejemplos:
    1_RSN730_SPITAK_GUK000_N.txt  -> base: 1_RSN730, dir: N
    1_RSN730_SPITAK_GUK090_E.txt  -> base: 1_RSN730, dir: E
    """

    stem = Path(filename).stem.strip()

    m_dir = re.search(r"[_\-\s]+([NE])$", stem, flags=re.IGNORECASE)

    if not m_dir:
        return None, None

    direction = m_dir.group(1).upper()

    stem_no_dir = re.sub(r"[_\-\s]+[NE]$", "", stem, flags=re.IGNORECASE)

    prefix = ""
    rest = stem_no_dir

    m_prefix = re.match(r"^(\d+_)(.*)$", stem_no_dir)

    if m_prefix:
        prefix = m_prefix.group(1)
        rest = m_prefix.group(2)

    m_rsn = re.search(r"(RSN\d+)", rest, flags=re.IGNORECASE)

    if m_rsn:
        base = prefix + m_rsn.group(1).upper()
        return base, direction

    base = re.sub(r"[_\-][A-Za-z]*\d{3}$", "", stem_no_dir)

    return base, direction


def build_pairs(uploaded_files):
    groups = {}

    for f in uploaded_files:
        base, direction = detect_direction_and_base(f.name)
        if base is None:
            groups.setdefault("__NO_RECONOCIDOS__", {}).setdefault("UNKNOWN", []).append(f)
            continue

        groups.setdefault(base, {}).setdefault(direction, []).append(f)

    pair_rows = []
    complete_pairs = []
    warnings = []

    for base, dirs in groups.items():
        if base == "__NO_RECONOCIDOS__":
            for f in dirs.get("UNKNOWN", []):
                warnings.append(f"No se pudo identificar N/E en: {f.name}")
                pair_rows.append({
                    "Par detectado": "No reconocido",
                    "Archivo N": "",
                    "Archivo E": "",
                    "Estado": f"No reconocido: {f.name}"
                })
            continue

        n_files = dirs.get("N", [])
        e_files = dirs.get("E", [])

        status_parts = []

        if len(n_files) == 0:
            status_parts.append("Falta N")
        if len(e_files) == 0:
            status_parts.append("Falta E")
        if len(n_files) > 1:
            status_parts.append(f"Duplicado N ({len(n_files)})")
        if len(e_files) > 1:
            status_parts.append(f"Duplicado E ({len(e_files)})")

        if len(status_parts) == 0:
            status = "Completo"
            complete_pairs.append({
                "base": base,
                "N": n_files[0],
                "E": e_files[0]
            })
        else:
            status = " / ".join(status_parts)
            warnings.append(f"Par {base}: {status}")

        pair_rows.append({
            "Par detectado": base,
            "Archivo N": ", ".join([f.name for f in n_files]),
            "Archivo E": ", ".join([f.name for f in e_files]),
            "Estado": status
        })

    return pd.DataFrame(pair_rows), complete_pairs, warnings


# ============================================================
# NEWMARK BETA ACELERADO CON NUMBA
# ============================================================
@njit(fastmath=True)
def spectral_sa_sd_newmark_numba(acc_g_m_s2, dt, period, damping):
    m = 1.0
    w = 2.0 * np.pi / period
    k = m * w * w
    c = 2.0 * damping * m * w

    beta = 0.25
    gamma = 0.5

    n = len(acc_g_m_s2)

    u = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)

    p0 = -m * acc_g_m_s2[0]
    a[0] = (p0 - c * v[0] - k * u[0]) / m

    a0 = 1.0 / (beta * dt * dt)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)

    k_eff = k + a0 * m + a1 * c

    max_abs_u = 0.0

    for i in range(1, n):
        p_i = -m * acc_g_m_s2[i]

        p_eff = (
            p_i
            + m * (a0 * u[i - 1] + a2 * v[i - 1] + a3 * a[i - 1])
            + c * (a1 * u[i - 1] + a4 * v[i - 1] + a5 * a[i - 1])
        )

        u[i] = p_eff / k_eff
        a[i] = a0 * (u[i] - u[i - 1]) - a2 * v[i - 1] - a3 * a[i - 1]
        v[i] = v[i - 1] + dt * ((1.0 - gamma) * a[i - 1] + gamma * a[i])

        au = abs(u[i])
        if au > max_abs_u:
            max_abs_u = au

    sd_m = max_abs_u
    psa_m_s2 = w * w * sd_m
    sa_g = psa_m_s2 / 9.80665

    return sa_g, sd_m


@njit(parallel=True, fastmath=True)
def response_spectrum_numba(acc, dt, periods, damping):
    nT = len(periods)
    sa_vals = np.zeros(nT)
    sd_vals = np.zeros(nT)

    for i in prange(nT):
        sa_g, sd_m = spectral_sa_sd_newmark_numba(acc, dt, periods[i], damping)
        sa_vals[i] = sa_g
        sd_vals[i] = sd_m

    return sa_vals, sd_vals


def spectral_sa_sd_newmark_python(acc_g_m_s2, dt, period, damping=0.05):
    if period <= 0:
        raise ValueError("El período debe ser mayor que cero.")
    if dt <= 0:
        raise ValueError("El dt debe ser mayor que cero.")
    if len(acc_g_m_s2) < 20:
        raise ValueError("El registro tiene pocos puntos.")

    m = 1.0
    w = 2.0 * np.pi / period
    k = m * w**2
    c = 2.0 * damping * m * w

    beta = 1.0 / 4.0
    gamma = 1.0 / 2.0

    n = len(acc_g_m_s2)

    u = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)

    p = -m * acc_g_m_s2
    a[0] = (p[0] - c * v[0] - k * u[0]) / m

    a0 = 1.0 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)

    k_eff = k + a0 * m + a1 * c

    max_abs_u = 0.0

    for i in range(1, n):
        p_eff = (
            p[i]
            + m * (a0 * u[i - 1] + a2 * v[i - 1] + a3 * a[i - 1])
            + c * (a1 * u[i - 1] + a4 * v[i - 1] + a5 * a[i - 1])
        )

        u[i] = p_eff / k_eff
        a[i] = a0 * (u[i] - u[i - 1]) - a2 * v[i - 1] - a3 * a[i - 1]
        v[i] = v[i - 1] + dt * ((1.0 - gamma) * a[i - 1] + gamma * a[i])

        au = abs(u[i])
        if au > max_abs_u:
            max_abs_u = au

    sd_m = max_abs_u
    psa_m_s2 = w**2 * sd_m
    sa_g = psa_m_s2 / G

    return float(sa_g), float(sd_m)


@st.cache_data(show_spinner=False)
def response_spectrum_cached(acc_tuple, dt, periods_tuple, damping):
    acc = np.asarray(acc_tuple, dtype=np.float64)
    periods = np.asarray(periods_tuple, dtype=np.float64)

    if NUMBA_AVAILABLE:
        sa_vals, sd_vals = response_spectrum_numba(
            acc,
            float(dt),
            periods,
            float(damping)
        )
    else:
        sa_vals = np.zeros_like(periods, dtype=float)
        sd_vals = np.zeros_like(periods, dtype=float)

        for i, T in enumerate(periods):
            sa_g, sd_m = spectral_sa_sd_newmark_python(acc, float(dt), float(T), float(damping))
            sa_vals[i] = sa_g
            sd_vals[i] = sd_m

    return pd.DataFrame({
        "T (s)": periods,
        "Sa (g)": sa_vals,
        "Sd (m)": sd_vals,
        "Sd (cm)": sd_vals * 100.0
    })


# ============================================================
# ESCALAMIENTO
# ============================================================
def compute_global_factor(mean_srss, target_sa, mask_range, criterion="100%"):
    target_multiplier = 1.0 if criterion == "100%" else 0.90

    ratios = (target_multiplier * target_sa[mask_range]) / mean_srss[mask_range]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]

    if len(ratios) == 0:
        raise ValueError("No hay datos válidos para calcular el factor global.")

    return float(np.max(ratios))


def compute_individual_pair_factor(srss, target_sa, mask_range, criterion="100%"):
    target_multiplier = 1.0 if criterion == "100%" else 0.90

    ratios = (target_multiplier * target_sa[mask_range]) / srss[mask_range]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]

    if len(ratios) == 0:
        return np.nan

    return float(np.max(ratios))


# ============================================================
# FRAGILIDAD
# ============================================================
def fit_lognormal_fragility(im_values, exceedance):
    im_values = np.asarray(im_values, dtype=float)
    y = np.asarray(exceedance, dtype=int)

    mask = np.isfinite(im_values) & (im_values > 0) & np.isfinite(y)
    im_values = im_values[mask]
    y = y[mask]

    if len(im_values) < 4:
        return None, None, "Se necesitan al menos 4 puntos."

    if len(np.unique(y)) < 2:
        return None, None, "No se puede ajustar: todos los puntos son 0 o todos son 1."

    def neg_loglike(params):
        ln_theta, ln_beta = params
        beta = np.exp(ln_beta)

        z = (np.log(im_values) - ln_theta) / beta
        p = norm.cdf(z)
        p = np.clip(p, 1e-8, 1.0 - 1e-8)

        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    x0 = np.array([np.log(np.median(im_values)), np.log(0.40)])
    res = minimize(neg_loglike, x0, method="Nelder-Mead")

    if not res.success:
        return None, None, "No se pudo ajustar."

    theta = float(np.exp(res.x[0]))
    beta = float(np.exp(res.x[1]))

    return theta, beta, "OK"


def fragility_probability(im_grid, theta, beta):
    return norm.cdf((np.log(im_grid) - np.log(theta)) / beta)


def build_fragility_dataframe(pair_results_df, drift_df, drift_limits):
    df = pair_results_df.copy()

    if drift_df is None:
        raise ValueError("Debes cargar una tabla de derivas máximas para generar curvas de fragilidad por deriva.")

    drift_df = drift_df.copy()
    drift_df["Par"] = drift_df["Par"].astype(str).str.strip()
    df["Par"] = df["Par"].astype(str).str.strip()

    df = df.merge(drift_df, on="Par", how="left")

    missing = df[df["Deriva_maxima_%"].isna()]["Par"].tolist()
    if missing:
        st.warning(
            "No se encontró deriva para estos pares. Se excluirán de la fragilidad: "
            + ", ".join(missing[:15])
            + ("..." if len(missing) > 15 else "")
        )

    df = df.dropna(subset=["Deriva_maxima_%"])

    if df.empty:
        raise ValueError("No hay pares con deriva máxima válida.")

    ds1, ds2, ds3, ds4 = drift_limits

    df["DS1 leve"] = (df["Deriva_maxima_%"] >= ds1).astype(int)
    df["DS2 moderado"] = (df["Deriva_maxima_%"] >= ds2).astype(int)
    df["DS3 severo"] = (df["Deriva_maxima_%"] >= ds3).astype(int)
    df["DS4 colapso"] = (df["Deriva_maxima_%"] >= ds4).astype(int)

    return df


# ============================================================
# STREAMLIT APP
# ============================================================
st.set_page_config(
    page_title="SRSS + Fragilidad",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Pares N/E: SRSS, escalamiento opcional y curvas de fragilidad")

st.markdown(
    """
Esta app permite:

```text
1. Leer pares N/E.
2. Calcular espectros Sa y SRSS.
3. Escalar opcionalmente contra espectro objetivo.
4. Usar registros ya escalados con factor = 1.0.
5. Generar curvas de fragilidad usando derivas máximas.
```
"""
)

with st.sidebar:
    st.header("0. Modo de trabajo")

    already_scaled = st.checkbox(
        "Mis registros ya están escalados",
        value=False,
        help="Activa esto si tus registros ya fueron escalados previamente, por ejemplo con RotD50."
    )

    st.header("1. Parámetros estructurales")

    T1 = st.number_input(
        "T1 período fundamental (s)",
        min_value=0.01,
        value=0.657,
        step=0.001,
        format="%.3f"
    )

    lower_mult = st.number_input(
        "Límite inferior del rango",
        min_value=0.01,
        value=0.20,
        step=0.05,
        format="%.2f"
    )

    upper_mult = st.number_input(
        "Límite superior del rango",
        min_value=0.10,
        value=1.50,
        step=0.05,
        format="%.2f"
    )

    T_low = lower_mult * T1
    T_high = upper_mult * T1

    st.info(f"Rango de referencia: {T_low:.4f} s a {T_high:.4f} s")

    damping = st.number_input(
        "Amortiguamiento ξ",
        min_value=0.00,
        max_value=0.30,
        value=0.05,
        step=0.01,
        format="%.3f"
    )

    st.header("2. Registros")

    unit = st.selectbox(
        "Unidad de aceleración en los archivos",
        ["g", "m/s²", "cm/s²"]
    )

    dt_manual = st.number_input(
        "dt manual si el archivo tiene una sola columna (s)",
        min_value=0.0001,
        value=0.005,
        step=0.001,
        format="%.4f"
    )

    st.header("3. Espectros")

    t_min = st.number_input(
        "T mínimo para cálculo de espectros (s)",
        min_value=0.01,
        value=0.01,
        step=0.01,
        format="%.2f"
    )

    t_max = st.number_input(
        "T máximo para cálculo de espectros (s)",
        min_value=0.10,
        value=5.00,
        step=0.10,
        format="%.2f"
    )

    n_periods = st.number_input(
        "Número de períodos",
        min_value=30,
        max_value=400,
        value=160,
        step=10
    )

    st.header("4. Método")

    if not already_scaled:
        scale_method = st.selectbox(
            "Método de escalamiento",
            [
                "Factor global: media SRSS escalada ≥ objetivo",
                "Factor por par: cada SRSS escalado ≥ objetivo",
                "Solo calcular, sin escalar"
            ]
        )

        criterion = st.radio(
            "Criterio de cumplimiento",
            ["100%", "90%"],
            index=0
        )
    else:
        scale_method = "Solo calcular, sin escalar"
        criterion = "No aplica"
        st.success("Modo registros ya escalados: factor = 1.0")

    show_individual = st.checkbox("Mostrar SRSS individuales", value=False)
    show_original_mean = st.checkbox("Mostrar media SRSS original", value=True)

    st.header("5. Fragilidad")

    enable_fragility = st.checkbox("Generar curvas de fragilidad", value=True)

    fragility_im_type = st.radio(
        "IM para curvas de fragilidad",
        ["Sa SRSS(T1) [g]", "Sd SRSS(T1) [cm]"],
        index=0,
        help="Escoge si la curva de fragilidad se ajusta usando aceleración espectral SRSS(T1) o desplazamiento espectral SRSS(T1)."
    )

    st.caption("Límites de daño por deriva máxima de entrepiso en %.")

    ds1_limit = st.number_input("DS1 leve: deriva ≥ (%)", min_value=0.001, value=0.50, step=0.10, format="%.3f")
    ds2_limit = st.number_input("DS2 moderado: deriva ≥ (%)", min_value=0.001, value=1.00, step=0.10, format="%.3f")
    ds3_limit = st.number_input("DS3 severo: deriva ≥ (%)", min_value=0.001, value=2.00, step=0.10, format="%.3f")
    ds4_limit = st.number_input("DS4 colapso: deriva ≥ (%)", min_value=0.001, value=4.00, step=0.10, format="%.3f")

    st.header("6. Exportación")

    generar_zip = st.checkbox(
        "Generar ZIP con acelerogramas procesados",
        value=not already_scaled,
        help="Si los registros ya están escalados, normalmente no necesitas generar ZIP."
    )

st.sidebar.caption(f"Numba disponible: {'Sí' if NUMBA_AVAILABLE else 'No'}")


uploaded_records = st.file_uploader(
    "Sube todos los registros N/E a la vez",
    type=["txt", "csv"],
    accept_multiple_files=True
)

uploaded_objective = None

if not already_scaled:
    uploaded_objective = st.file_uploader(
        "Sube el espectro objetivo en dos columnas: T, Sa(g)",
        type=["txt", "csv"]
    )
else:
    st.info("Modo registros ya escalados: no se requiere espectro objetivo.")

uploaded_drifts = None
if enable_fragility:
    uploaded_drifts = st.file_uploader(
        "Sube tabla de derivas máximas: Par, Deriva_maxima_%",
        type=["txt", "csv"],
        key="drifts"
    )


if uploaded_records:
    pair_df, complete_pairs, warnings = build_pairs(uploaded_records)

    st.subheader("1. Pares detectados")
    st.dataframe(pair_df, use_container_width=True)

    if warnings:
        with st.expander("Advertencias de detección"):
            for w in warnings:
                st.warning(w)

    if len(complete_pairs) == 0:
        st.error("No se detectaron pares completos N/E.")
        st.stop()

    st.success(f"Pares completos detectados: {len(complete_pairs)}")


can_process = False

if uploaded_records:
    if already_scaled:
        can_process = True
    elif uploaded_objective is not None:
        can_process = True

if can_process:
    try:
        periods = np.linspace(float(t_min), float(t_max), int(n_periods))
        mask_range = (periods >= T_low) & (periods <= T_high)

        if not np.any(mask_range):
            st.error("No hay períodos dentro del rango 0.2T1–1.5T1. Ajusta T mínimo, T máximo o T1.")
            st.stop()

        if already_scaled:
            target_sa = None
            obj_df_raw = None
        else:
            obj_df_raw = read_objective_spectrum(uploaded_objective)
            target_sa = np.interp(
                periods,
                obj_df_raw["T (s)"].to_numpy(dtype=float),
                obj_df_raw["Sa objetivo (g)"].to_numpy(dtype=float)
            )

        pair_results = []
        srss_matrix = []
        parsed_files_store = {}

        progress = st.progress(0, text="Calculando espectros de pares...")

        for idx, pair in enumerate(complete_pairs):
            base_name = pair["base"]
            fN = pair["N"]
            fE = pair["E"]

            time_N, acc_N_unit, dt_N_detected, fmt_N = read_accel_file(fN)
            dt_N = dt_N_detected if dt_N_detected is not None else dt_manual
            acc_N_m_s2 = convert_to_m_s2(acc_N_unit, unit)

            time_E, acc_E_unit, dt_E_detected, fmt_E = read_accel_file(fE)
            dt_E = dt_E_detected if dt_E_detected is not None else dt_manual
            acc_E_m_s2 = convert_to_m_s2(acc_E_unit, unit)

            if abs(dt_N - dt_E) > 1e-8:
                st.warning(f"El par {base_name} tiene dt diferente entre N y E: {dt_N} vs {dt_E}")

            spec_N = response_spectrum_cached(
                tuple(acc_N_m_s2.tolist()),
                float(dt_N),
                tuple(periods.tolist()),
                float(damping)
            )

            spec_E = response_spectrum_cached(
                tuple(acc_E_m_s2.tolist()),
                float(dt_E),
                tuple(periods.tolist()),
                float(damping)
            )

            sa_N = spec_N["Sa (g)"].to_numpy(dtype=float)
            sa_E = spec_E["Sa (g)"].to_numpy(dtype=float)
            sd_N_cm_arr = spec_N["Sd (cm)"].to_numpy(dtype=float)
            sd_E_cm_arr = spec_E["Sd (cm)"].to_numpy(dtype=float)

            srss = np.sqrt(sa_N**2 + sa_E**2)
            srss_sd_cm = np.sqrt(sd_N_cm_arr**2 + sd_E_cm_arr**2)
            srss_matrix.append(srss)

            pga_N = np.max(np.abs(acc_N_m_s2)) / G
            pga_E = np.max(np.abs(acc_E_m_s2)) / G

            saN_T1 = float(np.interp(T1, periods, sa_N))
            saE_T1 = float(np.interp(T1, periods, sa_E))
            srss_T1 = float(np.sqrt(saN_T1**2 + saE_T1**2))

            sdN_T1_cm = float(np.interp(T1, periods, sd_N_cm_arr))
            sdE_T1_cm = float(np.interp(T1, periods, sd_E_cm_arr))
            srss_sd_T1_cm = float(np.sqrt(sdN_T1_cm**2 + sdE_T1_cm**2))

            parsed_files_store[base_name] = {
                "N": {
                    "file_name": fN.name,
                    "time": time_N,
                    "acc_unit": acc_N_unit,
                    "dt": dt_N,
                    "format": fmt_N
                },
                "E": {
                    "file_name": fE.name,
                    "time": time_E,
                    "acc_unit": acc_E_unit,
                    "dt": dt_E,
                    "format": fmt_E
                }
            }

            pair_results.append({
                "Par": base_name,
                "Archivo N": fN.name,
                "Archivo E": fE.name,
                "dt N (s)": dt_N,
                "dt E (s)": dt_E,
                "PGA N (g)": pga_N,
                "PGA E (g)": pga_E,
                f"Sa_N(T1={T1:.3f}s) (g)": saN_T1,
                f"Sa_E(T1={T1:.3f}s) (g)": saE_T1,
                f"SRSS_Sa(T1={T1:.3f}s) (g)": srss_T1,
                f"Sd_N(T1={T1:.3f}s) (cm)": sdN_T1_cm,
                f"Sd_E(T1={T1:.3f}s) (cm)": sdE_T1_cm,
                f"SRSS_Sd(T1={T1:.3f}s) (cm)": srss_sd_T1_cm
            })

            progress.progress((idx + 1) / len(complete_pairs), text=f"Calculando {idx+1}/{len(complete_pairs)} pares")

        progress.empty()

        srss_matrix = np.vstack(srss_matrix)
        mean_srss = np.mean(srss_matrix, axis=0)

        if already_scaled:
            global_factor = 1.0
            pair_factors = np.ones(len(complete_pairs))
            mean_srss_scaled = mean_srss.copy()
            scaled_srss_matrix = srss_matrix.copy()
            factor_message = "Registros ya escalados: se usó factor = 1.0. No se solicitó espectro objetivo."

        else:
            if scale_method == "Factor global: media SRSS escalada ≥ objetivo":
                global_factor = compute_global_factor(mean_srss, target_sa, mask_range, criterion=criterion)
                pair_factors = np.full(len(complete_pairs), global_factor)
                factor_message = f"Factor global calculado = {global_factor:.4f}"

            elif scale_method == "Factor por par: cada SRSS escalado ≥ objetivo":
                global_factor = None
                pair_factors = []
                for i in range(len(complete_pairs)):
                    fac_i = compute_individual_pair_factor(srss_matrix[i, :], target_sa, mask_range, criterion=criterion)
                    pair_factors.append(fac_i)
                pair_factors = np.asarray(pair_factors, dtype=float)
                factor_message = "Se calculó un factor independiente para cada par."

            else:
                global_factor = 1.0
                pair_factors = np.ones(len(complete_pairs))
                factor_message = "Sin escalamiento. Factor = 1.0"

            scaled_srss_matrix = srss_matrix * pair_factors[:, None]
            mean_srss_scaled = np.mean(scaled_srss_matrix, axis=0)

        st.subheader("2. Resumen de pares y factores")
        pair_results_df = pd.DataFrame(pair_results)
        pair_results_df["Factor aplicado"] = pair_factors
        pair_results_df["Registros ya escalados"] = "Sí" if already_scaled else "No"

        # IM final por par para fragilidad: SRSS(T1) luego de aplicar factor
        srss_sa_t1_col = f"SRSS_Sa(T1={T1:.3f}s) (g)"
        srss_sd_t1_col = f"SRSS_Sd(T1={T1:.3f}s) (cm)"
        pair_results_df["IM_SRSS_Sa_T1_final (g)"] = pair_results_df[srss_sa_t1_col] * pair_results_df["Factor aplicado"]
        pair_results_df["IM_SRSS_Sd_T1_final (cm)"] = pair_results_df[srss_sd_t1_col] * pair_results_df["Factor aplicado"]

        st.dataframe(pair_results_df, use_container_width=True)
        st.info(factor_message)

        mean_srss_T1 = float(np.interp(T1, periods, mean_srss))
        mean_srss_scaled_T1 = float(np.interp(T1, periods, mean_srss_scaled))

        summary_rows = [
            {"Parámetro": "Modo", "Valor": "Registros ya escalados" if already_scaled else "Registros sin escalar"},
            {"Parámetro": "T1 (s)", "Valor": T1},
            {"Parámetro": f"{lower_mult:.2f}T1 (s)", "Valor": T_low},
            {"Parámetro": f"{upper_mult:.2f}T1 (s)", "Valor": T_high},
            {"Parámetro": "Media SRSS original/procesada en T1 (g)", "Valor": mean_srss_T1},
            {"Parámetro": "Media SRSS final en T1 (g)", "Valor": mean_srss_scaled_T1},
            {"Parámetro": "Factor global", "Valor": global_factor if global_factor is not None else np.nan},
        ]

        if not already_scaled:
            sa_obj_T1 = float(np.interp(T1, periods, target_sa))
            summary_rows.insert(4, {"Parámetro": "Sa objetivo en T1 (g)", "Valor": sa_obj_T1})

        summary_df = pd.DataFrame(summary_rows)

        st.subheader("3. Resumen general")
        st.dataframe(summary_df, use_container_width=True)

        # ====================================================
        # GRÁFICA SRSS
        # ====================================================
        st.subheader("4. Gráfica SRSS")

        fig = go.Figure()

        if show_individual:
            for i, pair in enumerate(complete_pairs):
                base_name = pair["base"]

                fig.add_trace(go.Scatter(
                    x=periods,
                    y=srss_matrix[i, :],
                    mode="lines",
                    line=dict(width=1),
                    opacity=0.35,
                    name=f"SRSS original {base_name}"
                ))

                if not already_scaled:
                    fig.add_trace(go.Scatter(
                        x=periods,
                        y=scaled_srss_matrix[i, :],
                        mode="lines",
                        line=dict(width=1, dash="dot"),
                        opacity=0.35,
                        name=f"SRSS escalado {base_name}"
                    ))

        if show_original_mean and not already_scaled:
            fig.add_trace(go.Scatter(
                x=periods,
                y=mean_srss,
                mode="lines",
                line=dict(width=3, dash="dot"),
                name="Media SRSS original"
            ))

        if already_scaled:
            fig.add_trace(go.Scatter(
                x=periods,
                y=mean_srss,
                mode="lines",
                line=dict(width=4),
                name="Media SRSS registros ya escalados"
            ))
        else:
            fig.add_trace(go.Scatter(
                x=periods,
                y=mean_srss_scaled,
                mode="lines",
                line=dict(width=4),
                name="Media SRSS escalada"
            ))

            fig.add_trace(go.Scatter(
                x=periods,
                y=target_sa,
                mode="lines",
                line=dict(width=4),
                name="Espectro objetivo"
            ))

        fig.add_vrect(
            x0=T_low,
            x1=T_high,
            fillcolor="LightGray",
            opacity=0.20,
            line_width=0,
            annotation_text=f"Rango {lower_mult:.2f}T1–{upper_mult:.2f}T1",
            annotation_position="top left"
        )

        fig.add_vline(
            x=T_low,
            line_dash="dash",
            line_width=2,
            annotation_text=f"{lower_mult:.2f}T1",
            annotation_position="top"
        )

        fig.add_vline(
            x=T1,
            line_dash="solid",
            line_width=2,
            annotation_text="T1",
            annotation_position="top"
        )

        fig.add_vline(
            x=T_high,
            line_dash="dash",
            line_width=2,
            annotation_text=f"{upper_mult:.2f}T1",
            annotation_position="top"
        )

        title = "Media SRSS de registros ya escalados" if already_scaled else "Escalamiento de registros por media SRSS"

        fig.update_layout(
            title=title,
            xaxis_title="T [s]",
            yaxis_title="Sa [g]",
            legend_title="Curvas",
            height=650
        )

        st.plotly_chart(fig, use_container_width=True)

        # ====================================================
        # VERIFICACIÓN
        # ====================================================
        if not already_scaled:
            multiplier = 1.0 if criterion == "100%" else 0.90
            ratio_scaled_to_target = mean_srss_scaled[mask_range] / (multiplier * target_sa[mask_range])
            min_ratio = float(np.min(ratio_scaled_to_target))
            min_idx_local = int(np.argmin(ratio_scaled_to_target))
            T_crit = float(periods[mask_range][min_idx_local])

            st.subheader("5. Verificación en el rango de escalamiento")

            check_df = pd.DataFrame({
                "T (s)": periods[mask_range],
                "Sa objetivo (g)": target_sa[mask_range],
                "Media SRSS escalada (g)": mean_srss_scaled[mask_range],
                "Relación media/objetivo": mean_srss_scaled[mask_range] / target_sa[mask_range]
            })

            st.write(f"Relación mínima respecto al criterio {criterion}: **{min_ratio:.4f}** en T = **{T_crit:.4f} s**")

            if min_ratio >= 0.999:
                st.success("Cumple: la media SRSS escalada no queda por debajo del criterio en el rango.")
            else:
                st.error("No cumple: la media SRSS escalada queda por debajo del criterio en algún punto del rango.")

            st.dataframe(check_df, use_container_width=True)

        else:
            st.subheader("5. Verificación")
            st.info("Modo registros ya escalados: se omitió la verificación contra espectro objetivo porque no se cargó espectro objetivo.")

            check_df = pd.DataFrame({
                "T (s)": periods[mask_range],
                "Media SRSS registros ya escalados (g)": mean_srss[mask_range]
            })
            st.dataframe(check_df, use_container_width=True)

        # ====================================================
        # FRAGILIDAD
        # ====================================================
        st.subheader("6. Curvas de fragilidad")

        if enable_fragility:
            if uploaded_drifts is None:
                st.warning("Para generar curvas de fragilidad debes subir la tabla de derivas máximas.")
                st.markdown(
                    """
Formato esperado:

```text
Par,Deriva_maxima_%
1_RSN730,0.45
1_RSN755,0.82
1_RSN767,1.35
```

El nombre de `Par` debe coincidir con la tabla de pares detectados.
"""
                )
            else:
                drift_df = read_drift_file(uploaded_drifts)

                frag_df = build_fragility_dataframe(
                    pair_results_df,
                    drift_df,
                    drift_limits=(ds1_limit, ds2_limit, ds3_limit, ds4_limit)
                )

                st.markdown("**Datos usados para fragilidad**")
                st.dataframe(frag_df, use_container_width=True)

                if fragility_im_type == "Sa SRSS(T1) [g]":
                    im_col = "IM_SRSS_Sa_T1_final (g)"
                    im_axis_label = "IM = Sa SRSS(T1) final [g]"
                    theta_label = "θ mediana Sa SRSS(T1) [g]"
                else:
                    im_col = "IM_SRSS_Sd_T1_final (cm)"
                    im_axis_label = "IM = Sd SRSS(T1) final [cm]"
                    theta_label = "θ mediana Sd SRSS(T1) [cm]"

                im_values = frag_df[im_col].to_numpy(dtype=float)

                damage_states = {
                    "DS1 Leve": frag_df["DS1 leve"].to_numpy(dtype=int),
                    "DS2 Moderado": frag_df["DS2 moderado"].to_numpy(dtype=int),
                    "DS3 Severo": frag_df["DS3 severo"].to_numpy(dtype=int),
                    "DS4 Colapso": frag_df["DS4 colapso"].to_numpy(dtype=int),
                }

                im_min = max(0.001, np.nanmin(im_values) * 0.50)
                im_max = max(np.nanmax(im_values) * 1.50, 0.01)
                im_grid = np.linspace(im_min, im_max, 400)

                fig_frag = go.Figure()
                fit_rows = []

                for ds_name, exceedance in damage_states.items():
                    theta, beta, status = fit_lognormal_fragility(im_values, exceedance)

                    zeros = int(np.sum(exceedance == 0))
                    ones = int(np.sum(exceedance == 1))

                    if theta is not None:
                        prob = fragility_probability(im_grid, theta, beta)
                        fig_frag.add_trace(go.Scatter(
                            x=im_grid,
                            y=prob,
                            mode="lines",
                            name=ds_name
                        ))

                    fit_rows.append({
                        "Estado de daño": ds_name,
                        "Cantidad 0": zeros,
                        "Cantidad 1": ones,
                        theta_label: theta,
                        "β dispersión lognormal": beta,
                        "Estado ajuste": status
                    })

                fig_frag.update_layout(
                    title="Curvas de fragilidad por deriva máxima",
                    xaxis_title=im_axis_label,
                    yaxis_title="P(DS ≥ ds | IM)",
                    yaxis=dict(range=[0, 1]),
                    legend_title="Estados de daño",
                    height=600
                )

                st.plotly_chart(fig_frag, use_container_width=True)

                fit_df = pd.DataFrame(fit_rows)
                st.markdown("**Parámetros ajustados**")
                st.dataframe(fit_df, use_container_width=True)

                csv_frag = frag_df.to_csv(index=False).encode("utf-8")
                csv_fit = fit_df.to_csv(index=False).encode("utf-8")

                cfrag1, cfrag2 = st.columns(2)
                with cfrag1:
                    st.download_button(
                        "Descargar datos fragilidad CSV",
                        data=csv_frag,
                        file_name="datos_fragilidad_derivas.csv",
                        mime="text/csv"
                    )
                with cfrag2:
                    st.download_button(
                        "Descargar parametros fragilidad CSV",
                        data=csv_fit,
                        file_name="parametros_fragilidad.csv",
                        mime="text/csv"
                    )
        else:
            st.info("Módulo de fragilidad desactivado.")

        # ====================================================
        # DESCARGAS GENERALES
        # ====================================================
        st.subheader("7. Descargas generales")

        csv_pairs = pair_results_df.to_csv(index=False).encode("utf-8")

        if already_scaled:
            spectra_out = pd.DataFrame({
                "T (s)": periods,
                "Media SRSS registros ya escalados (g)": mean_srss
            })
        else:
            spectra_out = pd.DataFrame({
                "T (s)": periods,
                "Sa objetivo (g)": target_sa,
                "Media SRSS original (g)": mean_srss,
                "Media SRSS escalada (g)": mean_srss_scaled
            })

        csv_spectra = spectra_out.to_csv(index=False).encode("utf-8")
        csv_check = check_df.to_csv(index=False).encode("utf-8")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.download_button(
                "Descargar resumen pares CSV",
                data=csv_pairs,
                file_name="resumen_pares_factores.csv",
                mime="text/csv"
            )

        with c2:
            st.download_button(
                "Descargar espectros CSV",
                data=csv_spectra,
                file_name="media_srss_resultados.csv",
                mime="text/csv"
            )

        with c3:
            st.download_button(
                "Descargar verificación CSV",
                data=csv_check,
                file_name="verificacion_rango.csv",
                mime="text/csv"
            )

        if generar_zip:
            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for i, pair in enumerate(complete_pairs):
                    base_name = pair["base"]
                    fac = pair_factors[i]

                    dataN = parsed_files_store[base_name]["N"]
                    dataE = parsed_files_store[base_name]["E"]

                    accN_processed = dataN["acc_unit"] * fac
                    accE_processed = dataE["acc_unit"] * fac

                    if already_scaled:
                        suffix = "_PROCESADO_FAC_1p0000.txt"
                    else:
                        suffix = f"_ESC_FAC_{fac:.4f}.txt"

                    nameN = Path(dataN["file_name"]).stem + suffix
                    nameE = Path(dataE["file_name"]).stem + suffix

                    zf.writestr(nameN, make_scaled_txt(dataN["time"], accN_processed))
                    zf.writestr(nameE, make_scaled_txt(dataE["time"], accE_processed))

            zip_name = "acelerogramas_procesados_fac_1.zip" if already_scaled else "acelerogramas_escalados_pares_srss.zip"

            st.download_button(
                "Descargar acelerogramas procesados ZIP",
                data=zip_buffer.getvalue(),
                file_name=zip_name,
                mime="application/zip"
            )

        with st.expander("Formato tabla de derivas"):
            st.markdown(
                """
Para curvas de fragilidad debes cargar una tabla así:

```text
Par,Deriva_maxima_%
1_RSN730,0.45
1_RSN755,0.82
1_RSN767,1.35
```

La deriva debe estar en porcentaje.  
Por ejemplo:

```text
0.50 = 0.50 %
1.20 = 1.20 %
```
"""
            )

        with st.expander("Criterio usado para fragilidad"):
            st.markdown(
                r"""
La intensidad usada puede ser:

\[
IM = Sa_{SRSS}(T_1)
\]

 o también:

\[
IM = Sd_{SRSS}(T_1)
\]

Para cada par, si usas aceleración espectral:

\[
Sa_{SRSS}(T_1)=\sqrt{Sa_N(T_1)^2+Sa_E(T_1)^2}
\]

Si usas desplazamiento espectral:

\[
Sd_{SRSS}(T_1)=\sqrt{Sd_N(T_1)^2+Sd_E(T_1)^2}
\]

Si los registros fueron escalados en el programa:

\[
IM_{final}=IM(T_1)\cdot Factor
\]

Si los registros ya estaban escalados:

\[
Factor=1.0
\]

Luego, para cada deriva máxima:

\[
DS_i = 1 \quad \text{si} \quad IDR_{max} \geq IDR_{DS_i}
\]

y se ajusta:

\[
P(DS \geq ds|IM)=\Phi\left(\frac{\ln(IM)-\ln(\theta)}{\beta}\right)
\]
"""
            )

    except Exception as e:
        st.error(f"Error durante el procesamiento: {e}")

elif uploaded_records and not already_scaled and uploaded_objective is None:
    st.info("Sube el espectro objetivo o activa 'Mis registros ya están escalados'.")
else:
    st.warning("Sube los registros N/E para iniciar.")
