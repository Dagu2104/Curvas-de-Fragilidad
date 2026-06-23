
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

G = 9.80665


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

    # Convierte coma decimal a punto: 0,123 -> 0.123
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

    # Remover períodos duplicados promediando
    out = pd.DataFrame({"T (s)": t, "Sa objetivo (g)": sa})
    out = out.groupby("T (s)", as_index=False).mean()

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
    Detecta nombres que terminan en:
    _N, _E, -N, -E, N, E antes de la extensión.

    Ejemplos:
    RSN4031_xxx_N.txt -> base RSN4031_xxx, dir N
    RSN4031_xxx_E.txt -> base RSN4031_xxx, dir E
    """
    stem = Path(filename).stem.strip()

    # Casos con separador _N, _E, -N, -E
    m = re.match(r"^(.*?)[_\-\s]+([NE])$", stem, flags=re.IGNORECASE)
    if m:
        base = m.group(1).strip()
        direction = m.group(2).upper()
        return base, direction

    # Caso final directo N/E, pero solo si antes hay número o letra.
    # Es menos recomendado, pero ayuda con nombres tipo Registro01N.txt
    m = re.match(r"^(.*?)([NE])$", stem, flags=re.IGNORECASE)
    if m and len(m.group(1)) > 3:
        base = m.group(1).rstrip("_- ").strip()
        direction = m.group(2).upper()
        return base, direction

    return None, None


def build_pairs(uploaded_files):
    """
    Agrupa archivos por nombre base y dirección N/E.
    """
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
# NEWMARK BETA
# ============================================================
def spectral_sa_sd_newmark(acc_g_m_s2, dt, period, damping=0.05):
    """
    Calcula pseudo-aceleración espectral Sa en g y desplazamiento espectral Sd en m.
    """
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
    acc = np.asarray(acc_tuple, dtype=float)
    periods = np.asarray(periods_tuple, dtype=float)

    sa_vals = np.zeros_like(periods, dtype=float)
    sd_vals = np.zeros_like(periods, dtype=float)

    for i, T in enumerate(periods):
        sa_g, sd_m = spectral_sa_sd_newmark(acc, float(dt), float(T), float(damping))
        sa_vals[i] = sa_g
        sd_vals[i] = sd_m

    return pd.DataFrame({
        "T (s)": periods,
        "Sa (g)": sa_vals,
        "Sd (m)": sd_vals,
        "Sd (cm)": sd_vals * 100.0
    })


# ============================================================
# ESCALAMIENTO TIPO EXCEL
# ============================================================
def compute_global_factor(mean_srss, target_sa, mask_range, criterion="100%"):
    """
    Calcula factor global para que la media SRSS escalada cumpla el objetivo en el rango.

    criterion:
    - "100%": media escalada >= 1.00 objetivo
    - "90%": media escalada >= 0.90 objetivo
    """
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
# STREAMLIT APP
# ============================================================
st.set_page_config(
    page_title="Escalamiento SRSS Pares",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Escalamiento de pares N/E por media SRSS contra espectro objetivo")

st.markdown(
    """
Esta app replica el flujo típico de Excel:

```text
Pares N/E → Sa_N(T), Sa_E(T) → SRSS → media SRSS → escalamiento en 0.2T1–1.5T1
```

Los pares se detectan automáticamente usando archivos que terminan en `_N` y `_E`.
"""
)

with st.sidebar:
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

    st.info(f"Rango de escalamiento: {T_low:.4f} s a {T_high:.4f} s")

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

    st.header("3. Espectro")

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

    st.header("4. Método de escalamiento")

    scale_method = st.selectbox(
        "Método",
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

    show_individual = st.checkbox("Mostrar SRSS individuales", value=False)
    show_original_mean = st.checkbox("Mostrar media SRSS original", value=True)

    st.header("5. Exportación")

    generar_zip = st.checkbox("Generar ZIP con acelerogramas escalados", value=True)


uploaded_records = st.file_uploader(
    "Sube todos los registros N/E a la vez",
    type=["txt", "csv"],
    accept_multiple_files=True
)

uploaded_objective = st.file_uploader(
    "Sube el espectro objetivo en dos columnas: T, Sa(g)",
    type=["txt", "csv"]
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

if uploaded_records and uploaded_objective:
    try:
        obj_df_raw = read_objective_spectrum(uploaded_objective)

        periods = np.linspace(float(t_min), float(t_max), int(n_periods))

        if T_low < periods.min() or T_high > periods.max():
            st.warning(
                f"El rango {T_low:.3f}–{T_high:.3f} s queda parcialmente fuera del rango de espectros calculado "
                f"{periods.min():.3f}–{periods.max():.3f} s."
            )

        target_sa = np.interp(
            periods,
            obj_df_raw["T (s)"].to_numpy(dtype=float),
            obj_df_raw["Sa objetivo (g)"].to_numpy(dtype=float)
        )

        mask_range = (periods >= T_low) & (periods <= T_high)

        if not np.any(mask_range):
            st.error("No hay períodos dentro del rango 0.2T1–1.5T1. Ajusta T mínimo, T máximo o T1.")
            st.stop()

        pair_results = []
        srss_matrix = []
        spectra_store = {}
        parsed_files_store = {}

        progress = st.progress(0, text="Calculando espectros de pares...")

        for idx, pair in enumerate(complete_pairs):
            base_name = pair["base"]
            fN = pair["N"]
            fE = pair["E"]

            # Leer N
            time_N, acc_N_unit, dt_N_detected, fmt_N = read_accel_file(fN)
            dt_N = dt_N_detected if dt_N_detected is not None else dt_manual
            acc_N_m_s2 = convert_to_m_s2(acc_N_unit, unit)

            # Leer E
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

            srss = np.sqrt(sa_N**2 + sa_E**2)
            srss_matrix.append(srss)

            pga_N = np.max(np.abs(acc_N_m_s2)) / G
            pga_E = np.max(np.abs(acc_E_m_s2)) / G

            # Sa SRSS en T1 interpolado
            saN_T1 = float(np.interp(T1, periods, sa_N))
            saE_T1 = float(np.interp(T1, periods, sa_E))
            srss_T1 = float(np.sqrt(saN_T1**2 + saE_T1**2))

            spectra_store[base_name] = {
                "T": periods,
                "Sa_N": sa_N,
                "Sa_E": sa_E,
                "SRSS": srss
            }

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
                f"SRSS(T1={T1:.3f}s) (g)": srss_T1
            })

            progress.progress((idx + 1) / len(complete_pairs), text=f"Calculando {idx+1}/{len(complete_pairs)} pares")

        progress.empty()

        srss_matrix = np.vstack(srss_matrix)
        mean_srss = np.mean(srss_matrix, axis=0)

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
        st.dataframe(pair_results_df, use_container_width=True)
        st.info(factor_message)

        # Sa objetivo en T1
        sa_obj_T1 = float(np.interp(T1, periods, target_sa))
        mean_srss_T1 = float(np.interp(T1, periods, mean_srss))
        mean_srss_scaled_T1 = float(np.interp(T1, periods, mean_srss_scaled))

        summary_df = pd.DataFrame([
            {"Parámetro": "T1 (s)", "Valor": T1},
            {"Parámetro": f"{lower_mult:.2f}T1 (s)", "Valor": T_low},
            {"Parámetro": f"{upper_mult:.2f}T1 (s)", "Valor": T_high},
            {"Parámetro": "Sa objetivo en T1 (g)", "Valor": sa_obj_T1},
            {"Parámetro": "Media SRSS original en T1 (g)", "Valor": mean_srss_T1},
            {"Parámetro": "Media SRSS escalada en T1 (g)", "Valor": mean_srss_scaled_T1},
            {"Parámetro": "Factor global", "Valor": global_factor if global_factor is not None else np.nan},
        ])

        st.subheader("3. Resumen general")
        st.dataframe(summary_df, use_container_width=True)

        # ====================================================
        # GRÁFICA TIPO EXCEL
        # ====================================================
        st.subheader("4. Gráfica tipo Excel: media SRSS vs espectro objetivo")

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

                fig.add_trace(go.Scatter(
                    x=periods,
                    y=scaled_srss_matrix[i, :],
                    mode="lines",
                    line=dict(width=1, dash="dot"),
                    opacity=0.35,
                    name=f"SRSS escalado {base_name}"
                ))

        if show_original_mean:
            fig.add_trace(go.Scatter(
                x=periods,
                y=mean_srss,
                mode="lines",
                line=dict(width=3, dash="dot"),
                name="Media SRSS original"
            ))

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

        # Rango sombreado
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

        fig.update_layout(
            title="Escalamiento de registros por media SRSS",
            xaxis_title="T [s]",
            yaxis_title="Sa [g]",
            legend_title="Curvas",
            height=650
        )

        st.plotly_chart(fig, use_container_width=True)

        # ====================================================
        # VERIFICACIÓN DEL CUMPLIMIENTO
        # ====================================================
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

        # ====================================================
        # DESCARGAS
        # ====================================================
        st.subheader("6. Descargas")

        csv_pairs = pair_results_df.to_csv(index=False).encode("utf-8")

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
                file_name="media_srss_escalada_vs_objetivo.csv",
                mime="text/csv"
            )

        with c3:
            st.download_button(
                "Descargar verificación CSV",
                data=csv_check,
                file_name="verificacion_rango_escalamiento.csv",
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

                    accN_scaled = dataN["acc_unit"] * fac
                    accE_scaled = dataE["acc_unit"] * fac

                    nameN = Path(dataN["file_name"]).stem + f"_ESC_FAC_{fac:.4f}.txt"
                    nameE = Path(dataE["file_name"]).stem + f"_ESC_FAC_{fac:.4f}.txt"

                    zf.writestr(nameN, make_scaled_txt(dataN["time"], accN_scaled))
                    zf.writestr(nameE, make_scaled_txt(dataE["time"], accE_scaled))

            st.download_button(
                "Descargar acelerogramas escalados ZIP",
                data=zip_buffer.getvalue(),
                file_name="acelerogramas_escalados_pares_srss.zip",
                mime="application/zip"
            )

        with st.expander("Formato esperado del espectro objetivo"):
            st.markdown(
                """
El archivo del espectro objetivo debe tener dos columnas:

```text
T      Sa
0.01   0.80
0.02   0.95
0.10   1.75
0.50   1.75
1.00   1.30
2.00   0.65
```

- Columna 1: período `T` en segundos.
- Columna 2: aceleración espectral `Sa` en `g`.
"""
            )

        with st.expander("Criterio usado"):
            st.markdown(
                r"""
Para cada par se calcula:

\[
Sa_{SRSS}(T)=\sqrt{Sa_N(T)^2+Sa_E(T)^2}
\]

Luego se calcula la media:

\[
\overline{Sa}_{SRSS}(T)=\frac{1}{n}\sum_{i=1}^{n}Sa_{SRSS,i}(T)
\]

Para el factor global:

\[
FAC=\max\left(\frac{Sa_{objetivo}(T)}{\overline{Sa}_{SRSS}(T)}\right)
\]

dentro del rango:

\[
0.2T_1 \leq T \leq 1.5T_1
\]

Si escoges criterio 90%, el numerador se cambia a:

\[
0.90 \cdot Sa_{objetivo}(T)
\]
"""
            )

    except Exception as e:
        st.error(f"Error durante el procesamiento: {e}")

elif uploaded_records and not uploaded_objective:
    st.info("Ahora sube el espectro objetivo para calcular el escalamiento.")
else:
    st.warning("Sube los registros N/E y el espectro objetivo para iniciar.")
