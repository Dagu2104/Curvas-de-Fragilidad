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


def read_accel_file(uploaded_file):
    raw = uploaded_file.read()
    text = raw.decode("utf-8", errors="ignore")
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)

    df = pd.read_csv(io.StringIO(text), sep=r"[\s,;]+", engine="python", header=None, comment="#")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

    if df.empty:
        raise ValueError("No se encontraron datos numéricos.")

    numeric_cols = []
    for col in df.columns:
        values = df[col].dropna().to_numpy(dtype=float)
        if len(values) > 10:
            numeric_cols.append(col)

    if not numeric_cols:
        raise ValueError("No hay columnas numéricas suficientes.")

    if len(numeric_cols) == 1:
        acc = df[numeric_cols[0]].dropna().to_numpy(dtype=float)
        return None, acc, None, "Una columna: aceleración"

    col0, col1 = numeric_cols[0], numeric_cols[1]
    x0 = df[col0].dropna().to_numpy(dtype=float)
    x1 = df[col1].dropna().to_numpy(dtype=float)
    n = min(len(x0), len(x1))
    x0, x1 = x0[:n], x1[:n]

    diffs = np.diff(x0)
    positive_ratio = np.mean(diffs > 0)
    dt_median = np.median(diffs)

    if positive_ratio > 0.95 and dt_median > 0:
        return x0, x1, float(dt_median), "Dos columnas: tiempo + aceleración"

    return None, x1, None, "Varias columnas: se usó segunda columna como aceleración"


def convert_to_m_s2(accel, unit):
    if unit == "g":
        return accel * G
    if unit == "m/s²":
        return accel
    if unit == "cm/s²":
        return accel / 100.0
    raise ValueError("Unidad no reconocida.")


def spectral_sa_sd_newmark(acc_g_m_s2, dt, period, damping=0.05):
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

    for i in range(1, n):
        p_eff = (
            p[i]
            + m * (a0 * u[i - 1] + a2 * v[i - 1] + a3 * a[i - 1])
            + c * (a1 * u[i - 1] + a4 * v[i - 1] + a5 * a[i - 1])
        )
        u[i] = p_eff / k_eff
        a[i] = a0 * (u[i] - u[i - 1]) - a2 * v[i - 1] - a3 * a[i - 1]
        v[i] = v[i - 1] + dt * ((1.0 - gamma) * a[i - 1] + gamma * a[i])

    sd_m = float(np.max(np.abs(u)))
    psa_m_s2 = w**2 * sd_m
    sa_g = float(psa_m_s2 / G)
    return sa_g, sd_m


def response_spectrum(acc_m_s2, dt, periods, damping=0.05):
    rows = []
    for T in periods:
        sa_g, sd_m = spectral_sa_sd_newmark(acc_m_s2, dt=dt, period=float(T), damping=damping)
        rows.append({"T (s)": float(T), "Sa (g)": sa_g, "Sd (m)": sd_m, "Sd (cm)": sd_m * 100.0})
    return pd.DataFrame(rows)


def fit_lognormal_fragility(im_values, exceedance):
    im_values = np.asarray(im_values, dtype=float)
    y = np.asarray(exceedance, dtype=int)
    mask = np.isfinite(im_values) & (im_values > 0) & np.isfinite(y)
    im_values = im_values[mask]
    y = y[mask]

    if len(im_values) < 4:
        return None, None, "Se necesitan más puntos."
    if len(np.unique(y)) < 2:
        return None, None, "No se puede ajustar: todos los puntos son 0 o todos son 1."

    def neg_loglike(params):
        ln_theta, ln_beta = params
        beta = np.exp(ln_beta)
        z = (np.log(im_values) - ln_theta) / beta
        p = np.clip(norm.cdf(z), 1e-8, 1.0 - 1e-8)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    x0 = np.array([np.log(np.median(im_values)), np.log(0.40)])
    res = minimize(neg_loglike, x0, method="Nelder-Mead")
    if not res.success:
        return None, None, "No se pudo ajustar."
    return float(np.exp(res.x[0])), float(np.exp(res.x[1])), "OK"


def fragility_probability(im_grid, theta, beta):
    return norm.cdf((np.log(im_grid) - np.log(theta)) / beta)


def parse_targets(text):
    text = text.replace(";", ",")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    values = []
    for p in parts:
        p = p.replace(",", ".")
        try:
            value = float(p)
            if value > 0:
                values.append(value)
        except ValueError:
            pass
    return sorted(list(set(values)))


def make_scaled_txt(time, accel_scaled_original_unit):
    if time is not None:
        lines = [f"{t:.6f}\t{a:.8e}" for t, a in zip(time, accel_scaled_original_unit)]
    else:
        lines = [f"{a:.8e}" for a in accel_scaled_original_unit]
    return "\n".join(lines)


st.set_page_config(page_title="Fragilidad Sa/Sd", page_icon="📈", layout="wide")
st.title("📈 Curvas de fragilidad con Sa o Sd + espectros de respuesta")
st.markdown(
    """
Esta app permite subir acelerogramas, calcular **Sa(T)** y **Sd(T)**, escoger si la curva de fragilidad se arma con **Sa** o con **Sd**, escalar los registros a niveles objetivo y graficar los espectros de respuesta en una sola figura interactiva.
"""
)

with st.sidebar:
    st.header("1. Estructura")
    period = st.number_input("Período de la estructura T (s)", min_value=0.01, value=0.50, step=0.01, format="%.3f")
    damping = st.number_input("Amortiguamiento ξ", min_value=0.00, max_value=0.30, value=0.05, step=0.01, format="%.3f")

    st.header("2. Registro")
    dt_manual = st.number_input("dt manual si el archivo tiene una columna (s)", min_value=0.0001, value=0.005, step=0.001, format="%.4f")
    unit = st.selectbox("Unidad de aceleración en archivos", ["g", "m/s²", "cm/s²"])

    st.header("3. Medida de intensidad")
    im_type = st.radio("Escoger medida de intensidad para fragilidad", ["Sa(T) en g", "Sd(T) en cm"], index=0)

    if im_type == "Sa(T) en g":
        targets_text = st.text_area("Niveles objetivo de Sa(T) en g", value="0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00, 1.20", height=90)
        target_label = "Sa objetivo (g)"
        im_label = "Sa(T) escalado (g)"
    else:
        targets_text = st.text_area("Niveles objetivo de Sd(T) en cm", value="1, 2, 3, 4, 5, 7, 10, 15, 20", height=90)
        target_label = "Sd objetivo (cm)"
        im_label = "Sd(T) escalado (cm)"

    targets = parse_targets(targets_text)

    st.header("4. Límites de daño")
    st.caption("Los límites deben estar en la misma unidad de la medida escogida.")
    if im_type == "Sa(T) en g":
        ds_leve = st.number_input("DS1 leve: Sa ≥", min_value=0.001, value=0.15, step=0.01, format="%.3f")
        ds_moderado = st.number_input("DS2 moderado: Sa ≥", min_value=0.001, value=0.30, step=0.01, format="%.3f")
        ds_severo = st.number_input("DS3 severo: Sa ≥", min_value=0.001, value=0.50, step=0.01, format="%.3f")
        ds_colapso = st.number_input("DS4 colapso: Sa ≥", min_value=0.001, value=0.80, step=0.01, format="%.3f")
    else:
        ds_leve = st.number_input("DS1 leve: Sd ≥ cm", min_value=0.001, value=1.0, step=0.5, format="%.3f")
        ds_moderado = st.number_input("DS2 moderado: Sd ≥ cm", min_value=0.001, value=3.0, step=0.5, format="%.3f")
        ds_severo = st.number_input("DS3 severo: Sd ≥ cm", min_value=0.001, value=7.0, step=0.5, format="%.3f")
        ds_colapso = st.number_input("DS4 colapso: Sd ≥ cm", min_value=0.001, value=15.0, step=0.5, format="%.3f")

    st.header("5. Espectros")
    t_min = st.number_input("T mínimo del espectro (s)", min_value=0.01, value=0.01, step=0.01, format="%.2f")
    t_max = st.number_input("T máximo del espectro (s)", min_value=0.10, value=4.00, step=0.10, format="%.2f")
    n_periods = st.number_input("Número de puntos del espectro", min_value=20, max_value=400, value=120, step=10)
    spectrum_to_show = st.radio("Tipo de espectro a graficar", ["Sa", "Sd", "Sa y Sd"], index=0)

    st.header("6. Exportación")
    generar_zip = st.checkbox("Generar ZIP con acelerogramas escalados", value=False)

uploaded_files = st.file_uploader("Sube acelerogramas de una dirección", type=["txt", "csv"], accept_multiple_files=True)

if uploaded_files and targets:
    original_rows, scaled_rows = [], []
    spectra_dict, scaled_files = {}, {}
    periods = np.linspace(float(t_min), float(t_max), int(n_periods))
    progress = st.progress(0, text="Procesando registros...")

    for i, file in enumerate(uploaded_files):
        try:
            time, accel_raw_original_unit, dt_detected, formato = read_accel_file(file)
            dt_usado = dt_detected if dt_detected is not None else dt_manual
            accel_original_m_s2 = convert_to_m_s2(accel_raw_original_unit, unit)
            pga_original_g = np.max(np.abs(accel_original_m_s2)) / G
            sa_original_g, sd_original_m = spectral_sa_sd_newmark(accel_original_m_s2, dt=dt_usado, period=period, damping=damping)
            sd_original_cm = sd_original_m * 100.0

            spec_df = response_spectrum(accel_original_m_s2, dt=dt_usado, periods=periods, damping=damping)
            spectra_dict[file.name] = spec_df

            original_rows.append({
                "Registro": file.name,
                "Formato detectado": formato,
                "Puntos": len(accel_raw_original_unit),
                "dt usado (s)": dt_usado,
                "PGA original (g)": pga_original_g,
                f"Sa original T={period:.3f}s (g)": sa_original_g,
                f"Sd original T={period:.3f}s (cm)": sd_original_cm,
                "Error": ""
            })

            im_original = sa_original_g if im_type == "Sa(T) en g" else sd_original_cm
            if im_original <= 0:
                raise ValueError("La intensidad original es cero o negativa. No se puede escalar.")

            for target in targets:
                factor = target / im_original
                accel_scaled_original_unit = accel_raw_original_unit * factor
                pga_scaled_g = pga_original_g * factor
                sa_scaled_g = sa_original_g * factor
                sd_scaled_cm = sd_original_cm * factor
                im_scaled = sa_scaled_g if im_type == "Sa(T) en g" else sd_scaled_cm

                scaled_rows.append({
                    "Registro": file.name,
                    target_label: target,
                    "Factor escala": factor,
                    "PGA escalado (g)": pga_scaled_g,
                    f"Sa escalado T={period:.3f}s (g)": sa_scaled_g,
                    f"Sd escalado T={period:.3f}s (cm)": sd_scaled_cm,
                    im_label: im_scaled,
                    "DS1 leve": int(im_scaled >= ds_leve),
                    "DS2 moderado": int(im_scaled >= ds_moderado),
                    "DS3 severo": int(im_scaled >= ds_severo),
                    "DS4 colapso": int(im_scaled >= ds_colapso),
                })

                if generar_zip:
                    stem = Path(file.name).stem
                    safe_target = f"Sa_{str(target).replace('.', 'p')}g" if im_type == "Sa(T) en g" else f"Sd_{str(target).replace('.', 'p')}cm"
                    out_name = f"{stem}_{safe_target}_FE_{factor:.4f}.txt"
                    scaled_files[out_name] = make_scaled_txt(time, accel_scaled_original_unit)

        except Exception as e:
            original_rows.append({
                "Registro": file.name,
                "Formato detectado": "",
                "Puntos": None,
                "dt usado (s)": None,
                "PGA original (g)": None,
                f"Sa original T={period:.3f}s (g)": None,
                f"Sd original T={period:.3f}s (cm)": None,
                "Error": str(e)
            })

        progress.progress((i + 1) / len(uploaded_files), text=f"Procesando {i+1}/{len(uploaded_files)}")
    progress.empty()

    df_original = pd.DataFrame(original_rows)
    df_scaled = pd.DataFrame(scaled_rows)

    st.subheader("1. Registros originales")
    st.dataframe(df_original, use_container_width=True)

    st.subheader("2. Puntos generados por escalamiento")
    st.dataframe(df_scaled, use_container_width=True)

    if not df_scaled.empty:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.download_button("Descargar originales CSV", data=df_original.to_csv(index=False).encode("utf-8"), file_name="registros_originales_sa_sd.csv", mime="text/csv")
        with col_b:
            st.download_button("Descargar puntos escalados CSV", data=df_scaled.to_csv(index=False).encode("utf-8"), file_name="puntos_escalados_sa_sd.csv", mime="text/csv")
        if generar_zip and scaled_files:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for name, txt in scaled_files.items():
                    zf.writestr(name, txt)
            with col_c:
                st.download_button("Descargar acelerogramas escalados ZIP", data=zip_buffer.getvalue(), file_name="acelerogramas_escalados_sa_sd.zip", mime="application/zip")

        st.subheader("3. Espectros de respuesta")
        available_records = list(spectra_dict.keys())
        selected_records = st.multiselect("Escoge los registros que quieres ver en el espectro", available_records, default=available_records)

        if selected_records:
            if spectrum_to_show in ["Sa", "Sa y Sd"]:
                fig_sa = go.Figure()
                for rec in selected_records:
                    spec_df = spectra_dict[rec]
                    fig_sa.add_trace(go.Scatter(x=spec_df["T (s)"], y=spec_df["Sa (g)"], mode="lines", name=rec))
                fig_sa.add_vline(x=period, line_dash="dash", annotation_text=f"T = {period:.3f}s", annotation_position="top")
                fig_sa.update_layout(title="Espectro de aceleración Sa", xaxis_title="Período T (s)", yaxis_title="Sa (g)", legend_title="Registros")
                st.plotly_chart(fig_sa, use_container_width=True)

            if spectrum_to_show in ["Sd", "Sa y Sd"]:
                fig_sd = go.Figure()
                for rec in selected_records:
                    spec_df = spectra_dict[rec]
                    fig_sd.add_trace(go.Scatter(x=spec_df["T (s)"], y=spec_df["Sd (cm)"], mode="lines", name=rec))
                fig_sd.add_vline(x=period, line_dash="dash", annotation_text=f"T = {period:.3f}s", annotation_position="top")
                fig_sd.update_layout(title="Espectro de desplazamiento Sd", xaxis_title="Período T (s)", yaxis_title="Sd (cm)", legend_title="Registros")
                st.plotly_chart(fig_sd, use_container_width=True)
        else:
            st.info("Selecciona al menos un registro para graficar espectros.")

        st.subheader("4. Curvas de fragilidad")
        im_values = df_scaled[im_label].to_numpy(dtype=float)
        damage_states = {
            "DS1 Leve": df_scaled["DS1 leve"].to_numpy(dtype=int),
            "DS2 Moderado": df_scaled["DS2 moderado"].to_numpy(dtype=int),
            "DS3 Severo": df_scaled["DS3 severo"].to_numpy(dtype=int),
            "DS4 Colapso": df_scaled["DS4 colapso"].to_numpy(dtype=int),
        }

        im_min = max(0.0001, min(targets) * 0.50)
        im_max = max(max(targets) * 1.30, ds_colapso * 1.30)
        im_grid = np.linspace(im_min, im_max, 400)
        fig_frag = go.Figure()
        fit_rows = []

        for ds_name, exceedance in damage_states.items():
            theta, beta, status = fit_lognormal_fragility(im_values, exceedance)
            zeros = int(np.sum(exceedance == 0))
            ones = int(np.sum(exceedance == 1))
            if theta is not None:
                prob = fragility_probability(im_grid, theta, beta)
                fig_frag.add_trace(go.Scatter(x=im_grid, y=prob, mode="lines", name=ds_name))
            fit_rows.append({
                "Estado de daño": ds_name,
                "Cantidad 0": zeros,
                "Cantidad 1": ones,
                f"θ / mediana {im_label}": theta,
                "β / dispersión lognormal": beta,
                "Estado ajuste": status
            })

        fig_frag.update_layout(xaxis_title=f"IM = {im_label}", yaxis_title="P(DS ≥ ds | IM)", yaxis=dict(range=[0, 1]), legend_title="Curvas")
        st.plotly_chart(fig_frag, use_container_width=True)

        st.subheader("5. Parámetros ajustados")
        st.dataframe(pd.DataFrame(fit_rows), use_container_width=True)
        st.warning("Esta versión decide el estado de daño comparando Sa o Sd con límites definidos por el usuario. Para una curva estructural más realista, usa derivas máximas de entrepiso obtenidas del análisis tiempo-historia.")

else:
    st.warning("Sube registros y define niveles objetivo para iniciar.")

with st.expander("Cómo escoger entre Sa y Sd"):
    st.markdown(r"""
- Usa **Sa(T)** si quieres trabajar con aceleración espectral como intensidad sísmica.
- Usa **Sd(T)** si quieres trabajar con desplazamiento espectral.

La relación es:

$$Sa = \omega^2 Sd$$

$$Sd = \frac{Sa \cdot g}{\omega^2}$$

con:

$$\omega = \frac{2\pi}{T}$$

En el programa, **Sd** se obtiene directamente como el desplazamiento máximo relativo del oscilador SDOF.
""")

with st.expander("Siguiente mejora recomendada"):
    st.markdown("""
La mejora más importante sería agregar un modo con derivas:

```text
Registro + escala + Sa/Sd objetivo
→ análisis estructural externo
→ cargas una tabla con deriva máxima
→ el programa compara con límites de deriva
→ curva de fragilidad
```
""")
