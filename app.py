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

    x0 = df[numeric_cols[0]].dropna().to_numpy(dtype=float)
    x1 = df[numeric_cols[1]].dropna().to_numpy(dtype=float)
    n = min(len(x0), len(x1))
    x0 = x0[:n]
    x1 = x1[:n]

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


def spectral_acceleration_newmark(acc_g_m_s2, dt, period, damping=0.05):
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

    sd = np.max(np.abs(u))
    psa_m_s2 = w**2 * sd
    return float(psa_m_s2 / G)


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
        p = norm.cdf(z)
        p = np.clip(p, 1e-8, 1.0 - 1e-8)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    x0 = np.array([np.log(np.median(im_values)), np.log(0.40)])
    res = minimize(neg_loglike, x0, method="Nelder-Mead")

    if not res.success:
        return None, None, "No se pudo ajustar."

    return float(np.exp(res.x[0])), float(np.exp(res.x[1])), "OK"


def fragility_probability(im_grid, theta, beta):
    return norm.cdf((np.log(im_grid) - np.log(theta)) / beta)


def parse_sa_targets(text):
    text = text.replace(";", ",")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    values = []
    for p in parts:
        try:
            val = float(p)
            if val > 0:
                values.append(val)
        except ValueError:
            pass
    return sorted(list(set(values)))


def make_scaled_txt(time, accel_scaled_original_unit):
    if time is not None:
        return "\n".join(f"{t:.6f}\t{a:.8e}" for t, a in zip(time, accel_scaled_original_unit))
    return "\n".join(f"{a:.8e}" for a in accel_scaled_original_unit)


st.set_page_config(page_title="Fragilidad con Escalamiento Sa", page_icon="📈", layout="wide")
st.title("📈 Curvas de fragilidad con escalamiento a Sa objetivo")

st.markdown(r"""
Esta versión permite subir acelerogramas, calcular su **Sa(T)** y escalarlos a varios niveles de **Sa objetivo**.

El factor de escala es:

$$FE = \frac{Sa_{objetivo}}{Sa_{registro}(T)}$$

Luego:

$$a_{escalado}(t)=FE \cdot a_{original}(t)$$
""")

with st.sidebar:
    st.header("1. Estructura")
    period = st.number_input("Período T (s)", min_value=0.01, value=0.50, step=0.01, format="%.3f")
    damping = st.number_input("Amortiguamiento ξ", min_value=0.00, max_value=0.30, value=0.05, step=0.01, format="%.3f")

    st.header("2. Registro")
    dt_manual = st.number_input("dt manual si el archivo tiene una columna (s)", min_value=0.0001, value=0.005, step=0.001, format="%.4f")
    unit = st.selectbox("Unidad de aceleración en los archivos", ["g", "m/s²", "cm/s²"])

    st.header("3. Sa objetivo")
    sa_targets_text = st.text_area(
        "Niveles de Sa objetivo en g, separados por coma",
        value="0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00, 1.20",
        height=100,
    )

    st.header("4. Límites de daño")
    st.caption("Versión simplificada: el daño se define con límites de Sa(T).")
    ds_leve = st.number_input("Daño leve: Sa ≥", min_value=0.001, value=0.15, step=0.01, format="%.3f")
    ds_moderado = st.number_input("Daño moderado: Sa ≥", min_value=0.001, value=0.30, step=0.01, format="%.3f")
    ds_severo = st.number_input("Daño severo: Sa ≥", min_value=0.001, value=0.50, step=0.01, format="%.3f")
    ds_colapso = st.number_input("Colapso: Sa ≥", min_value=0.001, value=0.80, step=0.01, format="%.3f")

    st.header("5. Archivos escalados")
    generar_zip = st.checkbox("Generar ZIP con acelerogramas escalados", value=False)

uploaded_files = st.file_uploader("Sube acelerogramas de una sola dirección", type=["txt", "csv"], accept_multiple_files=True)
sa_targets = parse_sa_targets(sa_targets_text)

if not sa_targets:
    st.error("Ingresa al menos un Sa objetivo válido. Ejemplo: 0.10, 0.20, 0.30")

if uploaded_files and sa_targets:
    original_rows = []
    scaled_rows = []
    scaled_files = {}
    progress = st.progress(0, text="Procesando registros...")

    for i, file in enumerate(uploaded_files):
        try:
            time, accel_raw_original_unit, dt_detected, formato = read_accel_file(file)
            dt_usado = dt_detected if dt_detected is not None else dt_manual
            accel_original_m_s2 = convert_to_m_s2(accel_raw_original_unit, unit)
            pga_original_g = np.max(np.abs(accel_original_m_s2)) / G
            sa_original_g = spectral_acceleration_newmark(accel_original_m_s2, dt=dt_usado, period=period, damping=damping)

            original_rows.append({
                "Registro": file.name,
                "Formato detectado": formato,
                "Puntos": len(accel_raw_original_unit),
                "dt usado (s)": dt_usado,
                "Acel. mín original": np.min(accel_raw_original_unit),
                "Acel. máx original": np.max(accel_raw_original_unit),
                "PGA original (g)": pga_original_g,
                f"Sa original T={period:.3f}s (g)": sa_original_g,
                "Error": "",
            })

            if sa_original_g <= 0:
                raise ValueError("Sa original es cero o negativa. No se puede escalar.")

            for sa_obj in sa_targets:
                factor = sa_obj / sa_original_g
                accel_scaled_original_unit = accel_raw_original_unit * factor
                pga_scaled_g = pga_original_g * factor
                sa_scaled_g = sa_original_g * factor

                scaled_rows.append({
                    "Registro": file.name,
                    "Sa objetivo (g)": sa_obj,
                    "Factor escala": factor,
                    "PGA escalado (g)": pga_scaled_g,
                    f"Sa escalado T={period:.3f}s (g)": sa_scaled_g,
                    "DS leve": int(sa_scaled_g >= ds_leve),
                    "DS moderado": int(sa_scaled_g >= ds_moderado),
                    "DS severo": int(sa_scaled_g >= ds_severo),
                    "DS colapso": int(sa_scaled_g >= ds_colapso),
                })

                if generar_zip:
                    stem = Path(file.name).stem
                    safe_sa = str(sa_obj).replace(".", "p")
                    out_name = f"{stem}_SaObj_{safe_sa}g_FE_{factor:.4f}.txt"
                    scaled_files[out_name] = make_scaled_txt(time, accel_scaled_original_unit)

        except Exception as e:
            original_rows.append({
                "Registro": file.name,
                "Formato detectado": "",
                "Puntos": None,
                "dt usado (s)": None,
                "Acel. mín original": None,
                "Acel. máx original": None,
                "PGA original (g)": None,
                f"Sa original T={period:.3f}s (g)": None,
                "Error": str(e),
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
            st.download_button("Descargar originales CSV", data=df_original.to_csv(index=False).encode("utf-8"), file_name="registros_originales.csv", mime="text/csv")
        with col_b:
            st.download_button("Descargar puntos escalados CSV", data=df_scaled.to_csv(index=False).encode("utf-8"), file_name="puntos_escalados_fragilidad.csv", mime="text/csv")

        if generar_zip and scaled_files:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for name, txt in scaled_files.items():
                    zf.writestr(name, txt)
            with col_c:
                st.download_button("Descargar acelerogramas escalados ZIP", data=zip_buffer.getvalue(), file_name="acelerogramas_escalados.zip", mime="application/zip")

        sa_scaled_col = f"Sa escalado T={period:.3f}s (g)"

        st.subheader("3. PGA escalado vs Sa objetivo")
        fig_points = go.Figure()
        fig_points.add_trace(go.Scatter(x=df_scaled["Sa objetivo (g)"], y=df_scaled["PGA escalado (g)"], mode="markers", text=df_scaled["Registro"], name="Puntos escalados"))
        fig_points.update_layout(xaxis_title="Sa objetivo = Sa(T) escalado [g]", yaxis_title="PGA escalado [g]")
        st.plotly_chart(fig_points, use_container_width=True)

        st.subheader("4. Curvas de fragilidad")
        im_values = df_scaled[sa_scaled_col].to_numpy(dtype=float)
        damage_states = {
            "Leve": df_scaled["DS leve"].to_numpy(dtype=int),
            "Moderado": df_scaled["DS moderado"].to_numpy(dtype=int),
            "Severo": df_scaled["DS severo"].to_numpy(dtype=int),
            "Colapso": df_scaled["DS colapso"].to_numpy(dtype=int),
        }

        im_min = max(0.001, min(sa_targets) * 0.50)
        im_max = max(max(sa_targets) * 1.30, ds_colapso * 1.30)
        im_grid = np.linspace(im_min, im_max, 400)
        fig_frag = go.Figure()
        fit_rows = []

        for ds_name, exceedance in damage_states.items():
            theta, beta, status = fit_lognormal_fragility(im_values, exceedance)
            zeros = int(np.sum(exceedance == 0))
            ones = int(np.sum(exceedance == 1))
            if theta is not None:
                prob = fragility_probability(im_grid, theta, beta)
                fig_frag.add_trace(go.Scatter(x=im_grid, y=prob, mode="lines", name=f"DS {ds_name}"))
            fit_rows.append({
                "Estado de daño": ds_name,
                "Cantidad 0": zeros,
                "Cantidad 1": ones,
                "θ / mediana Sa (g)": theta,
                "β / dispersión lognormal": beta,
                "Estado ajuste": status,
            })

        fig_frag.update_layout(xaxis_title="IM = Sa(T) escalado [g]", yaxis_title="P(DS ≥ ds | Sa)", yaxis=dict(range=[0, 1]), legend_title="Curvas")
        st.plotly_chart(fig_frag, use_container_width=True)

        st.subheader("5. Parámetros ajustados")
        st.dataframe(pd.DataFrame(fit_rows), use_container_width=True)

        st.warning("""
Importante: esta versión genera más puntos al escalar los acelerogramas, pero el daño todavía se define con límites de Sa.
Para una curva de fragilidad estructural más realista, debes usar la respuesta de la estructura: deriva máxima, desplazamiento de techo, rotaciones plásticas o daño de elementos.
""")
else:
    st.warning("Sube registros y define niveles de Sa objetivo para iniciar.")

with st.expander("¿Cómo usar esta versión?"):
    st.markdown(r"""
Ejemplo:

```text
0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00
```

Si subes 11 registros:

```text
11 registros × 8 Sa objetivo = 88 puntos
```

Cada punto tiene registro original, Sa objetivo, factor de escala, PGA escalado y estado de daño.
""")

with st.expander("Advertencia técnica importante"):
    st.markdown("""
Esta app todavía no reemplaza un análisis dinámico no lineal.

Para una curva de fragilidad más seria, el flujo debería ser:

```text
Registro original
→ escalamiento a Sa objetivo
→ análisis estructural tiempo-historia
→ obtención de deriva máxima
→ comparación con límites de daño por deriva
→ ajuste de curva lognormal
```

En esta versión simplificada, el estado de daño se decide directamente por Sa(T).
Eso sirve para probar el procedimiento y automatizar el escalamiento, pero no representa por sí solo el daño real de la estructura.
""")
