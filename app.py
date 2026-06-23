
import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy.stats import norm

G = 9.80665


# ============================================================
# LECTURA INTELIGENTE DE ACELEROGRAMAS
# ============================================================
def read_accel_file(uploaded_file):
    """
    Lee un acelerograma .txt/.csv.

    Casos aceptados:

    1) Una columna:
       aceleracion

    2) Dos columnas:
       tiempo    aceleracion

    3) Más columnas:
       toma columna 1 como tiempo y columna 2 como aceleración,
       si la primera columna parece ser tiempo creciente.

    Retorna:
       time_array, accel_array, dt_detected, formato_detectado
    """

    raw = uploaded_file.read()
    text = raw.decode("utf-8", errors="ignore")

    # Reemplaza coma decimal por punto, si existe.
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

    # Caso de una sola columna: solo aceleración
    if len(numeric_cols) == 1:
        acc = df[numeric_cols[0]].dropna().to_numpy(dtype=float)
        return None, acc, None, "Una columna: aceleración"

    # Caso de dos o más columnas
    col0 = numeric_cols[0]
    col1 = numeric_cols[1]

    x0 = df[col0].dropna().to_numpy(dtype=float)
    x1 = df[col1].dropna().to_numpy(dtype=float)

    n = min(len(x0), len(x1))
    x0 = x0[:n]
    x1 = x1[:n]

    # Detectar si la primera columna es tiempo:
    # - debe ser creciente
    # - debe tener incrementos casi constantes
    diffs = np.diff(x0)
    positive_ratio = np.mean(diffs > 0)
    dt_median = np.median(diffs)

    if positive_ratio > 0.95 and dt_median > 0:
        time = x0
        acc = x1
        dt_detected = float(dt_median)
        return time, acc, dt_detected, "Dos columnas: tiempo + aceleración"

    # Si no parece tiempo, toma segunda columna como aceleración igual,
    # porque en registros PEER/RSN suele venir tiempo + aceleración.
    acc = x1
    return None, acc, None, "Varias columnas: se usó la segunda columna como aceleración"


def convert_to_m_s2(accel, unit):
    if unit == "g":
        return accel * G
    elif unit == "m/s²":
        return accel
    elif unit == "cm/s²":
        return accel / 100.0
    else:
        raise ValueError("Unidad no reconocida.")


# ============================================================
# CÁLCULO DE Sa(T) CON NEWMARK BETA
# ============================================================
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
            + m * (a0 * u[i-1] + a2 * v[i-1] + a3 * a[i-1])
            + c * (a1 * u[i-1] + a4 * v[i-1] + a5 * a[i-1])
        )

        u[i] = p_eff / k_eff
        a[i] = a0 * (u[i] - u[i-1]) - a2 * v[i-1] - a3 * a[i-1]
        v[i] = v[i-1] + dt * ((1.0 - gamma) * a[i-1] + gamma * a[i])

    sd = np.max(np.abs(u))
    psa_m_s2 = w**2 * sd
    psa_g = psa_m_s2 / G

    return float(psa_g)


# ============================================================
# FRAGILIDAD LOGNORMAL
# ============================================================
def fit_lognormal_fragility(im_values, exceedance):
    im_values = np.asarray(im_values, dtype=float)
    y = np.asarray(exceedance, dtype=int)

    mask = np.isfinite(im_values) & (im_values > 0) & np.isfinite(y)
    im_values = im_values[mask]
    y = y[mask]

    if len(im_values) < 3:
        return None, None, "Se necesitan más registros."

    if len(np.unique(y)) < 2:
        return None, None, "No se puede ajustar: todos los registros están del mismo lado del estado de daño."

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


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
st.set_page_config(
    page_title="Curvas de Fragilidad Sísmica",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Curvas de fragilidad sísmica con registros en una dirección")

st.markdown(
    """
Sube registros sísmicos `.txt` o `.csv`.  
El programa detecta automáticamente si el archivo viene como:

- **una columna:** aceleración
- **dos columnas:** tiempo + aceleración

Si detecta columna de tiempo, calcula automáticamente el `dt`.
"""
)

with st.sidebar:
    st.header("Datos de la estructura")

    period = st.number_input(
        "Período de la estructura, T (s)",
        min_value=0.01,
        value=0.50,
        step=0.01,
        format="%.3f"
    )

    damping = st.number_input(
        "Amortiguamiento ξ",
        min_value=0.00,
        max_value=0.30,
        value=0.05,
        step=0.01,
        format="%.3f"
    )

    st.header("Datos del registro")

    dt_manual = st.number_input(
        "dt manual, solo si el archivo tiene una columna (s)",
        min_value=0.0001,
        value=0.005,
        step=0.001,
        format="%.4f"
    )

    unit = st.selectbox(
        "Unidad de aceleración en los archivos",
        ["g", "m/s²", "cm/s²"]
    )

    st.divider()

    st.header("Estados de daño")
    st.caption("Límites en función de Sa(T), en g.")

    ds_leve = st.number_input("Daño leve: Sa ≥", min_value=0.001, value=0.15, step=0.01, format="%.3f")
    ds_moderado = st.number_input("Daño moderado: Sa ≥", min_value=0.001, value=0.30, step=0.01, format="%.3f")
    ds_severo = st.number_input("Daño severo: Sa ≥", min_value=0.001, value=0.50, step=0.01, format="%.3f")
    ds_colapso = st.number_input("Colapso: Sa ≥", min_value=0.001, value=0.80, step=0.01, format="%.3f")


uploaded_files = st.file_uploader(
    "Sube acelerogramas de una sola dirección",
    type=["txt", "csv"],
    accept_multiple_files=True
)


if uploaded_files:
    results = []
    progress = st.progress(0, text="Procesando registros...")

    for i, file in enumerate(uploaded_files):
        try:
            time, accel_raw, dt_detected, formato = read_accel_file(file)

            if dt_detected is not None:
                dt_usado = dt_detected
            else:
                dt_usado = dt_manual

            accel_m_s2 = convert_to_m_s2(accel_raw, unit)

            pga_g = np.max(np.abs(accel_m_s2)) / G

            sa_g = spectral_acceleration_newmark(
                accel_m_s2,
                dt=dt_usado,
                period=period,
                damping=damping
            )

            results.append({
                "Registro": file.name,
                "Formato detectado": formato,
                "Puntos": len(accel_raw),
                "dt usado (s)": dt_usado,
                "Acel. mín": np.min(accel_raw),
                "Acel. máx": np.max(accel_raw),
                "PGA (g)": pga_g,
                f"Sa(T={period:.3f}s) (g)": sa_g,
                "DS leve": int(sa_g >= ds_leve),
                "DS moderado": int(sa_g >= ds_moderado),
                "DS severo": int(sa_g >= ds_severo),
                "DS colapso": int(sa_g >= ds_colapso),
                "Error": ""
            })

        except Exception as e:
            results.append({
                "Registro": file.name,
                "Formato detectado": "",
                "Puntos": None,
                "dt usado (s)": None,
                "Acel. mín": None,
                "Acel. máx": None,
                "PGA (g)": None,
                f"Sa(T={period:.3f}s) (g)": None,
                "DS leve": None,
                "DS moderado": None,
                "DS severo": None,
                "DS colapso": None,
                "Error": str(e)
            })

        progress.progress((i + 1) / len(uploaded_files), text=f"Procesando {i+1}/{len(uploaded_files)}")

    progress.empty()

    df = pd.DataFrame(results)

    st.subheader("Resultados por registro")
    st.dataframe(df, use_container_width=True)

    sa_col = f"Sa(T={period:.3f}s) (g)"
    valid_df = df.dropna(subset=[sa_col]).copy()

    if not valid_df.empty:
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar resultados CSV",
            data=csv_data,
            file_name="resultados_fragilidad.csv",
            mime="text/csv"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Sa(T) por registro")
            fig_sa = go.Figure()
            fig_sa.add_trace(go.Bar(
                x=valid_df["Registro"],
                y=valid_df[sa_col],
                name="Sa(T)"
            ))
            fig_sa.update_layout(
                xaxis_title="Registro",
                yaxis_title="Sa(T) [g]",
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_sa, use_container_width=True)

        with col2:
            st.subheader("PGA vs Sa(T)")
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=valid_df["PGA (g)"],
                y=valid_df[sa_col],
                mode="markers+text",
                text=valid_df["Registro"],
                textposition="top center"
            ))
            fig_scatter.update_layout(
                xaxis_title="PGA [g]",
                yaxis_title="Sa(T) [g]"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("Curvas de fragilidad")

        im_values = valid_df[sa_col].to_numpy(dtype=float)

        damage_states = {
            "Leve": valid_df["DS leve"].to_numpy(dtype=int),
            "Moderado": valid_df["DS moderado"].to_numpy(dtype=int),
            "Severo": valid_df["DS severo"].to_numpy(dtype=int),
            "Colapso": valid_df["DS colapso"].to_numpy(dtype=int),
        }

        im_min = max(0.001, np.min(im_values) * 0.50)
        im_max = max(np.max(im_values) * 1.80, ds_colapso * 1.50)
        im_grid = np.linspace(im_min, im_max, 300)

        fig_frag = go.Figure()
        fit_rows = []

        for ds_name, exceedance in damage_states.items():
            theta, beta, status = fit_lognormal_fragility(im_values, exceedance)

            if theta is not None:
                prob = fragility_probability(im_grid, theta, beta)
                fig_frag.add_trace(go.Scatter(
                    x=im_grid,
                    y=prob,
                    mode="lines",
                    name=f"DS {ds_name}"
                ))

            fit_rows.append({
                "Estado de daño": ds_name,
                "θ / mediana Sa (g)": theta,
                "β / dispersión lognormal": beta,
                "Estado": status
            })

        fig_frag.update_layout(
            xaxis_title="IM = Sa(T) [g]",
            yaxis_title="P(DS ≥ ds | Sa)",
            yaxis=dict(range=[0, 1]),
            legend_title="Curvas"
        )

        st.plotly_chart(fig_frag, use_container_width=True)

        st.subheader("Parámetros ajustados")
        st.dataframe(pd.DataFrame(fit_rows), use_container_width=True)

        st.info(
            """
Revisa las columnas **Acel. mín**, **Acel. máx**, **PGA** y **dt usado**.
Si tu registro está en g, los valores de PGA deberían estar en un rango razonable, por ejemplo 0.01 g a 2 g,
dependiendo del registro. Si sale 100 g, todavía se está leyendo mal el archivo.
"""
        )

else:
    st.warning("Sube uno o varios acelerogramas para iniciar.")


with st.expander("Formato recomendado"):
    st.markdown(
        """
Formato de dos columnas recomendado:

```txt
0.000000    2.177299e-04
0.005000    2.177415e-04
0.010000    2.177215e-04
```

En ese caso:

- columna 1 = tiempo
- columna 2 = aceleración

Formato de una columna aceptado:

```txt
2.177299e-04
2.177415e-04
2.177215e-04
```

En ese caso debes ingresar manualmente el `dt`.
"""
    )
