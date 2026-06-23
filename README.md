
# SRSS + Numba + Curvas de fragilidad Sa/Sd

## Qué hace esta versión

1. Lee pares N/E.
2. Calcula espectros Sa y Sd.
3. Combina pares mediante SRSS:
   - Sa_SRSS(T1)
   - Sd_SRSS(T1)
4. Permite usar registros ya escalados o escalar contra espectro objetivo.
5. Acelera el cálculo de espectros con Numba.
6. Genera curvas de fragilidad usando derivas máximas.
7. Permite escoger la medida de intensidad para la curva:
   - Sa SRSS(T1) [g]
   - Sd SRSS(T1) [cm]

## Ejecutar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tabla de derivas

Formato recomendado:

```text
Par,Deriva_maxima_%
1_RSN730,0.45
1_RSN755,0.82
1_RSN767,1.35
```

La columna `Par` debe coincidir con los nombres detectados por la app.

## Intensidades disponibles

Para aceleración:

```text
IM = Sa_SRSS(T1) final [g]
```

Para desplazamiento:

```text
IM = Sd_SRSS(T1) final [cm]
```
