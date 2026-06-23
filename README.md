
# SRSS + Numba + Curvas de fragilidad

## Qué hace

1. Lee pares N/E.
2. Calcula espectros Sa.
3. Combina pares mediante SRSS.
4. Permite usar registros ya escalados o escalar contra espectro objetivo.
5. Acelera el cálculo de espectros con Numba.
6. Genera curvas de fragilidad usando derivas máximas.

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

## Intensidad para fragilidad

La app usa:

```text
IM = SRSS(T1) final [g]
```

y compara la deriva máxima con límites DS1, DS2, DS3 y DS4.
