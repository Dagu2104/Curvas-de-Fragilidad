
# Escalamiento de pares N/E por media SRSS - versión 2

Esta versión permite dos modos:

## 1. Registros sin escalar

- Subes registros N/E.
- Subes espectro objetivo T-Sa.
- Calcula SRSS.
- Calcula media SRSS.
- Calcula factor de escala.
- Genera acelerogramas escalados.

## 2. Registros ya escalados

Activa:

```text
Mis registros ya están escalados
```

En ese modo:

- No se solicita espectro objetivo.
- No se calcula factor de escala.
- Se usa factor = 1.0.
- Se calculan espectros, SRSS y media SRSS directamente.
- Se grafica la media SRSS de los registros ya escalados.

## Ejecutar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Nombres de archivos

Los pares se detectan con terminaciones:

```text
_N
_E
```

Ejemplo:

```text
RSN4031_SANSIMEO_36695090_N.txt
RSN4031_SANSIMEO_36695090_E.txt
```
