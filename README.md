# Streamlit - Curvas de fragilidad sísmica

Versión corregida.

## Corrección principal

La versión anterior tomaba la primera columna del archivo como aceleración.  
Pero muchos acelerogramas vienen con dos columnas:

- columna 1: tiempo
- columna 2: aceleración

Esta versión detecta eso y usa correctamente la segunda columna como aceleración.

## Ejecutar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Archivos

- app.py
- requirements.txt
- README.md
