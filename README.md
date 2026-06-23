
# Escalamiento de pares N/E por media SRSS

Esta aplicación replica el procedimiento típico hecho en Excel:

1. Subir todos los acelerogramas N/E.
2. Detectar pares automáticamente por terminación `_N` y `_E`.
3. Calcular espectros de aceleración Sa para cada componente.
4. Combinar cada par con SRSS.
5. Calcular la media SRSS.
6. Comparar la media SRSS con un espectro objetivo dentro de 0.2T1–1.5T1.
7. Calcular factor de escala.
8. Graficar:
   - Espectro objetivo
   - Media SRSS original
   - Media SRSS escalada
   - Líneas T1, 0.2T1 y 1.5T1
9. Descargar acelerogramas escalados.

## Ejecutar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Nombre de archivos

Los archivos deben terminar en `_N` y `_E`, por ejemplo:

```text
RSN4031_SANSIMEO_36695090_N.txt
RSN4031_SANSIMEO_36695090_E.txt
```

## Espectro objetivo

Debe tener dos columnas:

```text
T Sa
0.01 0.80
0.10 1.75
0.50 1.75
1.00 1.30
```

Sin encabezado o con encabezado numérico no importa; la app usa las primeras dos columnas numéricas.
