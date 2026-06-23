
# App Streamlit - Curvas de fragilidad sísmica

## ¿Qué hace?

Esta app permite:

1. Subir registros sísmicos en una sola dirección.
2. Ingresar el período estructural T.
3. Calcular la aceleración espectral Sa(T).
4. Definir límites de daño.
5. Generar curvas de fragilidad lognormales.

## Archivos incluidos

- `app.py`: aplicación principal de Streamlit.
- `requirements.txt`: librerías necesarias.

## Cómo ejecutar localmente

Instala las dependencias:

```bash
pip install -r requirements.txt
```

Ejecuta la app:

```bash
streamlit run app.py
```

## Formato de los registros

Cada archivo debe tener aceleraciones en una columna, por ejemplo:

```txt
0.001
0.003
-0.002
-0.004
```

También acepta varias columnas, pero toma la primera columna numérica válida.

## Importante

Todos los registros deben tener el mismo:

- `dt`
- unidad de aceleración
- dirección de análisis

## Nota técnica

Con solo acelerogramas y período se calcula Sa(T).  
Para curvas de fragilidad más rigurosas, se recomienda usar derivas máximas o demandas estructurales obtenidas por análisis dinámico.
