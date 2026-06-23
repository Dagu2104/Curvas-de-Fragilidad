# Streamlit - Curvas de fragilidad con Sa o Sd

## Funciones principales

Esta versión permite:

1. Subir acelerogramas.
2. Calcular Sa(T) y Sd(T).
3. Elegir la medida de intensidad:
   - Sa(T) en g
   - Sd(T) en cm
4. Escalar registros a varios niveles objetivo de Sa o Sd.
5. Graficar espectros de respuesta:
   - Espectro de aceleración Sa
   - Espectro de desplazamiento Sd
6. Escoger qué registros ver en las gráficas.
7. Generar curvas de fragilidad lognormales.

## Ejecutar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Nota técnica

Esta versión define estados de daño con límites de Sa o Sd. Para curvas estructurales más rigurosas, se recomienda usar derivas máximas de entrepiso.
