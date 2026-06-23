# Streamlit - Curvas de fragilidad con escalamiento a Sa objetivo

## Qué hace esta versión

Permite:

1. Subir acelerogramas.
2. Calcular Sa(T) original.
3. Ingresar varios niveles de Sa objetivo.
4. Escalar cada registro a cada Sa objetivo.
5. Generar varios puntos para curvas de fragilidad.
6. Descargar tablas CSV.
7. Opcionalmente descargar los acelerogramas escalados en ZIP.

## Ejecución

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Factor de escala

```text
FE = Sa_objetivo / Sa_original(T)
```

El acelerograma escalado es:

```text
a_escalado(t) = FE * a_original(t)
```

## Nota técnica

La versión actual define estados de daño con límites de Sa. Para curvas estructurales más rigurosas se debe usar respuesta estructural como deriva máxima.
