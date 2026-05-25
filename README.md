# Herramienta de análisis de satisfacción - AVE UVG

Aplicación local en Streamlit para procesar reportes PDF individuales de Canvas/SpeedGrader y consolidar estadísticas de satisfacción de una sección.

## Funciones principales

- Carga múltiple de PDF, hasta 600 archivos.
- Lectura automática de 15 ítems tipo Likert.
- Extracción de 2 respuestas abiertas.
- Indicadores globales de satisfacción.
- Estadística por ítem y por categoría.
- Gráficas profesionales dentro de la app.
- Exportación a Excel y PDF.
- Encabezado institucional AVE-UVG, logos y marca de agua en reportes.
- Crédito visible: Ing. Christian Pocol, Ingeniero Electrónico.

## Instalación

1. Instalar Python 3.10 o superior.
2. Abrir una terminal dentro de esta carpeta.
3. Ejecutar:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Uso

1. Ingresar nombre del curso y sección.
2. Cargar los PDF exportados desde Canvas/SpeedGrader.
3. Revisar el resumen ejecutivo, estadística por ítem, distribución Likert y opiniones.
4. Descargar el Excel o PDF ejecutivo.

## Nota técnica

La aplicación detecta las respuestas seleccionadas en los PDF por análisis visual del radio button marcado, ya que Canvas/SpeedGrader no siempre almacena la opción seleccionada como texto extraíble.
