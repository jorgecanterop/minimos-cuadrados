# Visualizador de Mínimos Cuadrados — versión nativa y robusta

Esta versión fue reescrita para Streamlit y no depende de la notebook de Google Colab.

## Cambios de arquitectura

- Cálculos implementados con la biblioteca estándar de Python.
- Sin Matplotlib, NumPy, pandas, SciPy ni scikit-learn.
- Gráfico Altair de altura fija.
- Parámetros manuales dentro de un `st.form`: los valores se envían juntos al pulsar **Aplicar ajuste manual**.
- Sin callbacks en sliders y sin modificación de un widget después de crearlo.
- Estado de sesión formado únicamente por números, textos, listas y booleanos serializables.
- Datos propios ingresados como texto, evitando la complejidad de `st.data_editor`.
- Pruebas unitarias del cálculo, simulación y lectura de datos.

## Estructura

```text
app.py
least_squares.py
requirements.txt
.streamlit/config.toml
tests/test_least_squares.py
```

## Probar localmente

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
python -m unittest discover -s tests -v
streamlit run app.py
```

## Despliegue limpio en Streamlit Community Cloud

El registro de la versión anterior muestra Python 3.14.6, Matplotlib 3.11.0 y un `Segmentation fault`. Para evitar que Streamlit reutilice ese entorno:

1. Elimine del repositorio los archivos antiguos de dependencias, especialmente `environment.yml`, `Pipfile`, `pyproject.toml`, `setup.py` o un `requirements.txt` anterior.
2. Deje en el repositorio solamente los archivos de este paquete.
3. Elimine el despliegue anterior en Streamlit Community Cloud.
4. Cree nuevamente la aplicación desde el repositorio.
5. En **Advanced settings**, seleccione **Python 3.12**.
6. Use `app.py` como archivo principal.

No basta con renombrar `requirements_v4_altair.txt`: el archivo debe llamarse exactamente `requirements.txt` y estar en la raíz del repositorio.

## Integración en Google Sites

Después del despliegue use:

```text
https://NOMBRE-DE-LA-APP.streamlit.app/?embed=true
```
