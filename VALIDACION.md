# Validación técnica

## Pruebas automatizadas

- Compilación sintáctica de `app.py` y `least_squares.py`: correcta.
- Cinco pruebas unitarias: correctas.
- Arranque local de Streamlit y endpoint de salud: `200 OK`.
- Prueba con `streamlit.testing.v1.AppTest`: sin excepciones.

## Flujos comprobados

1. Ajuste manual en datos simulados con β₀ = 0.150 y β₁ = 1.200.
2. Mostrar mínimos cuadrados después del ajuste manual.
3. Alinear los parámetros manuales al ajuste por mínimos cuadrados.
4. Cambiar al modo de datos propios.
5. Cargar los datos de ejemplo.
6. Ajustar manualmente β₀ = 1.100 y β₁ = 0.350.
7. Mostrar el diagnóstico manual y el diagnóstico MC simultáneamente.

## Decisiones de robustez

- No se usan callbacks de sliders.
- No se modifica el estado de un widget después de instanciarlo.
- Los parámetros se envían en bloque mediante `st.form`.
- El estado de sesión contiene solamente tipos serializables.
- No se usan Matplotlib, NumPy, pandas, SciPy ni scikit-learn.
- El gráfico tiene una altura fija de 245 píxeles.
