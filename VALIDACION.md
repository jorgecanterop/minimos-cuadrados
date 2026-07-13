# Validación técnica v7

## Cálculo

- 7 pruebas unitarias aprobadas.
- La recta inicial manual tiene una SSE al menos cuatro veces mayor que MC en el conjunto de ejemplo.
- La recta inicial permanece dentro del dominio vertical del gráfico.
- Los rangos de ambos deslizadores incluyen la recta inicial y MC.
- Se validan planillas vacías y valores X sin variación.

## Streamlit 1.59.1

Pruebas realizadas con `streamlit.testing.v1.AppTest`:

- Arranque inicial sin excepciones.
- Movimiento de β₀ y β₁ hasta los extremos sin excepciones.
- Cambio al modo de ingreso manual sin excepciones.
- `Vaciar planilla` no produce excepciones.
- `Generar gráfico` sobre una planilla vacía muestra un mensaje de validación y no falla.
- `Cargar ejemplo` seguido de `Generar gráfico` crea los deslizadores y el gráfico sin excepciones.
- La recta inicial del ejemplo usa β₀ = 4.810 y β₁ = −0.632, claramente separada de MC.

## Validación sintáctica

- `app.py`: compilación correcta.
- `least_squares.py`: compilación correcta.
