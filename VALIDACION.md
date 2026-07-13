# Validación técnica v6

## Pruebas de cálculo

- Cálculo de mínimos cuadrados sobre una recta exacta.
- Recta manual inicial cercana a MC y dentro del dominio vertical didáctico.
- Rangos de los deslizadores contienen tanto MC como el inicio manual.
- Filas incompletas de la planilla se ignoran temporalmente sin interrumpir la app.

## Pruebas con Streamlit 1.59.1

- Arranque inicial sin excepciones.
- Detección de dos deslizadores nativos.
- Cambio de β₀ y β₁ hasta los extremos permitidos sin excepciones.
- Callback «Alinear con MC» sin errores de session_state.
- Cambio al modo «Ingresar datos» sin excepciones.
- Planilla dinámica creada mediante st.data_editor.
- «Vaciar tabla» elimina el ajuste y muestra el mensaje correspondiente.
- «Cargar ejemplo» reconstruye la planilla y los dos deslizadores.
- Ejes gráficos independientes de los parámetros manuales.
- Dependencias directas limitadas a Streamlit y Altair.
