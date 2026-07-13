# Validación técnica v10

- Compilación sintáctica de `app.py`: correcta.
- Pruebas unitarias del módulo de mínimos cuadrados: 7/7 correctas.
- Ejecución mediante `streamlit.testing.v1.AppTest`: sin excepciones.
- Inicio del servidor Streamlit 1.59.1 y respuesta de salud: correcta.
- Tema claro y oscuro configurados en `.streamlit/config.toml`.
- Colores del gráfico adaptados mediante `st.context.theme.type`.
- Tarjetas métricas, textos secundarios y bordes usan variables CSS del tema activo.
- Ejes, cuadrícula, etiquetas, puntos y rectas conservan contraste en ambos temas.
- No se modificó la lógica estadística ni la arquitectura estable de ingreso de datos.

La captura automatizada con navegador no pudo ejecutarse en el entorno de validación porque el acceso del navegador local estaba restringido, pero la aplicación fue comprobada por AppTest y por inicio real del servidor.
