# Visualizador interactivo de mínimos cuadrados

Esta aplicación está pensada para comprender, de forma visual y dinámica, cómo se construye una recta de regresión y cómo el método de mínimos cuadrados encuentra el mejor ajuste posible.

## ¿Qué permite hacer esta aplicación?

La aplicación muestra un conjunto de puntos y una recta de ajuste que puede modificarse manualmente mediante los parámetros:

- **β₀**: intercepto de la recta.
- **β₁**: pendiente de la recta.

Al mover los controles, la recta cambia de forma inmediata y la aplicación recalcula los residuos, la suma de cuadrados del error y el coeficiente de determinación.

Esto permite observar directamente cómo cada cambio en la pendiente o en el intercepto afecta la calidad del ajuste.

## ¿Qué se puede explorar?

La aplicación permite:

- modificar manualmente la recta de ajuste;
- observar los residuos de cada punto;
- visualizar la contribución individual de cada observación al error total;
- comparar el ajuste manual con la recta obtenida por mínimos cuadrados;
- analizar cómo cambia la **Suma Cuadrada del Error (SCE)**;
- interpretar el valor de **R²**;
- trabajar con datos simulados;
- ingresar datos propios y generar un gráfico nuevo.

## Modos de uso

### Datos simulados

La aplicación genera un conjunto de observaciones a partir de una relación lineal con variabilidad aleatoria.

Este modo es útil para experimentar con diferentes situaciones y observar cómo se comporta el ajuste cuando cambian los datos.

### Ingreso de datos

También es posible ingresar valores propios de **X** e **Y** en una planilla.

Una vez completados los datos, se debe pulsar **Generar gráfico**. A partir de ese momento se habilitan los mismos controles interactivos disponibles para los datos simulados.

## Ajuste manual

Los controles de **β₀** y **β₁** permiten desplazar y rotar la recta.

El objetivo es intentar reducir el error del ajuste observando simultáneamente:

- la posición de la recta;
- los residuos;
- la Suma Cuadrada del Error;
- el valor de R².

La recta inicial se presenta deliberadamente alejada del ajuste óptimo para que el proceso de mejora pueda observarse con claridad.

## Ajuste por mínimos cuadrados

La aplicación permite mostrar la recta calculada mediante el método de mínimos cuadrados.

Esta recta es la que minimiza:

$$
\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

Es decir, minimiza la suma de los cuadrados de las diferencias entre los valores observados y los valores predichos por la recta.

## Diagnóstico del ajuste

Para cada recta se muestran los siguientes resultados:

- **β₀**: intercepto estimado;
- **β₁**: pendiente estimada;
- **SCE**: suma de cuadrados del error;
- **R²**: proporción de la variabilidad de Y explicada por la recta.

Cuando el valor crudo de R² es negativo, la aplicación muestra **0** en el diagnóstico e indica en una nota que esa recta ajusta peor que utilizar la media de Y como predicción.

## Contribuciones individuales al error

En la sección **Contribuciones individuales a la Suma Cuadrada del Error** se puede revisar, para cada observación:

- el valor de X;
- el valor observado de Y;
- el valor predicho;
- el residuo;
- el residuo al cuadrado.

Esto permite comprender cómo se forma la SCE total a partir de los errores individuales.


## Acceso a la aplicación

La aplicación puede utilizarse desde:

**https://minimos-cuadrados.streamlit.app/**

---
