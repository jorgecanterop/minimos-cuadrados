from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class FitResult:
    beta0: float
    beta1: float
    predictions: tuple[float, ...]
    residuals: tuple[float, ...]
    sse: float
    r2: float | None


def _as_finite_tuple(values: Iterable[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} no puede estar vacío.")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} contiene valores no finitos.")
    return result


def validate_xy(
    x: Iterable[float], y: Iterable[float]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    x_values = _as_finite_tuple(x, "X")
    y_values = _as_finite_tuple(y, "Y")
    if len(x_values) != len(y_values):
        raise ValueError("X e Y deben tener la misma cantidad de observaciones.")
    return x_values, y_values


def generate_simulated_data(
    beta0: float,
    beta1: float,
    sigma: float,
    seed: int,
    x_values: Sequence[float] = (0, 1, 2, 3, 4, 5),
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not all(math.isfinite(float(v)) for v in (beta0, beta1, sigma)):
        raise ValueError("Los parámetros de simulación deben ser finitos.")
    if sigma < 0:
        raise ValueError("σ no puede ser negativo.")
    x = tuple(float(value) for value in x_values)
    if not x:
        raise ValueError("Se necesita al menos un valor de X.")
    rng = random.Random(int(seed))
    y = tuple(float(beta0 + beta1 * xi + rng.gauss(0.0, sigma)) for xi in x)
    return x, y


def least_squares(x: Iterable[float], y: Iterable[float]) -> tuple[float, float]:
    x_values, y_values = validate_xy(x, y)
    if len(x_values) < 2:
        raise ValueError("Se necesitan al menos dos observaciones.")

    mean_x = sum(x_values) / len(x_values)
    mean_y = sum(y_values) / len(y_values)
    sxx = sum((xi - mean_x) ** 2 for xi in x_values)
    if sxx <= 1e-15:
        raise ValueError("Se necesitan al menos dos valores distintos de X.")

    sxy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x_values, y_values))
    beta1 = sxy / sxx
    beta0 = mean_y - beta1 * mean_x
    return float(beta0), float(beta1)


def evaluate_fit(
    x: Iterable[float],
    y: Iterable[float],
    beta0: float,
    beta1: float,
) -> FitResult:
    x_values, y_values = validate_xy(x, y)
    beta0 = float(beta0)
    beta1 = float(beta1)
    if not math.isfinite(beta0) or not math.isfinite(beta1):
        raise ValueError("β₀ y β₁ deben ser finitos.")

    predictions = tuple(beta0 + beta1 * xi for xi in x_values)
    residuals = tuple(yi - yhat for yi, yhat in zip(y_values, predictions))
    sse = float(sum(residual**2 for residual in residuals))
    mean_y = sum(y_values) / len(y_values)
    sst = float(sum((yi - mean_y) ** 2 for yi in y_values))
    r2 = None if sst <= 1e-15 else float(1.0 - sse / sst)
    return FitResult(beta0, beta1, predictions, residuals, sse, r2)


def initial_manual_parameters(
    x: Iterable[float], y: Iterable[float]
) -> tuple[float, float]:
    """Devuelve una recta inicial deliberadamente imperfecta, pero visible.

    La recta se coloca cerca de MC y se comprueba contra un dominio vertical
    basado en los datos. Si se sale, se interpola progresivamente hacia MC.
    """
    x_values, y_values = validate_xy(x, y)
    beta0_mc, beta1_mc = least_squares(x_values, y_values)

    mean_x = sum(x_values) / len(x_values)
    mean_y = sum(y_values) / len(y_values)
    x_span = max(max(x_values) - min(x_values), 1.0)
    y_span = max(max(y_values) - min(y_values), 1.0)
    slope_scale = max(y_span / x_span, abs(beta1_mc) * 0.35, 0.10)

    direction = 1.0 if beta1_mc >= 0 else -1.0
    candidate_beta1 = beta1_mc - direction * 0.24 * slope_scale
    candidate_beta0 = (mean_y + 0.08 * y_span) - candidate_beta1 * mean_x

    lower = min(y_values) - 0.30 * y_span
    upper = max(y_values) + 0.30 * y_span
    for fraction in (1.0, 0.8, 0.6, 0.4, 0.2, 0.0):
        beta0 = beta0_mc + fraction * (candidate_beta0 - beta0_mc)
        beta1 = beta1_mc + fraction * (candidate_beta1 - beta1_mc)
        predictions = (beta0 + beta1 * xi for xi in x_values)
        if all(lower <= value <= upper for value in predictions):
            return float(beta0), float(beta1)

    return float(beta0_mc), float(beta1_mc)


def slider_parameter_ranges(
    x: Iterable[float], y: Iterable[float]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Rangos didácticos centrados en MC, suficientemente amplios y estables."""
    x_values, y_values = validate_xy(x, y)
    beta0_mc, beta1_mc = least_squares(x_values, y_values)
    beta0_start, beta1_start = initial_manual_parameters(x_values, y_values)

    mean_x = sum(x_values) / len(x_values)
    x_span = max(max(x_values) - min(x_values), 1.0)
    y_span = max(max(y_values) - min(y_values), 1.0)
    slope_scale = max(y_span / x_span, abs(beta1_mc) * 0.50, 0.25)

    beta1_margin = 1.35 * slope_scale
    beta1_low = min(beta1_mc - beta1_margin, beta1_start)
    beta1_high = max(beta1_mc + beta1_margin, beta1_start)

    beta0_margin = max(
        1.10 * y_span,
        0.30 * abs(mean_x) * beta1_margin,
        0.35 * abs(beta0_mc),
        1.0,
    )
    beta0_low = min(beta0_mc - beta0_margin, beta0_start)
    beta0_high = max(beta0_mc + beta0_margin, beta0_start)

    return (
        (float(beta0_low), float(beta0_high)),
        (float(beta1_low), float(beta1_high)),
    )


def editor_rows_to_xy(
    rows: Sequence[Mapping[str, object]],
    max_rows: int = 500,
) -> tuple[tuple[float, ...], tuple[float, ...], int]:
    """Convierte filas de st.data_editor a X/Y.

    Las filas vacías o parcialmente completadas se ignoran para permitir que el
    usuario agregue observaciones de forma gradual. Devuelve cuántas filas
    parciales fueron omitidas.
    """
    pairs: list[tuple[float, float]] = []
    incomplete = 0

    for row in rows:
        raw_x = row.get("X")
        raw_y = row.get("Y")
        empty_x = raw_x is None or raw_x == ""
        empty_y = raw_y is None or raw_y == ""

        if empty_x and empty_y:
            continue
        if empty_x or empty_y:
            incomplete += 1
            continue

        try:
            x_value = float(raw_x)
            y_value = float(raw_y)
        except (TypeError, ValueError):
            incomplete += 1
            continue
        if not math.isfinite(x_value) or not math.isfinite(y_value):
            incomplete += 1
            continue

        pairs.append((x_value, y_value))
        if len(pairs) > max_rows:
            raise ValueError(f"La aplicación admite como máximo {max_rows} observaciones.")

    if len(pairs) < 2:
        raise ValueError("Ingrese al menos dos filas completas con valores X e Y.")

    x, y = zip(*pairs)
    x_values, y_values = validate_xy(x, y)
    return x_values, y_values, incomplete
