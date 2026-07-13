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


def stable_chart_domains(
    x: Iterable[float], y: Iterable[float]
) -> tuple[list[float], list[float]]:
    """Dominios basados solo en datos y MC; no cambian al mover la recta manual."""
    x_values, y_values = validate_xy(x, y)
    x_low = min(x_values)
    x_high = max(x_values)
    x_span = max(x_high - x_low, 1.0)

    reference_y = list(y_values)
    try:
        beta0_mc, beta1_mc = least_squares(x_values, y_values)
    except ValueError:
        pass
    else:
        reference_y.extend(beta0_mc + beta1_mc * xi for xi in x_values)

    y_low = min(reference_y)
    y_high = max(reference_y)
    y_span = max(y_high - y_low, 1.0)
    return (
        [x_low - max(0.35, 0.08 * x_span), x_high + max(0.35, 0.08 * x_span)],
        [y_low - max(0.55, 0.28 * y_span), y_high + max(0.55, 0.28 * y_span)],
    )


def initial_manual_parameters(
    x: Iterable[float], y: Iterable[float]
) -> tuple[float, float]:
    """Selecciona una recta deliberadamente peor que MC, pero visible.

    Se exploran varias pendientes y desplazamientos alrededor del centro de los
    datos. Entre las rectas cuyas predicciones permanecen dentro del dominio
    gráfico estable, se elige la que produce mayor SSE. Así el punto de partida
    es didácticamente imperfecto sin desaparecer fuera del gráfico.
    """
    x_values, y_values = validate_xy(x, y)
    beta0_mc, beta1_mc = least_squares(x_values, y_values)
    mc_fit = evaluate_fit(x_values, y_values, beta0_mc, beta1_mc)

    mean_x = sum(x_values) / len(x_values)
    mean_y = sum(y_values) / len(y_values)
    x_span = max(max(x_values) - min(x_values), 1.0)
    y_span = max(max(y_values) - min(y_values), 1.0)
    natural_slope = max(y_span / x_span, abs(beta1_mc), 0.25)
    _, y_domain = stable_chart_domains(x_values, y_values)

    slope_candidates = (
        -0.55 * natural_slope,
        -0.20 * natural_slope,
        0.05 * natural_slope,
        0.30 * natural_slope,
        1.70 * natural_slope,
        2.10 * natural_slope,
    )
    if beta1_mc < 0:
        slope_candidates = tuple(-value for value in slope_candidates)

    center_offsets = (-0.22 * y_span, 0.0, 0.22 * y_span)
    candidates: list[tuple[float, float, float]] = []

    for beta1 in slope_candidates:
        for offset in center_offsets:
            target_at_mean = mean_y + offset
            beta0 = target_at_mean - beta1 * mean_x
            predictions = tuple(beta0 + beta1 * xi for xi in x_values)
            if all(y_domain[0] <= value <= y_domain[1] for value in predictions):
                fit = evaluate_fit(x_values, y_values, beta0, beta1)
                candidates.append((fit.sse, float(beta0), float(beta1)))

    if not candidates:
        # Respaldo visible: recta horizontal desplazada respecto a la media.
        beta1 = 0.0
        beta0 = mean_y + 0.18 * y_span
        return float(beta0), float(beta1)

    candidates.sort(reverse=True, key=lambda item: item[0])
    best_sse, best_beta0, best_beta1 = candidates[0]

    # Garantía adicional de separación respecto a MC en muestras casi exactas.
    sst = sum((yi - mean_y) ** 2 for yi in y_values)
    target_sse = max(mc_fit.sse * 4.0, 0.35 * sst, 0.08 * len(y_values) * y_span**2)
    if best_sse < target_sse:
        beta1 = 0.0
        beta0 = mean_y + 0.20 * y_span
        fallback = evaluate_fit(x_values, y_values, beta0, beta1)
        if fallback.sse > best_sse and all(
            y_domain[0] <= value <= y_domain[1] for value in fallback.predictions
        ):
            return float(beta0), float(beta1)

    return best_beta0, best_beta1


def slider_parameter_ranges(
    x: Iterable[float], y: Iterable[float]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Rangos estables que incluyen MC y la recta inicial deliberadamente mala."""
    x_values, y_values = validate_xy(x, y)
    beta0_mc, beta1_mc = least_squares(x_values, y_values)
    beta0_start, beta1_start = initial_manual_parameters(x_values, y_values)

    mean_x = sum(x_values) / len(x_values)
    x_span = max(max(x_values) - min(x_values), 1.0)
    y_span = max(max(y_values) - min(y_values), 1.0)
    slope_scale = max(y_span / x_span, abs(beta1_mc), abs(beta1_start), 0.25)

    beta1_margin = 0.70 * slope_scale
    beta1_low = min(beta1_mc, beta1_start) - beta1_margin
    beta1_high = max(beta1_mc, beta1_start) + beta1_margin

    beta0_margin = max(
        0.75 * y_span,
        0.25 * abs(mean_x) * (beta1_high - beta1_low),
        0.25 * abs(beta0_mc),
        1.0,
    )
    beta0_low = min(beta0_mc, beta0_start) - beta0_margin
    beta0_high = max(beta0_mc, beta0_start) + beta0_margin

    return (
        (float(beta0_low), float(beta0_high)),
        (float(beta1_low), float(beta1_high)),
    )


def editor_rows_to_xy(
    rows: Sequence[Mapping[str, object]],
    max_rows: int = 500,
) -> tuple[tuple[float, ...], tuple[float, ...], int]:
    """Convierte filas de st.data_editor a X/Y al confirmar la planilla."""
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
    if len(set(x_values)) < 2:
        raise ValueError("Ingrese al menos dos valores distintos de X.")
    return x_values, y_values, incomplete
