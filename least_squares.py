from __future__ import annotations

from dataclasses import dataclass
import math
import random
import re
from typing import Iterable, Sequence


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


def validate_xy(x: Iterable[float], y: Iterable[float]) -> tuple[tuple[float, ...], tuple[float, ...]]:
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


def parse_xy_text(text: str, max_rows: int = 500) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Parsea pares X-Y separados por coma, punto y coma, tabulación o espacios."""
    if not text or not text.strip():
        raise ValueError("Ingrese al menos dos filas con pares X e Y.")

    pairs: list[tuple[float, float]] = []
    errors: list[str] = []

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        pieces = [piece for piece in re.split(r"[;,\t ]+", line) if piece]
        if len(pieces) != 2:
            # Permite una cabecera simple, por ejemplo: X,Y
            lowered = "".join(pieces).lower()
            if line_number == 1 and "x" in lowered and "y" in lowered:
                continue
            errors.append(f"línea {line_number}")
            continue

        try:
            xi = float(pieces[0].replace(",", "."))
            yi = float(pieces[1].replace(",", "."))
        except ValueError:
            if line_number == 1 and pieces[0].lower().startswith("x") and pieces[1].lower().startswith("y"):
                continue
            errors.append(f"línea {line_number}")
            continue

        if not math.isfinite(xi) or not math.isfinite(yi):
            errors.append(f"línea {line_number}")
            continue

        pairs.append((xi, yi))
        if len(pairs) > max_rows:
            raise ValueError(f"La aplicación admite como máximo {max_rows} observaciones.")

    if errors:
        shown = ", ".join(errors[:5])
        suffix = "…" if len(errors) > 5 else ""
        raise ValueError(f"No se pudieron interpretar: {shown}{suffix}. Use dos números por fila.")
    if len(pairs) < 2:
        raise ValueError("Se necesitan al menos dos pares X-Y completos.")

    x, y = zip(*pairs)
    return validate_xy(x, y)
