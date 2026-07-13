import math

import pytest

from least_squares import (
    editor_rows_to_xy,
    evaluate_fit,
    initial_manual_parameters,
    least_squares,
    slider_parameter_ranges,
    stable_chart_domains,
)


SAMPLE_X = [0, 1, 2, 3, 4, 5]
SAMPLE_Y = [1.405, 3.180, 2.578, 6.047, 6.479, 6.781]


def test_least_squares_exact_line():
    beta0, beta1 = least_squares([0, 1, 2], [1, 3, 5])
    assert math.isclose(beta0, 1.0)
    assert math.isclose(beta1, 2.0)


def test_initial_manual_line_is_clearly_worse_but_visible():
    beta0_mc, beta1_mc = least_squares(SAMPLE_X, SAMPLE_Y)
    mc = evaluate_fit(SAMPLE_X, SAMPLE_Y, beta0_mc, beta1_mc)
    beta0, beta1 = initial_manual_parameters(SAMPLE_X, SAMPLE_Y)
    manual = evaluate_fit(SAMPLE_X, SAMPLE_Y, beta0, beta1)
    _, y_domain = stable_chart_domains(SAMPLE_X, SAMPLE_Y)

    assert manual.sse >= 4.0 * mc.sse
    assert all(y_domain[0] <= value <= y_domain[1] for value in manual.predictions)


def test_initial_manual_exact_line_still_has_visible_error():
    x = [0, 1, 2, 3, 4, 5]
    y = [1, 3, 5, 7, 9, 11]
    beta0, beta1 = initial_manual_parameters(x, y)
    manual = evaluate_fit(x, y, beta0, beta1)
    _, y_domain = stable_chart_domains(x, y)

    assert manual.sse > 0
    assert all(y_domain[0] <= value <= y_domain[1] for value in manual.predictions)


def test_slider_ranges_include_mc_and_initial():
    b0_mc, b1_mc = least_squares(SAMPLE_X, SAMPLE_Y)
    b0_start, b1_start = initial_manual_parameters(SAMPLE_X, SAMPLE_Y)
    b0_range, b1_range = slider_parameter_ranges(SAMPLE_X, SAMPLE_Y)
    assert b0_range[0] <= b0_mc <= b0_range[1]
    assert b0_range[0] <= b0_start <= b0_range[1]
    assert b1_range[0] <= b1_mc <= b1_range[1]
    assert b1_range[0] <= b1_start <= b1_range[1]


def test_editor_rows_dynamic_and_incomplete():
    rows = [
        {"X": 0.0, "Y": 1.0},
        {"X": 1.0, "Y": 3.0},
        {"X": 2.0, "Y": None},
        {"X": None, "Y": None},
    ]
    x, y, incomplete = editor_rows_to_xy(rows)
    assert x == (0.0, 1.0)
    assert y == (1.0, 3.0)
    assert incomplete == 1


def test_empty_editor_requires_confirmation_with_data():
    rows = [{"X": None, "Y": None} for _ in range(6)]
    with pytest.raises(ValueError, match="al menos dos filas"):
        editor_rows_to_xy(rows)


def test_editor_requires_distinct_x_values():
    rows = [{"X": 1.0, "Y": 2.0}, {"X": 1.0, "Y": 3.0}]
    with pytest.raises(ValueError, match="dos valores distintos de X"):
        editor_rows_to_xy(rows)
