from __future__ import annotations

import math
from typing import Any, Iterable

import altair as alt
import streamlit as st

from least_squares import (
    FitResult,
    evaluate_fit,
    generate_simulated_data,
    least_squares,
    parse_xy_text,
)


APP_TITLE = "Visualizador de Mínimos Cuadrados"
CHART_HEIGHT = 245
MAX_SQUARES = 60
MAX_POINT_LABELS = 20
MAX_DETAIL_ROWS = 50

COLORS = {
    "points": "#D8756B",
    "manual": "#2F7FC1",
    "mc": "#2E9D62",
    "grid": "#D6DEE8",
    "text": "#273444",
}

SAMPLE_DATA = """X,Y
0,1.405
1,3.180
2,2.578
3,6.047
4,6.479
5,6.781"""


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def apply_readable_css() -> None:
    """Aumenta la legibilidad sin alterar la estructura interna de los widgets."""
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1450px;
            padding-top: 0.75rem;
            padding-bottom: 1rem;
        }
        html, body, .stApp {
            font-size: 18px;
        }
        h1 { font-size: 2.05rem !important; line-height: 1.15 !important; }
        h2 { font-size: 1.55rem !important; }
        h3 { font-size: 1.30rem !important; }
        div[data-testid="stWidgetLabel"] p {
            font-size: 1.05rem !important;
        }
        div[data-testid="stMetricLabel"] p {
            font-size: 1rem !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.42rem !important;
        }
        div[data-testid="stCaptionContainer"] p {
            font-size: 0.96rem !important;
        }
        button p {
            font-size: 1rem !important;
        }
        @media (max-width: 900px) {
            html, body, .stApp { font-size: 16px; }
            .block-container { padding-left: 0.65rem; padding-right: 0.65rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    if "sim_x" not in st.session_state or "sim_y" not in st.session_state:
        x, y = generate_simulated_data(2.0, 1.0, 0.75, 2026)
        st.session_state.sim_x = list(x)
        st.session_state.sim_y = list(y)

    defaults: dict[str, Any] = {
        "mode": "Datos simulados",
        "sim_target_beta0": 2.0,
        "sim_target_beta1": 1.0,
        "sim_sigma": 0.75,
        "sim_seed": 2026,
        "sim_manual_beta0": 0.0,
        "sim_manual_beta1": 0.0,
        "sim_show_manual": True,
        "sim_show_mc": False,
        "sim_manual_form_version": 0,
        "sim_show_e2_labels": False,
        "own_x": [],
        "own_y": [],
        "own_data_text": SAMPLE_DATA,
        "own_data_error": "",
        "own_manual_beta0": 0.0,
        "own_manual_beta1": 0.0,
        "own_show_manual": True,
        "own_show_mc": False,
        "own_manual_form_version": 0,
        "own_show_e2_labels": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def finite_range(values: Iterable[float], minimum_span: float = 1.0) -> tuple[float, float, float]:
    values = tuple(float(v) for v in values)
    low = min(values)
    high = max(values)
    span = max(high - low, minimum_span)
    return low, high, span


def chart_domains(
    x: tuple[float, ...],
    y: tuple[float, ...],
    manual: FitResult | None,
    mc: FitResult | None,
) -> tuple[list[float], list[float]]:
    x_low, x_high, x_span = finite_range(x)
    all_y = list(y)
    if manual is not None:
        all_y.extend(manual.predictions)
    if mc is not None:
        all_y.extend(mc.predictions)
    y_low, y_high, y_span = finite_range(all_y)
    return (
        [x_low - max(0.35, 0.08 * x_span), x_high + max(0.35, 0.08 * x_span)],
        [y_low - max(0.40, 0.10 * y_span), y_high + max(0.40, 0.10 * y_span)],
    )


def residual_records(
    x: tuple[float, ...],
    y: tuple[float, ...],
    fit: FitResult,
    x_domain: list[float],
    y_domain: list[float],
    side: str,
) -> list[dict[str, float | str]]:
    # Conversión aproximada para que el área se perciba cuadrada en pantalla.
    x_span = max(x_domain[1] - x_domain[0], 1e-12)
    y_span = max(y_domain[1] - y_domain[0], 1e-12)
    screen_factor = (x_span / y_span) * (CHART_HEIGHT / 900.0)
    records: list[dict[str, float | str]] = []

    for index, (xi, yi, yhat, residual) in enumerate(
        zip(x, y, fit.predictions, fit.residuals), start=1
    ):
        side_x = abs(residual) * screen_factor
        x0 = xi if side == "right" else xi - side_x
        records.append(
            {
                "point": f"P{index}",
                "x": xi,
                "y": yi,
                "yhat": yhat,
                "x0": x0,
                "x1": x0 + side_x,
                "y0": min(yi, yhat),
                "y1": max(yi, yhat),
                "e2": residual**2,
                "e2_label": f"e²={residual**2:.2f}",
            }
        )
    return records


def build_chart(
    x: tuple[float, ...],
    y: tuple[float, ...],
    manual: FitResult | None,
    mc: FitResult | None,
    show_e2_labels: bool,
) -> alt.LayerChart:
    x_domain, y_domain = chart_domains(x, y, manual, mc)
    x_scale = alt.Scale(domain=x_domain, nice=False)
    y_scale = alt.Scale(domain=y_domain, nice=False)

    points = [
        {"point": f"P{i}", "x": xi, "y": yi}
        for i, (xi, yi) in enumerate(zip(x, y), start=1)
    ]
    layers: list[alt.Chart] = []

    def add_fit_layers(fit: FitResult, color: str, square_side: str, dashed: bool) -> None:
        residuals = residual_records(x, y, fit, x_domain, y_domain, square_side)
        residual_data = alt.Data(values=residuals)

        if len(x) <= MAX_SQUARES:
            layers.append(
                alt.Chart(residual_data)
                .mark_rect(color=color, opacity=0.14, stroke=color, strokeWidth=1, clip=True)
                .encode(
                    x=alt.X("x0:Q", scale=x_scale, title="X"),
                    x2="x1:Q",
                    y=alt.Y("y0:Q", scale=y_scale, title="Y"),
                    y2="y1:Q",
                    tooltip=[
                        alt.Tooltip("point:N", title="Punto"),
                        alt.Tooltip("e2:Q", title="e²", format=".3f"),
                    ],
                )
            )

        rule = alt.Chart(residual_data).mark_rule(
            color=color,
            strokeWidth=1.3,
            strokeDash=[5, 3] if dashed else [3, 2],
            clip=True,
        ).encode(
            x=alt.X("x:Q", scale=x_scale),
            y=alt.Y("y:Q", scale=y_scale),
            y2="yhat:Q",
        )
        layers.append(rule)

        line_data = alt.Data(
            values=[
                {"x": x_domain[0], "y": fit.beta0 + fit.beta1 * x_domain[0]},
                {"x": x_domain[1], "y": fit.beta0 + fit.beta1 * x_domain[1]},
            ]
        )
        line_mark = {
            "color": color,
            "strokeWidth": 2.5 if dashed else 3,
            "clip": True,
        }
        if dashed:
            line_mark["strokeDash"] = [8, 5]
        layers.append(
            alt.Chart(line_data)
            .mark_line(**line_mark)
            .encode(
                x=alt.X("x:Q", scale=x_scale, title="X"),
                y=alt.Y("y:Q", scale=y_scale, title="Y"),
            )
        )

        if show_e2_labels and len(x) <= MAX_POINT_LABELS:
            layers.append(
                alt.Chart(residual_data)
                .mark_text(color=color, fontSize=13, align="left", dx=6, clip=True)
                .encode(
                    x=alt.X("x:Q", scale=x_scale),
                    y=alt.Y("y0:Q", scale=y_scale),
                    text="e2_label:N",
                )
            )

    if manual is not None:
        add_fit_layers(manual, COLORS["manual"], "right", dashed=False)
    if mc is not None:
        add_fit_layers(mc, COLORS["mc"], "left", dashed=True)

    point_data = alt.Data(values=points)
    layers.append(
        alt.Chart(point_data)
        .mark_circle(size=95, color="white", stroke=COLORS["points"], strokeWidth=2.2)
        .encode(
            x=alt.X("x:Q", scale=x_scale, title="X"),
            y=alt.Y("y:Q", scale=y_scale, title="Y"),
            tooltip=[
                alt.Tooltip("point:N", title="Punto"),
                alt.Tooltip("x:Q", title="X", format=".3f"),
                alt.Tooltip("y:Q", title="Y", format=".3f"),
            ],
        )
    )

    if len(x) <= MAX_POINT_LABELS:
        layers.append(
            alt.Chart(point_data)
            .mark_text(dy=-14, fontSize=13, color=COLORS["text"], fontWeight="bold")
            .encode(
                x=alt.X("x:Q", scale=x_scale),
                y=alt.Y("y:Q", scale=y_scale),
                text="point:N",
            )
        )

    return (
        alt.layer(*layers)
        .properties(height=CHART_HEIGHT)
        .configure_axis(
            labelFontSize=14,
            titleFontSize=16,
            titleFontWeight="normal",
            labelColor=COLORS["text"],
            titleColor=COLORS["text"],
            gridColor=COLORS["grid"],
            gridOpacity=0.7,
        )
        .configure_view(stroke="#AAB7C4", strokeWidth=0.8)
    )


def format_r2(value: float | None) -> str:
    return "—" if value is None or not math.isfinite(value) else f"{value:.4f}"


def show_fit_metrics(title: str, fit: FitResult, color_label: str) -> None:
    st.markdown(f"**{color_label} {title}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("β₀", f"{fit.beta0:.3f}")
    c2.metric("β₁", f"{fit.beta1:.3f}")
    c3.metric("SSE = Σeᵢ²", f"{fit.sse:.3f}")
    c4.metric("R²", format_r2(fit.r2))
    if fit.r2 is not None and fit.r2 < 0:
        st.caption("El R² es negativo porque esta recta ajusta peor que usar la media de Y.")


def markdown_detail_table(
    x: tuple[float, ...],
    y: tuple[float, ...],
    manual: FitResult | None,
    mc: FitResult | None,
) -> str:
    headers = ["Punto", "X", "Y"]
    if manual is not None:
        headers += ["Ŷ manual", "e manual", "e² manual"]
    if mc is not None:
        headers += ["Ŷ MC", "e MC", "e² MC"]

    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    shown = min(len(x), MAX_DETAIL_ROWS)
    for i in range(shown):
        row = [f"P{i + 1}", f"{x[i]:.3f}", f"{y[i]:.3f}"]
        if manual is not None:
            row += [
                f"{manual.predictions[i]:.3f}",
                f"{manual.residuals[i]:.3f}",
                f"{manual.residuals[i] ** 2:.3f}",
            ]
        if mc is not None:
            row += [
                f"{mc.predictions[i]:.3f}",
                f"{mc.residuals[i]:.3f}",
                f"{mc.residuals[i] ** 2:.3f}",
            ]
        lines.append("| " + " | ".join(row) + " |")
    if len(x) > shown:
        lines.append(f"\nSe muestran las primeras {shown} de {len(x)} observaciones.")
    return "\n".join(lines)


def render_visualization(prefix: str, x_values: list[float], y_values: list[float]) -> None:
    x = tuple(float(v) for v in x_values)
    y = tuple(float(v) for v in y_values)
    manual: FitResult | None = None
    mc: FitResult | None = None

    if st.session_state[f"{prefix}_show_manual"]:
        manual = evaluate_fit(
            x,
            y,
            st.session_state[f"{prefix}_manual_beta0"],
            st.session_state[f"{prefix}_manual_beta1"],
        )

    mc_available = len(x) >= 2 and max(x) - min(x) > 1e-15
    if st.session_state[f"{prefix}_show_mc"] and mc_available:
        beta0_mc, beta1_mc = least_squares(x, y)
        mc = evaluate_fit(x, y, beta0_mc, beta1_mc)

    legend = ["● Datos observados"]
    if manual is not None:
        legend.append("━ Ajuste manual")
    if mc is not None:
        legend.append("┄ Mínimos cuadrados")
    st.caption("   ·   ".join(legend))

    chart = build_chart(
        x,
        y,
        manual,
        mc,
        bool(st.session_state[f"{prefix}_show_e2_labels"]),
    )
    st.altair_chart(chart, width="stretch", theme=None, key=f"{prefix}_chart")

    with st.container(border=True):
        st.subheader("Diagnóstico del ajuste")
        if manual is not None:
            show_fit_metrics("Ajuste manual", manual, "🔵")
        if manual is not None and mc is not None:
            st.divider()
        if mc is not None:
            show_fit_metrics("Mínimos cuadrados", mc, "🟢")
        if manual is None and mc is None:
            st.info("Aplique un ajuste manual o muestre el ajuste por mínimos cuadrados.")

    if manual is not None or mc is not None:
        with st.expander("Detalle numérico por punto", expanded=False):
            st.markdown(markdown_detail_table(x, y, manual, mc))


def reset_manual_state(prefix: str) -> None:
    st.session_state[f"{prefix}_manual_beta0"] = 0.0
    st.session_state[f"{prefix}_manual_beta1"] = 0.0
    st.session_state[f"{prefix}_show_manual"] = True
    st.session_state[f"{prefix}_show_mc"] = False
    st.session_state[f"{prefix}_manual_form_version"] += 1


def render_manual_controls(prefix: str, x_values: list[float], y_values: list[float]) -> None:
    x = tuple(float(v) for v in x_values)
    y = tuple(float(v) for v in y_values)
    mc_available = len(x) >= 2 and max(x) - min(x) > 1e-15
    version = st.session_state[f"{prefix}_manual_form_version"]

    st.subheader("Ajuste manual")
    st.caption("Los parámetros se aplican juntos al pulsar el botón; no hay recálculos durante la edición.")

    with st.form(f"{prefix}_manual_form_{version}", enter_to_submit=False, border=True):
        beta0_input = st.number_input(
            "β₀ — intercepto",
            value=float(st.session_state[f"{prefix}_manual_beta0"]),
            step=0.05,
            format="%.3f",
            key=f"{prefix}_beta0_input_{version}",
        )
        beta1_input = st.number_input(
            "β₁ — pendiente",
            value=float(st.session_state[f"{prefix}_manual_beta1"]),
            step=0.05,
            format="%.3f",
            key=f"{prefix}_beta1_input_{version}",
        )
        apply_manual = st.form_submit_button("Aplicar ajuste manual", width="stretch", type="primary")

    if apply_manual:
        st.session_state[f"{prefix}_manual_beta0"] = float(beta0_input)
        st.session_state[f"{prefix}_manual_beta1"] = float(beta1_input)
        st.session_state[f"{prefix}_show_manual"] = True

    a1, a2 = st.columns(2)
    show_mc = a1.button(
        "Mostrar MC" if not st.session_state[f"{prefix}_show_mc"] else "Ocultar MC",
        key=f"{prefix}_toggle_mc",
        width="stretch",
        disabled=not mc_available,
    )
    align_mc = a2.button(
        "Alinear manual a MC",
        key=f"{prefix}_align_mc",
        width="stretch",
        disabled=not mc_available,
    )
    reset = st.button("Reiniciar parámetros", key=f"{prefix}_reset", width="stretch")

    if show_mc:
        st.session_state[f"{prefix}_show_mc"] = not st.session_state[f"{prefix}_show_mc"]
        st.rerun()
    if align_mc:
        beta0_mc, beta1_mc = least_squares(x, y)
        st.session_state[f"{prefix}_manual_beta0"] = beta0_mc
        st.session_state[f"{prefix}_manual_beta1"] = beta1_mc
        st.session_state[f"{prefix}_show_manual"] = True
        st.session_state[f"{prefix}_show_mc"] = True
        st.session_state[f"{prefix}_manual_form_version"] += 1
        st.rerun()
    if reset:
        reset_manual_state(prefix)
        st.rerun()

    st.checkbox(
        "Mostrar valores e² sobre el gráfico",
        key=f"{prefix}_show_e2_labels",
        disabled=len(x) > MAX_POINT_LABELS,
    )
    if len(x) > MAX_POINT_LABELS:
        st.caption(f"Las etiquetas e² se desactivan automáticamente con más de {MAX_POINT_LABELS} puntos.")


def render_simulation() -> None:
    controls, visual = st.columns([0.34, 0.66], gap="large")

    with controls:
        with st.container(border=True):
            render_manual_controls("sim", st.session_state.sim_x, st.session_state.sim_y)

        with st.expander("Generación de los datos simulados", expanded=False):
            with st.form("simulation_form", enter_to_submit=False):
                target_beta0 = st.number_input(
                    "β₀ verdadero",
                    value=float(st.session_state.sim_target_beta0),
                    step=0.10,
                    format="%.2f",
                )
                target_beta1 = st.number_input(
                    "β₁ verdadero",
                    value=float(st.session_state.sim_target_beta1),
                    step=0.10,
                    format="%.2f",
                )
                sigma = st.number_input(
                    "Desvío σ",
                    min_value=0.0,
                    value=float(st.session_state.sim_sigma),
                    step=0.05,
                    format="%.2f",
                )
                seed = st.number_input(
                    "Semilla",
                    min_value=0,
                    value=int(st.session_state.sim_seed),
                    step=1,
                )
                regenerate = st.form_submit_button("Regenerar datos", width="stretch")

            if regenerate:
                x, y = generate_simulated_data(target_beta0, target_beta1, sigma, int(seed))
                st.session_state.sim_target_beta0 = float(target_beta0)
                st.session_state.sim_target_beta1 = float(target_beta1)
                st.session_state.sim_sigma = float(sigma)
                st.session_state.sim_seed = int(seed)
                st.session_state.sim_x = list(x)
                st.session_state.sim_y = list(y)
                reset_manual_state("sim")
                st.rerun()

    with visual:
        render_visualization("sim", st.session_state.sim_x, st.session_state.sim_y)


def render_own_data() -> None:
    controls, visual = st.columns([0.36, 0.64], gap="large")

    with controls:
        with st.expander("Ingresar o reemplazar los datos", expanded=not bool(st.session_state.own_x)):
            st.caption("Use una fila por observación. Se aceptan coma, punto y coma, tabulación o espacio como separadores.")
            with st.form("own_data_form", enter_to_submit=False):
                text = st.text_area(
                    "Pares X-Y",
                    key="own_data_text",
                    height=185,
                    label_visibility="collapsed",
                )
                load_data = st.form_submit_button("Cargar datos", width="stretch", type="primary")

            if load_data:
                try:
                    x, y = parse_xy_text(text)
                except ValueError as exc:
                    st.session_state.own_data_error = str(exc)
                else:
                    st.session_state.own_x = list(x)
                    st.session_state.own_y = list(y)
                    st.session_state.own_data_error = ""
                    reset_manual_state("own")
                    st.rerun()

            if st.session_state.own_data_error:
                st.error(st.session_state.own_data_error)

            if st.button("Quitar datos cargados", width="stretch", disabled=not bool(st.session_state.own_x)):
                st.session_state.own_x = []
                st.session_state.own_y = []
                st.session_state.own_data_error = ""
                reset_manual_state("own")
                st.rerun()

        if st.session_state.own_x:
            with st.container(border=True):
                render_manual_controls("own", st.session_state.own_x, st.session_state.own_y)

    with visual:
        if not st.session_state.own_x:
            st.info("Ingrese los pares X-Y y pulse **Cargar datos**.", icon="ℹ️")
        else:
            render_visualization("own", st.session_state.own_x, st.session_state.own_y)


def main() -> None:
    apply_readable_css()
    initialize_state()

    st.title("📐 Visualizador de Mínimos Cuadrados")
    st.caption("Ajuste lineal interactivo, residuos y suma de cuadrados del error")

    st.radio(
        "Origen de los datos",
        options=("Datos simulados", "Ingresar datos"),
        horizontal=True,
        key="mode",
    )

    try:
        if st.session_state.mode == "Ingresar datos":
            render_own_data()
        else:
            render_simulation()
    except ValueError as exc:
        st.error(str(exc), icon="⚠️")
    except Exception:
        st.error(
            "Ocurrió un error inesperado al construir la vista. Reinicie la aplicación y revise el registro técnico.",
            icon="⚠️",
        )
        raise


if __name__ == "__main__":
    main()
