from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

import altair as alt
import streamlit as st

from least_squares import (
    FitResult,
    editor_rows_to_xy,
    evaluate_fit,
    generate_simulated_data,
    initial_manual_parameters,
    least_squares,
    slider_parameter_ranges,
    stable_chart_domains,
)


APP_TITLE = "Visualizador de Mínimos Cuadrados"
CHART_HEIGHT = 360
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

SAMPLE_ROWS = [
    {"X": 0.0, "Y": 1.405},
    {"X": 1.0, "Y": 3.180},
    {"X": 2.0, "Y": 2.578},
    {"X": 3.0, "Y": 6.047},
    {"X": 4.0, "Y": 6.479},
    {"X": 5.0, "Y": 6.781},
]

EMPTY_ROWS = [{"X": None, "Y": None} for _ in range(6)]

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def apply_balanced_css() -> None:
    """Jerarquía tipográfica equilibrada y botones sin saltos de línea."""
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1480px;
            padding-top: 0.65rem;
            padding-bottom: 1rem;
        }
        html, body, .stApp {
            font-size: 17px;
        }
        h1 {
            font-size: 1.78rem !important;
            line-height: 1.15 !important;
            margin-bottom: 0.20rem !important;
        }
        h2 {
            font-size: 1.36rem !important;
            line-height: 1.20 !important;
        }
        h3 {
            font-size: 1.15rem !important;
            line-height: 1.20 !important;
        }
        div[data-testid="stWidgetLabel"] p,
        div[data-testid="stMarkdownContainer"] p {
            font-size: 0.98rem;
        }
        div[data-testid="stMetricLabel"] p {
            font-size: 0.90rem !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.28rem !important;
        }
        div[data-testid="stCaptionContainer"] p {
            font-size: 0.88rem !important;
            line-height: 1.30 !important;
        }
        div[data-testid="stButton"] button,
        div[data-testid="stFormSubmitButton"] button {
            min-height: 2.55rem;
            padding: 0.42rem 0.72rem;
            white-space: nowrap !important;
        }
        div[data-testid="stButton"] button p,
        div[data-testid="stFormSubmitButton"] button p {
            font-size: 0.92rem !important;
            white-space: nowrap !important;
            line-height: 1.05 !important;
        }
        div[data-testid="stDataEditor"] {
            font-size: 0.94rem;
        }
        div[data-testid="stExpander"] summary p {
            font-size: 0.95rem !important;
        }
        @media (max-width: 980px) {
            html, body, .stApp { font-size: 16px; }
            .block-container {
                padding-left: 0.65rem;
                padding-right: 0.65rem;
            }
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
        "sim_show_mc": False,
        "sim_show_e2_labels": False,
        "sim_slider_version": 0,
        "sim_data_signature": "",
        "own_show_mc": False,
        "own_show_e2_labels": False,
        "own_slider_version": 0,
        "own_data_signature": "",
        "own_editor_version": 0,
        "own_editor_seed": [dict(row) for row in SAMPLE_ROWS],
        "own_committed_x": None,
        "own_committed_y": None,
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


def data_signature(x: Sequence[float], y: Sequence[float]) -> str:
    payload = "|".join(f"{a:.12g},{b:.12g}" for a, b in zip(x, y))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def nice_step(span: float, target_steps: int = 90) -> float:
    raw = max(float(span) / target_steps, 1e-9)
    magnitude = 10 ** math.floor(math.log10(raw))
    normalized = raw / magnitude
    if normalized <= 1:
        factor = 1
    elif normalized <= 2:
        factor = 2
    elif normalized <= 5:
        factor = 5
    else:
        factor = 10
    return float(factor * magnitude)


def clamp(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def chart_domains(x: tuple[float, ...], y: tuple[float, ...]) -> tuple[list[float], list[float]]:
    """Dominios estables compartidos con la lógica de inicialización manual."""
    return stable_chart_domains(x, y)


def residual_records(
    x: tuple[float, ...],
    y: tuple[float, ...],
    fit: FitResult,
    x_domain: list[float],
    y_domain: list[float],
    side: str,
) -> list[dict[str, float | str]]:
    x_span = max(x_domain[1] - x_domain[0], 1e-12)
    y_span = max(y_domain[1] - y_domain[0], 1e-12)
    screen_factor = (x_span / y_span) * (CHART_HEIGHT / 930.0)
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
    manual: FitResult,
    mc: FitResult | None,
    show_e2_labels: bool,
) -> alt.LayerChart:
    x_domain, y_domain = chart_domains(x, y)
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

        layers.append(
            alt.Chart(residual_data)
            .mark_rule(
                color=color,
                strokeWidth=1.3,
                strokeDash=[5, 3] if dashed else [3, 2],
                clip=True,
            )
            .encode(
                x=alt.X("x:Q", scale=x_scale),
                y=alt.Y("y:Q", scale=y_scale),
                y2="yhat:Q",
            )
        )

        line_data = alt.Data(
            values=[
                {"x": x_domain[0], "y": fit.beta0 + fit.beta1 * x_domain[0]},
                {"x": x_domain[1], "y": fit.beta0 + fit.beta1 * x_domain[1]},
            ]
        )
        line_mark: dict[str, Any] = {
            "color": color,
            "strokeWidth": 2.7 if dashed else 3.2,
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
                .mark_text(color=color, fontSize=12, align="left", dx=6, clip=True)
                .encode(
                    x=alt.X("x:Q", scale=x_scale),
                    y=alt.Y("y0:Q", scale=y_scale),
                    text="e2_label:N",
                )
            )

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
            .mark_text(dy=-14, fontSize=12, color=COLORS["text"], fontWeight="bold")
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
            titleFontSize=15,
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
        st.caption("R² negativo: esta recta ajusta peor que utilizar la media de Y.")


def markdown_detail_table(
    x: tuple[float, ...],
    y: tuple[float, ...],
    manual: FitResult,
    mc: FitResult | None,
) -> str:
    headers = ["Punto", "X", "Y", "Ŷ manual", "e manual", "e² manual"]
    if mc is not None:
        headers += ["Ŷ MC", "e MC", "e² MC"]

    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    shown = min(len(x), MAX_DETAIL_ROWS)
    for i in range(shown):
        row = [
            f"P{i + 1}",
            f"{x[i]:.3f}",
            f"{y[i]:.3f}",
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


def ensure_slider_state(
    prefix: str,
    x: tuple[float, ...],
    y: tuple[float, ...],
) -> tuple[str, str, float, float, tuple[float, float], tuple[float, float]]:
    signature = data_signature(x, y)
    signature_key = f"{prefix}_data_signature"
    version_key = f"{prefix}_slider_version"

    if st.session_state[signature_key] != signature:
        st.session_state[signature_key] = signature
        st.session_state[version_key] += 1
        st.session_state[f"{prefix}_show_mc"] = False

    version = int(st.session_state[version_key])
    beta0_key = f"{prefix}_beta0_slider_{version}"
    beta1_key = f"{prefix}_beta1_slider_{version}"
    beta0_start, beta1_start = initial_manual_parameters(x, y)
    beta0_range, beta1_range = slider_parameter_ranges(x, y)

    if beta0_key not in st.session_state:
        st.session_state[beta0_key] = clamp(beta0_start, *beta0_range)
    if beta1_key not in st.session_state:
        st.session_state[beta1_key] = clamp(beta1_start, *beta1_range)

    return beta0_key, beta1_key, beta0_start, beta1_start, beta0_range, beta1_range


def set_slider_values(
    beta0_key: str,
    beta1_key: str,
    beta0: float,
    beta1: float,
    show_mc_key: str | None = None,
    show_mc: bool | None = None,
) -> None:
    st.session_state[beta0_key] = float(beta0)
    st.session_state[beta1_key] = float(beta1)
    if show_mc_key is not None and show_mc is not None:
        st.session_state[show_mc_key] = bool(show_mc)


def render_manual_controls(
    prefix: str,
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> tuple[float, float]:
    x = tuple(float(v) for v in x_values)
    y = tuple(float(v) for v in y_values)
    beta0_mc, beta1_mc = least_squares(x, y)
    (
        beta0_key,
        beta1_key,
        beta0_start,
        beta1_start,
        beta0_range,
        beta1_range,
    ) = ensure_slider_state(prefix, x, y)

    st.subheader("Ajuste manual")
    st.caption("Mueva los controles: la recta y los diagnósticos se actualizan automáticamente.")

    beta0_step = nice_step(beta0_range[1] - beta0_range[0])
    beta1_step = nice_step(beta1_range[1] - beta1_range[0])
    beta0 = st.slider(
        "β₀ — intercepto",
        min_value=float(beta0_range[0]),
        max_value=float(beta0_range[1]),
        step=float(beta0_step),
        format="%.3f",
        key=beta0_key,
    )
    beta1 = st.slider(
        "β₁ — pendiente",
        min_value=float(beta1_range[0]),
        max_value=float(beta1_range[1]),
        step=float(beta1_step),
        format="%.3f",
        key=beta1_key,
    )

    show_mc_key = f"{prefix}_show_mc"
    st.toggle("Mostrar mínimos cuadrados", key=show_mc_key)

    b1, b2 = st.columns(2)
    b1.button(
        "Alinear con MC",
        key=f"{prefix}_align_mc",
        width="stretch",
        on_click=set_slider_values,
        args=(beta0_key, beta1_key, beta0_mc, beta1_mc, show_mc_key, True),
    )
    b2.button(
        "Reiniciar",
        key=f"{prefix}_reset_manual",
        width="stretch",
        on_click=set_slider_values,
        args=(beta0_key, beta1_key, beta0_start, beta1_start, show_mc_key, False),
    )

    st.checkbox(
        "Mostrar valores e²",
        key=f"{prefix}_show_e2_labels",
        disabled=len(x) > MAX_POINT_LABELS,
    )
    if len(x) > MAX_POINT_LABELS:
        st.caption(f"Las etiquetas e² se desactivan con más de {MAX_POINT_LABELS} puntos.")

    return float(beta0), float(beta1)


def render_visualization(
    prefix: str,
    x_values: Sequence[float],
    y_values: Sequence[float],
    beta0_manual: float,
    beta1_manual: float,
) -> None:
    x = tuple(float(v) for v in x_values)
    y = tuple(float(v) for v in y_values)
    manual = evaluate_fit(x, y, beta0_manual, beta1_manual)

    mc: FitResult | None = None
    if st.session_state[f"{prefix}_show_mc"]:
        beta0_mc, beta1_mc = least_squares(x, y)
        mc = evaluate_fit(x, y, beta0_mc, beta1_mc)

    legend = ["● Datos", "━ Ajuste manual"]
    if mc is not None:
        legend.append("┄ Mínimos cuadrados")
    st.caption("   ·   ".join(legend) + "   ·   Ejes fijos para facilitar la comparación")

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
        show_fit_metrics("Ajuste manual", manual, "🔵")
        if mc is not None:
            st.divider()
            show_fit_metrics("Mínimos cuadrados", mc, "🟢")

    with st.expander("Detalle numérico por punto", expanded=False):
        st.markdown(markdown_detail_table(x, y, manual, mc))


def render_workspace(prefix: str, x: Sequence[float], y: Sequence[float]) -> None:
    controls, visual = st.columns([0.39, 0.61], gap="large")
    with controls:
        with st.container(border=True):
            beta0, beta1 = render_manual_controls(prefix, x, y)
    with visual:
        render_visualization(prefix, x, y, beta0, beta1)


def regenerate_simulation(
    beta0: float,
    beta1: float,
    sigma: float,
    seed: int,
) -> None:
    x, y = generate_simulated_data(beta0, beta1, sigma, seed)
    st.session_state.sim_target_beta0 = float(beta0)
    st.session_state.sim_target_beta1 = float(beta1)
    st.session_state.sim_sigma = float(sigma)
    st.session_state.sim_seed = int(seed)
    st.session_state.sim_x = list(x)
    st.session_state.sim_y = list(y)


def render_simulation() -> None:
    with st.expander("Generación de los datos simulados", expanded=False):
        with st.form("simulation_form", enter_to_submit=False):
            c1, c2, c3, c4 = st.columns(4)
            target_beta0 = c1.number_input(
                "β₀ verdadero",
                value=float(st.session_state.sim_target_beta0),
                step=0.10,
                format="%.2f",
            )
            target_beta1 = c2.number_input(
                "β₁ verdadero",
                value=float(st.session_state.sim_target_beta1),
                step=0.10,
                format="%.2f",
            )
            sigma = c3.number_input(
                "Desvío σ",
                min_value=0.0,
                value=float(st.session_state.sim_sigma),
                step=0.05,
                format="%.2f",
            )
            seed = c4.number_input(
                "Semilla",
                min_value=0,
                value=int(st.session_state.sim_seed),
                step=1,
            )
            regenerate = st.form_submit_button("Regenerar muestra", width="stretch")

        if regenerate:
            regenerate_simulation(target_beta0, target_beta1, sigma, int(seed))
            st.rerun()

    render_workspace("sim", st.session_state.sim_x, st.session_state.sim_y)


def normalize_editor_records(data: object) -> list[dict[str, object]]:
    if isinstance(data, list):
        return [dict(row) for row in data if isinstance(row, Mapping)]
    if hasattr(data, "to_dict"):
        try:
            records = data.to_dict("records")
        except TypeError:
            records = data.to_dict()
        if isinstance(records, list):
            return [dict(row) for row in records]
    if isinstance(data, Mapping):
        keys = list(data.keys())
        if not keys:
            return []
        columns = [data[key] for key in keys]
        if all(isinstance(column, Sequence) and not isinstance(column, (str, bytes)) for column in columns):
            row_count = max((len(column) for column in columns), default=0)
            return [
                {key: data[key][i] if i < len(data[key]) else None for key in keys}
                for i in range(row_count)
            ]
    return []


def replace_editor_data(rows: list[dict[str, object]]) -> None:
    """Reconstruye la planilla sin modificar un widget ya instanciado."""
    safe_rows = [dict(row) for row in rows] if rows else [dict(row) for row in EMPTY_ROWS]
    st.session_state.own_editor_seed = safe_rows
    st.session_state.own_editor_version += 1
    st.session_state.own_committed_x = None
    st.session_state.own_committed_y = None
    st.session_state.own_show_mc = False


def render_own_data() -> None:
    st.subheader("Planilla de datos")
    st.caption(
        "Edite la planilla y pulse **Generar gráfico**. Los cambios no afectan el gráfico "
        "hasta que se confirman, lo que evita recálculos mientras se escribe."
    )

    actions = st.columns([0.20, 0.20, 0.60])
    actions[0].button(
        "Cargar ejemplo",
        width="stretch",
        on_click=replace_editor_data,
        args=([dict(row) for row in SAMPLE_ROWS],),
    )
    actions[1].button(
        "Vaciar planilla",
        width="stretch",
        on_click=replace_editor_data,
        args=([dict(row) for row in EMPTY_ROWS],),
    )

    editor_key = f"own_data_editor_{st.session_state.own_editor_version}"
    with st.form(
        key=f"own_data_form_{st.session_state.own_editor_version}",
        clear_on_submit=False,
        enter_to_submit=False,
    ):
        edited = st.data_editor(
            st.session_state.own_editor_seed,
            key=editor_key,
            width="stretch",
            height=285,
            hide_index=True,
            num_rows="dynamic",
            row_height=36,
            column_order=("X", "Y"),
            column_config={
                "X": st.column_config.NumberColumn("X", format="%.3f"),
                "Y": st.column_config.NumberColumn("Y", format="%.3f"),
            },
        )
        generate = st.form_submit_button("Generar gráfico", width="stretch")

    if generate:
        records = normalize_editor_records(edited)
        # Nunca se almacena una tabla de cero filas: se conservan filas vacías seguras.
        st.session_state.own_editor_seed = (
            [dict(row) for row in records] if records else [dict(row) for row in EMPTY_ROWS]
        )
        try:
            x, y, incomplete = editor_rows_to_xy(records)
        except ValueError as exc:
            st.session_state.own_committed_x = None
            st.session_state.own_committed_y = None
            st.error(str(exc), icon="⚠️")
        else:
            st.session_state.own_committed_x = list(x)
            st.session_state.own_committed_y = list(y)
            st.session_state.own_show_mc = False
            if incomplete:
                st.warning(
                    f"Se omitieron {incomplete} fila(s) incompleta(s) al generar el gráfico.",
                    icon="⚠️",
                )
            else:
                st.success("Gráfico generado con los datos confirmados.", icon="✅")

    committed_x = st.session_state.own_committed_x
    committed_y = st.session_state.own_committed_y
    if committed_x is None or committed_y is None:
        st.info(
            "Complete al menos dos filas con valores distintos de X y pulse **Generar gráfico**.",
            icon="ℹ️",
        )
        return

    st.divider()
    render_workspace("own", committed_x, committed_y)


def main() -> None:
    apply_balanced_css()
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
