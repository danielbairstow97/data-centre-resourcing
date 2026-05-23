"""
Data Centre — Report Plots
==========================

Plotly-based visualisations for the data centre model.

Current plots
-------------
solar_heatmap        Monthly × hourly average CF heatmap for any p_max_pu series.
                     Used for both PPA farm profiles and rooftop profiles.
metrics_bar          Annual metric summary bar chart.
metrics_timeseries   PUE / CUE / REF over the simulation horizon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    raise ImportError("plotly is required for report plots: pip install plotly")


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_PALETTE = {
    "solar":     "#f39c12",
    "wind":      "#2980b9",
    "gas":       "#e74c3c",
    "grid":      "#95a5a6",
    "battery":   "#8e44ad",
    "cooling":   "#1abc9c",
    "water":     "#3498db",
    "co2":       "#c0392b",
    "renewable": "#27ae60",
}

_TEMPLATE = "plotly_white"


# ---------------------------------------------------------------------------
# Solar profile heatmap
# ---------------------------------------------------------------------------

def solar_heatmap(
    p_max_pu: pd.Series,
    title: str = "Solar Generation Profile",
    subtitle: str = "",
    colorscale: str = "YlOrRd",
) -> "go.Figure":
    """
    Monthly × hour-of-day average capacity factor heatmap.

    Rows = hour of day (0–23), Columns = month (Jan–Dec).
    Each cell = mean p_max_pu for that month × hour combination.

    Parameters
    ----------
    p_max_pu : pd.Series
        Hourly capacity factor series (0–1), indexed to DatetimeIndex.
    title : str
        Main plot title.
    subtitle : str
        Annotation shown below the title (e.g. location, PR, yield).
    colorscale : str
        Plotly colorscale name.

    Returns
    -------
    go.Figure
    """
    df = pd.DataFrame({
        "value": p_max_pu.values,
        "month": p_max_pu.index.month,
        "hour": p_max_pu.index.hour,
    })

    pivot = (
        df.groupby(["hour", "month"])["value"]
        .mean()
        .unstack("month")
        .reindex(index=range(24), columns=range(1, 13))
        .fillna(0.0)
    )

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    hour_labels = [f"{h:02d}:00" for h in range(24)]

    annual_yield = p_max_pu.mean() * 8760
    peak_cf = p_max_pu.max()

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=month_labels,
            y=hour_labels,
            colorscale=colorscale,
            colorbar=dict(
                title="Capacity<br>Factor",
                tickformat=".0%",
            ),
            hovertemplate=(
                "<b>%{x} %{y}</b><br>"
                "Mean CF: %{z:.1%}<br>"
                "<extra></extra>"
            ),
            zmin=0,
            zmax=min(peak_cf * 1.05, 1.0),
        )
    )

    annotations = []
    if subtitle:
        annotations.append(dict(
            text=subtitle,
            xref="paper", yref="paper",
            x=0.5, y=1.04,
            xanchor="center", yanchor="bottom",
            showarrow=False,
            font=dict(size=11, color="#666"),
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=16)),
        annotations=annotations,
        xaxis=dict(title="Month", side="bottom"),
        yaxis=dict(
            title="Hour of Day",
            autorange="reversed",   # 00:00 at top
            tickmode="array",
            tickvals=list(range(0, 24, 2)),
            ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)],
        ),
        template=_TEMPLATE,
        margin=dict(t=80, b=60, l=80, r=60),
        height=480,
        width=700,
    )

    # Add yield annotation in bottom-right
    fig.add_annotation(
        text=f"Annual yield: {annual_yield:.0f} MWh/MWp  |  Peak CF: {peak_cf:.1%}",
        xref="paper", yref="paper",
        x=1.0, y=-0.10,
        xanchor="right", yanchor="top",
        showarrow=False,
        font=dict(size=10, color="#888"),
    )

    return fig


def solar_heatmap_comparison(
    profiles: dict[str, pd.Series],
    title: str = "Solar Profile Comparison",
) -> "go.Figure":
    """
    Side-by-side heatmaps for multiple solar profiles (e.g. PPA vs rooftop).

    Parameters
    ----------
    profiles : dict[str, pd.Series]
        Mapping of label → p_max_pu series.
    """
    n = len(profiles)
    fig = make_subplots(
        rows=1, cols=n,
        subplot_titles=list(profiles.keys()),
        horizontal_spacing=0.08,
    )

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    for col_idx, (label, p_max_pu) in enumerate(profiles.items(), start=1):
        df = pd.DataFrame({
            "value": p_max_pu.values,
            "month": p_max_pu.index.month,
            "hour": p_max_pu.index.hour,
        })
        pivot = (
            df.groupby(["hour", "month"])["value"]
            .mean()
            .unstack("month")
            .reindex(index=range(24), columns=range(1, 13))
            .fillna(0.0)
        )

        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=month_labels,
                y=[f"{h:02d}:00" for h in range(24)],
                colorscale="YlOrRd",
                showscale=(col_idx == n),
                colorbar=dict(
                    title="CF",
                    tickformat=".0%",
                    x=1.02,
                ),
                zmin=0, zmax=1,
                hovertemplate=f"<b>{label}</b><br>%{{x}} %{{y}}<br>CF: %{{z:.1%}}<extra></extra>",
                name=label,
            ),
            row=1, col=col_idx,
        )

        annual_yield = p_max_pu.mean() * 8760
        fig.add_annotation(
            text=f"{annual_yield:.0f} MWh/MWp/yr",
            xref=f"x{col_idx}" if col_idx > 1 else "x",
            yref="paper",
            x=month_labels[5],  # June
            y=-0.08,
            showarrow=False,
            font=dict(size=10, color="#555"),
        )

    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=15)),
        template=_TEMPLATE,
        height=480,
        margin=dict(t=80, b=70, l=70, r=80),
    )

    return fig


# ---------------------------------------------------------------------------
# Metrics plots
# ---------------------------------------------------------------------------

def metrics_summary_bar(metrics: "DCMetrics") -> "go.Figure":
    """
    Horizontal bar chart of the six annual DC metrics against benchmark ranges.

    Each metric is shown with its computed value and a shaded benchmark band.
    """
    from dc.network.metrics import DCMetrics

    items = [
        {
            "name": "PUE",
            "label": "Power Usage<br>Effectiveness",
            "value": metrics.pue,
            "unit": "",
            "good": 1.15,
            "ok": 1.40,
            "note": "lower is better (ideal = 1.0)",
            "reverse": True,
        },
        {
            "name": "WUE",
            "label": "Water Usage<br>Effectiveness",
            "value": metrics.wue,
            "unit": " L/MWh",
            "good": 500,
            "ok": 2000,
            "note": "lower is better",
            "reverse": True,
        },
        {
            "name": "CUE",
            "label": "Carbon Usage<br>Effectiveness",
            "value": metrics.cue,
            "unit": " tCO₂e/MWh",
            "good": 0.1,
            "ok": 0.5,
            "note": "lower is better",
            "reverse": True,
        },
        {
            "name": "REF",
            "label": "Renewable Energy<br>Factor",
            "value": metrics.ref,
            "unit": "",
            "good": 0.75,
            "ok": 0.40,
            "note": "higher is better",
            "reverse": False,
            "format": ".0%",
        },
        {
            "name": "CER",
            "label": "Cooling Efficiency<br>Ratio",
            "value": metrics.cer,
            "unit": "",
            "good": 10,
            "ok": 5,
            "note": "higher is better",
            "reverse": False,
        },
    ]

    fig = go.Figure()

    bar_colours = []
    for item in items:
        v = item["value"]
        if np.isnan(v):
            bar_colours.append("#cccccc")
            continue
        if item["reverse"]:
            colour = "#27ae60" if v <= item["good"] else ("#f39c12" if v <= item["ok"] else "#e74c3c")
        else:
            colour = "#27ae60" if v >= item["good"] else ("#f39c12" if v >= item["ok"] else "#e74c3c")
        bar_colours.append(colour)

    fmt_values = []
    for item in items:
        v = item["value"]
        fmt = item.get("format", ".3f")
        if np.isnan(v):
            fmt_values.append("N/A")
        else:
            fmt_values.append(f"{v:{fmt}}{item['unit']}")

    fig.add_trace(go.Bar(
        x=[item["value"] if not np.isnan(item["value"]) else 0 for item in items],
        y=[item["label"] for item in items],
        orientation="h",
        marker_color=bar_colours,
        text=fmt_values,
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Value: %{text}<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(
            text="Data Centre Efficiency Metrics",
            x=0.5, xanchor="center", font=dict(size=16),
        ),
        xaxis=dict(visible=False),
        yaxis=dict(autorange="reversed"),
        template=_TEMPLATE,
        height=380,
        margin=dict(l=160, r=120, t=60, b=40),
        showlegend=False,
    )

    # Colour legend annotation
    fig.add_annotation(
        text="🟢 Good  🟡 Acceptable  🔴 Needs improvement",
        xref="paper", yref="paper",
        x=0.5, y=-0.08,
        xanchor="center", showarrow=False,
        font=dict(size=10, color="#666"),
    )

    return fig


def metrics_timeseries(
    metrics: "DCMetrics",
    resample: str = "D",
) -> "go.Figure":
    """
    Multi-panel time-series of PUE, CUE, and REF resampled to daily averages.

    Parameters
    ----------
    metrics : DCMetrics
    resample : str
        Pandas resample frequency for smoothing (default 'D' = daily).
    """
    pue = metrics.pue_t.resample(resample).mean()
    cue = metrics.cue_t.resample(resample).mean()
    ref = metrics.ref_t.resample(resample).mean()

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=["PUE — Power Usage Effectiveness",
                        "CUE — Carbon Usage Effectiveness (tCO₂e/MWh_it)",
                        "REF — Renewable Energy Factor"],
        vertical_spacing=0.08,
    )

    fig.add_trace(
        go.Scatter(x=pue.index, y=pue.values, mode="lines",
                   line=dict(color=_PALETTE["solar"], width=1.5),
                   name="PUE", hovertemplate="PUE: %{y:.3f}<extra></extra>"),
        row=1, col=1,
    )
    fig.add_hline(y=1.2, line_dash="dot", line_color="#27ae60",
                  annotation_text="1.2 target", row=1, col=1)

    fig.add_trace(
        go.Scatter(x=cue.index, y=cue.values, mode="lines",
                   line=dict(color=_PALETTE["co2"], width=1.5),
                   name="CUE", hovertemplate="CUE: %{y:.3f}<extra></extra>"),
        row=2, col=1,
    )

    fig.add_trace(
        go.Scatter(x=ref.index, y=ref.values, mode="lines",
                   line=dict(color=_PALETTE["renewable"], width=1.5),
                   fill="tozeroy", fillcolor="rgba(39,174,96,0.15)",
                   name="REF", hovertemplate="REF: %{y:.1%}<extra></extra>"),
        row=3, col=1,
    )
    fig.update_yaxes(tickformat=".0%", row=3, col=1)

    fig.update_layout(
        template=_TEMPLATE,
        height=600,
        showlegend=False,
        margin=dict(t=80, b=40, l=70, r=40),
        title=dict(
            text="Data Centre Metrics — Annual Time Series",
            x=0.5, xanchor="center", font=dict(size=15),
        ),
    )

    return fig
