"""Chart payload creation and Streamlit-compatible Plotly rendering."""

from __future__ import annotations

from app.schemas import ChartData, ChartType, ResultMetadata


def chart_payload(result: ResultMetadata) -> ChartData | None:
    """Build chart JSON from operation output; never recalculate dataframe values."""
    if result.operation.value == "histogram":
        return result.chart
    if result.operation.value == "categorical_counts" and result.groups is not None:
        column = result.column.value if result.column else "category"
        return ChartData(
            chart_type=ChartType.BAR,
            title=f"Passenger counts by {column}",
            x_label=column,
            y_label="Passenger count",
            labels=[group.label for group in result.groups],
            values=[group.count for group in result.groups],
        )
    return None


def to_plotly_figure(chart: ChartData):
    """Convert serializable chart data to a Plotly figure for Streamlit."""
    import plotly.graph_objects as go

    if chart.chart_type == ChartType.BAR:
        figure = go.Figure(data=[go.Bar(x=chart.labels, y=chart.values)])
    else:
        edges = chart.bin_edges
        centers = [(left + right) / 2 for left, right in zip(edges, edges[1:])]
        widths = [right - left for left, right in zip(edges, edges[1:])]
        figure = go.Figure(data=[go.Bar(x=centers, y=chart.values, width=widths)])
    figure.update_layout(title=chart.title, xaxis_title=chart.x_label, yaxis_title=chart.y_label)
    return figure
