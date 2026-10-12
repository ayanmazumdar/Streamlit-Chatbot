"""Allowlisted, deterministic dataframe operations."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from app.schemas import ChartData, ChartType, Column, Filter, FilterOperator, GroupCount, Operation, Query, ResultMetadata


NUMERIC_OPERATIONS = {
    Operation.SUM,
    Operation.MEAN,
    Operation.MEDIAN,
    Operation.MIN,
    Operation.MAX,
    Operation.RANGE,
    Operation.HISTOGRAM,
}
NUMERIC_COLUMNS = {"survived", "pclass", "age", "sibsp", "parch", "fare"}
MISSING_LABEL = "Missing / unknown"


class AnalyticsError(ValueError):
    """A validated query cannot be computed for the selected data."""


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_filter(frame: pd.DataFrame, item: Filter) -> None:
    name = item.column.value
    series = frame[name]
    numeric = pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(series.dtype)
    boolean = pd.api.types.is_bool_dtype(series.dtype)
    string = pd.api.types.is_string_dtype(series.dtype)
    if numeric:
        if not _is_number(item.value) or (item.upper_value is not None and not _is_number(item.upper_value)):
            raise AnalyticsError(f"Filter for {name} requires numeric values")
        return
    if boolean:
        if item.operator.value not in {"eq", "ne"} or not isinstance(item.value, bool):
            raise AnalyticsError(f"Filter for {name} requires a boolean equality value")
        return
    if string:
        if item.operator.value not in {"eq", "ne"} or not isinstance(item.value, str):
            raise AnalyticsError(f"Filter for {name} requires a string equality value")
        categories = set(series.dropna().unique().tolist())
        if item.value not in categories:
            allowed = ", ".join(sorted(categories))
            raise AnalyticsError(f"Invalid value {item.value!r} for {name}; allowed values: {allowed}")
        return
    raise AnalyticsError(f"Filtering is not supported for {name}")


def _apply_filter(frame: pd.DataFrame, item: Filter) -> pd.DataFrame:
    name = item.column.value
    series = frame[name]
    value = item.value
    op = item.operator.value
    if op == "eq":
        mask = series == value
    elif op == "ne":
        mask = series != value
    elif op == "gt":
        mask = series > value
    elif op == "gte":
        mask = series >= value
    elif op == "lt":
        mask = series < value
    elif op == "lte":
        mask = series <= value
    else:
        mask = series.between(value, item.upper_value, inclusive="both")
    return frame.loc[mask.fillna(False)]


def _metadata(query: Query, sample_count: int, missing_count: int, policy: str, **values: object) -> ResultMetadata:
    return ResultMetadata(
        operation=query.operation,
        column=query.column,
        sample_count=sample_count,
        missing_count=missing_count,
        filters_applied=query.filters,
        missing_policy=policy,
        **values,
    )


def compute(frame: pd.DataFrame, query: Query) -> ResultMetadata:
    """Apply validated filters and compute one allowlisted operation."""
    filtered = frame
    for item in query.filters:
        _validate_filter(frame, item)
        filtered = _apply_filter(filtered, item)
    if filtered.empty:
        raise AnalyticsError("No rows match the requested filters")

    operation = query.operation.value
    if operation == Operation.ROW_COUNT.value:
        return _metadata(query, len(filtered), 0, "All filtered rows")

    assert query.column is not None
    name = query.column.value
    series = filtered[name]
    valid = series.dropna()
    missing_count = int(series.isna().sum())

    if query.operation in NUMERIC_OPERATIONS and name not in NUMERIC_COLUMNS:
        raise AnalyticsError(f"Operation {operation} requires a numeric column")

    if operation == Operation.COUNT.value:
        return _metadata(query, len(valid), missing_count, "Missing values excluded", value=len(valid))
    if operation == Operation.PERCENTAGE.value:
        target = query.target_value
        if pd.api.types.is_string_dtype(series.dtype):
            categories = set(valid.unique().tolist())
            if target not in categories:
                allowed = ", ".join(sorted(categories))
                raise AnalyticsError(f"Invalid value {target!r} for {name}; allowed values: {allowed}")
        elif pd.api.types.is_bool_dtype(series.dtype):
            if not isinstance(target, bool):
                raise AnalyticsError(f"Percentage target for {name} must be boolean")
        elif not _is_number(target):
            raise AnalyticsError(f"Percentage target for {name} must be numeric")
        denominator = len(filtered)
        numerator = int((series == target).sum())
        percentage = (100.0 * numerator / denominator) if denominator else 0.0
        return _metadata(
            query,
            denominator,
            missing_count,
            "Denominator includes all filtered rows; missing category values remain in denominator",
            value=percentage,
            numerator=numerator,
            denominator=denominator,
        )
    if operation == Operation.CATEGORICAL_COUNTS.value:
        counts = series.value_counts(dropna=False, sort=False)
        groups = [
            GroupCount(label=MISSING_LABEL if pd.isna(label) else str(label), count=int(count))
            for label, count in counts.items()
        ]
        groups.sort(key=lambda group: (group.label == MISSING_LABEL, group.label))
        return _metadata(query, len(filtered), missing_count, "Missing values included as Missing / unknown", groups=groups)
    if operation == Operation.HISTOGRAM.value:
        values = valid.to_numpy(dtype=float)
        if not len(values):
            raise AnalyticsError(f"No valid values remain for {name}")
        counts, edges = np.histogram(values, bins=20)
        x_label = {"age": "Age (years)", "fare": "Fare (dataset fare units)"}.get(name, name)
        chart = ChartData(
            chart_type=ChartType.HISTOGRAM,
            title=f"Distribution of {name}",
            x_label=x_label,
            y_label="Passenger count",
            values=[int(count) for count in counts],
            bin_edges=[float(edge) for edge in edges],
        )
        return _metadata(query, len(valid), missing_count, "Missing values excluded", chart=chart)

    if operation in NUMERIC_OPERATIONS:
        if valid.empty:
            raise AnalyticsError(f"No valid values remain for {name}")
        if operation == Operation.SUM.value:
            result = float(valid.sum())
        elif operation == Operation.MEAN.value:
            result = float(valid.mean())
        elif operation == Operation.MEDIAN.value:
            result = float(valid.median())
        elif operation == Operation.MIN.value:
            result = float(valid.min())
        elif operation == Operation.MAX.value:
            result = float(valid.max())
        else:
            result = float(valid.max() - valid.min())
        if not math.isfinite(result):
            raise AnalyticsError(f"Operation {operation} produced a non-finite result")
        return _metadata(query, len(valid), missing_count, "Missing values excluded", value=result)

    raise AnalyticsError(f"Unsupported operation: {operation}")
