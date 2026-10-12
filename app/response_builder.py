"""Template-based user messages from deterministic results."""

from __future__ import annotations

from app.schemas import Filter, ResultMetadata


def _filter_text(item: Filter) -> str:
    upper = f" and {item.upper_value}" if item.upper_value is not None else ""
    return f"{item.column.value} {item.operator.value} {item.value}{upper}"


def format_result(result: ResultMetadata) -> str:
    """Describe supplied result values without recomputing them."""
    column = result.column.value if result.column else "passengers"
    operation = result.operation.value
    if operation == "row_count":
        message = f"There are {result.sample_count} passengers in the filtered data."
    elif operation == "percentage":
        percentage = float(result.value or 0.0)
        message = f"{percentage:.2f}% matched ({result.numerator}/{result.denominator})."
    elif operation == "histogram":
        message = (
            f"Histogram of {column} uses {result.sample_count} valid observations; "
            f"{result.missing_count} missing values were excluded."
        )
    elif operation == "categorical_counts":
        groups = ", ".join(f"{group.label}: {group.count}" for group in (result.groups or []))
        message = f"Counts by {column}: {groups}. Total rows: {result.sample_count}."
    elif operation == "mean" and column == "fare":
        message = f"Mean fare is {float(result.value):.4f} dataset fare units (n={result.sample_count})."
    elif operation == "count":
        message = f"Count of non-missing {column}: {result.value} (n={result.sample_count})."
    else:
        message = f"{operation.capitalize()} of {column}: {result.value} (n={result.sample_count})."
    if result.missing_count and operation not in ("histogram", "categorical_counts"):
        message += f" Missing values: {result.missing_count}."
    if result.filters_applied:
        message += " Filters: " + "; ".join(_filter_text(item) for item in result.filters_applied) + "."
    return message
