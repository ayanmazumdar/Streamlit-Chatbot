"""Validated request and response contracts for the chat API."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr, model_validator


class Column(StrEnum):
    SURVIVED = "survived"
    PCLASS = "pclass"
    SEX = "sex"
    AGE = "age"
    SIBSP = "sibsp"
    PARCH = "parch"
    FARE = "fare"
    EMBARKED = "embarked"
    TICKET_CLASS = "class"
    WHO = "who"
    ADULT_MALE = "adult_male"
    DECK = "deck"
    EMBARK_TOWN = "embark_town"
    ALIVE = "alive"
    ALONE = "alone"


class Operation(StrEnum):
    ROW_COUNT = "row_count"
    COUNT = "count"
    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    RANGE = "range"
    PERCENTAGE = "percentage"
    CATEGORICAL_COUNTS = "categorical_counts"
    HISTOGRAM = "histogram"


class FilterOperator(StrEnum):
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    BETWEEN = "between"


class ChartType(StrEnum):
    BAR = "bar"
    HISTOGRAM = "histogram"


Scalar = Annotated[StrictStr | StrictInt | StrictFloat | StrictBool, Field()]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Filter(StrictModel):
    column: Column
    operator: FilterOperator
    value: Scalar
    upper_value: Scalar | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> Filter:
        if (self.operator == FilterOperator.BETWEEN) != (self.upper_value is not None):
            raise ValueError("between filters require upper_value; other filters forbid it")
        return self


class Query(StrictModel):
    operation: Operation
    column: Column | None = None
    filters: list[Filter] = Field(default_factory=list, max_length=10)
    target_value: Scalar | None = None
    chart: ChartType | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> Query:
        needs_column = self.operation != Operation.ROW_COUNT
        if needs_column and self.column is None:
            raise ValueError(f"{self.operation} requires a column")
        if self.operation == Operation.PERCENTAGE and self.target_value is None:
            raise ValueError("percentage requires target_value")
        if self.operation != Operation.PERCENTAGE and self.target_value is not None:
            raise ValueError("target_value is only valid for percentage")
        if self.operation == Operation.HISTOGRAM and self.chart not in (None, ChartType.HISTOGRAM):
            raise ValueError("histogram operation requires a histogram chart")
        return self


class ChatRequest(StrictModel):
    question: Annotated[StrictStr, Field(min_length=1, max_length=2000)]
    request_id: Annotated[StrictStr, Field(min_length=1, max_length=100)] | None = None


class GroupCount(StrictModel):
    label: str
    count: int


class ChartData(StrictModel):
    chart_type: ChartType
    title: str
    x_label: str
    y_label: str
    labels: list[str] = Field(default_factory=list)
    values: list[int | float] = Field(default_factory=list)
    bin_edges: list[float] = Field(default_factory=list)


class ResultMetadata(StrictModel):
    operation: Operation
    column: Column | None = None
    value: int | float | str | None = None
    groups: list[GroupCount] | None = None
    numerator: int | None = None
    denominator: int | None = None
    sample_count: int
    missing_count: int
    filters_applied: list[Filter] = Field(default_factory=list)
    missing_policy: str
    chart: ChartData | None = None


class ResponseStatus(StrEnum):
    ANSWER = "answer"
    CLARIFICATION = "clarification"
    ERROR = "error"
    UNSUPPORTED = "unsupported"


class ChatResponse(StrictModel):
    status: ResponseStatus
    message: str
    result: ResultMetadata | None = None
    request_id: str | None = None


class Interpretation(StrictModel):
    status: ResponseStatus
    query: Query | None = None
    message: str | None = None

    @model_validator(mode="after")
    def validate_interpretation(self) -> Interpretation:
        if self.status == ResponseStatus.ANSWER and self.query is None:
            raise ValueError("answer interpretation requires a query")
        if self.status != ResponseStatus.ANSWER and (self.query is not None or not self.message):
            raise ValueError("non-answer interpretation requires a message and forbids a query")
        return self
