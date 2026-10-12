"""Load and validate the vendored Seaborn Titanic dataset."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "titanic.csv"
DATASET_SOURCE = "https://github.com/mwaskom/seaborn-data/blob/a29a0141d20e156043ec257a64c8de3b3a03fd6e/titanic.csv"
DATASET_REVISION = "a29a0141d20e156043ec257a64c8de3b3a03fd6e"
DATASET_SHA256 = "81787d320d7f7b03df935e91de8bd19e11d45c5bbcab86ef4d4a76dc91b7d4f2"

EXPECTED_COLUMNS = (
    "survived",
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked",
    "class",
    "who",
    "adult_male",
    "deck",
    "embark_town",
    "alive",
    "alone",
)
EXPECTED_MISSING = {
    "survived": 0,
    "pclass": 0,
    "sex": 0,
    "age": 177,
    "sibsp": 0,
    "parch": 0,
    "fare": 0,
    "embarked": 2,
    "class": 0,
    "who": 0,
    "adult_male": 0,
    "deck": 688,
    "embark_town": 2,
    "alive": 0,
    "alone": 0,
}
INTEGER_COLUMNS = ("survived", "pclass", "sibsp", "parch")
FLOAT_COLUMNS = ("age", "fare")
BOOLEAN_COLUMNS = ("adult_male", "alone")
STRING_COLUMNS = tuple(
    column
    for column in EXPECTED_COLUMNS
    if column not in INTEGER_COLUMNS + FLOAT_COLUMNS + BOOLEAN_COLUMNS
)


def validate_dataset(frame: pd.DataFrame, raw_bytes: bytes) -> dict[str, object]:
    """Validate the canonical dataset and return non-sensitive profile metadata."""
    if len(frame) != 891:
        raise ValueError(f"Expected 891 rows, found {len(frame)}")
    if tuple(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected columns: {list(frame.columns)}")

    for column in INTEGER_COLUMNS:
        if not pd.api.types.is_integer_dtype(frame[column].dtype):
            raise ValueError(f"Expected integer column {column}, found {frame[column].dtype}")
    for column in FLOAT_COLUMNS:
        if not pd.api.types.is_float_dtype(frame[column].dtype):
            raise ValueError(f"Expected floating-point column {column}, found {frame[column].dtype}")
    for column in BOOLEAN_COLUMNS:
        if not pd.api.types.is_bool_dtype(frame[column].dtype):
            raise ValueError(f"Expected boolean column {column}, found {frame[column].dtype}")
    for column in STRING_COLUMNS:
        if not pd.api.types.is_string_dtype(frame[column].dtype):
            raise ValueError(f"Expected string column {column}, found {frame[column].dtype}")

    missing = {column: int(count) for column, count in frame.isna().sum().items()}
    if missing != EXPECTED_MISSING:
        raise ValueError(f"Unexpected missing-value profile: {missing}")

    checksum = hashlib.sha256(raw_bytes).hexdigest()
    if checksum != DATASET_SHA256:
        raise ValueError(f"Dataset checksum mismatch: {checksum}")

    return {
        "rows": len(frame),
        "columns": list(frame.columns),
        "missing": missing,
        "revision": DATASET_REVISION,
        "sha256": checksum,
    }


def load_dataset(path: Path = DATA_PATH) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load the pinned CSV, treating empty fields as missing, then validate it."""
    raw_bytes = path.read_bytes()
    frame = pd.read_csv(path, keep_default_na=True)
    profile = validate_dataset(frame, raw_bytes)
    frame.attrs["profile"] = profile
    return frame, profile
