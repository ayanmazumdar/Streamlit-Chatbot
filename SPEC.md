# Titanic Dataset Chat Agent Specification

## 1. Objective

Build a friendly chatbot that answers supported questions about Titanic passenger data and displays useful visualizations. Users ask questions in plain English and receive clear text answers and, when requested or useful, a chart in a clean Streamlit interface.

Required stack from the assignment:

- Backend: Python with FastAPI
- Agent framework: LangChain
- Frontend: Streamlit

## 2. Data

### Dataset choice

The client did not specify a Titanic data variant. This specification selects the **Seaborn `titanic` dataset**, distributed as `titanic.csv` by the [`mwaskom/seaborn-data` repository](https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv). Use that file/version as the canonical source and pin or vendor a copy for reproducible deployments. It is the 891-row Seaborn teaching dataset, not the separate Kaggle train/test competition files. The file contains 891 passenger records and 15 columns.

### Columns, types, and missingness

Types below describe the CSV values as loaded into a dataframe (integer, floating point, string/category, or boolean). Missing counts are for the selected 891-row file.

| Column | Type | Missing | Meaning |
|---|---|---:|---|
| `survived` | integer (0/1) | 0 | Survival indicator |
| `pclass` | integer (1–3) | 0 | Ticket class code |
| `sex` | string/category | 0 | Recorded sex category |
| `age` | floating point | 177 | Age in years |
| `sibsp` | integer | 0 | Siblings/spouses aboard |
| `parch` | integer | 0 | Parents/children aboard |
| `fare` | floating point | 0 | Passenger fare |
| `embarked` | string/category | 2 | Embarkation port code |
| `class` | string/category | 0 | Ticket class label |
| `who` | string/category | 0 | Seaborn-derived person category (man/woman/child) |
| `adult_male` | boolean | 0 | Seaborn-derived adult-male indicator |
| `deck` | string/category | 688 | Cabin deck |
| `embark_town` | string/category | 2 | Embarkation town name |
| `alive` | string/category | 0 | Survival label (`yes`/`no`) |
| `alone` | boolean | 0 | Seaborn-derived alone indicator |

Treat blank values as missing. Preserve the raw records; do not silently impute missing values. `class`, `who`, `adult_male`, `embark_town`, `alive`, and `alone` are derived/duplicative views included in this Seaborn variant, not additional independent source measurements. The displayed source should be identified in the app or its data/about panel.

## 3. Supported operations

The first release supports deterministic dataframe operations on the selected dataset:

- Scalar statistics: row counts, sums/counts, mean, median, minimum, maximum, and basic range for supported numeric columns.
- Percentages and proportions: category share over a clearly stated denominator, such as the percentage of all passengers recorded as male.
- Categorical counts: counts by one categorical column, such as passengers by embarkation port.
- Distributions: histogram of a numeric column and bar chart of category counts. Label axes, units where known, and missing-value treatment.
- Simple filtering: equality/inequality and numeric ranges on known columns, optionally combined with a supported statistic/count/distribution (for example, average fare for first-class passengers).

The four assignment examples are supported as follows:

| Example question | Required interpretation and output |
|---|---|
| “What percentage of passengers were male on the Titanic?” | Count `sex == male` divided by all 891 passenger rows; show percentage and numerator/denominator. |
| “Show me a histogram of passenger ages” | Histogram using non-missing `age` values (714 observations); state that 177 missing ages are excluded. |
| “What was the average ticket fare?” | Mean of non-missing `fare` over the 891 rows; show currency/unit as “dataset fare units” unless a supported source defines currency. |
| “How many passengers embarked from each port?” | Counts grouped by `embark_town` or port code; show missing embarkation separately (2 records), and clarify which representation if the user asks for a particular code/name. |

Ask a brief clarification when the requested column, category, denominator, or operation has multiple plausible interpretations, such as “average passenger” without a metric, “port” when code versus town label matters, or an unspecified comparison group. Use the data dictionary and likely intent to resolve ordinary synonyms when there is one clear match.

Out of scope initially: predictions/classification, causal claims, free-form joins, arbitrary code execution, complex multi-step statistical tests, inferred facts absent from the dataset, and external/live data. Explain that these are unsupported and offer the closest supported count, summary, filter, or chart when possible.

## 4. Reliability

- **Missing values:** For each operation, use non-missing values for the measured column and report the valid sample size and excluded missing count. For categorical counts, include missing as its own “Missing / unknown” group unless the user explicitly asks to exclude it. Never impute or treat missing values as zero. State the denominator for percentages.
- **Invalid columns or values:** Validate requested fields against the 15-column allowlist and validate filter values/types. Do not execute generated code or arbitrary SQL. If a field/value is invalid, identify the issue and suggest valid fields or values.
- **Ambiguity:** Ask one concise clarification instead of guessing when reasonable interpretations would change the result. If a common interpretation is unambiguous, answer and state the interpretation used.
- **Unsupported operations:** Clearly say the request is not supported, give a short reason, and offer a supported alternative. Do not fabricate a result.
- **Model failures:** Keep computation and chart generation in validated deterministic code; the language model may interpret the question and phrase the response, but must not invent computed values. On model, backend, or parsing failure, show a concise recoverable error and invite a rephrased question. Do not present partial/stale results as current answers.
- **Result traceability:** Include the statistic, applied filters, denominator/sample size, and missing-value rule in the response or chart caption so users can check how the answer was produced.

## 5. Acceptance criteria

### Accuracy and query behavior

- The app loads the specified Seaborn `titanic.csv` variant with exactly 891 records and the 15 named columns above; its missing-value counts match the table.
- For “What percentage of passengers were male on the Titanic?”, the answer equals `100 × count(sex == 'male') / 891`, includes the numerator and denominator, and does not count unknown sex as male.
- For “Show me a histogram of passenger ages”, a histogram is rendered from exactly the non-missing `age` values; the response/caption reports 714 included and 177 missing observations.
- For “What was the average ticket fare?”, the reported value equals the arithmetic mean of non-missing `fare` values from the selected dataset, within display rounding; the answer states the sample size and does not imply an unsupported currency.
- For “How many passengers embarked from each port?”, the returned groups and counts equal the dataset counts for `embark_town` (or clearly identified `embarked` codes), with the two missing values disclosed separately.
- A valid simple filter returns results computed only from matching rows and displays the filter used. An ambiguous request triggers clarification; an invalid column/value, unsupported operation, or model/backend failure produces the defined safe response without a fabricated answer.

### Visualization

- The age request produces a visible, non-empty histogram with a descriptive title and labeled age/count axes.
- Categorical count requests produce a readable bar chart when a chart is requested or useful; category labels and counts are legible and missing categories follow the reliability rules.
- Charts are based on the same filtered rows and missing-value rules as the accompanying text answer. A chart-generation failure does not suppress a valid text result and is reported clearly.

### Application and deployment

- A user can enter a natural-language question in Streamlit, submit it, and see the textual answer and any requested chart.
- The Streamlit frontend successfully calls the FastAPI backend, and the backend routes supported questions through the LangChain agent/framework to validated data operations.
- The submitted Streamlit URL is publicly reachable and usable for testing; the deployed app starts with the pinned dataset and dependencies and returns a graceful message if the backend is unavailable.
- The final submission includes the working Streamlit URL requested by the assignment.
