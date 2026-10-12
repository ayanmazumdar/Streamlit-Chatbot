# Titanic Dataset Chat Agent Implementation Plan

This document turns [SPEC.md](SPEC.md) into an implementation sequence for a Python, FastAPI, LangChain, and Streamlit application. Each phase defines the method, its output, and the conditions to verify before proceeding.

## 1. Implementation principles

- Keep all data calculations in deterministic Python code. LangChain can interpret a question into a constrained request and help phrase a response, but it must not calculate or invent results.
- Load the selected Seaborn `titanic.csv` dataset once per backend process, validate its schema and missing-value profile at startup, and keep it read-only for user requests.
- Use an explicit operation schema and column allowlist. Do not expose Python, pandas, SQL, or arbitrary tool execution to the model.
- Carry sample counts, denominators, filters, and missing-value handling alongside every result so the response and chart describe the same data.
- Keep frontend, API, query interpretation, analytics, and chart rendering responsibilities in separate modules.

## 2. Suggested project structure

```text
.
├── app/
│   ├── api.py                 # FastAPI app and HTTP routes
│   ├── schemas.py             # Pydantic request/response models
│   ├── data.py                # Dataset loading and startup validation
│   ├── query_parser.py        # LangChain-backed question interpretation
│   ├── operations.py          # Allowlisted deterministic dataframe operations
│   ├── response_builder.py    # Text explanation from computed results
│   └── charts.py              # Chart specifications or serialization
├── streamlit_app.py           # Streamlit user interface
├── data/
│   └── titanic.csv            # Pinned canonical dataset copy
├── requirements.txt           # Or pyproject.toml with pinned dependencies
├── .env.example               # Names of required configuration variables only
├── Dockerfile                 # Optional, if supported by chosen host
└── README.md                  # Local run and deployment instructions
```

The final names may change to fit the repository, but module boundaries should remain clear. Keep secrets out of source control. A deployed application should use a fixed dataset file or immutable source revision, not download a changing `master` branch at every startup.

## 3. Phases of implementation

### Phase 1 — Repository and runtime setup

**Method**

1. Create the application package, entry points, dependency manifest, and configuration template.
2. Pin compatible versions of FastAPI, Uvicorn, Streamlit, LangChain integration packages, pandas, and a charting library. Pin the model provider SDK if one is required.
3. Configure the model provider through environment variables. Document the required key and model setting in `.env.example`, but never put a real key in that file or the repository.
4. Define local commands to start the API and Streamlit app separately. Use configuration for the API base URL so local development and deployment can use different endpoints.

**Deliverables**

- A reproducible Python environment and basic project layout.
- Local startup instructions and documented configuration variables.

**Completion check**

- Both app entry points start in the documented environment, and the Streamlit service can be configured to reach the API.

### Phase 2 — Dataset acquisition and validation

**Method**

1. Obtain the CSV selected in `SPEC.md` from the `mwaskom/seaborn-data` Titanic dataset source, then vendor it under `data/` or pin an immutable revision and retrieve that exact revision during build/deploy.
2. Record the source URL, selected revision or retrieval date, and a checksum in project documentation/configuration so the deployed data can be reproduced.
3. Load the CSV with pandas, treating empty cells as missing. Keep the source rows unchanged; do not fill nulls or modify source categories.
4. At startup, validate that the dataframe has 891 records, the exact 15 expected columns, expected column types/coercions, and the missing-value counts in `SPEC.md`.
5. If validation fails, fail startup with a clear log message. Do not run with partially parsed or silently substituted data.
6. Expose a small data-profile function for the health endpoint and diagnostics; do not send passenger rows to logs or to the language model.

**Deliverables**

- Reproducible dataset artifact and loader.
- Startup schema, row-count, type, and missingness checks.

**Completion check**

- The canonical data profile is 891 records, 15 expected fields, with 177 missing `age`, 2 missing `embarked`, 688 missing `deck`, and 2 missing `embark_town`; all other listed fields have zero missing values.

### Phase 3 — Typed request and response contracts

**Method**

Define Pydantic models shared by API routing and internal components. The natural language request should contain the user question and optionally a request identifier. The response should have an explicit status such as `answer`, `clarification`, or `error`, a user-facing message, and structured result metadata.

Represent an interpreted operation using a constrained shape, for example:

```json
{
  "operation": "mean",
  "column": "fare",
  "filters": [],
  "group_by": null,
  "chart": null
}
```

Supported operation names, fields, filter operators, chart types, and output types must be enums or validated values. A result payload should include the computed value or groups, valid row count, missing row count where relevant, denominator where relevant, applied filters, and any chart data/specification. Unknown keys and malformed types should be rejected before computation.

**Deliverables**

- Stable internal query, result, error, and clarification schemas.
- Consistent JSON API representation independent of UI rendering.

**Completion check**

- Invalid operation names, columns, operators, and payload shapes cannot reach dataframe execution.

### Phase 4 — Deterministic analytics engine

**Method**

1. Implement a dispatcher that maps the validated operation schema to a small set of pandas functions. Do not dynamically evaluate a model-generated expression.
2. Implement scalar operations: row count, count, sum, mean, median, minimum, maximum, and range on allowed numeric fields where meaningful.
3. Implement percentages using explicit target category and denominator. For the male percentage example, count `sex == "male"` and divide by all passenger rows; include the numerator and denominator.
4. Implement categorical counts for one selected category field. Include null values as a named “Missing / unknown” bucket unless the user explicitly requests missing values be excluded.
5. Implement numeric distributions using only valid numeric values. Return valid and excluded counts with histogram bin counts/edges or a chart-ready equivalent.
6. Apply filters before aggregation and chart creation. Validate filter values against field type and observed or allowed categories. Ensure all downstream outputs use the same filtered dataframe.
7. Return structured errors for invalid fields, invalid values, unsupported operations, and empty filtered results. Never convert a failed calculation into a guessed value.

**Deliverables**

- Small, testable deterministic operation functions.
- Result metadata that fully records filters and missing-value treatment.

**Completion check**

- Each operation produces a reproducible result from the pinned dataframe and preserves denominator/sample-size metadata.

### Phase 5 — Question interpretation and clarification with LangChain

**Method**

1. Build a LangChain interpretation chain that receives the question, the supported-operation catalog, the 15-column data dictionary, and the constrained query schema.
2. Request a structured output matching the typed query model. The model should choose an operation and normalized column/filter values or return a clarification/unsupported intent.
3. Validate the structured output locally. Treat malformed output, unknown columns, invalid operators, and out-of-scope requests as controlled errors; never pass unchecked model output to pandas.
4. Define deterministic synonym mappings for common phrases (for example, “ticket price” to `fare`, “age histogram” to a histogram of `age`) before or alongside model interpretation.
5. If two plausible interpretations would change the answer (for example, embarkation code versus town name), return one concise clarification question and do not calculate until the user clarifies.
6. After deterministic computation, pass only the computed result and its metadata to a response formatter. Prefer template-based phrasing; if a language model is used for wording, instruct it to preserve all numeric values and verify that the returned answer references the supplied result only.
7. Provide safe fallback responses for provider timeouts, rate limits, invalid structured output, and unavailable model configuration. The core data functions should remain callable without the model for known typed requests and operational diagnostics.

**Deliverables**

- LangChain-backed parser producing validated operations or clarification/error statuses.
- Safe response formatting with no model-sourced calculations.

**Completion check**

- The four assignment example questions map to the operations and interpretations defined in `SPEC.md`; unsupported prompts do not cause arbitrary execution or fabricated answers.

### Phase 6 — Chart generation

**Method**

1. Select one chart library supported by Streamlit (for example, Plotly or Altair) and use it consistently.
2. Build charts only from the computed result or the same already-filtered dataframe used by the analytics operation.
3. For age distribution, create a histogram over 714 non-missing ages and report the 177 excluded rows in the caption/associated answer.
4. For categorical counts, render a bar chart with readable category labels and counts, including “Missing / unknown” when applicable.
5. Add descriptive titles, labeled axes, and units only where the source supports them. For `fare`, say “dataset fare units”; do not assert currency.
6. Return a chart payload or chart-ready structured data from the API. Avoid relying on unserializable Python objects in API JSON. If chart rendering fails, return the computed text result and a chart-specific warning.

**Deliverables**

- Reusable chart builders for histograms and categorical count plots.
- Chart metadata aligned with the numeric answer and missing-data rules.

**Completion check**

- The age example produces a non-empty, labeled histogram; categorical bars match the counts in the response.

### Phase 7 — FastAPI backend

**Method**

1. Create the FastAPI application with a startup lifespan hook that loads and validates the dataset once.
2. Add `GET /health` for liveness and dataset readiness, returning service status and dataset profile/version metadata without exposing passenger-level data.
3. Add `POST /chat` accepting a question and returning the typed answer, clarification, or error response. Keep API errors structured and avoid returning stack traces or secrets to the client.
4. Configure request size limits, timeouts, CORS for the deployed Streamlit origin, and appropriate HTTP status codes. Do not allow broad wildcard CORS in production when a known frontend origin can be configured.
5. Log request IDs, operation names, durations, and error categories. Avoid logging full user data, provider credentials, or passenger records.
6. Keep model/API provider failures separate from deterministic calculation errors so the UI can explain whether the issue is interpretation, computation, or service availability.

**Deliverables**

- Deployable FastAPI service, documented endpoint schemas, and health/readiness route.

**Completion check**

- Valid question requests return structured results, and malformed, unavailable, or unsupported requests return clear controlled responses.

### Phase 8 — Streamlit user interface

**Method**

1. Add a chat-style question input, submit action, conversation history, and clear pending/loading state.
2. Send each submitted question to the configured FastAPI `/chat` endpoint with a bounded timeout. Keep UI code focused on presentation and do not duplicate analytics logic in Streamlit.
3. Render the response message and any structured chart. Show the sample size, denominator, applied filters, and missing-value disclosure near the result.
4. Render clarification prompts as a request for the user's next input. Render backend/model errors as recoverable messages with a clear retry or rephrase path.
5. Add a concise About/Data section naming the Seaborn dataset variant, 891 records, source link, and known missingness. Avoid presenting the chatbot as a source of facts beyond this dataset.
6. Use environment-based backend URL configuration and provide an understandable unavailable-service state when the API cannot be reached.

**Deliverables**

- Streamlit chat experience that presents text answers and visualizations.

**Completion check**

- A user can submit a natural-language query and view its answer/chart, clarification, or recovery message without leaving the interface.

### Phase 9 — Integration and acceptance verification

**Method**

Run the acceptance scenarios from `SPEC.md` against the deployed component boundaries. Use known dataframe calculations as expected values and compare numerical results before display rounding. Verify charts and error states as well as successful answers.

Required scenarios:

1. Male percentage uses `count(sex == "male") / 891`, displays the exact numerator/denominator, and excludes no unknown sex values silently.
2. Age histogram uses 714 valid observations, states that 177 age values are missing, and labels the axes.
3. Mean fare equals the dataframe mean of non-missing `fare` values, with sample size and no unsupported currency assertion.
4. Port counts match the selected representation (`embark_town` or `embarked`), with two unknown/missing records reported separately.
5. A valid filter affects the answer and chart consistently and is shown in result metadata.
6. Ambiguous intent asks for clarification; invalid column/value and unsupported operations are handled without calculations or invented results.
7. Simulated model timeout/invalid output, API unavailability, and chart failure produce the specified recoverable behavior.

**Deliverables**

- Recorded results for each acceptance scenario and a resolved issue list.

**Completion check**

- All behavioral and chart acceptance criteria pass before deployment. Verification artifacts should record expected/observed values and the code/data revision used.

### Phase 10 — Deployment and handoff

**Method**

1. Choose hosts that can run Streamlit and FastAPI, or a single host that can run both services. Configure the API base URL and allowed frontend origin for the production URLs.
2. Build from pinned dependencies and the pinned dataset artifact. Configure secrets in the host's secret manager/environment settings.
3. Set health checks, restart behavior, and startup failure reporting. Ensure the API dataset readiness check passes before routing user traffic.
4. Deploy the backend and frontend, then verify the public Streamlit URL can reach the API and complete each of the four example questions.
5. Confirm that cold start, unavailable backend, and model-provider failure states are usable and do not display stale results.
6. Document the public Streamlit URL, deployment configuration, data source/revision, and known limitations in the handoff.

**Deliverables**

- Public working Streamlit URL requested by the assignment.
- Deployment notes, data version, and configuration requirements.

**Completion check**

- The public URL is reachable and supports end-to-end question, answer, and chart flows using the pinned data and deployed API.

## 4. Operational error behavior

| Condition | Backend behavior | User-facing behavior |
|---|---|---|
| Missing values in measured column | Exclude nulls from the calculation and return valid/excluded counts. | State the sample size and excluded count. |
| Missing grouping category | Include a “Missing / unknown” group by default. | Show the group and its count. |
| Invalid column or filter | Reject before computation and return allowed fields/categories where practical. | Explain the invalid input and offer a correction. |
| Ambiguous question | Return a clarification status with one question. | Ask for the specific choice needed. |
| Unsupported request | Return an unsupported status; do not invoke arbitrary execution. | Explain the limit and suggest a supported operation. |
| Model/parser error | Return a controlled interpretation error; preserve no stale answer. | Invite retry or rephrasing. |
| Analytics error or empty result | Return a controlled computation error or empty-result status. | Explain that no matching data/result was available. |
| Chart error | Preserve valid text result and attach chart warning. | Display answer and explain chart could not be rendered. |
| API unavailable | Let the frontend request fail within a timeout and return a service error state. | Explain that the service is temporarily unavailable and allow retry. |
| Dataset validation failure | Fail readiness/startup and log a clear diagnostic. | Keep the service out of rotation; do not serve potentially incorrect answers. |

## 5. Scope boundaries and implementation risks

- The Seaborn dataset includes derived/duplicative columns and differs from Kaggle competition data. Keep the selected variant visible so users know which values are being analyzed.
- The GitHub branch URL can change. Pin a revision or vendor the CSV and record its checksum to prevent deployments from silently changing data.
- Natural-language requests can be interpreted inconsistently by a model. Constrained schemas, deterministic synonym mappings, validation, and clarification handling limit that risk.
- A chart and its text summary can disagree if separate filters are applied. Build both from one validated operation result and carry filter/missingness metadata through the API.
- Public deployment depends on host support for the chosen services and configured model credentials. Keep configuration documented and report startup/readiness failures clearly.

## 6. Release checklist

- [ ] Dataset source/version/checksum is recorded and the startup profile matches `SPEC.md`.
- [ ] All supported operations are deterministic, validated, and covered by acceptance scenarios.
- [ ] LangChain output is schema-validated and cannot trigger arbitrary code or SQL execution.
- [ ] Missingness, denominators, filters, and sample sizes appear in results where applicable.
- [ ] Text answers and charts use the same filtered data and missing-value rules.
- [ ] FastAPI health and chat routes work with production configuration and frontend CORS.
- [ ] Streamlit handles answer, chart, clarification, and recoverable error states.
- [ ] Required acceptance scenarios pass against the release revision.
- [ ] Public Streamlit URL is reachable and included in the handoff.
