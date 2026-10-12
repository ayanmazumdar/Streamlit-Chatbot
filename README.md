# Titanic Dataset Chat

Python application scaffold for a Streamlit chat interface and FastAPI backend.
The selected dataset and behavior are described in [SPEC.md](SPEC.md).

## Local setup

Create and activate a project-local virtual environment, then install the pinned
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env`. Add your Groq API key as `GROQ_API_KEY` for
LangChain interpretation beyond the built-in common question mappings. The
sample model is `openai/gpt-oss-120b`; set `GROQ_MODEL` to another active Groq
model if preferred. Get a key from the Groq Console. Never commit `.env`. The
API base URL defaults to `http://localhost:8000`; set
`API_BASE_URL` to a different backend URL when needed. Set `CORS_ORIGINS` to
the Streamlit origin for deployment.

Start the API and Streamlit UI in separate terminals, with the virtual
environment activated in each:

```bash
uvicorn app.api:app --reload
```

```bash
streamlit run streamlit_app.py
```

The API health endpoint is available at `http://localhost:8000/health`.

## Deploy on Render

The [`render.yaml`](render.yaml) Blueprint creates separate free web services
for the API and Streamlit UI. Push this project to a GitHub repository, then
create a Blueprint in Render from that repository. Enter a newly rotated
`GROQ_API_KEY` when Render prompts for the API service secret; never add it to
this repository. After both services deploy, copy the API service's public
HTTPS URL into the UI service's `API_BASE_URL` environment variable in the
Render dashboard, then redeploy the UI. The free services sleep after 15
minutes without traffic, so the first request after idle can take about a
minute to start.

## Dataset source

The application vendors the Seaborn teaching dataset from
[`mwaskom/seaborn-data`](https://github.com/mwaskom/seaborn-data/blob/a29a0141d20e156043ec257a64c8de3b3a03fd6e/titanic.csv).
It is pinned to commit `a29a0141d20e156043ec257a64c8de3b3a03fd6e` and retrieved
on 2026-10-12. The vendored file is `data/titanic.csv`; its SHA-256 is
`81787d320d7f7b03df935e91de8bd19e11d45c5bbcab86ef4d4a76dc91b7d4f2`.
