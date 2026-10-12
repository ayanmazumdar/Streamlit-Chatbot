"""Streamlit chat interface for the Titanic dataset API."""

from __future__ import annotations

import os

import httpx
import streamlit as st
from dotenv import load_dotenv

from app.charts import to_plotly_figure
from app.schemas import ChartData


load_dotenv()
st.set_page_config(page_title="Titanic Dataset Chat", page_icon="🚢")
st.title("Titanic Dataset Chat")
st.caption("Ask questions about the Seaborn Titanic teaching dataset.")

api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("About this data")
    st.markdown(
        "Seaborn `titanic.csv`: 891 passenger records. "
        "Missing values: age 177, embarked 2, deck 688, embark town 2. "
        "This is not Kaggle competition data."
    )
    st.markdown(
        "[Source and pinned revision](https://github.com/mwaskom/seaborn-data/blob/"
        "a29a0141d20e156043ec257a64c8de3b3a03fd6e/titanic.csv)"
    )


def render_message(message: dict[str, object]) -> None:
    with st.chat_message(str(message["role"])):
        st.markdown(str(message["content"]))
        chart_data = message.get("chart")
        if isinstance(chart_data, dict):
            try:
                chart = ChartData.model_validate(chart_data)
                st.plotly_chart(to_plotly_figure(chart), use_container_width=True)
            except Exception:
                st.warning("Answer is available, but chart could not be rendered.")
        metadata = message.get("metadata")
        if isinstance(metadata, dict):
            st.caption(
                f"Rows used: {metadata.get('sample_count', 0)} · "
                f"Missing: {metadata.get('missing_count', 0)} · "
                f"Missing-value rule: {metadata.get('missing_policy', 'not reported')}"
            )


for item in st.session_state.messages:
    render_message(item)

question = st.chat_input("Ask about passenger counts, fares, ages, or ports")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    render_message(st.session_state.messages[-1])
    with st.chat_message("assistant"):
        with st.spinner("Checking the dataset…"):
            try:
                response = httpx.post(
                    f"{api_base_url}/chat",
                    json={"question": question},
                    timeout=httpx.Timeout(25.0),
                )
                response.raise_for_status()
                payload = response.json()
                assistant_message = {
                    "role": "assistant",
                    "content": payload.get("message", "The API returned no message."),
                    "chart": (payload.get("result") or {}).get("chart"),
                    "metadata": payload.get("result"),
                }
            except (httpx.HTTPError, ValueError):
                assistant_message = {
                    "role": "assistant",
                    "content": "The data service is unavailable or returned an invalid response. Please retry.",
                }
        st.session_state.messages.append(assistant_message)
        render_message(assistant_message)
