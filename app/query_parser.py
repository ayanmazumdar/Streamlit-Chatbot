"""Constrained question interpretation with deterministic common mappings."""

from __future__ import annotations

import os
import re

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from app.schemas import Column, Interpretation, Operation, Query, ResponseStatus


DATA_DICTIONARY = {
    "survived": "integer survival indicator (0 or 1)",
    "pclass": "integer passenger ticket class (1, 2, or 3)",
    "sex": "recorded passenger sex category",
    "age": "passenger age in years; 177 values missing",
    "sibsp": "siblings and spouses aboard",
    "parch": "parents and children aboard",
    "fare": "ticket fare in dataset fare units",
    "embarked": "embarkation port code; 2 values missing",
    "class": "ticket class label",
    "who": "derived person category",
    "adult_male": "boolean adult-male indicator",
    "deck": "cabin deck; 688 values missing",
    "embark_town": "embarkation town; 2 values missing",
    "alive": "derived survival label",
    "alone": "derived alone indicator",
}


def _known_mapping(question: str) -> Interpretation | None:
    text = question.casefold()
    if any(term in text for term in ("percentage", "percent", "proportion", "share")) and "male" in text:
        return Interpretation(
            status=ResponseStatus.ANSWER,
            query=Query(operation=Operation.PERCENTAGE, column=Column.SEX, target_value="male"),
        )
    if "histogram" in text and "age" in text:
        return Interpretation(
            status=ResponseStatus.ANSWER,
            query=Query(operation=Operation.HISTOGRAM, column=Column.AGE),
        )
    if any(term in text for term in ("average", "mean")) and any(
        term in text for term in ("fare", "ticket price", "ticket fare")
    ):
        return Interpretation(
            status=ResponseStatus.ANSWER,
            query=Query(operation=Operation.MEAN, column=Column.FARE),
        )
    if "embark" in text and any(term in text for term in ("how many", "count", "number")):
        if "code" in text:
            column = Column.EMBARKED
        elif any(term in text for term in ("town", "city", "name")):
            column = Column.EMBARK_TOWN
        else:
            column = Column.EMBARK_TOWN
        return Interpretation(
            status=ResponseStatus.ANSWER,
            query=Query(operation=Operation.CATEGORICAL_COUNTS, column=column),
        )
    if re.search(r"\bhow many passengers\b", text):
        return Interpretation(status=ResponseStatus.ANSWER, query=Query(operation=Operation.ROW_COUNT))
    return None


def _chain():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL")
    if not api_key or not model_name:
        raise RuntimeError("Model configuration is missing; set GROQ_API_KEY and GROQ_MODEL")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Interpret the user question as one allowlisted deterministic dataframe query. "
                "Do not calculate values. Use only the operation, columns, filters, and categories "
                "listed below. If intent is ambiguous, return status=clarification with one concise "
                "question. If outside scope, return status=unsupported with a short reason. "
                "Return status=answer and a valid query only when intent is clear. "
                "Columns and meanings: {dictionary}. Query JSON schema: {schema}",
            ),
            ("human", "Question: {question}"),
        ]
    )
    model = ChatGroq(model=model_name, api_key=api_key, timeout=20, max_retries=1)
    return prompt | model.with_structured_output(Interpretation)


def parse_question(question: str) -> Interpretation:
    """Return validated intent, clarification, or a controlled parser error."""
    known = _known_mapping(question)
    if known is not None:
        return known
    try:
        parsed = _chain().invoke(
            {
                "question": question,
                "dictionary": DATA_DICTIONARY,
                "schema": Query.model_json_schema(),
            }
        )
        return Interpretation.model_validate(parsed)
    except Exception as exc:
        return Interpretation(
            status=ResponseStatus.ERROR,
            message=f"Could not interpret this question. Check model configuration or rephrase it. ({type(exc).__name__})",
        )
