"""FastAPI application and HTTP routes."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.charts import chart_payload
from app.data import load_dataset
from app.operations import AnalyticsError, compute
from app.query_parser import parse_question
from app.response_builder import format_result
from app.schemas import ChatRequest, ChatResponse, ResponseStatus


load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("titanic_chat.api")
MAX_REQUEST_BYTES = 16 * 1024


@asynccontextmanager
async def lifespan(application: FastAPI):
    try:
        frame, profile = load_dataset()
    except Exception as exc:
        logger.exception("Dataset validation failed: %s", type(exc).__name__)
        raise RuntimeError("Dataset failed startup validation") from exc
    application.state.dataset = frame
    application.state.dataset_profile = profile
    yield


app = FastAPI(title="Titanic Dataset Chat API", version="0.1.0", lifespan=lifespan)
origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Request-ID"],
)


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit() and int(content_length) > MAX_REQUEST_BYTES:
            return JSONResponse(status_code=413, content={"detail": "Request body is too large"})
        return await call_next(request)


app.add_middleware(RequestSizeLimitMiddleware)


@app.exception_handler(RequestValidationError)
async def invalid_request_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=ChatResponse(status=ResponseStatus.ERROR, message="Invalid request payload").model_dump(mode="json"),
    )


@app.get("/health")
def health(request: Request) -> dict[str, object]:
    """Report readiness and non-sensitive dataset profile."""
    return {"status": "ready", "dataset": request.app.state.dataset_profile}


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, request: Request) -> ChatResponse:
    """Interpret a question and return a deterministic result or controlled status."""
    started = time.perf_counter()
    request_id = body.request_id or str(uuid4())
    interpretation = parse_question(body.question)
    if interpretation.status != ResponseStatus.ANSWER:
        logger.info(
            "request_id=%s status=%s duration_ms=%.1f",
            request_id,
            interpretation.status.value,
            (time.perf_counter() - started) * 1000,
        )
        return ChatResponse(
            status=interpretation.status,
            message=interpretation.message or "Please clarify your question.",
            request_id=request_id,
        )

    assert interpretation.query is not None
    try:
        result = compute(request.app.state.dataset, interpretation.query)
        chart = chart_payload(result)
        if chart is not None:
            result = result.model_copy(update={"chart": chart})
    except AnalyticsError as exc:
        logger.info("request_id=%s category=computation_error", request_id)
        return ChatResponse(
            status=ResponseStatus.ERROR,
            message=str(exc),
            request_id=request_id,
        )
    except Exception as exc:
        logger.exception("request_id=%s category=internal_error type=%s", request_id, type(exc).__name__)
        return ChatResponse(
            status=ResponseStatus.ERROR,
            message="The request could not be completed. Please retry.",
            request_id=request_id,
        )

    logger.info(
        "request_id=%s operation=%s duration_ms=%.1f",
        request_id,
        interpretation.query.operation.value,
        (time.perf_counter() - started) * 1000,
    )
    return ChatResponse(
        status=ResponseStatus.ANSWER,
        message=format_result(result),
        result=result,
        request_id=request_id,
    )
