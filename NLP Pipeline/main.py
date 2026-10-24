"""
FastAPI service for the document NLP pipeline.

Endpoints:
  POST /process               -- submit a document for async processing, returns a job id
  GET  /status/{pythonJobId}  -- poll job status/progress/result
  POST /index                 -- index already-extracted text into Qdrant for RAG retrieval
  POST /retrieve               -- RAG similarity search, scoped to a workspace
  POST /answer                 -- RAG question answering: retrieve, then answer with citations
  GET  /health                -- health check (Redis/Celery/Ollama/Qdrant reachability)

Processing is asynchronous: a single document can take 15-20+ minutes, which
is well past what any HTTP client/gateway/load balancer will hold a
connection open for. The actual OCR/table/LLM pipeline runs in a Celery
worker process (see tasks.py) with a single worker (see celery_app.py) --
the model stack doesn't fit multiple copies in a typical single-GPU VRAM
budget, so the queue provides safe serialization of concurrent requests,
not parallel throughput. This FastAPI process only enqueues jobs and reports
status -- it deliberately never imports tasks.py directly (jobs are
dispatched by name via Celery's send_task), so it never loads the heavy
OCR/ML stack itself and stays fast to start.
"""

import os
import sys
import tempfile
import uuid
from typing import Any, Dict, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from celery_app import celery_app  # noqa: E402
import llm_pipeline  # noqa: E402
import rag_pipeline  # noqa: E402

VALID_MODES = {"FULL", "SUMMARY", "KEY_INSIGHTS", "CUSTOM_FIELDS"}

UPLOAD_DIR = os.getenv("SHARED_UPLOAD_DIR", os.path.join(tempfile.gettempdir(), "docupipeline_uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Document NLP Pipeline")


class ProcessResponse(BaseModel):
    pythonJobId: str


class IndexRequest(BaseModel):
    docId: str
    text: str
    workspaceId: str
    sessionId: str = ""
    fileName: str = ""


class IndexResponse(BaseModel):
    docId: str
    chunksIndexed: int


class RetrieveRequest(BaseModel):
    query: str
    topK: int = 5
    workspaceId: str


class AnswerRequest(BaseModel):
    question: str
    topK: int = 5
    workspaceId: str


class DeleteDocsRequest(BaseModel):
    docIds: List[str]
    workspaceId: str


class ResetCollectionRequest(BaseModel):
    workspaceId: str


JOB_REGISTRY_TTL_SECONDS = int(os.getenv("JOB_REGISTRY_TTL", "86400") or "86400")
_REDIS_CLIENT = None


def _get_redis():
    global _REDIS_CLIENT
    if _REDIS_CLIENT is None:
        import redis

        _REDIS_CLIENT = redis.Redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379/0")
        )
    return _REDIS_CLIENT


def _remember_job(job_id: str) -> None:
    """Record that we accepted this job id. Advisory only -- never fail a
    submission because the registry write failed."""
    try:
        _get_redis().setex(f"docupipeline:job:{job_id}", JOB_REGISTRY_TTL_SECONDS, "1")
    except Exception:
        pass


def _job_known(job_id: str) -> bool:
    """False only when we can positively say we never saw this id. If Redis is
    unreachable we assume the job is known, so a transient outage can never
    make live jobs look lost."""
    try:
        return bool(_get_redis().exists(f"docupipeline:job:{job_id}"))
    except Exception:
        return True


@app.post("/reset-collection")
async def reset_collection(payload: ResetCollectionRequest) -> Dict[str, Any]:
    """Drop a workspace's entire collection. It is recreated automatically on
    the next /index call."""
    try:
        client = rag_pipeline.get_qdrant_client()
        if client.collection_exists(payload.workspaceId):
            client.delete_collection(payload.workspaceId)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reset failed: {exc}")
    return {"reset": True}


@app.get("/health")
async def health() -> Dict[str, Any]:
    status: Dict[str, Any] = {"status": "ok"}

    try:
        import redis

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        redis.Redis.from_url(redis_url).ping()
        status["queueReachable"] = True
    except Exception:
        status["queueReachable"] = False

    try:
        models = llm_pipeline.list_available_models()
        status["ollamaReachable"] = True
        status["ollamaModels"] = sorted(models)
    except Exception:
        status["ollamaReachable"] = False

    try:
        client = rag_pipeline.get_qdrant_client()
        client.get_collections()
        status["qdrantReachable"] = True
    except Exception:
        status["qdrantReachable"] = False

    return status


@app.post("/process", response_model=ProcessResponse, status_code=202)
async def process(
    file: UploadFile = File(...),
    mode: str = Form("FULL"),
    customPrompt: str = Form(""),
    workspaceId: str = Form(""),
) -> ProcessResponse:
    mode_upper = (mode or "FULL").upper()
    if mode_upper not in VALID_MODES:
        raise HTTPException(status_code=400, detail=f"Invalid mode. Expected one of {sorted(VALID_MODES)}")

    safe_name = f"{uuid.uuid4()}_{file.filename or 'upload'}"
    saved_path = os.path.join(UPLOAD_DIR, safe_name)
    content = await file.read()
    with open(saved_path, "wb") as f:
        f.write(content)

    task = celery_app.send_task(
        "tasks.process_document_task",
        kwargs={"file_path": saved_path, "mode": mode_upper, "custom_prompt": customPrompt},
    )
    _remember_job(task.id)
    return ProcessResponse(pythonJobId=task.id)


@app.get("/status/{python_job_id}")
async def get_status(python_job_id: str) -> Dict[str, Any]:
    result = celery_app.AsyncResult(python_job_id)

    if result.state == "PENDING":
        if not _job_known(python_job_id):
            return {"status": "NOT_FOUND"}
        return {"status": "QUEUED"}
    if result.state == "PROGRESS":
        progress = (result.info or {}).get("progress", 0.0)
        return {"status": "PROCESSING", "progress": progress}
    if result.state == "STARTED":
        return {"status": "PROCESSING", "progress": 0.0}
    if result.state == "SUCCESS":
        return {"status": "COMPLETED", "result": result.result}
    if result.state == "FAILURE":
        return {"status": "FAILED", "error": str(result.info)}
    return {"status": "PROCESSING"}


@app.post("/index", response_model=IndexResponse)
async def index(payload: IndexRequest) -> IndexResponse:
    metadata = {"session_id": payload.sessionId, "file_name": payload.fileName}
    try:
        n = rag_pipeline.index_document(
            payload.docId,
            payload.text,
            metadata=metadata,
            collection_name=payload.workspaceId,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}")
    return IndexResponse(docId=payload.docId, chunksIndexed=n)


@app.post("/retrieve")
async def retrieve(payload: RetrieveRequest) -> List[Dict[str, Any]]:
    try:
        return rag_pipeline.retrieve(
            payload.query,
            top_k=payload.topK,
            collection_name=payload.workspaceId,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}")
    

@app.post("/answer")
async def answer(payload: AnswerRequest) -> Dict[str, Any]:
    """Retrieve the workspace's most relevant passages, then have the LLM
    answer from them behind the guardrails in chat_guardrails.py. The
    passages returned are the ones the model saw, numbered from 1 so the
    caller can resolve the [n] citations. Both steps are blocking
    (embedding model, Ollama), so they run off the event loop."""
    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")
    if len(question) > 1000:
        raise HTTPException(status_code=400, detail="Question is too long.")
    try:
        passages = await run_in_threadpool(
            rag_pipeline.retrieve, question, payload.topK, payload.workspaceId
        )
        result = await run_in_threadpool(llm_pipeline.answer_question, question, passages)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Answering failed: {exc}")
    return result


@app.post("/delete-docs")
async def delete_docs(payload: DeleteDocsRequest) -> Dict[str, Any]:
    try:
        n = rag_pipeline.delete_documents(
            payload.docIds,
            collection_name=payload.workspaceId,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Delete failed: {exc}")
    return {"deleted": n}