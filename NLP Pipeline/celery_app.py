"""
Celery application config for the async document-processing queue.

Worker concurrency is pinned to 1 -- the model stack (DeepSeek-OCR + Table
Transformer) already uses ~8GB VRAM on an 8GB GPU when loaded once, so
running multiple worker processes (each loading its own copy of the models)
doesn't fit in a typical single-GPU budget. This queue exists for safe
serialization and async job status, not parallel throughput.

Redis is used as both broker and result backend (matches the free,
self-hosted, Docker-friendly pattern already used for Qdrant).

Run the worker from inside the "NLP Pipeline" directory with:
    celery -A celery_app worker --pool=solo --loglevel=info
"pool=solo" is required on Windows (no fork()) and also naturally enforces
one job at a time, which is what we want anyway given the VRAM budget.
"""

import os

from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "docupipeline",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["tasks"],
)

celery_app.conf.update(
    task_track_started=True,
    worker_concurrency=1,
    worker_pool="solo",
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    result_expires=3600,
)