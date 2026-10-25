"""
Serverless GPU runner for the pipeline, on Modal (https://modal.com).

The local setup runs the heavy OCR/table/LLM stack in a single Celery worker
on one 8 GB GPU, so documents queue one at a time. Here each document gets
its own GPU container, containers start only when a job arrives and stop
about a minute after the last one, so an idle deployment costs nothing.

Two classes, so a chat question never pays to load the OCR stack:
  Processor -- the full pipeline (tasks.run_pipeline) for one document
  Answerer  -- RAG answer generation only (llm_pipeline.answer_question)

The API service (main.py with PIPELINE_BACKEND=modal) calls these by name:
  Processor.process.spawn(...)  -> returns a call id immediately
  FunctionCall.from_id(id)      -> polled from GET /status/{id}

Deploy:   modal deploy modal_app.py
Warm up:  modal run modal_app.py      (downloads model weights once, ~10 GB)

Cost controls -- keep these as they are:
  min_containers=0    never keep a GPU warm; a warm L4 bills ~24 h a day
  scaledown_window    stop a container shortly after its last job
  max_containers      cap how many GPUs one burst of uploads can start
"""

import os
import subprocess
import tempfile
import time
import uuid

import modal

APP_NAME = "documorph-pipeline"
GPU = "L4"  # 24 GB: holds DeepSeek-OCR, Table Transformer and the LLM together
LLM_MODEL = "qwen2.5:7b"
OLLAMA_MODELS = "/opt/ollama/models"  # baked into the image, outside the volume

app = modal.App(APP_NAME)

# Model weights downloaded at runtime (Hugging Face, PaddleOCR, DocTR) are kept
# here so they download once and survive image rebuilds.
cache = modal.Volume.from_name("documorph-model-cache", create_if_missing=True)
# Per-job progress (0-1), read by the API's /status endpoint.
progress = modal.Dict.from_name("documorph-progress", create_if_missing=True)

_LOCAL_ENV_FILES = [".env", ".env.extraction", ".env.environment", "**/__pycache__"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "libgomp1", "curl", "zstd", "ca-certificates")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .env({"OLLAMA_MODELS": OLLAMA_MODELS})
    .run_commands(
        f"bash -c 'ollama serve >/tmp/ollama.log 2>&1 & "
        f"for i in $(seq 1 30); do curl -sf http://127.0.0.1:11434 >/dev/null && break; sleep 1; done; "
        f"ollama pull {LLM_MODEL}'"
    )
    .pip_install_from_requirements("requirements.txt")
    .env({
        "LLM_PRIMARY_MODEL": LLM_MODEL,
        # one model in the image; an equal fallback makes a failure raise
        # instead of retrying against a model that is not there
        "LLM_FALLBACK_MODEL": LLM_MODEL,
        "PYTHONUNBUFFERED": "1",
    })
    # code last, so editing it rebuilds only this layer
    .add_local_dir("NLP Pipeline", "/app/NLP Pipeline", ignore=_LOCAL_ENV_FILES)
    .add_local_dir("Text Extraction", "/app/Text Extraction", ignore=_LOCAL_ENV_FILES)
)


def _prepare_runtime() -> None:
    """Point every model cache at the volume and put the code on sys.path.
    Must run before the pipeline is imported: PaddleOCR resolves ~ at import."""
    import sys

    os.environ["HOME"] = "/cache/home"
    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["TORCH_HOME"] = "/cache/torch"
    os.environ["DOCTR_CACHE_DIR"] = "/cache/doctr"
    for d in ("/cache/home", "/cache/hf", "/cache/torch", "/cache/doctr"):
        os.makedirs(d, exist_ok=True)
    sys.path.insert(0, "/app/NLP Pipeline")


def _start_ollama(timeout_s: int = 60) -> None:
    import urllib.request

    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "OLLAMA_MODELS": OLLAMA_MODELS},
    )
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            urllib.request.urlopen("http://127.0.0.1:11434", timeout=2)
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("Ollama did not start")


@app.cls(
    gpu=GPU,
    image=image,
    volumes={"/cache": cache},
    timeout=45 * 60,  # matches the backend's 45-minute job timeout
    scaledown_window=60,
    min_containers=0,
    max_containers=3,
)
class Processor:
    @modal.enter()
    def load(self):
        _prepare_runtime()
        _start_ollama()
        import tasks

        self.tasks = tasks
        tasks._get_engines()  # DeepSeek-OCR + PaddleOCR, once per container
        cache.commit()

    @modal.method()
    def process(self, file_bytes: bytes, filename: str, mode: str, custom_prompt: str = ""):
        call_id = modal.current_function_call_id()
        progress[call_id] = 0.0

        # run_pipeline deletes its input file when it finishes
        safe = os.path.basename(filename or "upload") or "upload"
        path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}_{safe}")
        with open(path, "wb") as f:
            f.write(file_bytes)

        def _report(fraction: float) -> None:
            try:
                progress[call_id] = fraction
            except Exception:
                pass  # progress is advisory; never fail a job over it

        try:
            return self.tasks.run_pipeline(path, mode, custom_prompt, progress_cb=_report)
        finally:
            try:
                progress.pop(call_id)
            except Exception:
                pass
            cache.commit()  # keeps weights a first run downloaded lazily

    @modal.method()
    def warm(self) -> dict:
        """Download every lazily-loaded model so real jobs start warm."""
        import doctr_extraction
        import table_transformer_extraction as tatr

        tatr._load_model(tatr.DETECTION_MODEL_NAME)
        tatr._load_model(tatr.STRUCTURE_MODEL_NAME)
        doctr_extraction._get_model()
        cache.commit()
        return {"engines": sorted(self.tasks._ENGINES), "llm": LLM_MODEL}


@app.cls(
    gpu=GPU,
    image=image,
    timeout=5 * 60,
    scaledown_window=120,  # a chat is usually several questions in a row
    min_containers=0,
    max_containers=1,
)
class Answerer:
    @modal.enter()
    def load(self):
        _prepare_runtime()
        _start_ollama()
        import llm_pipeline

        self.llm = llm_pipeline

    @modal.method()
    def answer(self, question: str, passages: list) -> dict:
        return self.llm.answer_question(question, passages)


@app.local_entrypoint()
def main():
    print("Warming up: downloading model weights into the volume...")
    print(Processor().warm.remote())
