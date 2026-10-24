# Document NLP Pipeline

A local, 100%-open-source document processing pipeline: image denoising →
OCR/table extraction → RAG indexing → local LLM inference, exposed as an
asynchronous FastAPI service. No paid APIs — everything runs on your own
machine.

## Architecture

- **Preprocessing** (`Preprocessing/`) — SwinUNet denoiser (currently a
  standalone, callable model — not yet wired into the automatic pipeline,
  see Known Limitations).
- **Text/table extraction** (`Text Extraction/`) — deskew, perspective
  correction, shadow removal, document-type classification (digital PDF /
  printer scan / phone scan), PaddleOCR text extraction (with DocTR as a
  fallback OCR engine), and a three-tier table extraction chain: Table
  Transformer (TATR) → DeepSeek-OCR (VLM fallback, gated by a
  plausibility/reliability check so a hallucinated non-table doesn't get
  accepted) → OCR-position-clustering (last resort, with real column
  alignment via x-position clustering), plus pdfplumber for ruling-line
  tables in digital PDFs.
- **RAG** (`NLP Pipeline/rag_pipeline.py`) — chunking (LangChain
  `RecursiveCharacterTextSplitter`), embeddings (`BAAI/bge-m3`), Qdrant
  vector storage and cosine-similarity retrieval. Indexed per-workspace:
  each `workspaceId` maps to its own Qdrant collection, so documents from
  different workspaces/tenants never mix in retrieval.
- **LLM** (`NLP Pipeline/llm_pipeline.py`) — local inference via Ollama
  (Qwen2.5:7b primary, Llama3.1:8b fallback), temperature pinned to 0 for
  reproducible output on factual document tasks. Four modes: FULL (OCR
  cleanup, faithful reproduction), SUMMARY, KEY_INSIGHTS, and CUSTOM_FIELDS.
  CUSTOM_FIELDS always uses Ollama's schema-constrained structured output
  (never fully-loose JSON mode, which was measured to let the model default
  to null too easily) and always requires a chain-of-thought "reasoning"
  field before the actual answer, which measurably improves accuracy on
  multi-step questions (e.g. "which item has the least quantity") for a
  small local model -- the reasoning is stripped before the response
  reaches the caller. Two request shapes: a clean comma-separated field
  list ("Extract: invoice_number, total") gets a strict multi-key schema;
  anything else is treated as a natural-language question and gets a
  single strict `{"answer": ...}` schema instead.
- **Async job queue** (`NLP Pipeline/celery_app.py`, `NLP Pipeline/tasks.py`)
  — Celery + Redis. A single document can take 15-20+ minutes end to end,
  which is well past what any HTTP client/gateway/load balancer will hold a
  connection open for, so processing is asynchronous: submit a job, poll
  for status/result. The worker is pinned to a single process/concurrency
  of 1 -- the model stack (DeepSeek-OCR + Table Transformer) already uses
  ~8GB VRAM on an 8GB GPU when loaded once, so multiple workers each
  loading their own copy of the models doesn't fit a typical single-GPU
  budget. The queue exists for safe serialization of concurrent requests
  (so two uploads never race against the same shared model instances), not
  for parallel throughput.
- **API** (`NLP Pipeline/main.py`) — FastAPI service tying it all together.
  Deliberately never imports the OCR/ML stack directly (jobs are dispatched
  to Celery by task name), so it stays lightweight and starts fast. See API
  section below for endpoints.

## Setup

### 1. Python environment

Requires Python 3.10+ (developed against 3.11 -- note Python 3.13 does not
have wheels for several pinned dependencies here, including numpy 1.26.4
and torch 2.5.1; use 3.11 or 3.12).

    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    # macOS/Linux:
    source .venv/bin/activate

    pip install -r requirements.txt

If you don't have an NVIDIA GPU, remove the `--extra-index-url` line and the
`+cu121` suffixes from `torch`/`torchvision` in `requirements.txt` first, or
install a CPU build of PyTorch yourself before running the command above
(see https://pytorch.org/get-started/locally/).

### 2. Ollama (local LLM runtime)

Download and install from https://ollama.com/download.

**Windows-specific gotcha:** after installing, a terminal/IDE window opened
*before* the install won't see the new `ollama` command on PATH. Fully close
and reopen your terminal (or your IDE, if it hosts an integrated terminal)
rather than just opening a new tab.

Verify it's installed, then pull the models:

    ollama --version
    ollama pull qwen2.5:7b
    ollama pull llama3.1:8b

### 3. Qdrant and Redis (via Docker)

Needs Docker Desktop. On Windows this requires WSL2:

    wsl --install
    # restart your computer

Then install Docker Desktop from https://www.docker.com/products/docker-desktop/
(it should auto-detect WSL2). Verify:

    docker --version
    docker run hello-world

Start Qdrant (vector database) and Redis (Celery broker/result backend):

    docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
    docker run -d -p 6379:6379 --name redis redis:alpine

### 4. DeepSeek-OCR / Table Transformer / BGE-M3 model downloads

These download automatically from Hugging Face on first use (no separate
setup step) — but they're large (DeepSeek-OCR ~9GB, Table Transformer
models a few hundred MB, BGE-M3 ~2.3GB). First run of anything that touches
these will take a while.

### 5. Environment configuration

Copy `Text Extraction/.env.example` to `Text Extraction/.env.extraction` and
fill in any values you need (DeepSeek backend mode, table detection
thresholds, etc.). Defaults work for local/free operation. `REDIS_URL`
(default `redis://localhost:6379/0`) and `SHARED_UPLOAD_DIR` (default a
system temp folder) can also be overridden via environment variables if
needed.

## Running the service

Two processes need to run alongside Ollama and the Docker containers above:

    # Terminal 1 -- the Celery worker (does the actual OCR/table/LLM work)
    cd "NLP Pipeline"
    celery -A celery_app worker --pool=solo --loglevel=info

    # Terminal 2 -- the FastAPI service (just enqueues jobs and reports status)
    uvicorn main:app --app-dir "NLP Pipeline" --port 8000

`--pool=solo` is required on Windows (no `fork()`) and also naturally
enforces one job at a time, which matches the VRAM budget above regardless
of platform.

Check it's healthy:

    curl http://localhost:8000/health

## API

- `GET /health` — reports Redis (queue)/Ollama/Qdrant reachability.
- `POST /process` — multipart form: `file` (the document), `mode` (`FULL` |
  `SUMMARY` | `KEY_INSIGHTS` | `CUSTOM_FIELDS`), `customPrompt` (used for
  `CUSTOM_FIELDS`: either a comma-separated field list or a natural-language
  question), `workspaceId`. Returns `202` with `{ pythonJobId }` immediately
  -- does not wait for processing to finish.
- `GET /status/{pythonJobId}` — poll for job status. Returns
  `{ status: "QUEUED" | "PROCESSING" | "COMPLETED" | "FAILED", progress?,
  result?, error? }`. `result` (present once `status` is `COMPLETED`) has
  the shape `{ extractedText, structuredJson, tables, mode }`.
- `POST /index` — JSON body `{ docId, text, workspaceId, sessionId?,
  fileName? }`. Chunks, embeds, and indexes the text into a
  workspace-scoped Qdrant collection for later retrieval.
- `POST /retrieve` — JSON body `{ query, topK?, workspaceId }`. Returns a
  list of `{ text, score, docId, chunkIndex, sessionId, fileName }`,
  most relevant first, scoped to that workspace's indexed documents.
- `POST /answer` — JSON body `{ question, topK?, workspaceId }`. Retrieves
  the workspace's most relevant passages, then has the LLM answer from them
  alone (temperature 0, schema-constrained, reasoning stripped). Returns
  `{ answer, sources, model_used, passages }`: `passages` are the retrieved
  chunks numbered from 1, and `sources` lists the numbers the answer used.
  With no relevant passages it says so rather than guessing.

## Configuration

The following environment variables control where the service looks for
its dependencies — all have working defaults for local use, override them
when integrating/deploying elsewhere:

- `QDRANT_URL` (default `http://localhost:6333`)
- `REDIS_URL` (default `redis://localhost:6379/0`)
- `OLLAMA_HOST` (default: Ollama library default, `localhost:11434`)
- `LLM_PRIMARY_MODEL` / `LLM_FALLBACK_MODEL` (default `qwen2.5:7b` / `llama3.1:8b`)
- `SHARED_UPLOAD_DIR` (default a system temp folder)

There are also ~25+ fine-grained OCR/table-detection tuning thresholds
(`PADDLE_*`, `OCR_*`, `TABLE_REGION_*`, etc.) throughout
`Text Extraction/OCR_Text_Extraction.py` -- these are internal heuristics
already tuned during development, not part of the integration contract.
Leave them at their defaults unless specifically tuning OCR behavior.

## Known limitations

- SwinUNet denoising is wired into the automatic pipeline (`--swinunet-denoise`
  flag / `OCR_SWINUNET_DENOISE` env var) but **defaults to off**: measured
  against 20 real noisy/clean image pairs, it produced 3x worse CER and ~2x
  worse WER than classical OpenCV denoising (see `Text Extraction/accuracy_eval.py
  noisy-pairs`). Visually the denoised output looks cleaner, but it measurably
  hurts PaddleOCR's recognition accuracy -- likely because `denoise_image()`
  converts to grayscale (discarding color/contrast information the OCR model
  may rely on), and/or the checkpoint itself was trained in disconnected
  5-epoch bursts with no tracked total epoch count. Don't enable it without
  re-measuring first.
- Beyond SwinUNet specifically: the image path actually used for PaddleOCR
  (`preprocess_image_for_paddle`) has classical OpenCV denoising **off by
  default too** (`PADDLE_DENOISE` defaults to `"0"`) -- only resizing and
  deskew run by default, no noise removal at all. This is a deliberate
  existing default (see the function's own docstring, "avoid
  over-denoise"), consistent with the SwinUNet finding above: denoising
  measurably risked hurting PaddleOCR's accuracy more than it helped. Not
  yet re-measured against real noisy scans under this session's testing --
  our test documents weren't heavily degraded to begin with, so this
  default's effect hasn't actually been exercised or verified either way.
- DocTR is wired in as a fallback when PaddleOCR (including its own
  internal fallback/rescue passes) fails to produce usable text. No
  PyTesseract tier yet (spec's third-tier fallback).
- Borderless (no visible gridlines) tables in digital PDFs: pdfplumber
  only catches ruling-line tables, but when it finds none, the page is
  rendered to an image and run through the same TATR -> DeepSeek ->
  OCR-clustering chain used for scanned pages. Verified on a real customs
  form with no table gridlines at all -- correctly extracted structured
  data (port codes, GSTIN, item descriptions, duty amounts) that would
  previously have been silently skipped.
- Table rows whose cells span multiple visual lines (e.g. a line-item
  description followed by its own HSN/SAC code, serial number, and tax
  breakdown on separate lines within the same logical row) can get
  fragmented into several smaller rows by the OCR-position-clustering
  fallback tier, instead of being merged back into one row -- it clusters
  purely by vertical text-line position, with no concept of "these several
  lines belong to one row." Verified on a real single-line-item tax
  invoice: every individual value was still captured correctly, just
  spread across ~9 small rows instead of one clean row. Left as-is
  deliberately -- a general fix (merging visually-adjacent lines into one
  logical row) risks breaking documents where those lines really are
  separate rows.
- PaddleOCR runs CPU-only by default. The code already auto-detects and
  would use GPU automatically (`_paddle_can_use_gpu()` in
  `OCR_Text_Extraction.py`) if `paddlepaddle-gpu` were installed instead of
  the plain `paddlepaddle` pinned in `requirements.txt` -- no code change
  needed, just an environment-specific install swap. Not done by default
  because this project has previously hit real compatibility bugs when
  deviating from the current pinned paddleocr/paddlepaddle/paddlex
  combination (see the comment in `requirements.txt`).
- Processing is single-worker by design, not a scaling limitation to be
  fixed later: one Celery worker (concurrency 1) processes one document at
  a time. The model stack (DeepSeek-OCR + Table Transformer) already uses
  ~8GB VRAM on an 8GB GPU, so running multiple workers -- each loading its
  own copy of the models -- doesn't fit a typical single-GPU budget.
  Concurrent requests queue safely instead of running in parallel or
  racing against the shared model instances.
- Automated accuracy benchmarking exists (`Text Extraction/accuracy_eval.py`,
  `pdf` and `noisy-pairs` subcommands) but has only been run on a handful of
  sample documents. It hasn't been run against your actual production
  documents, so treat current numbers (measured in this repo's development)
  as a starting baseline, not a guarantee.