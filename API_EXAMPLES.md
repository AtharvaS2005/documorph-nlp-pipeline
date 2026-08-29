# API Examples — Real Request/Response Pairs

Every example below is a real, verified request/response captured during
actual testing of this pipeline (not hand-written/hypothetical). Field
names, casing, and nesting are exactly what the API returns.

## Health check

**Request:**
```bash
curl -s http://localhost:8000/health
```

**Response:**
```json
{
    "status": "ok",
    "queueReachable": true,
    "ollamaReachable": true,
    "ollamaModels": ["llama3.1:8b", "qwen2.5:7b"],
    "qdrantReachable": true
}
```

## Submitting a document — POST /process

**Request** (multipart form):
```bash
curl -s -X POST "http://localhost:8000/process" \
  -F "file=@invoice.pdf" \
  -F "mode=FULL" \
  -F "workspaceId=test-workspace"
```

**Response** (immediate, HTTP 202 -- does NOT wait for processing):
```json
{
    "pythonJobId": "b6167251-3e5f-4b92-ae48-dbbb106000a8"
}
```

`mode` is one of `FULL`, `SUMMARY`, `KEY_INSIGHTS`, `CUSTOM_FIELDS`.
`customPrompt` is only used for `CUSTOM_FIELDS` (see below). `workspaceId`
is required but not currently used by `/process` itself -- it matters for
`/index` and `/retrieve` (see those sections).

**Invalid mode** (HTTP 400, synchronous -- rejected before any job is
enqueued):
```json
{
    "detail": "Invalid mode. Expected one of ['CUSTOM_FIELDS', 'FULL', 'KEY_INSIGHTS', 'SUMMARY']"
}
```

## Polling — GET /status/{pythonJobId}

**While queued** (this also happens if you poll a job id that was never
actually submitted -- Celery can't distinguish "queued" from "never
existed" without extra bookkeeping, so an unknown id reads as QUEUED
forever, not a distinct "not found" error):
```json
{
    "status": "QUEUED"
}
```

**While running:**
```json
{
    "status": "PROCESSING",
    "progress": 0.45
}
```

**Once finished, mode=FULL:**
```json
{
    "status": "COMPLETED",
    "result": {
        "extractedText": "===== Page 1 =====\nInvoice no: 63042822\nDate of issue:\n06/04/2014\nSeller:\nClient:\nRamirez and Sons\nHarrison LLC\n057 Andrews Vista\n5446 Miller Rue\nJeffreyfort, AR 30246\nLyonsfort, ID 29652\nTax ld:961-81-3603\nTax ld:909-82-6817\nIBAN: GB72EJSX53160989293657\nITEMS\nSUMMARY",
        "structuredJson": {
            "result": "Invoice No.: 63042822\nDate of Issue: 06/04/2014\nSeller: Ramirez and Sons\nClient: Harrison LLC\nAddress:\nRamirez and Sons\n057 Andrews Vista\n5446 Miller Rue\nJeffreyfort, AR 30246\n\nHarrison LLC\nLyonsfort, ID 29652\nTax ID: 961-81-3603\nTax ID: 909-82-6817\nIBAN: GB72EJSX53160989293657\n\nITEMS\nSUMMARY"
        },
        "tables": [
            {"page": "1", "col_1": "No.", "col_2": "Description", "col_3": "Qty", "col_4": "UM", "col_5": "Net price", "col_6": "Net worth", "col_7": "VAT [%]", "col_8": "Gross\nworth"},
            {"page": "1", "col_1": "1.", "col_2": "Round Rug Carpets for Living\nRoom Decor Kids Room Long\nPlush Rugs Bedroom Mats*", "col_3": "3,00", "col_4": "each", "col_5": "4,38", "col_6": "13,14", "col_7": "10%", "col_8": "14,45"},
            {"page": "1", "col_1": "Total", "col_2": "", "col_3": "$51 527,97", "col_4": "$5 152,80", "col_5": "$56 680,77", "col_6": "", "col_7": "", "col_8": ""}
        ],
        "mode": "FULL"
    }
}
```
(Full `tables` array shortened here for readability -- in practice it has
one entry per extracted row.) Notice `structuredJson` for `FULL` mode is
always `{"result": "<cleaned text>"}` -- prose, not structured fields.
Same shape for `SUMMARY` and `KEY_INSIGHTS`, just different content:

**mode=SUMMARY**, `structuredJson`:
```json
{
    "result": "This tax invoice details a transaction between Consolidated Private Limited and Vian Enterprises for the sale of an MXCU3HN/A MBA 13 M38CPU10GPU16GB512GB SL computer, with a net amount of ₹121,410.00. The invoice includes tax details (SGST and CGST at 9% each) and specifies payment methods as Phonepe or cash. The transaction took place on January 26, 2025, and was handled by Sales Employee Manish Khemchandani."
}
```
(`₹` is the real UTF-8 rupee sign, ₹ -- verified directly from the raw
response bytes. It can display as `?` in a Windows PowerShell console
depending on codepage; that's a terminal display issue, not the actual
API data -- parse the JSON body directly and you'll get the real character.)

**mode=KEY_INSIGHTS**, `structuredJson`:
```json
{
    "result": "- Invoice number: 63042822\n- Date of issue: 06/04/2014\n- Seller: Ramirez and Sons\n- Client: Harrison LLC\n- Seller's address: 057 Andrews Vista, Jeffreyfort, AR 30246\n- Client's address: 5446 Miller Rue, Lyonsfort, ID 29652\n- Seller's tax ID: 961-81-3603\n- Client's tax ID: 909-82-6817\n- IBAN: GB72EJSX53160989293657"
}
```

## CUSTOM_FIELDS -- two different request shapes, two different response shapes

**Case A: a clean comma-separated field list.** `customPrompt` looks like
`"Extract: field_a, field_b"` -> the response is a flat object with exactly
those keys (snake_case, normalized from whatever you typed).

Request:
```bash
curl -s -X POST "http://localhost:8000/process" \
  -F "file=@invoice.pdf" -F "mode=CUSTOM_FIELDS" \
  -F "customPrompt=Extract: invoice_number, total_amount" \
  -F "workspaceId=test-workspace"
```

Completed result's `structuredJson`:
```json
{
    "invoice_number": "63042822",
    "total_amount": "56680.77"
}
```
A field that isn't found in the document returns JSON `null` (not the
string `"null"`), e.g. `{"invoice_number": "63042822", "total_amount": null}`.

**Case B: anything else (a natural-language question).** The response is
always exactly `{"answer": "..."}` or `{"answer": null}` -- one fixed key,
regardless of how the question is phrased.

Request:
```bash
curl -s -X POST "http://localhost:8000/process" \
  -F "file=@invoice.pdf" -F "mode=CUSTOM_FIELDS" \
  -F "customPrompt=Which item has the highest gross worth, and what is that amount?" \
  -F "workspaceId=test-workspace"
```

Completed result's `structuredJson`:
```json
{
    "answer": "The item with the highest gross worth is the 'Yilong 5'x8' Handknotted Silk Area Rug Living Room Black Classic Carpet L38B', with a gross worth of $44,000.00."
}
```
This mode can correctly answer comparison/aggregation questions across
multiple table rows (verified: correctly identified the maximum across 5
line items), not just single-value lookups.

**In both cases**, `extractedText` and `tables` are still populated in the
full result -- CUSTOM_FIELDS never omits them, by design (every job
returns a complete record: full text, full tables, plus whatever the
requested mode produced).

## Indexing for RAG -- POST /index

**Request** (JSON body, not multipart):
```bash
curl -s -X POST "http://localhost:8000/index" \
  -H "Content-Type: application/json" \
  -d '{
    "docId": "rug-invoice-001",
    "text": "Invoice no: 63042822 ... IBAN: GB72EJSX53160989293657",
    "workspaceId": "test-workspace",
    "sessionId": "session-001",
    "fileName": "batch1-0514.jpg"
  }'
```

**Response:**
```json
{
    "docId": "rug-invoice-001",
    "chunksIndexed": 1
}
```

`workspaceId` determines which Qdrant collection the text is chunked and
embedded into -- documents from different workspaces are never mixed.

## Retrieval -- POST /retrieve

**Request:**
```bash
curl -s -X POST "http://localhost:8000/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the seller'"'"'s address?",
    "topK": 3,
    "workspaceId": "test-workspace"
  }'
```

**Response** (a list, most relevant first, empty list `[]` if nothing
matches or the workspace has nothing indexed):
```json
[
    {
        "text": "Invoice no: 63042822\nDate of issue: 06/04/2014\nSeller: Ramirez and Sons\n057 Andrews Vista\nJeffreyfort, AR 30246\nClient: Harrison LLC\n5446 Miller Rue\nLyonsfort, ID 29652\nTax Id (Seller): 961-81-3603\nTax Id (Client): 909-82-6817\nIBAN: GB72EJSX53160989293657",
        "score": 0.5901879,
        "doc_id": "rug-invoice-001",
        "chunk_index": 0,
        "session_id": "session-001",
        "file_name": "batch1-0514.jpg"
    }
]
```
Note the response uses `doc_id`/`chunk_index`/`session_id`/`file_name`
(snake_case), while the request used `docId`/`sessionId`/`fileName`
(camelCase) -- this asymmetry is real and currently unchanged; map field
names accordingly on the Java side rather than assuming consistent casing
between request and response.

## Error / edge cases

- **Invalid `mode` on `/process`**: HTTP 400, synchronous, before any job
  is enqueued:
  ```json
  {"detail": "Invalid mode. Expected one of ['CUSTOM_FIELDS', 'FULL', 'KEY_INSIGHTS', 'SUMMARY']"}
  ```
- **Unknown/never-submitted `pythonJobId` on `/status`**: reads as
  `{"status": "QUEUED"}` indefinitely, not a 404 or distinct "not found"
  error. Don't build any logic that assumes this will eventually resolve
  on its own -- if a job id you're polling was never a real submission
  (e.g. a bug produced a bad id, or the record is very old), it will look
  identical to a job that's simply still queued, forever.
- **A corrupt/invalid/unsupported document upload** -- verified with a
  real test (a plain text file renamed to `.pdf`): the job completes with
  a genuine `FAILED` status and a clear error message:
  ```json
  {
      "status": "FAILED",
      "error": "No text or tables could be extracted from this document -- it may be corrupt, empty, or an unsupported format."
  }
  ```
  This is deliberately raised by the pipeline when *both* `extractedText`
  and `tables` come back empty -- earlier in testing, before this check
  existed, a bad upload instead completed "successfully" with empty
  content and a confused LLM response (e.g. `{"result": "Sure, please
  provide the text that needs cleaning up..."}` for FULL mode). If you
  ever see an empty `extractedText` with a `COMPLETED` status against the
  current code, that indicates a genuinely unusual case slipping past this
  check, not expected behavior -- worth investigating rather than assuming
  it's fine.