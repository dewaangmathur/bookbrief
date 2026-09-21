# BookBrief — Chapter-Wise Book Summarizer

Turn a book PDF into a dynamic, chapter-wise summary using GPT-4o-mini,
with full logs of time, tokens, and cost.

**Bring your own OpenAI API key** — it's entered in the browser and stored
only in your own `localStorage`. It's never written to this server's disk,
`.env`, logs, or any database; it's sent per-request and used in memory for
that one call only.

## How it works

1. **Extract chapters** — `backend/pdf_utils.py` reads the PDF's embedded
   table of contents (bookmarks) via PyMuPDF to get **exact chapter titles**
   and their page ranges. If a PDF has no bookmarks, it falls back to a
   regex heuristic ("Chapter 1", "Part II", etc.), and if that also fails,
   treats the whole book as one chapter.
2. **Summarize** — `backend/summarizer.py` sends each chapter's text to
   `gpt-4o-mini`. Long chapters are split into ~6000-token chunks,
   summarized individually, then combined into one coherent chapter
   summary (map-reduce), so it scales to a 50-page chapter just as well as
   a 5-page one.
3. **Log** — `backend/logger_utils.py` records, per chapter and in total:
   time taken, input/output tokens, and cost (using gpt-4o-mini pricing,
   configurable in `.env`). Every run writes a `.json` and a readable
   `.txt` table to `backend/logs/`.
4. **Frontend** — a single-page app (`frontend/index.html`) lets you
   upload a PDF, preview the detected table of contents (page counts per
   chapter) before spending any tokens, run the summarization, and read
   the result as a collapsible, book-like accordion.

## 1. Setup

```bash
cd book-summarizer/backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Copy the env template (no OpenAI key needed here anymore — see below):

```bash
cp .env.example .env
```

## 2. Where to put your book

Place your PDF here:

```
book-summarizer/backend/data/book.pdf
```

(Any filename works — you can also upload directly from the frontend,
which saves into the same folder.)

## 3. Run it

```bash
cd book-summarizer/backend
python app.py
```

Then open **http://localhost:5000** in your browser. The Flask app serves
both the API and the frontend, so that's the only URL you need.

In the UI:
1. Paste your OpenAI API key into the field at the top and click **"Save
   key"** — it's saved to your browser's `localStorage` only, and stays
   there across refreshes/restarts until you clear it. Get a key at
   [platform.openai.com/api-keys](https://platform.openai.com/api-keys).
2. Select `book.pdf` from the dropdown (or upload a new one).
3. Review the auto-detected table of contents (chapter names + page
   counts) — this step is free, no API calls yet.
4. Click **"Summarize this book"**. This is when the OpenAI calls happen,
   using your saved key.
5. Read the result: click any chapter to expand/collapse its summary.
   The stats bar at the top shows total time, tokens, and cost for the
   run.

## 4. Logs

Every summarization run writes to `backend/logs/`:
- `run_<timestamp>.json` — machine-readable, chapter list with tokens/cost/time
- `run_<timestamp>.txt` — human-readable table, e.g.:

```
Chapter                                        Pages    In Tok   Out Tok   Cost($)  Time(s)
-------------------------------------------------------------------------------------------
Chapter 1: The Beginning                          50      9142      612     0.0021    14.32
Chapter 2: Rising Action                          20      3781      340     0.0008     6.10
-------------------------------------------------------------------------------------------
TOTAL                                             70     12923      952     0.0029    20.42
```

A copy of the full summary text is also saved to
`backend/output/<book>_summary.json`.

## Notes / things you can tune

- `SUMMARY_MODEL` in `.env` — swap models if you want (must be an OpenAI
  chat model).
- `MAX_CHUNK_TOKENS` — lower it if you hit context-length issues on very
  dense chapters; raise it to reduce the number of API calls (and cost)
  per chapter.
- `PRICE_INPUT_PER_M` / `PRICE_OUTPUT_PER_M` — update if OpenAI's pricing
  changes, so cost figures in the logs stay accurate.
- Chapter detection quality depends on the PDF. Scanned/image-only PDFs
  have no extractable text layer — you'd need OCR first (not included
  here) since PyMuPDF only pulls text that's actually embedded in the PDF.
