import os
import json
import traceback
import concurrent.futures
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

from pdf_utils import extract_chapters
from summarizer import summarize_chapter
from logger_utils import RunLogger, PRICE_INPUT_PER_M, PRICE_OUTPUT_PER_M

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")
MAX_CHAPTER_WORKERS = int(os.getenv("MAX_CHAPTER_WORKERS", "4"))

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB hard limit
CORS(app)


@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"error": "PDF is too large — maximum file size is 50 MB."}), 413


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/books", methods=["GET"])
def list_books():
    """List PDF files currently sitting in backend/data/"""
    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    return jsonify({"books": files})


@app.route("/api/upload", methods=["POST"])
def upload_book():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400
    save_path = os.path.join(DATA_DIR, f.filename)
    f.save(save_path)
    return jsonify({"message": "Uploaded", "filename": f.filename})


@app.route("/api/preview-chapters", methods=["POST"])
def preview_chapters():
    """Quickly detect chapters (no summarization) so the frontend can show
    the chapter/page table before the user commits to running the (paid)
    summarization step."""
    body = request.get_json(force=True)
    filename = body.get("filename")
    if not filename:
        return jsonify({"error": "filename is required"}), 400
    pdf_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(pdf_path):
        return jsonify({"error": f"{filename} not found in backend/data/"}), 404

    # Use API key for LLM-assisted TOC extraction if the user has one saved
    preview_api_key = (request.headers.get("X-OpenAI-Key") or body.get("api_key", "")).strip() or None

    try:
        chapters, source, n_pages = extract_chapters(pdf_path, api_key=preview_api_key)
        table = [{
            "title":      c["title"],
            "part":       c.get("part"),
            "pages":      c["num_pages"],
            "start_page": c["start_page"],
            "end_page":   c["end_page"],
        } for c in chapters]
        return jsonify({
            "filename": filename,
            "total_pages": n_pages,
            "detection_method": source,
            "chapters": table,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/summarize", methods=["POST"])
def summarize_book():
    body = request.get_json(force=True)
    filename = body.get("filename")
    if not filename:
        return jsonify({"error": "filename is required"}), 400
    pdf_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(pdf_path):
        return jsonify({"error": f"{filename} not found in backend/data/"}), 404

    # BYOK: the user's own OpenAI key comes in per-request (header or body).
    # It is used only for this request's API calls and is never written to
    # disk, logs, or any database on this server.
    api_key = request.headers.get("X-OpenAI-Key") or body.get("api_key")
    if not api_key or not api_key.strip():
        return jsonify({"error": "No OpenAI API key provided"}), 400
    api_key = api_key.strip()

    # Optional user-tunable summary density params
    words_per_page = int(body.get("words_per_page", 55))
    max_words      = int(body.get("max_words", 2200))
    # Clamp to sane ranges
    words_per_page = max(10, min(words_per_page, 300))
    max_words      = max(100, min(max_words, 8000))

    # Summary output language
    summary_language = body.get("summary_language", "en").strip().lower()
    if summary_language not in ("en", "de"):
        summary_language = "en"

    # Model selection — whitelist to prevent arbitrary strings hitting OpenAI
    ALLOWED_MODELS = {
        "gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4",
        "gpt-3.5-turbo", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
        "gpt-4.5", "gpt-5",
        "o1", "o1-mini", "o3", "o3-mini", "o4-mini",
    }
    requested_model = body.get("model", "gpt-4o-mini").strip()
    model = requested_model if requested_model in ALLOWED_MODELS else "gpt-4o-mini"

    # If the frontend passes the already-detected chapter list (from preview),
    # skip re-running extract_chapters entirely — just extract text for each
    # chapter using the page ranges we already know. This avoids a redundant
    # LLM TOC call and saves 10-20s on every summarize request.
    cached_chapters = body.get("chapters")  # [{title, start_page, end_page, part}, ...]

    try:
        if cached_chapters and isinstance(cached_chapters, list) and len(cached_chapters) > 0:
            import fitz
            doc = fitz.open(pdf_path)
            n_pages = doc.page_count
            source  = body.get("detection_method", "cached")
            chapters = []
            for c in cached_chapters:
                start = int(c.get("start_page", 0))
                end   = int(c.get("end_page", n_pages - 1))
                pages_text = [doc[p].get_text("text") for p in range(start, end + 1)]
                chapters.append({
                    "title":      c.get("title", ""),
                    "part":       c.get("part"),
                    "start_page": start,
                    "end_page":   end,
                    "num_pages":  end - start + 1,
                    "text":       "\n".join(pages_text),
                })
            doc.close()
        else:
            chapters, source, n_pages = extract_chapters(pdf_path, api_key=api_key)

        run_logger = RunLogger(book_name=filename, num_pages=n_pages, source=source)

        # summarize all chapters concurrently, preserving original order in results
        results = [None] * len(chapters)

        def _work(idx, ch):
            summary, in_tok, out_tok, elapsed, chunks_used = summarize_chapter(
                ch["title"], ch["text"], api_key=api_key, num_pages=ch["num_pages"],
                words_per_page=words_per_page, max_words=max_words, model=model,
                summary_language=summary_language,
            )
            return idx, ch, summary, in_tok, out_tok, elapsed, chunks_used

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CHAPTER_WORKERS) as pool:
            futures = [pool.submit(_work, i, ch) for i, ch in enumerate(chapters)]
            for fut in concurrent.futures.as_completed(futures):
                idx, ch, summary, in_tok, out_tok, elapsed, chunks_used = fut.result()
                run_logger.log_chapter(
                    title=ch["title"],
                    num_pages=ch["num_pages"],
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    seconds=elapsed,
                    chunks_used=chunks_used,
                    part=ch.get("part"),
                    order=idx,
                )
                results[idx] = {
                    "title": ch["title"],
                    "part": ch.get("part"),
                    "pages": ch["num_pages"],
                    "summary": summary,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "time_seconds": round(elapsed, 2),
                    "cost_usd": round(
                        (in_tok / 1_000_000 * PRICE_INPUT_PER_M) + (out_tok / 1_000_000 * PRICE_OUTPUT_PER_M),
                        6,
                    ),
                }

        summary_stats, json_log_path, txt_log_path = run_logger.finalize(logs_dir=LOGS_DIR)

        # also save the full summary as a standalone json for convenience
        out_path = os.path.join(BASE_DIR, "output", f"{os.path.splitext(filename)[0]}_summary.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"filename": filename, "chapters": results, "stats": summary_stats}, f, indent=2)

        return jsonify({
            "filename": filename,
            "detection_method": source,
            "total_pages": n_pages,
            "chapters": results,
            "stats": summary_stats,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/export/docx", methods=["POST"])
def export_docx():
    """Generate and return a Word (.docx) document from summary data."""
    import io
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from flask import send_file

    body = request.get_json(silent=True) or {}
    data = body.get("data")
    if not data:
        return jsonify({"error": "No summary data provided"}), 400

    doc = Document()

    # Page margins
    for section in doc.sections:
        section.top_margin    = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin   = Inches(1.2)
        section.right_margin  = Inches(1.2)

    # Styles
    style = doc.styles['Normal']
    style.font.name = 'Georgia'
    style.font.size = Pt(11)

    # ── Cover info ───────────────────────────────────────────────────────
    book_name = (data.get("filename") or "Book Summary").replace(".pdf", "")
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = t.add_run("BOOKBRIEF — CHAPTER SUMMARY REPORT")
    run.font.size = Pt(8)
    run.font.color.rgb = RGBColor(0x6f, 0x66, 0x56)

    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    title_run = title_p.add_run(book_name)
    title_run.bold = True
    title_run.font.size = Pt(20)
    title_run.font.name = 'Georgia'

    s = data.get("stats", {})
    meta = doc.add_paragraph()
    meta.add_run(f"Chapters: {len(data.get('chapters', []))}   ·   "
                 f"Pages: {data.get('total_pages', '—')}   ·   "
                 f"Model: {s.get('model', 'gpt-4o-mini')}   ·   "
                 f"Cost: ${s.get('total_cost_usd', 0):.4f}")
    meta.runs[0].font.size = Pt(9)
    meta.runs[0].font.color.rgb = RGBColor(0x6f, 0x66, 0x56)

    doc.add_paragraph()  # spacer

    # ── Table of contents ────────────────────────────────────────────────
    toc_heading = doc.add_paragraph()
    toc_run = toc_heading.add_run("Contents")
    toc_run.bold = True
    toc_run.font.size = Pt(14)

    for i, ch in enumerate(data.get("chapters", [])):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.1)
        r = p.add_run(f"{str(i+1).zfill(2)}  {ch.get('title', '')}")
        r.font.size = Pt(10)

    doc.add_page_break()

    # ── Chapters ─────────────────────────────────────────────────────────
    for i, ch in enumerate(data.get("chapters", [])):
        # Chapter heading
        h = doc.add_paragraph()
        num_run = h.add_run(f"{str(i+1).zfill(2)} / {len(data['chapters'])}   ")
        num_run.font.size = Pt(9)
        num_run.font.color.rgb = RGBColor(0xA9, 0x78, 0x2F)
        title_run = h.add_run(ch.get("title", ""))
        title_run.bold = True
        title_run.font.size = Pt(15)

        # Meta
        m = doc.add_paragraph()
        m.add_run(f"{ch.get('pages', 0)} pages  ·  ${ch.get('cost_usd', 0):.4f}")
        m.runs[0].font.size = Pt(8)
        m.runs[0].font.color.rgb = RGBColor(0x6f, 0x66, 0x56)

        # Summary body
        for para_text in (ch.get("summary") or "").split("\n\n"):
            if para_text.strip():
                p = doc.add_paragraph(para_text.strip())
                p.paragraph_format.space_after = Pt(6)
                p.paragraph_format.line_spacing = Pt(16)

        doc.add_paragraph()  # spacing between chapters

    # ── Save to buffer ───────────────────────────────────────────────────
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)

    safe_name = book_name.replace(" ", "_")[:60]
    return send_file(
        buf,
        as_attachment=True,
        download_name=f"{safe_name}_summary.docx",
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


@app.route("/api/logs", methods=["GET"])
def list_logs():
    files = sorted(
        [f for f in os.listdir(LOGS_DIR) if f.endswith(".json")], reverse=True
    )
    return jsonify({"logs": files})


@app.route("/api/logs/<path:filename>", methods=["GET"])
def get_log(filename):
    return send_from_directory(LOGS_DIR, filename)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
