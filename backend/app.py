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
CORS(app)


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

    try:
        chapters, source, n_pages = extract_chapters(pdf_path)
        table = [{"title": c["title"], "part": c.get("part"), "pages": c["num_pages"]} for c in chapters]
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

    try:
        chapters, source, n_pages = extract_chapters(pdf_path)
        run_logger = RunLogger(book_name=filename, num_pages=n_pages, source=source)

        # summarize all chapters concurrently, preserving original order in results
        results = [None] * len(chapters)

        def _work(idx, ch):
            summary, in_tok, out_tok, elapsed, chunks_used = summarize_chapter(
                ch["title"], ch["text"], api_key=api_key, num_pages=ch["num_pages"]
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
