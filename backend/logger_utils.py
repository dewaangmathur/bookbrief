"""
logger_utils.py
Tracks per-chapter stats (pages, tokens, cost, time) and writes a
run log (json + a human-readable txt table) to backend/logs/.
"""

import json
import os
import time
from datetime import datetime

# gpt-4o-mini pricing (USD per 1M tokens) - update here if pricing changes
PRICE_INPUT_PER_M = float(os.getenv("PRICE_INPUT_PER_M", "0.15"))
PRICE_OUTPUT_PER_M = float(os.getenv("PRICE_OUTPUT_PER_M", "0.60"))


class RunLogger:
    def __init__(self, book_name: str, num_pages: int, source: str):
        self.book_name = book_name
        self.num_pages = num_pages
        self.source = source
        self.started_at = time.time()
        self.chapters = []  # list of per-chapter dict stats
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def log_chapter(self, title, num_pages, input_tokens, output_tokens, seconds, chunks_used=1, part=None, order=None):
        cost = (input_tokens / 1_000_000 * PRICE_INPUT_PER_M) + \
               (output_tokens / 1_000_000 * PRICE_OUTPUT_PER_M)
        entry = {
            "chapter": title,
            "part": part,
            "pages": num_pages,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 6),
            "time_seconds": round(seconds, 2),
            "chunks_used": chunks_used,
            "_order": order if order is not None else len(self.chapters),
        }
        self.chapters.append(entry)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        return entry

    def finalize(self, logs_dir="logs"):
        os.makedirs(logs_dir, exist_ok=True)
        self.chapters.sort(key=lambda c: c.get("_order", 0))
        for c in self.chapters:
            c.pop("_order", None)
        total_time = time.time() - self.started_at
        total_cost = (self.total_input_tokens / 1_000_000 * PRICE_INPUT_PER_M) + \
                     (self.total_output_tokens / 1_000_000 * PRICE_OUTPUT_PER_M)

        summary = {
            "book_name": self.book_name,
            "total_pages": self.num_pages,
            "chapter_detection_method": self.source,
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "total_time_seconds": round(total_time, 2),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(total_cost, 6),
            "model": os.getenv("SUMMARY_MODEL", "gpt-4o-mini"),
            "chapters": self.chapters,
        }

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(logs_dir, f"run_{ts}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        txt_path = os.path.join(logs_dir, f"run_{ts}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Book: {self.book_name}\n")
            f.write(f"Total pages: {self.num_pages}  |  Chapter detection: {self.source}\n")
            f.write(f"Run at: {summary['run_timestamp']}\n")
            f.write(f"Total time: {summary['total_time_seconds']}s\n")
            f.write(f"Model: {summary['model']}\n\n")
            f.write(f"{'Part':<20}{'Chapter':<38}{'Pages':>7}{'In Tok':>9}{'Out Tok':>9}{'Cost($)':>9}{'Time(s)':>8}\n")
            f.write("-" * 100 + "\n")
            for c in self.chapters:
                part = (c.get("part") or "-")
                part = (part[:17] + "...") if len(part) > 20 else part
                title = (c["chapter"][:35] + "...") if len(c["chapter"]) > 38 else c["chapter"]
                f.write(f"{part:<20}{title:<38}{c['pages']:>7}{c['input_tokens']:>9}{c['output_tokens']:>9}"
                        f"{c['cost_usd']:>9.4f}{c['time_seconds']:>8.2f}\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'TOTAL':<58}{self.num_pages:>7}{self.total_input_tokens:>9}"
                    f"{self.total_output_tokens:>9}{summary['total_cost_usd']:>9.4f}"
                    f"{summary['total_time_seconds']:>8.2f}\n")

        return summary, json_path, txt_path
