"""
pdf_utils.py — Chapter extraction with 4-tier strategy:

1. Embedded TOC bookmarks via PyMuPDF          (free, instant)
2. LLM reads first 30 pages → extracts TOC     (smart, cheap ~$0.001)
3. Regex scan → LLM cleans/validates output    (hybrid fallback)
4. Auto-split into 50-page chunks              (last resort)
"""

import re
import json
import os
import fitz  # PyMuPDF
from openai import OpenAI

LLM_TOC_MODEL  = "gpt-4o-mini"
TOC_SCAN_PAGES = 30   # how many front pages to send for LLM TOC extraction

CHAPTER_REGEXES = [
    re.compile(r'^\s*(chapter|chap\.?)\s+([ivxlcdm]+|\d+)\b.*$', re.IGNORECASE),
    re.compile(r'^\s*(part)\s+([ivxlcdm]+|\d+)\b.*$', re.IGNORECASE),
    re.compile(r'^\s*\d{1,3}[\.\)]\s+[A-Z][A-Za-z0-9 ,\'\-:]{2,80}$'),
]


def _clean(text: str) -> str:
    text = text.replace('\x00', ' ')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ── Tier 1: embedded bookmarks ────────────────────────────────────────────

def _get_toc_chapters(doc):
    toc = doc.get_toc(simple=True)  # [[level, title, page], ...] 1-indexed
    if not toc:
        return None

    n = len(toc)
    n_pages = doc.page_count
    min_level = min(e[0] for e in toc)

    ends = []
    for i, (level, _title, _page) in enumerate(toc):
        end = n_pages - 1
        for j in range(i + 1, n):
            if toc[j][0] <= level:
                end = toc[j][2] - 2  # next entry start (0-idx) - 1
                break
        ends.append(max(end, 0))

    leaves = []
    current_part = None

    for i, (level, title, page) in enumerate(toc):
        start = max(page - 1, 0)
        end   = max(ends[i], start)
        has_child = (i + 1 < n) and (toc[i + 1][0] > level)

        if level == min_level:
            if has_child:
                current_part = title.strip()
                continue
            else:
                current_part = None
        else:
            if has_child:
                continue

        leaves.append({
            "title":      title.strip(),
            "part":       current_part,
            "start_page": start,
            "end_page":   end,
        })

    return leaves if leaves else None


# ── Tier 2: LLM reads front matter → extracts TOC ────────────────────────

_LLM_TOC_SYSTEM = """You are a book structure parser.
You will be given the first pages of a book (including its table of contents).
Extract every chapter/section that represents a substantial content unit to summarize.

Return ONLY a JSON array, no markdown, no explanation:
[
  {"title": "Chapter title", "page": <1-indexed page number where chapter starts>},
  ...
]

Rules:
- Include chapters, parts, sections — anything a reader would want summarized separately
- Skip: table of contents page itself, index, bibliography, acknowledgements, permissions, 
  list of figures/illustrations, foreword/preface ONLY if very short (< 3 pages)
- For anthologies/readers: each essay/excerpt is its own entry
- Page numbers must be the actual content page number shown in the TOC
- If you cannot find a clear TOC or chapter structure, return: []
"""

def _get_llm_toc_chapters(doc, api_key: str):
    """Send first TOC_SCAN_PAGES pages to LLM, ask it to extract chapter structure."""
    n_pages = doc.page_count
    scan_up_to = min(TOC_SCAN_PAGES, n_pages)

    front_text = ""
    for pno in range(scan_up_to):
        page_text = doc[pno].get_text("text").strip()
        if page_text:
            front_text += f"\n--- Page {pno + 1} ---\n{page_text}"

    if not front_text.strip():
        return None

    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=LLM_TOC_MODEL,
            max_tokens=2000,
            temperature=0,
            messages=[
                {"role": "system", "content": _LLM_TOC_SYSTEM},
                {"role": "user",   "content": front_text[:12000]},  # ~8k tokens safe
            ],
        )
        raw = resp.choices[0].message.content.strip()

        # strip markdown fences if model added them
        raw = re.sub(r'^```[a-z]*\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw)

        entries = json.loads(raw)
        if not isinstance(entries, list) or len(entries) < 2:
            return None

        # map {title, page} → {title, start_page, end_page}
        # page in entries is 1-indexed as printed in book; map to 0-indexed PDF pages
        # We use a heuristic: scan pages around the stated page number for a match,
        # since printed page numbers may not align with PDF page indices.
        chapters = _map_llm_entries_to_pages(doc, entries, n_pages)
        return chapters if chapters and len(chapters) >= 2 else None

    except Exception:
        return None


def _map_llm_entries_to_pages(doc, entries, n_pages):
    """
    Map LLM-extracted {title, page} entries to 0-indexed PDF page ranges.
    The LLM gives us the printed page number from the TOC; we need to find
    the actual PDF page index. Strategy:
    - Compute an offset by scanning a few entries near known pages
    - Fall back to treating the stated page as an approximate 0-indexed hint
      with a ±15 page search window for the chapter heading text
    """
    results = []

    # Try to find the PDF-page offset by looking for the first chapter's text
    # near its stated page number (printed numbers often start after front matter)
    offset = _estimate_page_offset(doc, entries)

    for i, entry in enumerate(entries):
        title     = str(entry.get("title", "")).strip()
        stated_pg = int(entry.get("page", 1))

        # apply offset, clamp to valid range
        pdf_pg = max(0, min(stated_pg - 1 + offset, n_pages - 1))

        # end page = start of next chapter - 1
        if i + 1 < len(entries):
            next_stated = int(entries[i + 1].get("page", stated_pg + 1))
            next_pdf    = max(0, min(next_stated - 1 + offset, n_pages - 1))
            end_pg      = max(pdf_pg, next_pdf - 1)
        else:
            end_pg = n_pages - 1

        if end_pg < pdf_pg:
            end_pg = pdf_pg

        results.append({
            "title":      title,
            "part":       None,
            "start_page": pdf_pg,
            "end_page":   end_pg,
        })

    return results


def _estimate_page_offset(doc, entries):
    """
    Estimate the offset between printed page numbers (from LLM/TOC) and
    0-indexed PDF pages. Many books have roman-numeral front matter so
    chapter 1 starts at PDF page 10+ even though it's printed as page 1.
    We try up to 3 entries and return the most common offset found.
    """
    n_pages = doc.page_count
    offsets = []

    for entry in entries[:5]:
        title     = str(entry.get("title", "")).strip()
        stated_pg = int(entry.get("page", 1))
        # search a window of ±20 pages around the naive 0-indexed guess
        naive = stated_pg - 1
        window_start = max(0, naive - 20)
        window_end   = min(n_pages - 1, naive + 20)

        title_words = [w.lower() for w in title.split() if len(w) > 3][:4]
        if not title_words:
            continue

        for pno in range(window_start, window_end + 1):
            page_text = doc[pno].get_text("text").lower()
            if sum(1 for w in title_words if w in page_text) >= max(1, len(title_words) // 2):
                offsets.append(pno - (stated_pg - 1))
                break

    if not offsets:
        return 0
    # return median offset
    offsets.sort()
    return offsets[len(offsets) // 2]


# ── Tier 3: regex scan → LLM cleanup ─────────────────────────────────────

def _get_regex_chapters(doc):
    n_pages = doc.page_count
    candidates = []

    for pno in range(n_pages):
        page  = doc[pno]
        text  = page.get_text("text")
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        for line in lines[:12]:
            for rx in CHAPTER_REGEXES:
                if rx.match(line) and len(line) < 100:
                    candidates.append((pno, line))
                    break
            else:
                continue
            break

    if len(candidates) < 2:
        return None

    dedup = []
    seen  = set()
    for pno, title in candidates:
        if pno not in seen:
            seen.add(pno)
            dedup.append((pno, title))

    chapters = []
    for i, (pno, title) in enumerate(dedup):
        end = (dedup[i + 1][0] - 1) if i + 1 < len(dedup) else n_pages - 1
        chapters.append({
            "title": title, "part": None,
            "start_page": pno, "end_page": max(pno, end),
        })
    return chapters


_LLM_REGEX_CLEAN_SYSTEM = """You are a book structure validator.
You will be given a list of chapter headings extracted by regex from a book PDF,
along with their page numbers. The list may contain noise: running headers,
repeated titles, page headers, or false positives.

Clean and validate the list. Return ONLY a JSON array:
[
  {"title": "Clean chapter title", "start_page": <0-indexed>, "end_page": <0-indexed>},
  ...
]

Rules:
- Remove obvious noise (page headers, repeated consecutive entries, very short fragments)
- Merge entries that are clearly continuations of the same chapter heading split across lines
- Keep the original 0-indexed page numbers exactly as given
- If the list looks correct already, return it as-is
- If fewer than 2 valid chapters remain after cleaning, return: []
"""

def _llm_clean_regex_chapters(chapters: list, api_key: str):
    """Pass regex-detected chapters through LLM for cleanup and validation."""
    try:
        payload = [{"title": c["title"], "start_page": c["start_page"], "end_page": c["end_page"]}
                   for c in chapters]
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=LLM_TOC_MODEL,
            max_tokens=1500,
            temperature=0,
            messages=[
                {"role": "system", "content": _LLM_REGEX_CLEAN_SYSTEM},
                {"role": "user",   "content": json.dumps(payload)},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r'^```[a-z]*\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw)

        cleaned = json.loads(raw)
        if not isinstance(cleaned, list) or len(cleaned) < 2:
            return None

        # rebuild full chapter dicts with part=None
        result = []
        for c in cleaned:
            result.append({
                "title":      str(c.get("title", "")).strip(),
                "part":       None,
                "start_page": int(c["start_page"]),
                "end_page":   int(c["end_page"]),
            })
        return result

    except Exception:
        return None


# ── Tier 4: auto-split fallback ───────────────────────────────────────────

SPLIT_THRESHOLD = 80
SPLIT_CHUNK     = 50

def _auto_split(chapters, source):
    split = []
    for ch in chapters:
        num = ch["end_page"] - ch["start_page"] + 1
        if num > SPLIT_THRESHOLD:
            start, part_num = ch["start_page"], 1
            while start <= ch["end_page"]:
                end = min(start + SPLIT_CHUNK - 1, ch["end_page"])
                label = (f"Pages {start + 1}–{end + 1}"
                         if ch["title"] == "Full Book"
                         else f"{ch['title']} (Part {part_num})")
                split.append({
                    "title": label, "part": ch.get("part"),
                    "start_page": start, "end_page": end,
                })
                start = end + 1
                part_num += 1
            if source == "fallback_single":
                source = "fallback_split"
        else:
            split.append(ch)
    return split, source


# ── Main entry point ──────────────────────────────────────────────────────

def extract_chapters(pdf_path: str, api_key: str = None):
    """
    Returns (chapters, source, n_pages).
    api_key is optional — if omitted, LLM tiers are skipped.
    """
    doc     = fitz.open(pdf_path)
    n_pages = doc.page_count

    # Tier 1: embedded bookmarks
    chapters = _get_toc_chapters(doc)
    source   = "toc"

    # Tier 2: LLM reads front 30 pages
    if not chapters and api_key:
        chapters = _get_llm_toc_chapters(doc, api_key)
        source   = "llm_toc"

    # Tier 3a: regex scan
    if not chapters:
        regex_chapters = _get_regex_chapters(doc)
        if regex_chapters:
            # Tier 3b: LLM cleans regex output
            if api_key:
                cleaned = _llm_clean_regex_chapters(regex_chapters, api_key)
                chapters = cleaned if cleaned else regex_chapters
                source   = "llm_regex" if cleaned else "regex"
            else:
                chapters = regex_chapters
                source   = "regex"

    # Tier 4: single-chapter fallback
    if not chapters:
        chapters = [{"title": "Full Book", "part": None,
                     "start_page": 0, "end_page": n_pages - 1}]
        source   = "fallback_single"

    # Auto-split any chapter > 80 pages
    chapters, source = _auto_split(chapters, source)

    # Extract text for each chapter
    for ch in chapters:
        pages_text = []
        for pno in range(ch["start_page"], ch["end_page"] + 1):
            pages_text.append(doc[pno].get_text("text"))
        ch["text"]      = _clean("\n".join(pages_text))
        ch["num_pages"] = ch["end_page"] - ch["start_page"] + 1

    doc.close()
    return chapters, source, n_pages
