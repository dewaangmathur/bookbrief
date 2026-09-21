"""
pdf_utils.py
Extract chapter-wise text from a PDF book.

Strategy:
1. Try to read the embedded Table of Contents (bookmarks) via PyMuPDF.
   Use the shallowest bookmark level as "chapters" (exact titles preserved).
2. If no TOC exists, fall back to a regex-based heuristic that looks for
   lines like "Chapter 1", "CHAPTER ONE", "1. Introduction", etc. on their
   own line near the top of a page.
3. If neither works, the whole book is treated as a single chapter.
"""

import re
import fitz  # PyMuPDF


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


def _get_toc_chapters(doc):
    """
    Builds a flat list of "leaf" chapters to summarize, each tagged with its
    parent Part title (if the book's bookmarks are nested, e.g. Part I >
    Chapter 1, Chapter 2 ... Part II > Chapter 4, ...).

    A bookmark entry is treated as a "Part" (group header, not summarized
    directly) if the very next bookmark entry is nested one level deeper
    under it. Anything else (including top-level entries with no children,
    e.g. a standalone "Bibliography") becomes a leaf chapter.
    """
    toc = doc.get_toc(simple=True)  # [[level, title, page], ...] page is 1-indexed
    if not toc:
        return None

    n = len(toc)
    n_pages = doc.page_count
    min_level = min(entry[0] for entry in toc)

    # end page for every entry: right before the next entry at the same or
    # shallower level
    ends = []
    for i, (level, _title, _page) in enumerate(toc):
        end = n_pages - 1
        for j in range(i + 1, n):
            if toc[j][0] <= level:
                end = toc[j][2] - 1 - 1  # next entry's start (0-idx) - 1
                break
        ends.append(end)

    leaves = []
    current_part = None
    leaf_count = 0

    for i, (level, title, page) in enumerate(toc):
        start = max(page - 1, 0)
        end = max(ends[i], start)
        has_child = (i + 1 < n) and (toc[i + 1][0] > level)

        if level == min_level:
            if has_child:
                # this is a Part / group header - don't summarize it directly
                current_part = title.strip()
                continue
            else:
                # standalone top-level item (no children) - not under any part
                current_part = None
                leaves.append({
                    "title": title.strip(),
                    "part": None,
                    "start_page": start,
                    "end_page": end,
                })
                leaf_count += 1
        else:
            if has_child:
                # a nested item that itself has children (rare, 3+ levels
                # deep) - skip as a group header too, keep current_part as-is
                continue
            leaves.append({
                "title": title.strip(),
                "part": current_part,
                "start_page": start,
                "end_page": end,
            })
            leaf_count += 1

    if leaf_count < 1:
        return None
    return leaves


def _get_regex_chapters(doc):
    n_pages = doc.page_count
    candidates = []  # (page_idx, title)

    for pno in range(n_pages):
        page = doc[pno]
        text = page.get_text("text")
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        # only look at first ~6 non-empty lines of the page (chapter headings
        # are typically near the top)
        for line in lines[:6]:
            for rx in CHAPTER_REGEXES:
                if rx.match(line) and len(line) < 100:
                    candidates.append((pno, line))
                    break
            else:
                continue
            break

    if len(candidates) < 2:
        return None

    # de-duplicate consecutive pages picking up same chapter start
    dedup = []
    seen_pages = set()
    for pno, title in candidates:
        if pno in seen_pages:
            continue
        seen_pages.add(pno)
        dedup.append((pno, title))

    chapters = []
    for i, (pno, title) in enumerate(dedup):
        start = pno
        end = (dedup[i + 1][0] - 1) if i + 1 < len(dedup) else n_pages - 1
        if end < start:
            end = start
        chapters.append({"title": title, "part": None, "start_page": start, "end_page": end})
    return chapters


def extract_chapters(pdf_path: str):
    """
    Returns a list of dicts:
      { "title": str, "start_page": int(0idx), "end_page": int(0idx),
        "num_pages": int, "text": str }
    """
    doc = fitz.open(pdf_path)
    n_pages = doc.page_count

    chapters = _get_toc_chapters(doc)
    source = "toc"
    if not chapters:
        chapters = _get_regex_chapters(doc)
        source = "regex"
    if not chapters:
        chapters = [{"title": "Full Book", "part": None, "start_page": 0, "end_page": n_pages - 1}]
        source = "fallback_single"

    for ch in chapters:
        pages_text = []
        for pno in range(ch["start_page"], ch["end_page"] + 1):
            pages_text.append(doc[pno].get_text("text"))
        ch["text"] = _clean("\n".join(pages_text))
        ch["num_pages"] = ch["end_page"] - ch["start_page"] + 1

    doc.close()
    return chapters, source, n_pages
