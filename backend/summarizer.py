"""
summarizer.py
Summarizes chapter text using OpenAI gpt-4o-mini.
Long chapters are split into token-bounded chunks, each chunk is summarized,
then the chunk summaries are combined into one final chapter summary
(map-reduce). Tracks input/output token usage via the API response.
"""

import os
import time
import concurrent.futures
import tiktoken
from openai import OpenAI

MODEL = os.getenv("SUMMARY_MODEL", "gpt-4o-mini")
MAX_CHUNK_TOKENS = int(os.getenv("MAX_CHUNK_TOKENS", "6000"))
MAX_CHUNK_WORKERS = int(os.getenv("MAX_CHUNK_WORKERS", "6"))

# NOTE: no global OpenAI client here anymore. Each request brings its own
# API key (from the user's browser), so a fresh client is built per-call
# in summarize_chapter() and discarded right after - never stored.

try:
    _enc = tiktoken.encoding_for_model("gpt-4o-mini")
except Exception:
    _enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def chunk_text(text: str, max_tokens: int = MAX_CHUNK_TOKENS):
    tokens = _enc.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(_enc.decode(chunk_tokens))
    return chunks if chunks else [""]


def _call_openai(client, system_prompt: str, user_prompt: str, max_tokens: int = 1000):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=max_tokens,
    )
    text = resp.choices[0].message.content.strip()
    usage = resp.usage
    return text, usage.prompt_tokens, usage.completion_tokens


CHUNK_SYSTEM_PROMPT = (
    "You are an expert book summarizer. Summarize the given excerpt of a "
    "book chapter clearly and faithfully, preserving key ideas, arguments, "
    "characters, and events. Do not add outside information. Be thorough: "
    "make sure every distinct idea, argument, event, or claim in the excerpt "
    "is represented in the summary, not just the first or most obvious ones."
)

FINAL_SYSTEM_PROMPT = (
    "You are an expert book summarizer. You will be given partial summaries "
    "of consecutive sections of one chapter. Combine them into a single, "
    "coherent, well-structured chapter summary in flowing prose with clear "
    "paragraphs. Preserve every distinct point, argument, or event from the "
    "partial summaries so nothing is lost - do not compress by dropping "
    "content, only by removing redundancy. Keep the narrative/argument "
    "order, and do not mention that this was made from parts."
)


def _target_word_count(num_pages: int) -> int:
    """
    Scales the target summary length with how many pages the chapter/section
    actually has, so a 50-page chapter isn't reduced to the same length
    as a 5-page one.
    """
    words = int(num_pages * 55)
    return max(110, min(words, 2200))


def _max_tokens_for_words(word_count: int) -> int:
    # ~1.4 tokens per word is a safe upper bound for English prose,
    # plus headroom so the model doesn't get cut off mid-sentence.
    return min(int(word_count * 1.6) + 120, 4096)


def summarize_chapter(title: str, text: str, api_key: str, num_pages: int = None, logger=None):
    """
    api_key: the user's own OpenAI key, passed in per-request. A fresh
    client is built here and never persisted anywhere.
    Returns (summary_text, input_tokens, output_tokens, elapsed_seconds, chunks_used)
    """
    client = OpenAI(api_key=api_key)
    start = time.time()
    total_in, total_out = 0, 0

    if not text.strip():
        return "(No extractable text found for this chapter.)", 0, 0, 0.0, 0

    if num_pages is None:
        # rough fallback estimate: ~500 words/page, ~1.4 tokens/word
        num_pages = max(1, count_tokens(text) // 700)

    chunks = chunk_text(text)
    chapter_target_words = _target_word_count(num_pages)

    if len(chunks) == 1:
        max_tok = _max_tokens_for_words(chapter_target_words)
        user_prompt = (
            f"Chapter title: {title}\n"
            f"Target length: approximately {chapter_target_words} words - use the "
            f"full length available to cover every point in the text; do not "
            f"cut it short.\n\n"
            f"Chapter text:\n{chunks[0]}"
        )
        summary, in_tok, out_tok = _call_openai(client, CHUNK_SYSTEM_PROMPT, user_prompt, max_tok)
        total_in += in_tok
        total_out += out_tok
        elapsed = time.time() - start
        return summary, total_in, total_out, elapsed, 1

    # map step - run all chunk summaries concurrently
    # give each chunk a proportional slice of the overall word budget
    per_chunk_words = max(90, chapter_target_words // len(chunks))
    chunk_max_tok = _max_tokens_for_words(per_chunk_words)

    partial_summaries = [None] * len(chunks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CHUNK_WORKERS) as pool:
        future_to_idx = {}
        for idx, chunk in enumerate(chunks):
            user_prompt = (
                f"Chapter title: {title}\nSection {idx + 1} of {len(chunks)}\n"
                f"Target length: approximately {per_chunk_words} words - cover "
                f"every distinct point in this section, don't skip any.\n\n"
                f"Text:\n{chunk}"
            )
            fut = pool.submit(_call_openai, client, CHUNK_SYSTEM_PROMPT, user_prompt, chunk_max_tok)
            future_to_idx[fut] = idx

        for fut in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[fut]
            summ, in_tok, out_tok = fut.result()
            partial_summaries[idx] = summ
            total_in += in_tok
            total_out += out_tok

    # reduce step
    combined = "\n\n".join(
        f"Section {i + 1} summary:\n{s}" for i, s in enumerate(partial_summaries)
    )
    final_max_tok = _max_tokens_for_words(chapter_target_words)
    final_prompt = (
        f"Chapter title: {title}\n"
        f"Target length: approximately {chapter_target_words} words for the "
        f"final combined summary - use the full length to preserve every "
        f"point from the section summaries below.\n\n{combined}"
    )
    final_summary, in_tok, out_tok = _call_openai(client, FINAL_SYSTEM_PROMPT, final_prompt, final_max_tok)
    total_in += in_tok
    total_out += out_tok

    elapsed = time.time() - start
    return final_summary, total_in, total_out, elapsed, len(chunks)
