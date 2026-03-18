"""
pipeline/stage2_chunk.py — Split raw articles into overlapping chunks.

No LLM. Free stage.
Strategy:
  1. Split by == Section == headings
  2. Further split sections > max_tokens into overlapping windows
"""
import json
import logging
import os
import re
import uuid
from typing import List

from pipeline.checkpoint import CheckpointManager
from pipeline.config import settings
from pipeline.schema import Chunk

logger = logging.getLogger(__name__)

MAX_TOKENS = 400   # max tokens per chunk (approximated as words)
MIN_TOKENS = 80    # merge or skip chunks below this
OVERLAP    = 50    # overlap in tokens between adjacent chunks
SECTION_RE = re.compile(r"^={2,3}\s*(.+?)\s*={2,3}\s*$", re.MULTILINE)

# Navigation sections produce worthless assertions — skip them
SKIP_SECTIONS = {
    "see also", "references", "external links", "further reading",
    "notes", "bibliography", "footnotes", "citations", "sources",
    "works cited", "related articles", "navigation menu",
}


def should_skip_section(title: str) -> bool:
    return title.strip().lower() in SKIP_SECTIONS


def is_navigation_chunk(text: str) -> bool:
    """True if chunk is a list of short terms, not substantive prose."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return True
    if len(lines) < 3:
        # Only navigation if total word count is tiny (a few labels, not a paragraph)
        return sum(len(l.split()) for l in lines) <= 4
    short_lines = sum(1 for l in lines if len(l.split()) <= 4)
    return (short_lines / len(lines)) > 0.6


# ─── Chunking utilities (importable for tests) ────────────────────────────────

def split_by_sections(text: str) -> List[str]:
    """Split text on == Heading == markers. Returns list of section texts."""
    parts = SECTION_RE.split(text)
    # parts alternates: [pre_text, heading1, body1, heading2, body2, ...]
    sections: List[str] = []
    if parts[0].strip():
        sections.append(parts[0].strip())
    i = 1
    while i + 1 < len(parts):
        heading = parts[i].strip()
        body    = parts[i + 1].strip()
        if body:
            sections.append(f"{heading}\n{body}")
        i += 2
    return sections if sections else [text]


def chunk_text(text: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP) -> List[str]:
    """Split text into overlapping windows. Returns list of chunk strings."""
    words = text.split()
    if len(words) <= max_tokens:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += max_tokens - overlap

    return chunks


def chunk_article(document_id: str, article: dict) -> List[Chunk]:
    """Produce a list of Chunk objects from a raw article dict."""
    content = article.get("content", "")
    title   = article.get("title", "")
    sections = split_by_sections(content)

    chunks: List[Chunk] = []
    seq = 0
    for section in sections:
        # Extract section title if present (first line)
        lines = section.strip().splitlines()
        section_title = lines[0].strip() if lines else ""
        section_body  = "\n".join(lines[1:]).strip() if len(lines) > 1 else section

        # Skip navigation/reference sections
        if should_skip_section(section_title):
            continue
        if is_navigation_chunk(section_body or section):
            continue

        # Sub-chunk if the section is too long
        sub_chunks = chunk_text(section_body or section, MAX_TOKENS, OVERLAP)
        for text in sub_chunks:
            if not text.strip():
                continue
            # Skip micro-chunks below minimum
            if len(text.split()) < MIN_TOKENS:
                continue
            chunks.append(Chunk(
                chunk_id       = str(uuid.uuid4()),
                document_id    = document_id,
                sequence       = seq,
                section_title  = section_title,
                content        = text,
                token_estimate = len(text.split()),
            ))
            seq += 1

    return chunks


# ─── Stage runner ─────────────────────────────────────────────────────────────

def run() -> None:
    logging.basicConfig(level=getattr(logging, settings.log_level))

    checkpoint = CheckpointManager("stage2_chunk")
    already_done = checkpoint.get_completed_ids()

    raw_files = [f for f in os.listdir(settings.raw_dir) if f.endswith(".json")]
    work_queue = [f for f in raw_files if f not in already_done]
    checkpoint.set_total(len(raw_files))

    os.makedirs(settings.chunks_dir, exist_ok=True)

    logger.info(f"Stage 2: {len(work_queue)} articles to chunk ({len(already_done)} already done)")

    for filename in work_queue:
        path = os.path.join(settings.raw_dir, filename)
        try:
            with open(path) as f:
                article = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            checkpoint.mark_failed(filename, str(e))
            continue

        document_id = article.get("title", filename).replace(" ", "_")
        chunks = chunk_article(document_id, article)

        out_path = os.path.join(settings.chunks_dir, filename)
        chunk_dicts = []
        for c in chunks:
            chunk_dicts.append({
                "chunk_id":       c.chunk_id,
                "document_id":    document_id,
                "sequence":       c.sequence,
                "section_title":  c.section_title,
                "content":        c.content,
                "token_estimate": c.token_estimate,
                "title":          article.get("title", ""),
                "url":            article.get("url", ""),
                "domain":         article.get("domain", "finance"),
            })

        try:
            with open(out_path, "w") as f:
                json.dump(chunk_dicts, f, indent=2)
            checkpoint.mark_done(filename)
            logger.debug(f"Chunked: {filename} → {len(chunks)} chunks")
        except OSError as e:
            checkpoint.mark_failed(filename, str(e))

    checkpoint.complete()


if __name__ == "__main__":
    run()
