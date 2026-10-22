"""Split source documents into text chunks with provenance metadata."""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

PAGE_MARKER = re.compile(
    r"^\s*(?:page\s+)?(\d+)(?:\s+of\s+\d+)?\s*$", re.IGNORECASE
)
SECTION_HEADING = re.compile(
    r"^(?:section\s+[ivx0-9]+|coverage\s+[a-z0-9]+)"
    r"(?:\s*[-–—:]\s*.+)?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ChunkMetadata:
    """Stable provenance fields shared by every chunk."""

    source: str
    section: str | None
    page: int | None
    position: int
    start_char: int
    end_char: int


@dataclass(frozen=True)
class Chunk:
    """A source-text slice and its provenance metadata."""

    text: str
    metadata: ChunkMetadata

    def to_dict(self) -> dict[str, object]:
        """Return the consistent text-plus-metadata representation."""
        return {"text": self.text, "metadata": asdict(self.metadata)}


def _paragraph_spans(text: str) -> Iterator[tuple[int, int]]:
    offset = 0
    paragraph_start: int | None = None
    paragraph_end = 0

    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        if content.strip():
            if paragraph_start is None:
                paragraph_start = offset
            paragraph_end = offset + len(content)
        elif paragraph_start is not None:
            yield paragraph_start, paragraph_end
            paragraph_start = None
        offset += len(line)

    if paragraph_start is not None:
        yield paragraph_start, paragraph_end


def _section_heading(text: str) -> str | None:
    heading = text.strip()
    if not heading or len(heading) > 120:
        return None
    if SECTION_HEADING.fullmatch(heading):
        return heading
    if any(character.isalpha() for character in heading) and heading == heading.upper():
        if not any(mark in heading for mark in ".?!"):
            return heading
    return None


def _split_span(
    text: str, start: int, end: int, max_chars: int
) -> Iterator[tuple[int, int]]:
    while end - start > max_chars:
        limit = start + max_chars
        minimum = start + max_chars // 2
        sentence_boundaries = []
        for match in re.finditer(r"[.!?](?=\s|$)", text[minimum:limit]):
            punctuation = minimum + match.start()
            preceding_text = text[max(start, punctuation - 5) : punctuation + 1]
            if match.group() == "." and re.search(
                r"\b(?:[a-z]|\d{1,2})\.$", preceding_text, re.IGNORECASE
            ):
                continue
            sentence_boundaries.append(minimum + match.end())
        clause_boundaries = [
            minimum + match.end()
            for match in re.finditer(r";(?=\s|$)", text[minimum:limit])
        ]
        if sentence_boundaries:
            split_at = sentence_boundaries[-1]
        elif clause_boundaries:
            split_at = clause_boundaries[-1]
        else:
            split_at = limit
            preferred_split = text.rfind(" ", minimum, limit)
            if preferred_split > start:
                split_at = preferred_split + 1
        if split_at < end and text[split_at].isspace():
            split_at += 1
        yield start, split_at
        start = split_at
    if start < end:
        yield start, end


def chunk_document(text: str, source: str, max_chars: int = 500) -> list[Chunk]:
    """Chunk a document while retaining source-relative character positions.

    Positions are zero-based, half-open offsets into ``text``. Section headings
    and page markers update metadata; page-marker lines are not returned as text.
    """
    if not source.strip():
        raise ValueError("source must be a non-empty document identifier")
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")

    chunks: list[Chunk] = []
    section: str | None = None
    page: int | None = None
    pending_start: int | None = None
    pending_end = 0
    pending_section: str | None = None
    pending_page: int | None = None

    def flush() -> None:
        nonlocal pending_start, pending_end
        if pending_start is None:
            return
        chunk_text = text[pending_start:pending_end]
        if chunk_text.strip():
            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata=ChunkMetadata(
                        source=source,
                        section=pending_section,
                        page=pending_page,
                        position=len(chunks),
                        start_char=pending_start,
                        end_char=pending_end,
                    ),
                )
            )
        pending_start = None

    for start, end in _paragraph_spans(text):
        paragraph = text[start:end]
        page_match = PAGE_MARKER.fullmatch(paragraph)
        if page_match:
            flush()
            page = int(page_match.group(1))
            continue

        heading = _section_heading(paragraph)
        if heading is not None:
            flush()
            section = heading

        if end - start > max_chars:
            flush()
            for part_start, part_end in _split_span(text, start, end, max_chars):
                chunks.append(
                    Chunk(
                        text=text[part_start:part_end],
                        metadata=ChunkMetadata(
                            source=source,
                            section=section,
                            page=page,
                            position=len(chunks),
                            start_char=part_start,
                            end_char=part_end,
                        ),
                    )
                )
            continue

        if pending_start is None:
            pending_start = start
            pending_end = end
            pending_section = section
            pending_page = page
        elif (
            pending_section == section
            and pending_page == page
            and end - pending_start <= max_chars
        ):
            pending_end = end
        else:
            flush()
            pending_start = start
            pending_end = end
            pending_section = section
            pending_page = page

    flush()
    return chunks


def chunk_file(
    path: Path, *, source: str | None = None, max_chars: int = 500
) -> list[Chunk]:
    """Read and chunk a UTF-8 text file, defaulting its identifier to its name."""
    document = path.read_text(encoding="utf-8")
    return chunk_document(document, source or path.name, max_chars=max_chars)


def chunk_corpus(input_dir: Path, max_chars: int = 500) -> list[Chunk]:
    """Chunk all UTF-8 text files using their corpus-relative paths as IDs."""
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    chunks = []
    for path in sorted(input_dir.rglob("*.txt")):
        source = path.relative_to(input_dir).as_posix()
        chunks.extend(chunk_file(path, source=source, max_chars=max_chars))
    return chunks


def write_chunks_jsonl(chunks: Iterable[Chunk], output_path: Path) -> None:
    """Write chunks as JSON Lines with text and metadata kept together."""
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for chunk in chunks:
            output.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
