"""Normalize extracted text and remove recurring document boilerplate."""

import argparse
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

PAGE_MARKER = re.compile(r"^\s*(?:page\s+)?\d+\s*(?:of\s+\d+)?\s*$", re.IGNORECASE)
PAGE_COUNT = re.compile(r"^\s*page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE)
NAVIGATION = re.compile(
    r"^\s*(?:home\s*[|>›]\s*)+(?:about|contact|help|login|search|claims?)\s*$",
    re.IGNORECASE,
)
ARTIFACTS = str.maketrans({"\ufeff": "", "\u200b": "", "\u00ad": "", "\ufffd": " "})


def normalize_text(text: str) -> str:
    """Normalize Unicode, whitespace, and line wraps without joining paragraphs."""
    text = unicodedata.normalize("NFKC", text).translate(ARTIFACTS)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n+", text)
    normalized_blocks = []

    for block in blocks:
        lines = [re.sub(r"\s+", " ", line).strip() for line in block.splitlines()]
        lines = [line for line in lines if line]
        if not lines:
            continue

        joined = lines[0]
        for line in lines[1:]:
            if joined.endswith("-") and line[:1].islower():
                joined = joined[:-1] + line
            else:
                joined += " " + line
        normalized_blocks.append(joined)

    return "\n\n".join(normalized_blocks)


def _edge_lines(text: str) -> set[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return set(lines[:3] + lines[-3:])


def _repeated_edge_lines(documents: dict[str, str]) -> set[str]:
    if len(documents) < 2:
        return set()

    counts = Counter(line for text in documents.values() for line in _edge_lines(text))
    threshold = max(2, math.ceil(len(documents) / 2))
    return {
        line
        for line, count in counts.items()
        if count >= threshold and len(line) <= 120
    }


def clean_text(text: str, repeated_lines: set[str] | None = None) -> str:
    """Clean a document, optionally removing edge boilerplate shared by its corpus."""
    repeated_lines = repeated_lines or set()
    kept_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in repeated_lines or PAGE_MARKER.fullmatch(stripped):
            continue
        if PAGE_COUNT.fullmatch(stripped) or NAVIGATION.fullmatch(stripped):
            continue
        kept_lines.append(line)
    return normalize_text("\n".join(kept_lines))


def clean_corpus(input_dir: Path, output_dir: Path) -> int:
    """Clean every UTF-8 text file under input_dir into a mirrored output tree."""
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("Output directory must not be inside the input directory")

    source_files = sorted(input_dir.rglob("*.txt"))
    documents = {
        path.relative_to(input_dir).as_posix(): path.read_text(
            encoding="utf-8-sig", errors="replace"
        )
        for path in source_files
    }
    repeated_lines = _repeated_edge_lines(documents)

    for relative_path, text in documents.items():
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(clean_text(text, repeated_lines), encoding="utf-8")

    return len(source_files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean extracted corpus text files.")
    parser.add_argument("input_dir", nargs="?", type=Path, default=Path("data"))
    parser.add_argument(
        "output_dir", nargs="?", type=Path, default=Path("outputs/cleaned_data")
    )
    args = parser.parse_args()
    count = clean_corpus(args.input_dir, args.output_dir)
    print(f"Cleaned {count} text document(s) into {args.output_dir}")


if __name__ == "__main__":
    main()