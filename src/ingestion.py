"""Run and account for the complete text-corpus ingestion pipeline."""

import argparse
import json
import re
from pathlib import Path
from typing import Any

from src.chunking import Chunk, chunk_document
from src.text_cleaner import _repeated_edge_lines, clean_text

CONTROL_FILES = {".gitkeep"}
SENSITIVE_SAMPLE_MARKERS = re.compile(
    r"\b(?:policy number|named insured|insured location|policy period)\b|\b\d{3,}\b",
    re.IGNORECASE,
)


def ingest_corpus(
    input_dir: Path,
    *,
    max_chars: int = 500,
    sample_limit: int = 3,
) -> tuple[dict[str, Any], list[Chunk]]:
    """Load, clean, chunk, and account for every file under ``input_dir``.

    Text document failures are recorded per source and do not prevent other
    documents from being ingested. Non-document control files are reported as
    skipped; all other files participate in the document accounting.
    """
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")
    if sample_limit < 0:
        raise ValueError("sample_limit cannot be negative")

    all_files = sorted(path for path in input_dir.rglob("*") if path.is_file())
    skipped_files = [
        {
            "source": path.relative_to(input_dir).as_posix(),
            "reason": "repository control placeholder",
        }
        for path in all_files
        if path.name in CONTROL_FILES
    ]
    document_files = [path for path in all_files if path.name not in CONTROL_FILES]
    source_documents = len(document_files)

    loaded: dict[str, str] = {}
    failures: list[dict[str, str]] = []
    for path in document_files:
        source = path.relative_to(input_dir).as_posix()
        if path.suffix.lower() != ".txt":
            failures.append(
                {"source": source, "reason": "unsupported document type; expected .txt"}
            )
            continue
        try:
            loaded[source] = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as error:
            failures.append(
                {"source": source, "reason": f"{type(error).__name__}: {error}"}
            )

    repeated_lines = _repeated_edge_lines(loaded)
    document_results: list[dict[str, Any]] = []
    chunks: list[Chunk] = []

    for source, raw_text in loaded.items():
        try:
            cleaned = clean_text(raw_text, repeated_lines)
            if not cleaned.strip():
                raise ValueError("document is empty after cleaning")
            document_chunks = chunk_document(cleaned, source, max_chars=max_chars)
            if not document_chunks:
                raise ValueError("document produced no chunks")
            if any(
                chunk.metadata.source != source
                or cleaned[
                    chunk.metadata.start_char : chunk.metadata.end_char
                ]
                != chunk.text
                for chunk in document_chunks
            ):
                raise ValueError("chunk metadata does not trace to cleaned source")

            chunks.extend(document_chunks)
            document_results.append(
                {
                    "source": source,
                    "status": "ingested",
                    "raw_characters": len(raw_text),
                    "cleaned_characters": len(cleaned),
                    "chunks": len(document_chunks),
                }
            )
        except (OSError, UnicodeError, ValueError) as error:
            failures.append(
                {"source": source, "reason": f"{type(error).__name__}: {error}"}
            )

    ingested_documents = len(document_results)
    accounted_documents = ingested_documents + len(failures)
    accounting_valid = source_documents == accounted_documents
    if not accounting_valid:
        raise RuntimeError(
            f"Document accounting mismatch: {source_documents} source documents, "
            f"{ingested_documents} ingested, {len(failures)} failures"
        )

    selected_samples: list[Chunk] = []
    sampled_sources: set[str] = set()
    if sample_limit:
        for chunk in chunks:
            if (
                chunk.metadata.source not in sampled_sources
                and chunk.metadata.section is not None
                and chunk.metadata.section.casefold().startswith("coverage")
                and chunk.text.rstrip().endswith((".", "!", "?"))
                and not SENSITIVE_SAMPLE_MARKERS.search(chunk.text)
            ):
                selected_samples.append(chunk)
                sampled_sources.add(chunk.metadata.source)
                if len(selected_samples) == sample_limit:
                    break

    summary = {
        "input_directory": input_dir.name,
        "chunk_size_characters": max_chars,
        "total_source_documents": source_documents,
        "successfully_ingested_documents": ingested_documents,
        "failed_documents": len(failures),
        "total_chunks_created": len(chunks),
        "skipped_files": skipped_files,
        "failures": failures,
        "documents": document_results,
        "accounting": {
            "source_documents": source_documents,
            "ingested_plus_failures": accounted_documents,
            "valid": accounting_valid,
        },
        "sample_chunks_saved": len(selected_samples),
    }
    return summary, selected_samples


def run_ingestion(
    input_dir: Path,
    summary_path: Path,
    samples_path: Path,
    *,
    max_chars: int = 500,
    sample_limit: int = 3,
) -> dict[str, Any]:
    """Run ingestion and persist an aggregate summary and review samples."""
    input_dir = input_dir.resolve()
    summary_path = summary_path.resolve()
    samples_path = samples_path.resolve()
    for output_path in (summary_path, samples_path):
        if output_path == input_dir or input_dir in output_path.parents:
            raise ValueError("Ingestion outputs must be outside the input corpus")
    if summary_path == samples_path:
        raise ValueError("Summary and sample output paths must be different")

    summary, samples = ingest_corpus(
        input_dir, max_chars=max_chars, sample_limit=sample_limit
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with samples_path.open("w", encoding="utf-8", newline="\n") as output:
        for chunk in samples:
            output.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load, clean, chunk, and validate every text document in a corpus."
    )
    parser.add_argument("input_dir", nargs="?", type=Path, default=Path("data"))
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/ingestion_summary.json"),
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path("outputs/ingestion_sample_chunks.jsonl"),
    )
    parser.add_argument("--max-chars", type=int, default=500)
    parser.add_argument("--sample-limit", type=int, default=3)
    args = parser.parse_args()

    summary = run_ingestion(
        args.input_dir,
        args.summary,
        args.samples,
        max_chars=args.max_chars,
        sample_limit=args.sample_limit,
    )
    print(
        "Ingestion complete: "
        f"{summary['successfully_ingested_documents']}/"
        f"{summary['total_source_documents']} documents, "
        f"{summary['total_chunks_created']} chunks, "
        f"{summary['failed_documents']} failures, "
        f"{len(summary['skipped_files'])} skipped control files."
    )
    print(f"Summary: {args.summary}")
    print(f"Sample chunks: {args.samples}")
    if not summary["accounting"]["valid"]:
        raise RuntimeError("Ingestion accounting validation failed")


if __name__ == "__main__":
    main()
