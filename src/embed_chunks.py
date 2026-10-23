"""Generate and persist embeddings for chunks from a UTF-8 text document."""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from openai import OpenAI

from src.chunking import Chunk, ChunkMetadata, chunk_document
from src.config import BASE_DIR, config

DEFAULT_INPUT = BASE_DIR / "examples" / "chunk_metadata_source.txt"
DEFAULT_OUTPUT = config.OUTPUTS_DIR / "embedded_chunks.jsonl"
SAMPLE_VALUE_COUNT = 5


@dataclass(frozen=True)
class EmbeddedChunk:
    """A chunk with its source metadata and complete embedding vector."""

    text: str
    metadata: ChunkMetadata
    embedding: list[float]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable text, metadata, and vector record."""
        return {
            "text": self.text,
            "metadata": asdict(self.metadata),
            "embedding": self.embedding,
        }


def embed_chunks(
    chunks: Sequence[Chunk], *, client: OpenAI, model: str
) -> list[EmbeddedChunk]:
    """Embed chunks in one API request and preserve their source ordering."""
    if not chunks:
        raise ValueError("At least one text chunk is required")
    if not model.strip():
        raise ValueError("Embedding model must be a non-empty string")

    response = client.embeddings.create(
        model=model,
        input=[chunk.text for chunk in chunks],
    )
    if len(response.data) != len(chunks):
        raise RuntimeError(
            f"Embedding API returned {len(response.data)} vectors "
            f"for {len(chunks)} chunks"
        )

    vectors: list[list[float] | None] = [None] * len(chunks)
    for item in response.data:
        if item.index < 0 or item.index >= len(chunks) or vectors[item.index] is not None:
            raise RuntimeError(f"Embedding API returned invalid index {item.index}")
        vector = [float(value) for value in item.embedding]
        if not vector or any(not math.isfinite(value) for value in vector):
            raise RuntimeError(f"Embedding API returned an invalid vector at index {item.index}")
        vectors[item.index] = vector

    if any(vector is None for vector in vectors):
        raise RuntimeError("Embedding API response omitted one or more chunk vectors")
    dimensions = {len(vector) for vector in vectors if vector is not None}
    if len(dimensions) != 1:
        raise RuntimeError("Embedding API returned vectors with inconsistent dimensions")

    return [
        EmbeddedChunk(chunk.text, chunk.metadata, vector)
        for chunk, vector in zip(chunks, vectors)
        if vector is not None
    ]


def write_embeddings(records: Sequence[EmbeddedChunk], output_path: Path) -> None:
    """Persist complete vectors alongside chunk text and metadata as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def write_sample_report(
    records: Sequence[EmbeddedChunk], output_path: Path, *, model: str
) -> None:
    """Save a compact, human-reviewable report with trimmed vector previews."""
    if not records:
        raise ValueError("At least one embedded chunk is required for a sample report")
    dimensions = {len(record.embedding) for record in records}
    if len(dimensions) != 1:
        raise ValueError("Sample report requires vectors with consistent dimensions")

    report = {
        "model": model,
        "chunks_embedded": len(records),
        "vector_length": dimensions.pop(),
        "records": [
            {
                "text": record.text,
                "metadata": asdict(record.metadata),
                "vector_length": len(record.embedding),
                "vector_preview": record.embedding[:SAMPLE_VALUE_COUNT],
            }
            for record in records
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _source_identifier(path: Path) -> str:
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(BASE_DIR).as_posix()
    except ValueError:
        return resolved_path.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-output", type=Path)
    parser.add_argument("--max-chars", type=int, default=config.CHUNK_SIZE)
    args = parser.parse_args()
    if args.max_chars < 1:
        parser.error("--max-chars must be at least 1")
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    sample_output_path = (
        args.sample_output.resolve() if args.sample_output is not None else None
    )
    if input_path == output_path or input_path == sample_output_path:
        parser.error("Output paths must not overwrite the input document")
    if output_path == sample_output_path:
        parser.error("--output and --sample-output must be different paths")
    if not config.validate_api_keys():
        raise RuntimeError("Set OPENAI_API_KEY in the environment or repository .env file")

    source_text = input_path.read_text(encoding="utf-8")
    chunks = chunk_document(
        source_text, _source_identifier(input_path), max_chars=args.max_chars
    )
    client = OpenAI(
        base_url=config.OPENAI_BASE_URL,
        api_key=config.OPENAI_API_KEY,
    )
    records = embed_chunks(chunks, client=client, model=config.EMBED_MODEL)
    write_embeddings(records, output_path)
    if sample_output_path is not None:
        write_sample_report(records, sample_output_path, model=config.EMBED_MODEL)

    print(f"Chunks embedded: {len(records)}")
    print(f"Vector length: {len(records[0].embedding)}")
    print(f"Sample vector values: {records[0].embedding[:SAMPLE_VALUE_COUNT]}")
    print(f"Stored embeddings: {output_path}")
    if sample_output_path is not None:
        print(f"Sample report: {sample_output_path}")


if __name__ == "__main__":
    main()
