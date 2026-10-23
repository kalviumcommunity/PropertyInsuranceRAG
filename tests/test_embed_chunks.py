import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.chunking import Chunk, ChunkMetadata
from src.embed_chunks import embed_chunks, write_embeddings, write_sample_report


class FakeEmbeddings:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors
        self.arguments: dict[str, object] | None = None

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.arguments = kwargs
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=vector)
                for index, vector in enumerate(self.vectors)
            ]
        )


class FakeClient:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.embeddings = FakeEmbeddings(vectors)


def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            "We cover the dwelling.",
            ChunkMetadata("policy.txt", "COVERAGE A", 1, 0, 0, 23),
        ),
        Chunk(
            "We cover attached structures.",
            ChunkMetadata("policy.txt", "COVERAGE A", 2, 1, 24, 54),
        ),
    ]


def test_embed_chunks_sends_text_and_preserves_metadata_and_vector_order():
    client = FakeClient([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    records = embed_chunks(sample_chunks(), client=client, model="test-embedding-model")

    assert client.embeddings.arguments == {
        "model": "test-embedding-model",
        "input": ["We cover the dwelling.", "We cover attached structures."],
    }
    assert len(records) == 2
    assert records[0].text == "We cover the dwelling."
    assert records[0].metadata.page == 1
    assert records[1].embedding == [0.4, 0.5, 0.6]


def test_embed_chunks_rejects_mismatched_response_count():
    client = FakeClient([[0.1, 0.2, 0.3]])

    with pytest.raises(RuntimeError, match="1 vectors for 2 chunks"):
        embed_chunks(sample_chunks(), client=client, model="test-model")


def test_embed_chunks_rejects_inconsistent_vector_dimensions():
    client = FakeClient([[0.1, 0.2], [0.3]])

    with pytest.raises(RuntimeError, match="inconsistent dimensions"):
        embed_chunks(sample_chunks(), client=client, model="test-model")


def test_embed_chunks_rejects_empty_chunks_and_model():
    client = FakeClient([])

    with pytest.raises(ValueError, match="At least one"):
        embed_chunks([], client=client, model="test-model")
    with pytest.raises(ValueError, match="non-empty"):
        embed_chunks(sample_chunks(), client=client, model=" ")


def test_embedding_output_stores_full_vectors_and_compact_sample(tmp_path: Path):
    records = embed_chunks(
        sample_chunks(),
        client=FakeClient([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        model="test-model",
    )
    stored_path = tmp_path / "embedded_chunks.jsonl"
    report_path = tmp_path / "embedding_sample_output.json"

    write_embeddings(records, stored_path)
    write_sample_report(records, report_path, model="test-model")

    stored = [json.loads(line) for line in stored_path.read_text().splitlines()]
    report = json.loads(report_path.read_text())
    assert stored[0]["embedding"] == [0.1, 0.2, 0.3]
    assert stored[0]["text"] == "We cover the dwelling."
    assert stored[0]["metadata"]["page"] == 1
    assert report["chunks_embedded"] == 2
    assert report["vector_length"] == 3
    assert report["records"][0]["vector_preview"] == [0.1, 0.2, 0.3]
