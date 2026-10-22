import json
from pathlib import Path

import pytest

from src.chunking import (
    ChunkMetadata,
    chunk_corpus,
    chunk_document,
    chunk_file,
    write_chunks_jsonl,
)


def test_chunks_have_consistent_provenance_and_trace_to_source():
    text = (
        "COVERAGE A - DWELLING\n\n"
        "We cover the dwelling on the residence premises.\n\n"
        "Page 2\n\n"
        "We also cover attached structures."
    )

    chunks = chunk_document(text, "policy.txt", max_chars=100)

    assert chunks
    assert all(isinstance(chunk.metadata, ChunkMetadata) for chunk in chunks)
    assert [chunk.metadata.position for chunk in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        metadata = chunk.metadata
        assert metadata.source == "policy.txt"
        assert text[metadata.start_char : metadata.end_char] == chunk.text
        assert metadata.end_char > metadata.start_char
    assert chunks[0].metadata.section == "COVERAGE A - DWELLING"
    assert chunks[-1].metadata.page == 2


def test_coverage_sentence_is_not_mistaken_for_a_section_heading():
    text = "COVERAGE A - DWELLING\n\nCoverage applies to the dwelling."

    chunks = chunk_document(text, "policy.txt", max_chars=100)

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].metadata.section == "COVERAGE A - DWELLING"


def test_large_paragraphs_split_at_character_limit_with_exact_spans():
    text = "A policy statement with enough words to split across multiple chunks."

    chunks = chunk_document(text, "policy.txt", max_chars=20)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 20 for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == text
    assert all(
        text[chunk.metadata.start_char : chunk.metadata.end_char] == chunk.text
        for chunk in chunks
    )


def test_long_paragraphs_prefer_sentence_boundaries():
    text = "First complete clause. Second complete clause. Third complete clause."

    chunks = chunk_document(text, "policy.txt", max_chars=30)

    assert len(chunks) > 1
    assert all(chunk.text.rstrip().endswith((".", "!", "?")) for chunk in chunks[:-1])
    assert all(len(chunk.text) <= 30 for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == text


def test_long_paragraphs_do_not_split_after_single_letter_list_markers():
    text = (
        "We do not cover: a. land and soil; b. underground pipes outside the "
        "foundation; c. structures used for business purposes unless endorsed."
    )

    chunks = chunk_document(text, "policy.txt", max_chars=60)

    assert all(
        not chunk.text.rstrip().endswith(("a.", "b.", "c."))
        for chunk in chunks[:-1]
    )
    assert "".join(chunk.text for chunk in chunks) == text


def test_chunk_file_and_jsonl_keep_same_metadata_schema(tmp_path: Path):
    source_file = tmp_path / "policy.txt"
    source_file.write_text("DEFINITIONS\n\nA policy definition.", encoding="utf-8")
    chunks = chunk_file(source_file, max_chars=100)
    output_path = tmp_path / "chunks.jsonl"

    write_chunks_jsonl(chunks, output_path)

    serialized = [
        json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    expected_fields = {
        "source",
        "section",
        "page",
        "position",
        "start_char",
        "end_char",
    }
    assert serialized
    assert all(set(item) == {"text", "metadata"} for item in serialized)
    assert all(set(item["metadata"]) == expected_fields for item in serialized)
    assert all(item["metadata"]["source"] == "policy.txt" for item in serialized)


def test_committed_sample_chunks_trace_to_their_source_excerpt():
    repository = Path(__file__).resolve().parents[1]
    source_path = repository / "examples" / "chunk_metadata_source.txt"
    sample_path = repository / "examples" / "chunk_metadata_sample.jsonl"
    source = source_path.read_text(encoding="utf-8")
    sample_chunks = [
        json.loads(line)
        for line in sample_path.read_text(encoding="utf-8").splitlines()
    ]

    assert sample_chunks
    for chunk in sample_chunks:
        metadata = chunk["metadata"]
        assert metadata["source"] == "examples/chunk_metadata_source.txt"
        assert (
            source[metadata["start_char"] : metadata["end_char"]] == chunk["text"]
        )


def test_chunk_corpus_uses_unique_relative_source_identifiers(tmp_path: Path):
    (tmp_path / "first").mkdir()
    (tmp_path / "second").mkdir()
    (tmp_path / "first" / "policy.txt").write_text("Coverage A.", encoding="utf-8")
    (tmp_path / "second" / "policy.txt").write_text("Coverage B.", encoding="utf-8")

    chunks = chunk_corpus(tmp_path)

    assert {chunk.metadata.source for chunk in chunks} == {
        "first/policy.txt",
        "second/policy.txt",
    }
    assert all(chunk.metadata.position == 0 for chunk in chunks)


@pytest.mark.parametrize(
    ("source", "max_chars"),
    [("", 10), ("policy.txt", 0)],
)
def test_chunk_document_rejects_invalid_arguments(source: str, max_chars: int):
    with pytest.raises(ValueError):
        chunk_document("Policy text.", source, max_chars=max_chars)
