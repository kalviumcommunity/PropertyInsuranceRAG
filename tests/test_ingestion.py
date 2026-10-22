import json
from pathlib import Path

import pytest

from src.ingestion import ingest_corpus, run_ingestion


def test_ingestion_accounts_for_successes_failures_and_control_files(tmp_path: Path):
    (tmp_path / "good.txt").write_text(
        "COVERAGE A\n\nCoverage applies to the dwelling.", encoding="utf-8"
    )
    (tmp_path / "unsupported.pdf").write_bytes(b"not parsed")
    (tmp_path / ".gitkeep").write_text("placeholder", encoding="utf-8")

    summary, samples = ingest_corpus(tmp_path)

    assert summary["total_source_documents"] == 2
    assert summary["successfully_ingested_documents"] == 1
    assert summary["failed_documents"] == 1
    assert summary["total_chunks_created"] == 1
    assert summary["failures"] == [
        {
            "source": "unsupported.pdf",
            "reason": "unsupported document type; expected .txt",
        }
    ]
    assert summary["skipped_files"] == [
        {"source": ".gitkeep", "reason": "repository control placeholder"}
    ]
    assert summary["accounting"] == {
        "source_documents": 2,
        "ingested_plus_failures": 2,
        "valid": True,
    }
    assert samples[0].metadata.source == "good.txt"


def test_ingestion_records_decode_failures_instead_of_dropping_documents(
    tmp_path: Path,
):
    (tmp_path / "invalid.txt").write_bytes(b"\xff\xfe\xfa")

    summary, samples = ingest_corpus(tmp_path)

    assert summary["total_source_documents"] == 1
    assert summary["successfully_ingested_documents"] == 0
    assert summary["failed_documents"] == 1
    assert summary["failures"][0]["source"] == "invalid.txt"
    assert summary["accounting"]["valid"] is True
    assert samples == []


def test_run_ingestion_writes_summary_and_traceable_review_samples(tmp_path: Path):
    corpus = tmp_path / "corpus"
    output = tmp_path / "outputs"
    corpus.mkdir()
    source_text = (
        "COVERAGE A - DWELLING\n\n"
        "We cover direct physical loss to the dwelling."
    )
    (corpus / "policy.txt").write_text(source_text, encoding="utf-8")

    summary_path = output / "ingestion_summary.json"
    samples_path = output / "ingestion_sample_chunks.jsonl"
    summary = run_ingestion(corpus, summary_path, samples_path)

    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    saved_samples = [
        json.loads(line)
        for line in samples_path.read_text(encoding="utf-8").splitlines()
    ]
    assert saved_summary == summary
    assert saved_summary["accounting"]["valid"] is True
    assert saved_summary["total_chunks_created"] == len(saved_samples)
    for sample in saved_samples:
        metadata = sample["metadata"]
        assert metadata["source"] == "policy.txt"
        assert metadata["position"] == 0
        assert metadata["section"] == "COVERAGE A - DWELLING"
        assert source_text[metadata["start_char"] : metadata["end_char"]] == sample[
            "text"
        ]
        assert "COVERAGE A - DWELLING" in sample["text"]


def test_run_ingestion_rejects_outputs_inside_corpus(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    with pytest.raises(ValueError, match="outside the input corpus"):
        run_ingestion(corpus, corpus / "summary.json", tmp_path / "samples.jsonl")


def test_ingestion_rejects_empty_documents_after_cleaning(tmp_path: Path):
    (tmp_path / "empty.txt").write_text("Page 1\n", encoding="utf-8")

    summary, _ = ingest_corpus(tmp_path)

    assert summary["total_source_documents"] == 1
    assert summary["successfully_ingested_documents"] == 0
    assert summary["failed_documents"] == 1
    assert summary["failures"][0]["reason"] == (
        "ValueError: document is empty after cleaning"
    )
    assert summary["accounting"]["valid"] is True
