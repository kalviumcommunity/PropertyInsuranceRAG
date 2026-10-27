"""
Unit tests for TokenAwareChunker, controlled overlap, and boundary context demonstration.
"""

from pathlib import Path
import json
import pytest

from src.token_aware_chunker import (
    BoundaryInspection,
    TokenAwareChunker,
    TokenChunk,
    demonstrate_boundary_context,
    export_demonstration_artifacts,
    generate_full_report,
)


SAMPLE_DOC = """
SECTION I – PROPERTY COVERAGES
COVERAGE A – DWELLING
We cover the dwelling on the residence premises shown in the Declarations,
including structures attached to the dwelling. We do not cover land,
underground plumbing systems, or structures used for business.

COVERAGE B – OTHER STRUCTURES
We cover other structures on the residence premises set apart from the dwelling by clear space.
This includes detached garages, fences, retaining walls, and swimming pools.
The limit of liability for this coverage will not be more than 10% of Coverage A.

SPECIAL ENDORSEMENT: WATER BACK-UP (HO-04-95)
We will pay up to $10,000 for direct physical loss caused by sewer backup or sump discharge.
A separate special deductible of $1,000 applies to each covered loss under this endorsement.
"""


class TestTokenAwareChunkerInitialization:
    """Tests for parameter validation and tokenizer setup."""

    def test_default_initialization(self):
        chunker = TokenAwareChunker()
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 100
        assert chunker.encoding_name == "cl100k_base"

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TokenAwareChunker(chunk_size=0)
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TokenAwareChunker(chunk_size=-50)

    def test_invalid_overlap(self):
        with pytest.raises(ValueError, match="cannot be negative"):
            TokenAwareChunker(chunk_size=200, chunk_overlap=-10)
        with pytest.raises(ValueError, match="strictly less than chunk_size"):
            TokenAwareChunker(chunk_size=200, chunk_overlap=200)
        with pytest.raises(ValueError, match="strictly less than chunk_size"):
            TokenAwareChunker(chunk_size=200, chunk_overlap=250)


class TestTokenSizingAndControlledOverlap:
    """Tests verifying sizing strictly by token count and overlap mechanics."""

    def test_empty_and_whitespace_input(self):
        chunker = TokenAwareChunker(chunk_size=100, chunk_overlap=20)
        assert chunker.chunk("") == []
        assert chunker.chunk("   \n\t  ") == []

    def test_single_chunk_document(self):
        short_text = "This is a short policy statement."
        chunker = TokenAwareChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(short_text, doc_id="single_test")
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "single_test_tok_000"
        assert chunks[0].token_count <= 100
        assert chunks[0].overlap_tokens_with_prev == 0
        assert chunks[0].overlap_tokens_with_next == 0

    def test_controlled_overlap_token_matching(self):
        chunk_size = 50
        chunk_overlap = 15
        chunker = TokenAwareChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.chunk(SAMPLE_DOC, doc_id="overlap_test")

        assert len(chunks) > 1

        # Verify that each chunk is sized by token count
        for i, c in enumerate(chunks):
            assert c.token_count <= chunk_size
            if i < len(chunks) - 1:
                assert c.token_count == chunk_size

        # Verify token overlap between consecutive chunks
        tokenizer = chunker.tokenizer
        for i in range(len(chunks) - 1):
            curr_chunk = chunks[i]
            next_chunk = chunks[i + 1]

            curr_tokens = tokenizer.encode(curr_chunk.text)
            next_tokens = tokenizer.encode(next_chunk.text)

            # Trailing N tokens of current chunk must match leading N tokens of next chunk
            trailing_tokens = curr_tokens[-chunk_overlap:]
            leading_tokens = next_tokens[:chunk_overlap]

            assert trailing_tokens == leading_tokens
            assert next_chunk.overlap_tokens_with_prev == chunk_overlap
            assert curr_chunk.overlap_tokens_with_next == chunk_overlap


class TestBoundaryContextPreservation:
    """Tests demonstrating the difference between chunking with vs without overlap."""

    def test_boundary_demonstration(self):
        chunk_size = 60
        chunk_overlap = 20
        boundary = demonstrate_boundary_context(SAMPLE_DOC, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        assert isinstance(boundary, BoundaryInspection)
        assert boundary.boundary_token_index == chunk_size
        assert len(boundary.without_overlap_chunk0_end) > 0
        assert len(boundary.without_overlap_chunk1_start) > 0
        assert len(boundary.with_overlap_chunk1_start) > 0
        assert boundary.repeated_token_count == chunk_overlap
        assert boundary.repeated_tokens_match is True
        # Overlap chunk should include context preceding the boundary cutoff
        assert boundary.without_overlap_chunk0_end[-20:] in boundary.with_overlap_chunk1_start

    def test_compute_stats_comparison(self):
        chunker_zero = TokenAwareChunker(chunk_size=60, chunk_overlap=0)
        chunker_ovl = TokenAwareChunker(chunk_size=60, chunk_overlap=15)

        chunks_zero = chunker_zero.chunk(SAMPLE_DOC)
        chunks_ovl = chunker_ovl.chunk(SAMPLE_DOC)

        stats_zero = chunker_zero.compute_stats(SAMPLE_DOC, chunks_zero)
        stats_ovl = chunker_ovl.compute_stats(SAMPLE_DOC, chunks_ovl)

        assert stats_zero.chunk_overlap == 0
        assert stats_zero.overlap_overhead_ratio == 1.0
        assert stats_ovl.overlap_overhead_ratio > 1.0
        assert stats_ovl.chunk_count >= stats_zero.chunk_count


class TestReportAndArtifacts:
    """Tests for report generation and artifact exports."""

    def test_generate_full_report(self):
        report = generate_full_report(SAMPLE_DOC, chunk_size=60, chunk_overlap=15)
        assert "[TASK 1]" in report
        assert "[TASK 2]" in report
        assert "[TASK 3]" in report
        assert "[TASK 4]" in report
        assert "[TASK 5]" in report
        assert "gpt-4o-mini" in report

    def test_export_demonstration_artifacts(self, tmp_path: Path):
        txt_path = tmp_path / "test_report.txt"
        json_path = tmp_path / "test_chunks.json"
        md_path = tmp_path / "test_demo.md"

        export_demonstration_artifacts(
            SAMPLE_DOC,
            output_txt_path=txt_path,
            output_json_path=json_path,
            output_md_path=md_path,
            chunk_size=60,
            chunk_overlap=15,
        )

        assert txt_path.exists()
        assert json_path.exists()
        assert md_path.exists()

        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "boundary_inspection" in data
        assert "without_overlap" in data
        assert "with_overlap" in data
