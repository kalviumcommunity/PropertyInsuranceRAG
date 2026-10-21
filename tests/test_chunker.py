"""
Unit tests for document chunking strategies, boundary detection, statistics, and pipeline execution.
"""

from pathlib import Path
import json
import pytest

from src.chunker import (
    Chunk,
    FixedTokenChunker,
    HierarchicalPolicyChunker,
    ParagraphChunker,
    calculate_strategy_stats,
    compare_chunking_strategies,
    detect_mid_sentence_cuts,
    export_sample_chunks,
    generate_comparison_report,
    run_chunking_pipeline,
)


SAMPLE_TEXT = """
================================================================================
SECTION I – PROPERTY COVERAGES
================================================================================

COVERAGE A – DWELLING
We cover the dwelling on the residence premises, including attached structures.
Wall-to-wall carpeting and machinery attached to the dwelling are also covered.

COVERAGE B – OTHER STRUCTURES
We cover other structures on the residence premises set apart from the dwelling.
The limit of liability for this coverage will not be more than 10% of Coverage A.

SPECIAL ENDORSEMENT: WATER BACK-UP
We will pay up to $10,000 for direct physical loss caused by water backing up through drains.
A separate deductible of $1,000 applies to each covered loss under this endorsement.
"""


class TestFixedTokenChunker:
    """Tests for fixed-size token chunking with overlap."""

    def test_empty_input(self):
        chunker = FixedTokenChunker(chunk_size=100, chunk_overlap=20)
        assert chunker.chunk("") == []
        assert chunker.chunk("   \n\t  ") == []

    def test_invalid_overlap(self):
        with pytest.raises(ValueError, match="strictly less"):
            FixedTokenChunker(chunk_size=100, chunk_overlap=100)
        with pytest.raises(ValueError, match="strictly less"):
            FixedTokenChunker(chunk_size=100, chunk_overlap=150)

    def test_chunking_output_structure(self):
        chunker = FixedTokenChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk(SAMPLE_TEXT, doc_id="test_doc")
        assert len(chunks) > 1
        for i, c in enumerate(chunks):
            assert c.chunk_id == f"test_doc_fixed_{i:03d}"
            assert c.token_count <= 50
            assert c.char_count > 0
            assert c.strategy == "fixed_token"
            assert "window_start_token" in c.metadata


class TestParagraphChunker:
    """Tests for paragraph-semantic boundary chunking."""

    def test_empty_input(self):
        chunker = ParagraphChunker(target_tokens=100, max_tokens=150)
        assert chunker.chunk("") == []

    def test_paragraph_preservation(self):
        chunker = ParagraphChunker(target_tokens=60, max_tokens=100, overlap_paragraphs=1)
        chunks = chunker.chunk(SAMPLE_TEXT, doc_id="para_test")
        assert len(chunks) >= 2
        for c in chunks:
            assert c.strategy == "paragraph_semantic"
            assert c.token_count > 0
            assert "paragraph_count" in c.metadata

    def test_oversized_paragraph_fallback(self):
        # Create an oversized paragraph with many sentences
        oversized = "This is a property coverage condition. " * 30
        chunker = ParagraphChunker(target_tokens=40, max_tokens=60)
        chunks = chunker.chunk(oversized, doc_id="oversized_test")
        assert len(chunks) > 1
        for c in chunks:
            assert c.token_count <= 65


class TestHierarchicalPolicyChunker:
    """Tests for policy hierarchy and section header preservation."""

    def test_empty_input(self):
        chunker = HierarchicalPolicyChunker()
        assert chunker.chunk("") == []

    def test_header_breadcrumb_injection(self):
        chunker = HierarchicalPolicyChunker(target_tokens=100, max_tokens=150)
        chunks = chunker.chunk(SAMPLE_TEXT, doc_id="hier_test")
        assert len(chunks) > 0
        # Check that breadcrumbs like [SECTION I...] or [COVERAGE A...] exist
        has_breadcrumb = any("[" in c.text and "]" in c.text for c in chunks)
        assert has_breadcrumb
        for c in chunks:
            assert "section_header" in c.metadata


class TestBoundaryCutDetection:
    """Tests for detecting mid-sentence or mid-clause boundary cuts."""

    def test_clean_boundaries(self):
        c1 = Chunk("c1", "We cover the dwelling against storm damage.", "test", 8, 44, 0, 44)
        c2 = Chunk("c2", "A separate deductible of $1,000 applies.", "test", 8, 41, 45, 86)
        cuts = detect_mid_sentence_cuts([c1, c2])
        assert cuts == 0

    def test_severed_boundary(self):
        c1 = Chunk("c1", "We cover the dwelling against storm", "test", 6, 35, 0, 35)
        c2 = Chunk("c2", "damage and hail events.", "test", 5, 23, 36, 59)
        cuts = detect_mid_sentence_cuts([c1, c2])
        assert cuts == 1

    def test_single_or_empty_chunk_list(self):
        assert detect_mid_sentence_cuts([]) == 0
        single = [Chunk("c1", "Just one chunk.", "test", 4, 15, 0, 15)]
        assert detect_mid_sentence_cuts(single) == 0


class TestComparisonAndStats:
    """Tests for computing statistics, comparing strategies, and generating reports."""

    def test_calculate_strategy_stats(self):
        c1 = Chunk("c1", "First complete sentence.", "test", 4, 24, 0, 24)
        c2 = Chunk("c2", "Second complete sentence.", "test", 4, 25, 25, 50)
        stats = calculate_strategy_stats("TestStrategy", [c1, c2], "First complete sentence. Second complete sentence.")
        assert stats.chunk_count == 2
        assert stats.avg_tokens == 4.0
        assert stats.min_tokens == 4
        assert stats.max_tokens == 4
        assert stats.mid_sentence_cuts == 0

    def test_compare_chunking_strategies(self):
        chunks, stats = compare_chunking_strategies(SAMPLE_TEXT, doc_id="comp_test")
        assert len(chunks) >= 2
        assert len(stats) >= 2
        assert "Fixed-Token (400 tok, 80 overlap)" in stats
        assert "Paragraph-Semantic (target 400, max 550)" in stats

    def test_generate_comparison_report(self):
        chunks, stats = compare_chunking_strategies(SAMPLE_TEXT, doc_id="report_test")
        report = generate_comparison_report(SAMPLE_TEXT, chunks, stats)
        assert "[TASK 3]" in report
        assert "[TASK 2]" in report
        assert "[TASK 4]" in report
        assert "[TASK 5]" in report
        assert "SELECTED STRATEGY" in report


class TestExportAndPipeline:
    """Tests for file export and running the full pipeline."""

    def test_export_sample_chunks(self, tmp_path: Path):
        chunks, _ = compare_chunking_strategies(SAMPLE_TEXT, doc_id="exp_test")
        json_out = tmp_path / "sample_chunks.json"
        md_out = tmp_path / "sample_chunks.md"

        export_sample_chunks(chunks, output_json_path=json_out, output_md_path=md_out)
        assert json_out.exists()
        assert md_out.exists()

        data = json.loads(json_out.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert len(data) >= 2

        md_content = md_out.read_text(encoding="utf-8")
        assert "# Sample Document Chunks" in md_content
        assert "Boundary Start:" in md_content

    def test_run_chunking_pipeline_with_custom_doc(self, tmp_path: Path):
        doc = tmp_path / "custom_policy.txt"
        doc.write_text(SAMPLE_TEXT, encoding="utf-8")
        report = run_chunking_pipeline(doc_path=doc, save_outputs=False)
        assert "Statistical Comparison of Chunking Strategies" in report
