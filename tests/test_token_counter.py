"""
Unit tests for token counter, cost estimation, and length-to-token relationship analysis.
"""

from pathlib import Path
import pytest

from src.token_counter import (
    DEFAULT_ENCODING,
    analyze_length_to_token_relationship,
    count_messages_tokens,
    count_tokens,
    estimate_corpus_scale,
    estimate_cost,
    get_tokenizer,
    run_token_analysis,
    save_sample_results,
)


class TestGetTokenizer:
    """Tests for tokenizer initialization and caching."""

    def test_get_default_tokenizer(self):
        enc = get_tokenizer(DEFAULT_ENCODING)
        assert enc is not None
        assert enc.name == "cl100k_base"

    def test_get_tokenizer_caching(self):
        enc1 = get_tokenizer("cl100k_base")
        enc2 = get_tokenizer("cl100k_base")
        assert enc1 is enc2

    def test_get_tokenizer_for_model(self):
        enc = get_tokenizer("gpt-4o-mini")
        assert enc is not None

    def test_get_tokenizer_unknown_model_fallback(self):
        enc = get_tokenizer("unknown-unsupported-model-xyz")
        assert enc is not None
        assert enc.name == "cl100k_base"


class TestCountTokens:
    """Tests for token counting on text inputs and chat messages."""

    def test_empty_and_none_text(self):
        assert count_tokens("") == 0
        assert count_tokens(None) == 0

    def test_sample_token_counts(self):
        # "refund" is typically 1 token in cl100k_base
        refund_count = count_tokens("refund")
        assert refund_count == 1

        # "refundable" splits into "refund" + "able" (2 tokens)
        refundable_count = count_tokens("refundable")
        assert refundable_count == 2

    def test_count_messages_tokens(self):
        messages = [
            {"role": "system", "content": "You are a property insurance assistant."},
            {"role": "user", "content": "What is covered under HO-3?"},
        ]
        tokens = count_messages_tokens(messages)
        # Check that overhead exists (more than just raw string token counts)
        raw_tokens = count_tokens(messages[0]["content"]) + count_tokens(messages[1]["content"])
        assert tokens > raw_tokens
        assert tokens >= raw_tokens + (len(messages) * 3) + 3


class TestEstimateCost:
    """Tests for cost estimation under asymmetric billing."""

    def test_cost_calculation_formula(self):
        # 1,000 prompt tokens @ $0.0005/1K = $0.0005
        # 2,000 completion tokens @ $0.0015/1K = $0.0030
        # Total = $0.0035
        estimate = estimate_cost(
            prompt_tokens=1000,
            completion_tokens=2000,
            input_rate_per_1k=0.0005,
            output_rate_per_1k=0.0015,
        )
        assert estimate.prompt_tokens == 1000
        assert estimate.completion_tokens == 2000
        assert estimate.total_tokens == 3000
        assert pytest.approx(estimate.input_cost, 0.00001) == 0.0005
        assert pytest.approx(estimate.output_cost, 0.00001) == 0.0030
        assert pytest.approx(estimate.total_cost, 0.00001) == 0.0035

    def test_zero_tokens_cost(self):
        estimate = estimate_cost(0, 0)
        assert estimate.total_cost == 0.0

    def test_to_dict_method(self):
        estimate = estimate_cost(500, 100)
        data = estimate.to_dict()
        assert isinstance(data, dict)
        assert data["total_tokens"] == 600
        assert "total_cost" in data


class TestCorpusScaleEstimation:
    """Tests for scaling estimates across 4,000 documents and runtime queries."""

    def test_corpus_scale_calculation(self):
        # 4,000 documents * 2,000 tokens = 8,000,000 tokens
        # 8M tokens @ $0.02 / 1M = $0.1600
        corpus = estimate_corpus_scale(
            num_documents=4000,
            avg_tokens_per_doc=2000,
            embed_rate_per_1m=0.02,
        )
        assert corpus.num_documents == 4000
        assert corpus.total_corpus_tokens == 8_000_000
        assert pytest.approx(corpus.one_time_embedding_cost, 0.001) == 0.1600
        assert corpus.cost_per_query > 0
        assert corpus.cost_per_1000_queries > 0


class TestLengthTokenRelationship:
    """Tests for character, word, and token ratio comparisons."""

    def test_length_token_analysis_metrics(self):
        samples = {
            "Simple": "Hello world",
            "Technical": "Indemnification and subrogation rights",
            "German": "Wohngebäudeversicherung",
            "Numbers": "POL-987214-X",
        }
        metrics = analyze_length_to_token_relationship(samples)
        assert len(metrics) == 4

        for m in metrics:
            assert m.char_count > 0
            assert m.word_count > 0
            assert m.token_count > 0
            assert m.chars_per_token > 0
            assert m.tokens_per_word > 0

        # German compound word should have higher tokens_per_word than English prose
        german_metric = next(m for m in metrics if m.category == "German")
        english_metric = next(m for m in metrics if m.category == "Simple")
        assert german_metric.tokens_per_word > english_metric.tokens_per_word


class TestReportAndPersistence:
    """Tests for running the full analysis and persisting output."""

    def test_run_token_analysis_output(self):
        report = run_token_analysis()
        assert "PROPERTY INSURANCE RAG" in report
        assert "[TASK 1]" in report
        assert "[TASK 2]" in report
        assert "[TASK 3]" in report
        assert "[TASK 4]" in report
        assert "4,000" in report

    def test_save_sample_results(self, tmp_path: Path):
        test_output = tmp_path / "test_sample_results.txt"
        saved_path = save_sample_results(output_path=test_output)
        assert saved_path.exists()
        content = saved_path.read_text(encoding="utf-8")
        assert len(content) > 200
        assert "PROPERTY INSURANCE RAG" in content
