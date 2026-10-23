"""
Unit tests for embedding generation, vector dimension verification, and semantic cosine similarity.
"""

from pathlib import Path
import json
import math
import pytest

from src.embeddings import (
    EmbeddingClient,
    SimilarityPairResult,
    TextEmbeddingSample,
    cosine_similarity,
    explain_vectors,
    export_embedding_artifacts,
    get_default_demonstration_samples,
    run_embedding_demonstration,
)


class TestCosineSimilarity:
    """Tests for cosine similarity calculation and mathematical properties."""

    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0, 4.0]
        sim = cosine_similarity(v, v)
        assert pytest.approx(sim, 0.0001) == 1.0

    def test_orthogonal_vectors(self):
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        sim = cosine_similarity(v1, v2)
        assert pytest.approx(sim, 0.0001) == 0.0

    def test_opposite_vectors(self):
        v1 = [1.0, 2.0, 3.0]
        v2 = [-1.0, -2.0, -3.0]
        sim = cosine_similarity(v1, v2)
        assert pytest.approx(sim, 0.0001) == -1.0

    def test_dimension_mismatch_raises_error(self):
        v1 = [1.0, 2.0]
        v2 = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match="dimensions do not match"):
            cosine_similarity(v1, v2)

    def test_zero_vector_raises_error(self):
        v1 = [0.0, 0.0, 0.0]
        v2 = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match="zero-magnitude vector"):
            cosine_similarity(v1, v2)


class TestEmbeddingClient:
    """Tests for embedding client initialization, dimension uniformity, and generation."""

    def test_client_initialization_defaults(self):
        client = EmbeddingClient(force_offline=True)
        assert client.dimensions == 1536
        assert not client.is_live

    def test_custom_dimensions(self):
        client = EmbeddingClient(dimensions=512, force_offline=True)
        assert client.dimensions == 512
        embeddings = client.get_embeddings(["Sample text"])
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 512

    def test_empty_input_list(self):
        client = EmbeddingClient(force_offline=True)
        assert client.get_embeddings([]) == []

    def test_dimension_uniformity_across_diverse_texts(self):
        client = EmbeddingClient(dimensions=1536, force_offline=True)
        texts = [
            "Short",
            "A medium length sentence regarding homeowners property insurance coverage.",
            "A very long legal clause detailing deductibles, perils, exclusions, and endorsements.",
            "12345 67890 !@#$%^&*()",
            "Special endorsement HO-04-95 for sewer water backup coverage.",
        ]
        embeddings = client.get_embeddings(texts)
        assert len(embeddings) == len(texts)
        for vec in embeddings:
            assert len(vec) == 1536
            # Verify vectors are non-empty and finite
            assert all(not math.isnan(x) and not math.isinf(x) for x in vec)


class TestDemonstrationAndComparisons:
    """Tests for similar vs dissimilar comparison (Task 3) and explanations (Task 4)."""

    def test_run_embedding_demonstration_similar_vs_dissimilar(self):
        client = EmbeddingClient(force_offline=True)
        samples, pairs, report = run_embedding_demonstration(client=client)

        assert len(samples) >= 4
        # Confirm every sample produced identical dimension
        dims = [s.vector_dim for s in samples]
        assert all(d == 1536 for d in dims)

        # Check pairwise comparison results
        similar_pair = next(p for p in pairs if "SIMILAR" in p.relationship)
        dissimilar_pair = next(p for p in pairs if "DISSIMILAR" in p.relationship)

        # Task 3 core condition: similar pair must score higher than dissimilar pair
        assert similar_pair.cosine_similarity > dissimilar_pair.cosine_similarity
        assert similar_pair.cosine_similarity > 0.70
        assert dissimilar_pair.cosine_similarity < 0.30

    def test_explanation_content_requirements(self):
        note = explain_vectors()
        # Must address: numeric representation of meaning, not random IDs, not keyword counts
        assert "Random IDs" in note or "random IDs" in note
        assert "Keyword Matching" in note or "keyword" in note
        assert "Meaning" in note or "meaning" in note
        assert "1,536" in note


class TestExportDeliverables:
    """Tests for persisting sample output and documentation (Task 5)."""

    def test_export_embedding_artifacts(self, tmp_path: Path):
        client = EmbeddingClient(dimensions=256, force_offline=True)
        samples, pairs, report = run_embedding_demonstration(client=client)

        txt_file = tmp_path / "embedding_demo.txt"
        json_file = tmp_path / "sample_embeddings.json"
        md_file = tmp_path / "embedding_demo.md"

        export_embedding_artifacts(
            samples,
            pairs,
            report,
            output_txt_path=txt_file,
            output_json_path=json_file,
            output_md_path=md_file,
        )

        assert txt_file.exists()
        assert json_file.exists()
        assert md_file.exists()

        # Validate JSON content
        data = json.loads(json_file.read_text(encoding="utf-8"))
        assert "metadata" in data
        assert data["metadata"]["vector_dimension"] == 256
        assert len(data["samples"]) == len(samples)
        assert len(data["pairwise_similarity_comparisons"]) == len(pairs)
        assert "explanation" in data

        # Validate Markdown content
        md_content = md_file.read_text(encoding="utf-8")
        assert "# Embedding Vectors & Semantic Similarity Demonstration" in md_content
        assert "Cosine Similarity" in md_content
