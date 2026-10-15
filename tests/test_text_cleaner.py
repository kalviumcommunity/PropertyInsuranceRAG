from src.text_cleaner import clean_corpus, clean_text, normalize_text


def test_normalize_text_fixes_unicode_whitespace_and_wrapped_lines():
    text = "\ufeffＣＯＶＥＲＡＧＥ\r\nmay be\u00ad\r\n\r\napplied   when   insured.\u200b"

    assert normalize_text(text) == "COVERAGE may be\n\napplied when insured."


def test_clean_text_removes_page_markers_navigation_and_repeated_lines():
    text = (
        "Property Policy\nHome | Claims\nCoverage applies to the dwelling.\n"
        "Page 2 of 8\nFooter Notice"
    )

    cleaned = clean_text(text, {"Property Policy", "Footer Notice"})

    assert cleaned == "Coverage applies to the dwelling."


def test_clean_corpus_cleans_every_text_file_and_preserves_tree(tmp_path):
    source = tmp_path / "raw"
    output = tmp_path / "cleaned"
    (source / "nested").mkdir(parents=True)
    (source / "policy-a.txt").write_text(
        "Property Policy\nCoverage applies.\nPage 1 of 4\nFooter Notice",
        encoding="utf-8",
    )
    (source / "nested" / "policy-b.txt").write_text(
        "Property Policy\nExclusions apply.\nPage 2 of 4\nFooter Notice",
        encoding="utf-8",
    )
    (source / "ignored.pdf").write_text("not processed", encoding="utf-8")

    assert clean_corpus(source, output) == 2
    assert (output / "policy-a.txt").read_text(encoding="utf-8") == "Coverage applies."
    assert (output / "nested" / "policy-b.txt").read_text(encoding="utf-8") == "Exclusions apply."


def test_clean_corpus_rejects_output_inside_input(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()

    try:
        clean_corpus(source, source / "cleaned")
    except ValueError as error:
        assert "inside the input directory" in str(error)
    else:
        raise AssertionError("Expected output inside input directory to be rejected")