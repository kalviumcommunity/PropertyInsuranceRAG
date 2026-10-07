"""
Tests for configuration management and workspace integrity.
"""

from pathlib import Path
from src.config import config, BASE_DIR


def test_project_directories_exist():
    """Verify that all core project directories are present."""
    assert (BASE_DIR / "data").exists(), "data directory missing"
    assert (BASE_DIR / "src").exists(), "src directory missing"
    assert (BASE_DIR / "prompts").exists(), "prompts directory missing"
    assert (BASE_DIR / "outputs").exists(), "outputs directory missing"


def test_env_example_exists():
    """Verify that .env.example exists and contains key placeholders."""
    env_example = BASE_DIR / ".env.example"
    assert env_example.exists(), ".env.example file missing"
    content = env_example.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" in content
    assert "CHAT_MODEL" in content
    assert "EMBED_MODEL" in content


def test_gitignore_protects_secrets():
    """Verify that .gitignore properly guards sensitive files."""
    gitignore = BASE_DIR / ".gitignore"
    assert gitignore.exists(), ".gitignore file missing"
    content = gitignore.read_text(encoding="utf-8")
    assert ".env" in content
    assert ".venv" in content
