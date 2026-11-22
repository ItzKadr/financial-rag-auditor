import pytest

from src.extractor import OfflineExtractor, build_extractor
from src.providers import ProviderUnavailable, available_providers, load_dotenv, resolve_provider

KEYS = ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "GROQ_API_KEY")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in KEYS + ("DEEPSEEK_MODEL", "GROQ_BASE_URL", "OPENAI_RESPONSE_MODE"):
        monkeypatch.delenv(k, raising=False)


def test_no_keys_means_offline_only() -> None:
    assert available_providers() == []
    with pytest.raises(ProviderUnavailable):
        resolve_provider("openai")
    assert isinstance(build_extractor("offline"), OfflineExtractor)


def test_provider_defaults_and_overrides(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-reasoner")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("GROQ_BASE_URL", "http://localhost:9999/v1")
    ds = resolve_provider("deepseek")
    assert ds.base_url == "https://api.deepseek.com" and ds.model == "deepseek-reasoner" and ds.response_mode == "json_object"
    gq = resolve_provider("groq")
    assert gq.base_url == "http://localhost:9999/v1"
    assert set(available_providers()) == {"deepseek", "groq"}
    with pytest.raises(ValueError):
        resolve_provider("mistral")


def test_invalid_response_mode(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_RESPONSE_MODE", "xml")
    with pytest.raises(ValueError):
        resolve_provider("openai")


def test_load_dotenv(tmp_path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text("# comment\nGROQ_API_KEY='gsk-from-file'\nEMPTY=\n", encoding="utf-8")
    # register GROQ_API_KEY with monkeypatch so the value loaded from the file is removed afterwards
    monkeypatch.setenv("GROQ_API_KEY", "placeholder")
    monkeypatch.delenv("GROQ_API_KEY")
    assert load_dotenv(env) == 1
    import os

    assert os.environ["GROQ_API_KEY"] == "gsk-from-file"
    assert load_dotenv(tmp_path / "missing.env") == 0
