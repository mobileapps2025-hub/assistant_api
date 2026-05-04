import importlib


def test_weaviate_url_defaults_to_embedded_on_azure(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("WEAVIATE_URL", raising=False)
    monkeypatch.delenv("APPSETTING_WEAVIATE_URL", raising=False)
    monkeypatch.setenv("WEBSITE_SITE_NAME", "spotplan-assistant")

    from app.core import config

    reloaded = importlib.reload(config)

    assert reloaded.WEAVIATE_URL == "embedded"


def test_weaviate_url_prefers_explicit_setting(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("WEAVIATE_URL", "https://cluster.weaviate.cloud")
    monkeypatch.setenv("WEBSITE_SITE_NAME", "spotplan-assistant")

    from app.core import config

    reloaded = importlib.reload(config)

    assert reloaded.WEAVIATE_URL == "https://cluster.weaviate.cloud"


def test_weaviate_url_reads_azure_appsetting(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("WEAVIATE_URL", raising=False)
    monkeypatch.setenv("APPSETTING_WEAVIATE_URL", "embedded")

    from app.core import config

    reloaded = importlib.reload(config)

    assert reloaded.WEAVIATE_URL == "embedded"
