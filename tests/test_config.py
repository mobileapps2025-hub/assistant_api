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
    monkeypatch.delenv("APPSETTING_WEAVIATE_URL", raising=False)
    monkeypatch.setenv("WEBSITE_SITE_NAME", "spotplan-assistant")

    from app.core import config

    reloaded = importlib.reload(config)

    assert reloaded.WEAVIATE_URL == "https://cluster.weaviate.cloud"


def test_weaviate_url_ignores_localhost_on_azure(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("WEAVIATE_URL", "http://localhost:8080")
    monkeypatch.delenv("APPSETTING_WEAVIATE_URL", raising=False)
    monkeypatch.setenv("WEBSITE_SITE_NAME", "spotplan-assistant")

    from app.core import config

    reloaded = importlib.reload(config)

    assert reloaded.WEAVIATE_URL == "embedded"


def test_weaviate_url_keeps_localhost_for_local_development(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("WEAVIATE_URL", "http://localhost:8080")
    monkeypatch.delenv("APPSETTING_WEAVIATE_URL", raising=False)
    monkeypatch.delenv("WEBSITE_SITE_NAME", raising=False)

    from app.core import config

    reloaded = importlib.reload(config)

    assert reloaded.WEAVIATE_URL == "http://localhost:8080"


def test_weaviate_url_reads_azure_appsetting(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("WEAVIATE_URL", raising=False)
    monkeypatch.setenv("APPSETTING_WEAVIATE_URL", "embedded")
    monkeypatch.setenv("WEBSITE_SITE_NAME", "spotplan-assistant")

    from app.core import config

    reloaded = importlib.reload(config)

    assert reloaded.WEAVIATE_URL == "embedded"


def test_weaviate_url_ignores_localhost_azure_appsetting(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("WEAVIATE_URL", raising=False)
    monkeypatch.setenv("APPSETTING_WEAVIATE_URL", "http://127.0.0.1:8080")
    monkeypatch.setenv("WEBSITE_SITE_NAME", "spotplan-assistant")

    from app.core import config

    reloaded = importlib.reload(config)

    assert reloaded.WEAVIATE_URL == "embedded"
