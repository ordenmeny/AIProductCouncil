import httpx

from ai_product_council.config import Settings
from ai_product_council.llm_client import LMStudioClient


def make_settings() -> Settings:
    return Settings(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        model="qwen3.5-9b",
    )


def test_list_models_reads_openai_compatible_response(monkeypatch):
    def fake_get(url, headers, timeout, trust_env):
        return httpx.Response(
            200,
            request=httpx.Request("GET", url),
            json={
                "object": "list",
                "data": [
                    {"id": "qwen3.5-9b", "object": "model"},
                    {"id": "google/gemma-4-e4b", "object": "model"},
                ],
            },
        )

    monkeypatch.setattr(httpx, "get", fake_get)

    client = LMStudioClient(make_settings())

    assert client.list_models() == ["qwen3.5-9b", "google/gemma-4-e4b"]
    assert client.is_model_available()
