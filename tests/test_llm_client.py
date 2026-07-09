import httpx

from ai_product_council.config import Settings
from ai_product_council.llm_client import LMStudioClient


def make_settings() -> Settings:
    return Settings(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        model="deepseek-r1-distill-qwen-7b-q4-k-m",
    )


def test_list_models_reads_openai_compatible_response(monkeypatch):
    def fake_get(url, headers, timeout, trust_env):
        return httpx.Response(
            200,
            request=httpx.Request("GET", url),
            json={
                "object": "list",
                "data": [
                    {"id": "deepseek-r1-distill-qwen-7b-q4-k-m", "object": "model"},
                    {"id": "google/gemma-4-e4b", "object": "model"},
                ],
            },
        )

    monkeypatch.setattr(httpx, "get", fake_get)

    client = LMStudioClient(make_settings())

    assert client.list_models() == ["deepseek-r1-distill-qwen-7b-q4-k-m", "google/gemma-4-e4b"]
    assert client.is_model_available()
