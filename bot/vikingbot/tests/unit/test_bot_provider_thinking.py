import contextlib
import sys
from types import SimpleNamespace

import pytest

from vikingbot.config.schema import AgentsConfig
from vikingbot.providers.vlm_adapter import VLMProviderAdapter


def test_agents_config_defaults_thinking_enabled():
    assert AgentsConfig().thinking is True
    assert AgentsConfig(thinking=False).thinking is False


def test_agents_config_defaults_max_tokens_unset():
    assert AgentsConfig().max_tokens is None
    assert AgentsConfig(max_tokens=8192).max_tokens == 8192


def test_agents_openviking_retention_defaults_to_turn_budget_values():
    config = AgentsConfig()

    assert config.commit_keep_recent_count == 10
    assert AgentsConfig(commit_keep_recent_count=7).commit_keep_recent_count == 7
    assert config.commit_keep_recent_turn_count == 3
    assert config.commit_retained_message_token_budget == 6_000
    assert config.commit_min_raw_tail_steps == 1


def test_make_provider_passes_default_thinking_to_vlm_adapter(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "prompt_toolkit",
        SimpleNamespace(PromptSession=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "prompt_toolkit.formatted_text",
        SimpleNamespace(HTML=lambda value: value, FormattedText=lambda value: value),
    )
    monkeypatch.setitem(
        sys.modules,
        "prompt_toolkit.history",
        SimpleNamespace(FileHistory=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "prompt_toolkit.patch_stdout",
        SimpleNamespace(patch_stdout=lambda: contextlib.nullcontext()),
    )
    monkeypatch.setitem(
        sys.modules,
        "prompt_toolkit.styles",
        SimpleNamespace(
            Style=SimpleNamespace(from_dict=lambda value: value),
        ),
    )

    from vikingbot.cli.commands import _make_provider

    captured = {}

    def fake_create(config):
        captured.update(config)
        return SimpleNamespace(
            provider=config["provider"],
            model=config["model"],
            thinking=config["thinking"],
        )

    monkeypatch.setattr("openviking.models.vlm.base.VLMFactory.create", fake_create)

    provider = _make_provider(
        SimpleNamespace(
            agents=SimpleNamespace(
                model="ep-test",
                temperature=0.0,
                thinking=True,
                api_key="ak-test",
                api_base="https://example.invalid",
                provider="volcengine",
                extra_headers={},
                timeout=None,
            )
        )
    )

    assert captured["thinking"] is True
    assert provider._vlm.thinking is True










@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("configured_max_tokens", "expected_max_tokens"),
    [(None, None), (8192, 8192)],
)
async def test_vlm_adapter_volcengine_stream_respects_optional_max_tokens(
    configured_max_tokens,
    expected_max_tokens,
):
    captured = {}

    class FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)

            async def chunks():
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason="stop",
                            delta=SimpleNamespace(
                                reasoning_content=None,
                                content="ok",
                                tool_calls=[],
                            ),
                        )
                    ],
                    usage=None,
                )

            return chunks()

    vlm = SimpleNamespace(
        provider="volcengine",
        model="ep-test",
        temperature=0.0,
        max_tokens=configured_max_tokens,
        thinking=False,
        extra_headers=None,
        get_async_client=lambda: SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        ),
    )
    provider = VLMProviderAdapter(
        vlm,
        default_model="ep-test",
        langfuse_client=SimpleNamespace(enabled=False, _client=None),
    )

    events = [
        event
        async for event in provider.chat_stream(
            messages=[{"role": "user", "content": "hi"}],
        )
    ]

    assert events[-1].response.content == "ok"
    if expected_max_tokens is None:
        assert "max_tokens" not in captured
    else:
        assert captured["max_tokens"] == expected_max_tokens
