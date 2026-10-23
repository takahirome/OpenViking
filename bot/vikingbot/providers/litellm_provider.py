"""Reject the retired LiteLLM provider without importing its SDK."""


class LiteLLMProvider:
    def __init__(self, *args, **kwargs):
        raise ValueError("LiteLLM support was retired; configure a supported direct provider.")
