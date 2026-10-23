# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Reject legacy configurations after retirement of the LiteLLM dependency."""


class LiteLLMRerankClient:
    def __init__(self, *args, **kwargs):
        raise ValueError("LiteLLM support was retired; configure a supported direct provider.")
