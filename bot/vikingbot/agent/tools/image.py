"""Keep the retired image tool unavailable without loading LiteLLM."""

from typing import Any

from vikingbot.agent.tools.base import Tool, ToolContext


class ImageGenerationTool(Tool):
    @property
    def name(self) -> str:
        return "generate_image"

    @property
    def description(self) -> str:
        return "Unavailable: the LiteLLM image generation integration has been retired."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    def __init__(self, *args, **kwargs):
        self._send_callback = None

    def set_send_callback(self, callback) -> None:
        self._send_callback = callback

    async def execute(self, tool_context: ToolContext, **kwargs: Any) -> str:
        return "Error: LiteLLM image generation was retired; no image request was sent."
