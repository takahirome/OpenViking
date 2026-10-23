"""Legacy provider configurations must fail without importing the retired SDK."""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

class RetiredProviderTest(unittest.TestCase):
    def test_legacy_adapters_reject_configuration_without_loading_sdk(self):
        for path, name in [
            ("openviking/models/vlm/backends/litellm_vlm.py", "LiteLLMVLMProvider"),
            ("openviking/models/embedder/litellm_embedders.py", "LiteLLMDenseEmbedder"),
            ("openviking/models/rerank/litellm_rerank.py", "LiteLLMRerankClient"),
        ]:
            with self.subTest(provider=name):
                spec = importlib.util.spec_from_file_location("retired_provider", ROOT / path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                with self.assertRaisesRegex(ValueError, "retired"):
                    getattr(module, name)(model="legacy")
                self.assertNotIn("litellm", sys.modules)

if __name__ == "__main__":
    unittest.main()
