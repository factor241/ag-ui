"""Keep the authoritative LangGraph table aligned with the strict Python fork."""

import unittest
from pathlib import Path


class TestDocumentationContract(unittest.TestCase):
    def test_root_interrupt_docs_do_not_advertise_legacy_python_behavior(self):
        repository_root = Path(__file__).resolve().parents[4]
        text = (repository_root / "docs/concepts/interrupts.mdx").read_text()
        langgraph_row = next(
            line for line in text.splitlines() if line.startswith("| LangGraph |")
        )

        self.assertIn("accepts only `RunAgentInput.resume[]`", langgraph_row)
        self.assertIn("always emits `RunFinishedEvent.outcome", langgraph_row)
        self.assertIn("remain accepted for source compatibility", langgraph_row)
        self.assertIn("are ignored", langgraph_row)
        self.assertIn("paths do not execute", langgraph_row)
        self.assertNotIn("default off", langgraph_row)
        self.assertNotIn("emitted by default", langgraph_row)

    def test_readme_documents_claim_retry_and_deployment_boundaries(self):
        repository_root = Path(__file__).resolve().parents[4]
        text = (
            repository_root / "integrations/langgraph/python/README.md"
        ).read_text()
        normalized = " ".join(text.split())

        self.assertIn("claim is not released", normalized)
        self.assertIn(
            "cannot be retried after cancellation or disconnect",
            normalized,
        )
        self.assertIn("bounded at 4096 live claims", normalized)
        self.assertIn(
            "fail closed rather than evicting replay protection",
            normalized,
        )
        self.assertIn("one FastAPI async worker/event loop", normalized)
        self.assertIn("durable atomic claim/CAS", normalized)


if __name__ == "__main__":
    unittest.main()
