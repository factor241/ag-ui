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


if __name__ == "__main__":
    unittest.main()
