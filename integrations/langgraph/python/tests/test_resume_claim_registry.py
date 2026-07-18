"""Bounded replay claims fail closed and retire only safe same-thread history."""

import unittest

import ag_ui_langgraph.agent as agent_module


class TestResumeClaimRegistry(unittest.TestCase):
    def test_duplicate_capacity_and_same_thread_progression(self):
        registry_type = getattr(agent_module, "_ResumeClaimRegistry", None)
        self.assertIsNotNone(registry_type)
        registry = registry_type(max_claims=2)
        first = ("thread-1", "checkpoint-1", ("interrupt-1",))
        second = ("thread-2", "checkpoint-2", ("interrupt-2",))
        advanced = ("thread-1", "checkpoint-3", ("interrupt-3",))
        overflow = ("thread-4", "checkpoint-4", ("interrupt-4",))

        self.assertEqual(registry.try_claim(first).value, "claimed")
        self.assertEqual(registry.try_claim(first).value, "duplicate")
        self.assertEqual(registry.try_claim(second).value, "claimed")
        self.assertEqual(len(registry), 2)

        # A newer observed checkpoint for the same thread safely retires its
        # old fingerprint, so bounded storage can keep making progress.
        self.assertEqual(registry.try_claim(advanced).value, "claimed")
        self.assertEqual(len(registry), 2)

        # Claims for other still-current threads are never evicted merely to
        # make room: capacity pressure rejects the new dispatch fail-closed.
        self.assertEqual(registry.try_claim(overflow).value, "capacity_exceeded")
        self.assertEqual(len(registry), 2)


if __name__ == "__main__":
    unittest.main()
