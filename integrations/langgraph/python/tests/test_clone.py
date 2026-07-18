"""Tests for LangGraphAgent.clone() subclass preservation."""

import unittest
import warnings
import threading
from unittest.mock import MagicMock

from langchain_core.callbacks import BaseCallbackHandler, CallbackManager

from ag_ui_langgraph import LangGraphAgent


class SubclassAgent(LangGraphAgent):
    """Test subclass that adds custom behavior."""

    def __init__(self, *, name, graph, description=None, config=None, custom_flag=False):
        super().__init__(name=name, graph=graph, description=description, config=config)
        self.custom_flag = custom_flag

    def custom_method(self):
        return "subclass behavior"


class TestClone(unittest.TestCase):
    """Test that clone() preserves subclass identity and behavior."""

    def _make_graph(self):
        """Create a mock compiled graph for testing."""
        graph = MagicMock()
        graph.config_specs = []
        return graph

    def test_clone_returns_same_class(self):
        """clone() should return an instance of the same class, not the base."""
        agent = SubclassAgent(name="test", graph=self._make_graph())
        cloned = agent.clone()
        self.assertIsInstance(cloned, SubclassAgent)

    def test_clone_base_class(self):
        """clone() on the base class should still return LangGraphAgent."""
        agent = LangGraphAgent(name="test", graph=self._make_graph())
        cloned = agent.clone()
        self.assertIsInstance(cloned, LangGraphAgent)

    def test_clone_copies_fields(self):
        """clone() should copy name, graph, description, and config."""
        graph = self._make_graph()
        config = {"recursion_limit": 50}
        agent = LangGraphAgent(
            name="my-agent",
            graph=graph,
            description="A test agent",
            config=config,
        )
        cloned = agent.clone()
        self.assertEqual(cloned.name, "my-agent")
        self.assertIs(cloned.graph, graph)
        self.assertEqual(cloned.description, "A test agent")
        self.assertEqual(cloned.config, config)

    def test_clone_deep_copies_nested_config(self):
        """clone() should isolate nested config mutations across requests."""
        config = {
            "recursion_limit": 50,
            "configurable": {"request_context": {"actor": "template"}},
        }
        agent = LangGraphAgent(name="test", graph=self._make_graph(), config=config)
        first = agent.clone()
        second = agent.clone()

        first.config["configurable"]["request_context"]["actor"] = "request-1"

        self.assertEqual(agent.config["configurable"]["request_context"]["actor"], "template")
        self.assertEqual(second.config["configurable"]["request_context"]["actor"], "template")
        self.assertIsNot(first.config["configurable"], agent.config["configurable"])

    def test_clone_structurally_copies_containers_but_preserves_opaque_leaves(self):
        class NonCopyableCallback:
            def __deepcopy__(self, memo):
                raise TypeError("callback must not be deep-copied")

        callback = NonCopyableCallback()
        runtime_lock = threading.Lock()
        config = {
            "callbacks": [callback],
            "configurable": {
                "runtime": (runtime_lock, {"labels": ["template"]}),
            },
        }
        agent = LangGraphAgent(name="test", graph=self._make_graph(), config=config)

        cloned = agent.clone()

        self.assertIs(cloned.config["callbacks"][0], callback)
        self.assertIs(cloned.config["configurable"]["runtime"][0], runtime_lock)
        self.assertIsNot(cloned.config, agent.config)
        self.assertIsNot(cloned.config["callbacks"], agent.config["callbacks"])
        self.assertIsNot(
            cloned.config["configurable"]["runtime"][1],
            agent.config["configurable"]["runtime"][1],
        )

    def test_clone_preserves_real_callback_manager_with_noncopyable_handler(self):
        class LockedHandler(BaseCallbackHandler):
            def __init__(self):
                self.runtime_lock = threading.Lock()

        handler = LockedHandler()
        callback_manager = CallbackManager([handler])
        agent = LangGraphAgent(
            name="test",
            graph=self._make_graph(),
            config={
                "callbacks": callback_manager,
                "configurable": {"request_context": {"actor": "template"}},
            },
        )

        cloned = agent.clone()

        self.assertIs(cloned.config["callbacks"], callback_manager)
        self.assertIs(callback_manager.handlers[0], handler)
        self.assertIs(handler.runtime_lock, callback_manager.handlers[0].runtime_lock)
        self.assertIsNot(cloned.config["configurable"], agent.config["configurable"])

    def test_clone_subclass_has_overridden_methods(self):
        """clone() of a subclass should have the subclass's methods."""
        agent = SubclassAgent(name="test", graph=self._make_graph())
        cloned = agent.clone()
        self.assertEqual(cloned.custom_method(), "subclass behavior")

    def test_clone_does_not_preserve_subclass_extra_state(self):
        """clone() only passes base-class params; subclass defaults apply."""
        agent = SubclassAgent(name="test", graph=self._make_graph(), custom_flag=True)
        cloned = agent.clone()
        # Documented limitation: custom_flag reverts to its default
        self.assertFalse(cloned.custom_flag)

    def test_clone_subclass_with_required_extra_param_raises(self):
        """Subclasses with extra required params must override clone()."""
        class StrictAgent(LangGraphAgent):
            def __init__(self, *, name, graph, api_key, description=None, config=None):
                super().__init__(name=name, graph=graph, description=description, config=config)
                self.api_key = api_key

        agent = StrictAgent(name="test", graph=self._make_graph(), api_key="sk-123")
        with self.assertRaises(TypeError) as ctx:
            agent.clone()
        self.assertIn("must override clone()", str(ctx.exception))

    def test_clone_with_no_config(self):
        """clone() with default (empty) config round-trips correctly."""
        agent = LangGraphAgent(name="test", graph=self._make_graph())
        cloned = agent.clone()
        self.assertEqual(cloned.config, {})

    def test_clone_isolates_mutable_state(self):
        """clone() should produce a separate instance (not the same object)."""
        agent = LangGraphAgent(name="test", graph=self._make_graph())
        cloned = agent.clone()
        self.assertIsNot(agent, cloned)
        self.assertIsNot(agent.messages_in_process, cloned.messages_in_process)

    def test_deprecated_interrupt_flags_are_accepted_but_standardized(self):
        graph = self._make_graph()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            agent = LangGraphAgent(
                name="test",
                graph=graph,
                enable_legacy_on_interrupt_event=True,
                emit_interrupt_outcome=False,
            )

        self.assertFalse(agent.enable_legacy_on_interrupt_event)
        self.assertTrue(agent.emit_interrupt_outcome)
        self.assertTrue(any(item.category is DeprecationWarning for item in caught))

        cloned = agent.clone()
        self.assertFalse(cloned.enable_legacy_on_interrupt_event)
        self.assertTrue(cloned.emit_interrupt_outcome)


if __name__ == "__main__":
    unittest.main()
