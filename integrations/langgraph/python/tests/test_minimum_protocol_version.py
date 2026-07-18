"""The integration's declared AG-UI floor must provide the resume API it imports."""

import unittest
from importlib.metadata import requires

from packaging.requirements import Requirement
from packaging.version import Version

from ag_ui.core import ResumeEntry, RunFinishedInterruptOutcome


class TestMinimumProtocolVersion(unittest.TestCase):
    def test_declared_floor_is_the_first_supported_resume_version(self):
        requirement = next(
            Requirement(value)
            for value in (requires("ag-ui-langgraph") or [])
            if Requirement(value).name == "ag-ui-protocol"
        )

        self.assertIn(Version("0.1.19"), requirement.specifier)
        self.assertNotIn(Version("0.1.18"), requirement.specifier)
        self.assertIsNotNone(ResumeEntry)
        self.assertIsNotNone(RunFinishedInterruptOutcome)


if __name__ == "__main__":
    unittest.main()
