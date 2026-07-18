"""Core package import must not require the optional FastAPI extra."""

import subprocess
import sys
import textwrap
import unittest


class TestOptionalFastAPIImport(unittest.TestCase):
    def test_bare_package_import_succeeds_when_fastapi_is_unavailable(self):
        script = textwrap.dedent(
            """
            import sys

            class BlockFastAPI:
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "fastapi" or fullname.startswith("fastapi."):
                        raise ModuleNotFoundError("blocked optional dependency", name=fullname)
                    return None

            sys.meta_path.insert(0, BlockFastAPI())
            import ag_ui_langgraph
            assert ag_ui_langgraph.LangGraphAgent is not None
            assert "add_langgraph_fastapi_endpoint" in ag_ui_langgraph.__all__
            try:
                ag_ui_langgraph.add_langgraph_fastapi_endpoint
            except ImportError as exc:
                assert "[fastapi]" in str(exc)
            else:
                raise AssertionError("optional endpoint export unexpectedly loaded")
            """
        )

        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
