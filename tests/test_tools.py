# Comprehensive tool manager test
import pytest

try:
    from src.ace.tools import manager
except ImportError:
    pytest.skip("src.ace.tools.manager not available", allow_module_level=True)


def test_tool_manager_load():
    tm = manager.ToolManager.from_manifest("tools/manifest.json")
    assert tm is not None
    assert len(tm.tools) > 0
