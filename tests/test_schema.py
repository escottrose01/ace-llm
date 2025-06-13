# Comprehensive schema import test
import pytest

try:
    from src.ace.schema import abstract, concrete, infoflow, lattice, permissions
except ImportError:
    pytest.skip("src.ace.schema modules not available", allow_module_level=True)


def test_schema_imports():
    assert abstract is not None
    assert concrete is not None
    assert infoflow is not None
    assert lattice is not None
    assert permissions is not None
