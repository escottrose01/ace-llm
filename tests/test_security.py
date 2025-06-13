# Comprehensive security/infoflow test
import pytest

try:
    from src.ace.security import infoflow
except ImportError:
    pytest.skip("src.ace.security.infoflow not available", allow_module_level=True)

# Comprehensive security/infoflow test
import pytest

try:
    from src.ace.schema.infoflow import MemoryModel
    from src.ace.schema.lattice import SubsetLattice
    from src.ace.security import infoflow
except ImportError:
    pytest.skip("src.ace.security.infoflow not available", allow_module_level=True)


def test_infoflow_analyzer():
    # Use a minimal valid memory model
    memory = MemoryModel(lattice_type=SubsetLattice, dynamic_vars={}, static_vars={})
    analyzer = infoflow.FlowAnalyzer(memory)
    assert analyzer is not None
