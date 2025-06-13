# Comprehensive infoflow example test
import pytest

try:
    from examples import info_flow_verify
except ImportError:
    pytest.skip("examples.info_flow_verify not available", allow_module_level=True)


def test_infoflow_runs():
    # Should not raise exceptions for all example cases
    for code in info_flow_verify.source_codes:
        info_flow_verify.run_case(code)
