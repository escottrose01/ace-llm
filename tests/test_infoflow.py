import ast

from src.ace.schema.infoflow import MemoryModel
from src.ace.schema.lattice import SubsetLattice
from src.ace.security.infoflow import FlowAnalyzer, FlowParser, Violation


def make_system_config():
    return MemoryModel(
        lattice_type=SubsetLattice,
        dynamic_vars={
            "pwd": SubsetLattice({"personal"}),
            "symptoms": SubsetLattice({"medical"}),
        },
        static_vars={
            "load_bank_details": SubsetLattice({"financial"}),
            "diagnose_symptoms": SubsetLattice({"medical"}),
            "pay_hospital_bill": SubsetLattice({"financial", "medical"}),
            "send_money": SubsetLattice({"financial"}),
            "send_email": SubsetLattice({"personal"}),
            "send_work_email": SubsetLattice({"work"}),
            "send_personal_email": SubsetLattice({"personal"}),
            "network_send": SubsetLattice(set()),
        },
    )


def test_linear_flow_with_violation():
    # This test checks for a violation when trying to send financial data to a personal email
    code = """
def main():
    data = load_bank_details()
    send_email(subject="Bank Details", content=data, recipient="eve@gmail.com")
"""
    tree = ast.parse(code)
    memory = make_system_config()
    analyzer = FlowAnalyzer(memory)
    parser = FlowParser()
    flow = parser.parse(tree)
    analyzer.analyze_flow(flow)

    # Verify violation: We expect a violation when financial data is sent to a personal channel
    assert analyzer.violations
    # Check that the flow function is send_email (personal channel)
    assert any(v.flow.function == "send_email" for v in analyzer.violations)
    # Print the violation details to help debug
    for v in analyzer.violations:
        print(f"Violation found: function={v.flow.function}, inputs={v.violating_inputs}")


def test_branching_flow_violation():
    code = """
def main():
    amt = load_bank_details()
    if amt > 0:
        network_send(amt)
"""
    tree = ast.parse(code)
    memory = make_system_config()
    analyzer = FlowAnalyzer(memory)
    parser = FlowParser()
    flow = parser.parse(tree)
    analyzer.analyze_flow(flow)
    assert any(isinstance(v, Violation) for v in analyzer.violations)


def test_loop_implicit_leakage():
    code = """
def main():
    for i in range(2):
        data = load_bank_details()
        network_send(data)
"""
    tree = ast.parse(code)
    memory = make_system_config()
    analyzer = FlowAnalyzer(memory)
    parser = FlowParser()
    flow = parser.parse(tree)
    analyzer.analyze_flow(flow)
    assert analyzer.violations
