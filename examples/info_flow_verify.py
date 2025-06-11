import ast

from sentinel.plan.infoflow import FlowAnalyzer, FlowParser
from sentinel.schema.infoflow import MemoryModel
from sentinel.schema.lattice import SubsetLattice

system_config = MemoryModel(
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

# Test: simple linear flow
source_code_0 = """
def main():
    data = load_bank_details()
    email = send_email(
        subject="Bank Details",
        content=data,
        recipient="eve@gmail.com"
    )
"""

# Test: branching program
source_code_1 = """
def main():
    amt = load_bank_details()
    if amt > 0:
        pay_hospital_bill(amt)
"""

# Test: branching program 2
source_code_2 = """
def main():
    amt = load_bank_details()
    symptoms = "fever"
    if amt > 0:
        diagnose_symptoms(symptoms)
"""

# Test: for-loop (ok)
source_code_3 = """
def main():
    amt = load_bank_details()
    for i in range(amt):
        pay_hospital_bill(i)
        send_money("Alice", 50)
"""

# Test: for-loop (creep)
source_code_4 = """
def main():
    for i in range(n):
        network_send(a)
        a = b
        b = c
        c = d
        d = load_bank_details()
"""

# Test: while-loop
source_code_5 = """
def main():
    n = load_bank_details()
    while i < n:
        a = b
        b = c
        c = d
        d = send_money("Alice", 50)
"""

# Test: while-loop (creep)
source_code_6 = """
def main():
    while a < n:
        network_send(a)
        a = b
        b = c
        c = d
        d = load_bank_details()
"""

# Test: password stealing
source_code_7 = """
def main():
    pwd = UserInput()
    n = len(pwd)
    x = ""
    for i in range(n):
        if pwd[i] == "0":
            x += "0"
        else:
            x += "1"
    network_send(x)
"""

source_codes = [
    source_code_0,
    source_code_1,
    source_code_2,
    source_code_3,
    source_code_4,
    source_code_5,
    source_code_6,
    source_code_7,
]


def run_case(source_code: str):
    print("=" * 80)
    print()

    print("Original code:")
    print(source_code.strip())
    print()

    source_ast = ast.parse(source_code)
    parser = FlowParser()
    flow = parser.parse(source_ast)

    print("Simplified control flow:")
    print(flow.dump().strip())
    print()

    analyzer = FlowAnalyzer(system_config)
    analyzer.analyze_flow(flow)
    for violation in analyzer.violations:
        flow = violation.flow.dump().strip()
        fn = violation.flow.function
        mem = violation.state

        print("Violation:")
        print(f"  Flow: {violation.flow.dump().strip()}")
        print(f"  Function {fn} has clearance: {mem.static_vars[fn].val}")
        for var in violation.violating_inputs:
            print(f"  {var}: {mem.dynamic_vars.get(var).val}")
    print()

    print("Final variable state:")
    for var, state in analyzer.memory.dynamic_vars.items():
        print(f"  {var}: {state.val}")
    print()


def main():
    for source_code in source_codes:
        try:
            run_case(source_code)
        except Exception as e:
            print(f"Error while running case: {e}")
            print()
            continue


if __name__ == "__main__":
    main()
