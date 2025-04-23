import ast

from sentinel.plan.infoflow import FlowParser

# Test: simple linear flow
source_code_0 = """
def main():
    x = f(a, b)
    y = g(x)
    z = h(y)
    r()
"""

# Test: branching program
source_code_1 = """
def main():
    if a > 0:
        b = 1
    elif b == 0:
        c = 1
    else:
        c = a + b
        f()
"""

# Test: single for loop
source_code_2 = """
def main():
    for i in range(n):
        x = f(i, a)
        y = g()
"""

# Test: single while loop
source_code_3 = """
def main():
    while i < n:
        x = f(i, a)
        y = g()
        i += 1
        n = h(x)
"""

# Test: nested loops
source_code_4 = """
def main():
    for i in range(n):
        m = h(i)
        for j in range(m):
            x = f(i, j, a)
            y = g()
            if y > 0:
                break
            else:
                m = 0
"""

# Test: nested loops 2
source_code_5 = """
def main():
    while i < n:
        while j < m:
            x = f(i, j, a)
            y = g()
            if y > 0:
                break
            else:
                m = 0
"""


source_codes = [
    source_code_0,
    source_code_1,
    source_code_2,
    source_code_3,
    source_code_4,
    source_code_5,
]


def run_case(source_code: str):
    source_ast = ast.parse(source_code)
    parser = FlowParser()
    flow = parser.parse(source_ast)

    print("Original code:")
    print(source_code.strip())

    print("Simplified control flow:")
    print(flow.dump().strip())


def main():
    for source_code in source_codes:
        run_case(source_code)


if __name__ == "__main__":
    main()
