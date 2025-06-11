import ast
from dataclasses import dataclass

from ..schema.infoflow import ExplicitFlow, Flow, MemoryModel, RepetitionFlow, SequenceFlow
from ..schema.lattice import Lattice

ExprType = ast.Name | ast.Constant | ast.BinOp | ast.Compare | ast.Subscript

OK_FUNCTIONS = {"len", "abs", "min", "max"}


def extract_vars_from_expr(node: ast.expr) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    elif isinstance(node, ast.Constant):
        return set()
    elif isinstance(node, ast.BinOp):
        left = extract_vars_from_expr(node.left)
        right = extract_vars_from_expr(node.right)
        return left | right
    elif isinstance(node, ast.Compare):
        left = extract_vars_from_expr(node.left)
        comparators = set()
        for comparator in node.comparators:
            comparators |= extract_vars_from_expr(comparator)
        return left | comparators
    elif isinstance(node, ast.Subscript):
        if not isinstance(node.value, (ast.Name, ast.Constant)):
            raise TypeError("Subscript base must be a variable name or constant")
        base = {node.value.id} if isinstance(node.value, ast.Name) else set()
        index = extract_vars_from_expr(node.slice)
        return base | index
    else:
        raise TypeError(f"Unsupported expression type: {type(node)}")


def extract_vars_from_call(node: ast.Call) -> set[str]:
    if not isinstance(node.func, ast.Name):
        raise TypeError("Function call must be a variable name")

    args = node.args
    kwargs = node.keywords

    inputs = set()
    for arg in args:
        inputs |= extract_vars_from_expr(arg)
    for kwarg in kwargs:
        inputs |= extract_vars_from_expr(kwarg.value)

    return inputs


class FlowParser(ast.NodeVisitor):
    taint: list[str]
    flow: Flow
    cond_counter: int

    def __init__(self):
        self.taint = []
        self.flow = None
        self.cond_counter = 0

    def parse(self, source_ast: ast.AST) -> Flow:
        self.flow = SequenceFlow([])

        main_defs = [node for node in source_ast.body if isinstance(node, ast.FunctionDef) and node.name == "main"]

        if len(main_defs) != 1:
            raise ValueError("There must be exactly one main() function")

        main_node = main_defs[0]
        for stmt in main_node.body:
            self.visit(stmt)

        return self.flow

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # Handle variable annotations (e.g., x: int = 0)
        if not isinstance(node.target, ast.Name):
            raise TypeError("Annotation target must be a variable name")

        outputs = {node.target.id}
        inputs = set(self.taint)

        # If there's an initial value, extract variables from it
        if isinstance(node.value, ast.Call):
            if not isinstance(node.value.func, ast.Name):
                raise TypeError("Function call must be a variable name")

            # OK functions are treated as "pure"
            func_name = "+" if node.value.func.id in OK_FUNCTIONS else node.value.func.id
            inputs |= extract_vars_from_call(node.value)
        elif isinstance(node.value, ExprType):
            func_name = "+"
            inputs |= extract_vars_from_expr(node.value)
        else:
            raise TypeError("Annotation value must be a function call or expression")

        self.flow.flows.append(ExplicitFlow(outputs, inputs, func_name))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1:
            raise ValueError("Assignment must have exactly one target")
        if not isinstance(node.targets[0], ast.Name):
            raise TypeError("Assignment target must be a variable name")

        outputs = {node.targets[0].id}
        inputs = set(self.taint)

        if isinstance(node.value, ast.Call):
            if not isinstance(node.value.func, ast.Name):
                raise TypeError("Function call must be a variable name")

            # OK functions are treated as "pure"
            func_name = "+" if node.value.func.id in OK_FUNCTIONS else node.value.func.id
            inputs |= extract_vars_from_call(node.value)
        elif isinstance(node.value, ExprType):
            func_name = "+"
            inputs |= extract_vars_from_expr(node.value)
        else:
            raise TypeError("Assignment value must be a function call or expression")

        self.flow.flows.append(ExplicitFlow(outputs, inputs, func_name))
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if not isinstance(node.target, ast.Name):
            raise TypeError("Augmented assignment target must be a variable name")

        outputs = {node.target.id}
        inputs = set(self.taint)

        if isinstance(node.value, ast.Call):
            if not isinstance(node.value.func, ast.Name):
                raise TypeError("Function call must be a variable name")

            # OK functions are treated as "pure"
            func_name = "+" if node.value.func.id in OK_FUNCTIONS else node.value.func.id
            inputs |= extract_vars_from_call(node.value)
        elif isinstance(node.value, ExprType):
            func_name = "+"
            inputs |= extract_vars_from_expr(node.value)
        else:
            raise TypeError("Augmented assignment value must be an expression")

        self.flow.flows.append(ExplicitFlow(outputs, inputs, func_name))
        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        if isinstance(node.value, ast.Call):
            if not isinstance(node.value.func, ast.Name):
                raise TypeError("Function call must be a variable name")
            func_name = node.value.func.id
            inputs = extract_vars_from_call(node.value) | set(self.taint)
            self.flow.flows.append(ExplicitFlow([], inputs, func_name))
        else:
            raise TypeError("Standalone expressions must be a function call")

        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        # Enforce: for loop is of the form `for i in range(n)`
        if (
            not isinstance(node.iter, ast.Call)
            or not isinstance(node.iter.func, ast.Name)
            or node.iter.func.id != "range"
            or not isinstance(node.iter.args[0], ast.Constant | ast.Name)
            or len(node.iter.args) != 1
        ):
            raise TypeError("For loop must be of the form `for i in range(n)`")

        if not isinstance(node.target, ast.Name):
            raise TypeError("For loop target must be a variable name")
        if len(node.target.id) != 1:
            raise ValueError("For loop target must be a single variable")

        # Extract the loop variable
        loop_var = node.target.id
        loop_arg = node.iter.args[0]

        # Store the current flow
        cur_flow = self.flow

        # Create a flow for the loop condition
        loop_inputs = set(self.taint)
        if isinstance(loop_arg, ast.Name):
            loop_inputs |= {loop_arg.id}
        loop_flow = ExplicitFlow([loop_var], loop_inputs, "+")
        self.flow = SequenceFlow([loop_flow])

        # Recursively visit the body of the for loop
        self.taint.append(loop_var)
        self.generic_visit(node)
        self.taint.pop()

        # Append the repeating flow to the current flow
        cur_flow.flows.append(RepetitionFlow(self.flow))
        self.flow = cur_flow

    def visit_If(self, node: ast.If) -> None:
        # Extract a unique identifier for the condition
        self.cond_counter += 1
        cond_id = f"&cond_{self.cond_counter}"
        cond = node.test

        # Create a flow for the branch condition
        cond_inputs = set(self.taint)
        if isinstance(cond, ast.Call):
            cond_inputs |= extract_vars_from_call(cond)
            func_name = cond.func.id
        elif isinstance(cond, ExprType):
            cond_inputs |= extract_vars_from_expr(cond)
            func_name = "+"
        else:
            raise TypeError(f"Unsupported condition type: {type(cond)}")
        cond_flow = ExplicitFlow([cond_id], cond_inputs, func_name)
        self.flow.flows.append(cond_flow)

        # Recursively visit the body of the if statement
        self.taint.append(cond_id)
        self.generic_visit(node)
        self.taint.pop()

    def visit_While(self, node: ast.While) -> None:
        # Extract a unique identifier for the condition
        self.cond_counter += 1
        cond_id = f"&cond_{self.cond_counter}"
        cond = node.test

        # Store the current flow
        cur_flow = self.flow

        # Create a flow for the branch condition
        cond_inputs = set(self.taint)
        if isinstance(cond, ast.Call):
            cond_inputs |= extract_vars_from_call(cond)
            func_name = cond.func.id
        elif isinstance(cond, ExprType):
            cond_inputs |= extract_vars_from_expr(cond)
            func_name = "+"
        cond_flow = ExplicitFlow([cond_id], cond_inputs, func_name)
        self.flow = SequenceFlow([cond_flow])

        # Recursively visit the body of the while statement
        self.taint.append(cond_id)
        self.generic_visit(node)
        self.taint.pop()

        # Append the repeating flow to the current flow
        cur_flow.flows.append(RepetitionFlow(self.flow))
        self.flow = cur_flow


@dataclass
class Violation:
    flow: ExplicitFlow
    violating_inputs: set[str]
    state: MemoryModel


class FlowAnalyzer:
    memory: MemoryModel
    _violations: list[Violation]

    def __init__(self, memory: MemoryModel):
        self.memory = memory.clone()
        self._violations = []  # list to record security or analysis violations

    @property
    def violations(self) -> list[Violation]:
        return self._violations.copy()

    @property
    def valid(self) -> bool:
        return len(self._violations) == 0

    def get_label(self, var: str) -> Lattice:
        if var in self.memory.dynamic_vars:
            return self.memory.dynamic_vars[var]
        elif var in self.memory.static_vars:
            return self.memory.static_vars[var]
        else:
            return self.memory.lattice_type.bottom()

    def apply_explicit_flow(self, flow: ExplicitFlow) -> None:
        joined_inputs = self.memory.lattice_type.bottom()
        for var in flow.inputs:
            joined_inputs = joined_inputs + self.get_label(var)

        if flow.function != "+":
            func_label = self.memory.static_vars.get(flow.function, self.memory.lattice_type.bottom())
            if not (joined_inputs <= func_label):
                violating_inputs = {var for var in flow.inputs if not (self.get_label(var) <= func_label)}
                self._violations.append(
                    Violation(flow=flow, violating_inputs=violating_inputs, state=self.memory.clone())
                )
            extra = func_label
        else:
            extra = self.memory.lattice_type.bottom()

        combined = joined_inputs + extra

        for var in flow.outputs:
            current = self.get_label(var)
            new_label = current + combined
            self.memory.dynamic_vars[var] = new_label

    def analyze_flow(self, flow: Flow) -> None:
        if isinstance(flow, ExplicitFlow):
            self.apply_explicit_flow(flow)
        elif isinstance(flow, SequenceFlow):
            for subflow in flow.flows:
                self.analyze_flow(subflow)
        elif isinstance(flow, RepetitionFlow):
            # For loops: perform fixpoint iteration until the dynamic labels stabilize.
            stable = False
            while not stable:
                before = self.memory.clone()
                self.analyze_flow(flow.body)
                after = self.memory
                stable = before.dynamic_vars == after.dynamic_vars

    def analyze_ast(self, source_ast: ast.AST) -> None:
        parser = FlowParser()
        flow = parser.parse(source_ast)
        self.analyze_flow(flow)
