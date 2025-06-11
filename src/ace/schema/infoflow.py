from abc import ABC, abstractmethod
from copy import deepcopy

from .lattice import Lattice


class MemoryModel:
    dynamic_vars: dict[str, Lattice]
    static_vars: dict[str, Lattice]
    lattice_type: type[Lattice]

    def __init__(
        self,
        lattice_type: type[Lattice],
        dynamic_vars: dict[str, Lattice] | None = None,
        static_vars: dict[str, Lattice] | None = None,
    ):
        self.lattice_type = lattice_type
        self.dynamic_vars = dynamic_vars if dynamic_vars is not None else {}
        self.static_vars = static_vars if static_vars is not None else {}

    def clone(self) -> "MemoryModel":
        cloned = MemoryModel(self.lattice_type)
        cloned.dynamic_vars = deepcopy(self.dynamic_vars)
        cloned.static_vars = deepcopy(self.static_vars)
        return cloned


class Flow(ABC):
    @abstractmethod
    def dump(self, indent: int = 0) -> str:
        pass


class ExplicitFlow(Flow):
    outputs: list[str]
    inputs: list[str]
    function: str

    def __init__(self, outputs: list[str], inputs: list[str], function: str):
        self.outputs = outputs
        self.inputs = inputs
        self.function = function

    def dump(self, indent: int = 0) -> str:
        indent_str = " " * indent
        inputs_str = ", ".join(self.inputs)
        outputs_str = ", ".join(self.outputs)
        if self.outputs:
            return f"{indent_str}{outputs_str} <- {self.function}({inputs_str})\n"
        else:
            return f"{indent_str}{self.function}({inputs_str})\n"


class SequenceFlow(Flow):
    flows: list[Flow]

    def __init__(self, flows: list[Flow]):
        self.flows = flows

    def dump(self, indent: int = 0) -> str:
        return "".join([f.dump(indent) for f in self.flows])


class RepetitionFlow(Flow):
    def __init__(self, body: Flow):
        self.body = body

    def dump(self, indent: int = 0) -> str:
        indent_str = " " * indent
        return f"{indent_str}LOOP:\n{self.body.dump(indent + 2)}"
