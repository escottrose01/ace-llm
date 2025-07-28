import pydantic
from ace.schema.concrete import ConcreteToolBase


class MockTool(ConcreteToolBase):
    return_val: str

    def generate_source(self) -> str:
        escaped_val = self.return_val.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
        return f"def main(*args, **kwargs):\n    return {{'return_val': '{escaped_val}'}}\n"


def to_pascal_case(snake_str: str) -> str:
    components = snake_str.split("_")
    return "".join(x[:1].upper() + x[1:] for x in components)


def make_mock_tool(name: str, description: str, return_val: str) -> MockTool:
    return MockTool(
        name=to_pascal_case(name),
        provider="asb",
        description=description,
        clearances=set(),
        permissions=set(),
        args_schema=pydantic.create_model("MockToolArgs"),
        output_schema=pydantic.create_model("MockToolReturn", return_val=(str, ...)),
        return_val=return_val,
    )
