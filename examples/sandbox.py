import argparse
import logging

from pydantic import BaseModel
from sentinel.execute.orchestrator import PlanOrchestrator
from sentinel.schema.abstract import AbstractPlan
from sentinel.schema.concrete import CustomTool


class MultiplierArgs(BaseModel):
    a: int
    b: int


class MultiplierOutput(BaseModel):
    product: int


multiplier_source = "def main(a : int, b : int):\n    # Multiply the two numbers\n    return a * b\n"


class CountRsArgs(BaseModel):
    text: str


class CountRsOutput(BaseModel):
    count: int


count_rs_source = (
    "def main(text : str):\n"
    "    # Count 'r' (case-insensitive) in the text\n"
    "    return sum(1 for c in text if c.lower() == 'r')\n"
)

multiplier_tool = CustomTool(
    name="Multiplier",
    id="multiplier_tool",
    description="Multiplies two numbers.",
    clearances={"basic"},
    permissions=set(),
    provider="demo_provider",
    args_schema=MultiplierArgs,
    output_schema=MultiplierOutput,
    source_code=multiplier_source,
)

count_rs_tool = CustomTool(
    name="CountRs",
    id="count_rs_tool",
    description="Counts the number of 'r' characters in a string.",
    clearances={"basic"},
    permissions=set(),
    provider="demo_provider",
    args_schema=CountRsArgs,
    output_schema=CountRsOutput,
    source_code=count_rs_source,
)

plan_script = """
def main():
    display("Plan started.")
    mult_result = invoke("Multiplier", a=3, b=7)
    display("Multiplier result: " + str(mult_result))
    count_result = invoke("CountRs", text="Rural road in the rain.")
    display("CountRs result: " + str(count_result))
    return "Success"
final_output = main()
"""


def main():
    # Create the plan
    plan = AbstractPlan(script=plan_script, abs_tools=[])
    tools = {
        "Multiplier": multiplier_tool,
        "CountRs": count_rs_tool,
    }

    # Initialize the Orchestrator with plan and tools
    with PlanOrchestrator(plan, tools) as orchestrator:
        orchestrator.launch()
        orchestrator.join()

        print("[Final Output]:", orchestrator.result)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-v", "--verbose", action="store_true")
    args.add_argument("-vv", "--very-verbose", action="store_true")

    args = args.parse_args()

    lvl = logging.DEBUG if args.very_verbose else logging.INFO if args.verbose else logging.WARNING

    logging.basicConfig(level=lvl)
    main()
