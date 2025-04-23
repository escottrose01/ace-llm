from pydantic import create_model

from sentinel.tools.manager import ToolManager
from sentinel.schema import AbstractPlan, AbstractTool
from sentinel.execute import PlanOrchestrator
from sentinel.plan.concrete import SchemaAdaptedTool


def main():
    tool_manager = ToolManager.from_manifest("tools/manifest.json")
    quickride = tool_manager.get_by_name("QuickRide")

    # Generate a dummy abstract plan and abstract tools
    arg_schema = create_model("RideShareArgs", src=(str, ...), dst=(str, ...))
    output_schema = create_model("RideShareOutput", fare=(float, ...))
    abs_tool = AbstractTool(
        name="RideShare",
        description="Generic rideshare service",
        args_schema=arg_schema,
        output_schema=output_schema
    )

    input_mapping = "def input_mapping(src, dst): return {'start_point': src, 'end_point': dst}"
    output_mapping = "def output_mapping(fare): return fare"

    print(input_mapping, output_mapping)

    wrapped_quickride = SchemaAdaptedTool(
        name="QuickRide",
        provider="QuickRide Inc.",
        description="Generic rideshare service",
        permissions=set(),
        clearances=set(),
        args_schema=abs_tool.args_schema,
        output_schema=abs_tool.output_schema,
        wrapped_tool=quickride,
        input_mapping_source=input_mapping,
        output_mapping_source=output_mapping
    )

    abs_tools = [
        AbstractTool(
            name="RideShare",
            description="Generic rideshare service",
            args_schema=arg_schema,
            output_schema=output_schema
        )
    ]

    plan_script = (
        "src : str = 'Main Street'\n"
        "dst : str = 'Cooper Street'\n"
        "v1 = RideShare(src=src, dst=dst)\n"
        "display(f'Fare: {v1}')\n"
    )

    plan = AbstractPlan(script=plan_script, abs_tools=abs_tools)

    tool_mapping = {
        "RideShare": wrapped_quickride
    }

    print("Abstract Plan script:")
    print(plan.script)

    print("Compiled plan script:")
    print(plan.compile_for_protocol())

    print("Base tool code:")
    print(quickride.generate_source())

    print("Type-conformed tool code:")
    print(wrapped_quickride.generate_source())
    print()

    with PlanOrchestrator(plan=plan, tools=tool_mapping) as orchestrator:
        orchestrator.launch()
        orchestrator.join()


if __name__ == "__main__":
    main()
