from pydantic import create_model
from sentinel.execute import PlanOrchestrator
from sentinel.schema import AbstractPlan, AbstractTool
from sentinel.tools.manager import ToolManager


def main():
    tool_manager = ToolManager.from_manifest("tools/manifest.json")

    # Print out all the loaded tools and their source codes
    for tool in tool_manager.tools:
        if tool is not None:
            print(tool.name)
            print(tool.description)
            print(tool.args_schema.model_fields)
            print(tool.output_schema.model_fields)
            print(tool.generate_source())
            print()

    # Generate a dummy abstract plan and abstract tools
    arg_schema = create_model("RideShareArgs", src=(str, ...), dst=(str, ...))
    output_schema = create_model("RideShareOutput", fare=(float, ...))
    abs_tools = [
        AbstractTool(
            name="RideShare__1",
            description="Generic rideshare service",
            args_schema=arg_schema,
            output_schema=output_schema,
        ),
        AbstractTool(
            name="RideShare__2",
            description="MetroHail rideshare service",
            args_schema=arg_schema,
            output_schema=output_schema,
        ),
    ]

    # TODO: In the "implementor", we still need to map the arguments appropriately
    # I.e., abstract args -> concrete args
    #       abstract outputs -> concrete outputs
    plan_script = (
        "src : str = 'Main Street'\n"
        "dst : str = 'Cooper Street'\n"
        "v1 = RideShare__1(start_point=src, end_point=dst)\n"
        "v2 = RideShare__2(start_point=src, end_point=dst)\n"
        "if v1 < v2:\n"
        "    display(f'v1 is cheaper at {v1}')\n"
        "else:\n"
        "    display(f'v2 is cheaper at {v2}')\n"
    )

    plan = AbstractPlan(script=plan_script, abs_tools=abs_tools)

    tool_mapping = {
        "RideShare__1": tool_manager.get_by_name("QuickRide"),
        "RideShare__2": tool_manager.get_by_name("MetroHail"),
    }

    print(f"Plan script: \n{plan.script}\n")
    print(f"Plan compiled for protocol: \n{plan.compile_for_protocol()}\n")

    with PlanOrchestrator(plan=plan, tools=tool_mapping) as orchestrator:
        orchestrator.launch()
        orchestrator.join()


if __name__ == "__main__":
    main()
