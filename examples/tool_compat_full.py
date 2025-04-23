from pydantic import create_model, Field
from dotenv import load_dotenv
import json

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from sentinel.tools.manager import ToolManager
from sentinel.schema import AbstractPlan, AbstractTool
from sentinel.execute import PlanOrchestrator
from sentinel.plan.concrete import SimpleConcretePlanner


def main():
    load_dotenv()

    tool_manager = ToolManager.from_manifest("tools/manifest.json")
    quickride = tool_manager.get_by_name("QuickRide")

    # Generate a dummy abstract plan and abstract tools
    arg_schema = create_model(
        "RideShareArgs",
        src=(str, Field(..., description="Starting location for the ride")),
        dst=(str, Field(..., description="Destination location for the ride"))
    )
    output_schema = create_model(
        "RideShareOutput",
        fare=(float, Field(..., description="Fare for the ride"))
    )

    abstract_tool = AbstractTool(
        name="RideShare",
        description="Generic rideshare service",
        args_schema=arg_schema,
        output_schema=output_schema
    )

    plan_script = (
        "src : str = 'Main Street'\n"
        "dst : str = 'Cooper Street'\n"
        "v1 = RideShare(src=src, dst=dst)\n"
        "display(f'Fare: {v1}')\n"
    )

    plan = AbstractPlan(script=plan_script, abs_tools=[abstract_tool])

    base_llm = ChatOpenAI(
        model="Qwen/Qwen2.5-72B-Instruct",
        temperature=0.0,
        openai_api_base="http://localhost:8000/v1"
    )
    embedding_model = OpenAIEmbeddings()
    concrete_planner = SimpleConcretePlanner(
        tool_manager=tool_manager,
        base_llm=base_llm,
        embedding_model=embedding_model
    )

    # Test matching: raw llm invocation
    result = concrete_planner.compat_chain.invoke({
        "abstract_tool": abstract_tool.as_json(),
        "concrete_tool": quickride.as_json()
    })
    print("Abstract JSON:")
    print(json.dumps(json.loads(abstract_tool.as_json()), indent=2))
    print("Concrete JSON:")
    print(json.dumps(json.loads(quickride.as_json()), indent=2))
    print("Compatibility result:")
    print(result)

    # Test matching capability
    matches = concrete_planner.generate_matches_for_tool(abstract_tool)

    print("Matched tools:")
    for match in matches:
        print(match.name)
        print(match.description)
        print(match.args_schema)
        print(match.output_schema)
        print(match.generate_source())
        print()

    tool_mapping = concrete_planner.implement_plan(plan)

    print("Abstract Plan script:")
    print(plan.script)

    print("Compiled plan script:")
    print(plan.compile_for_protocol())

    print("Base tool code:")
    print(quickride.generate_source())

    print("Type-conformed tool code:")
    for tool in tool_mapping.values():
        print(tool.generate_source())
        print()

    with PlanOrchestrator(plan=plan, tools=tool_mapping) as orchestrator:
        orchestrator.launch()
        orchestrator.join()


if __name__ == "__main__":
    main()
