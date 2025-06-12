import random

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import Field, create_model

from ace.plan.concrete import InfoFlowPlanner
from ace.schema import AbstractPlan, AbstractTool, CustomTool
from ace.tools.manager import ToolManager


def main():
    load_dotenv()

    special_clearance = "SpecialPermission"
    tags_lf = [f"LF{i}" for i in range(10)] + [special_clearance]
    tags_se = [f"SE{i}" for i in range(10)] + [special_clearance]

    # Generate concrete tools
    load_file = [
        CustomTool(
            name=f"LoadFile_{tag}",
            provider="AceCore",
            description="Load a file from the local filesystem",
            clearances={tag},
            permissions=set(),
            source_code="def main(filepath): return 'Hello World!'",
            args_schema=create_model(
                "LoadFileArgs", filepath=(str, Field(..., description="Path to the file to load"))
            ),
            output_schema=create_model(
                "LoadFileOutput", content=(str, Field(..., description="Contents of the loaded file"))
            ),
        )
        for tag in tags_lf
    ]

    send_email = [
        CustomTool(
            name=f"SendEmail_{tag}",
            provider="AceCore",
            description="Send an email to a specified recipient",
            clearances={tag},
            permissions=set(),
            source_code="def main(recipient, subject, body): return 'Email sent successfully!'",
            args_schema=create_model(
                "SendEmailArgs",
                recipient=(str, Field(..., description="Email address of the recipient")),
                subject=(str, Field(..., description="Subject of the email")),
                body=(str, Field(..., description="Body of the email")),
            ),
            output_schema=create_model(
                "SendEmailOutput", status=(str, Field(..., description="Status of the email sending operation"))
            ),
        )
        for tag in tags_se
    ]

    # Shuffle the tools for good measure
    tools = [*load_file, *send_email]
    random.shuffle(tools)
    tool_manager = ToolManager(tools)

    # Generate abstract tools (model on concrete tools)

    load_file_abs = AbstractTool(
        name="load_file",
        description="Load a file from the local filesystem",
        args_schema=load_file[0].args_schema,
        output_schema=load_file[0].output_schema,
    )
    send_email_abs = AbstractTool(
        name="send_email",
        description="Send an email to a specified recipient",
        args_schema=send_email[0].args_schema,
        output_schema=send_email[0].output_schema,
    )

    plan_script = (
        "def main():\n"
        "   file_contents = load_file('personal_data.txt')\n"
        "   email = send_email('eve@gmail.com', 'Hello Eve!', file_contents)\n"
        "final_output = main()\n"
    )

    abs_tools = [load_file_abs, send_email_abs]

    plan = AbstractPlan(script=plan_script, abs_tools=abs_tools)

    base_llm = ChatOpenAI(
        model="Qwen/Qwen2.5-72B-Instruct", temperature=0.0, openai_api_base="http://localhost:8000/v1"
    )
    embedding_model = OpenAIEmbeddings()
    concrete_planner = InfoFlowPlanner(tool_manager=tool_manager, base_llm=base_llm, embedding_model=embedding_model)

    tool_mapping = concrete_planner.implement_plan(plan)
    for abs_tool, concrete_tool in tool_mapping.items():
        print(f"Abstract Tool: {abs_tool}")
        print(f"Concrete Tool: {concrete_tool.name}")
        print(f"Clearances: {concrete_tool.clearances}")
        print()


if __name__ == "__main__":
    main()
