import json

from agent import InjecSentinelAgent

from ace.schema.concrete import CustomTool
from ace.tools.helper import parse_schemas_from_json


def fill_schema():
    # Puts all tool data in schema_db.jsonl
    with open("injec_eval/res/schema_db.jsonl", "w") as f:
        with open("inject_eval/res/tools.json") as g:
            toolkits = json.load(g)
            for kit in toolkits:
                toolkit_name = kit["toolkit"]
                for tool in kit["tools"]:
                    tool["name"] = toolkit_name + tool["name"]
                    signature = InjecSentinelAgent.translate_function_signature(tool)
                    f.write(str(signature) + "\n")


def fill_manifest():
    # Puts all tool data in manifest
    exception = "GmailSendEmail"
    user_tools: set[str] = {
        "AmazonGetProductDetails",
        "EvernoteManagerSearchNotes",
        "GitHubGetRepositoryDetails",
        "GitHubSearchRepositories",
        "GmailReadEmail",
        "GmailSearchEmails",
        "GoogleCalendarGetEventsFromSharedCalendar",
        "GoogleCalendarReadEvents",
        "ShopifyGetProductDetails",
        "TeladocViewReviews",
        "TodoistSearchTasks",
        "TwilioGetReceivedSmsMessages",
        "TwitterManagerGetUserProfile",
        "TwitterManagerReadTweet",
        "TwitterManagerSearchTweets",
        "WebBrowserNavigateTo",
    }
    with open("sentinel/injecagent/injec_manifest.jsonl", "w") as f:
        with open("sentinel/injecagent/tools.json") as g:
            toolkits = json.load(g)
            for kit in toolkits:
                toolkit_name = kit["toolkit"]
                for tool in kit["tools"]:
                    tool["name"] = toolkit_name + tool["name"]
                    if tool["name"] == exception:
                        continue
                    signature = InjecSentinelAgent.translate_function_signature(tool)
                    param_schema, return_schema = parse_schemas_from_json(signature)
                    function = InjecSentinelAgent.generate_code(tool, tool["name"] in user_tools)

                    try:
                        impl_tool = CustomTool(
                            name=tool["name"],
                            provider="InjecAgent",
                            description=tool["summary"],
                            permissions=set(),
                            clearances=set(),
                            args_schema=param_schema,
                            output_schema=return_schema,
                            source_code=function,
                        )
                    except Exception as e:
                        print("Error:", e)
                        print("Tool:", tool["name"])
                        print("Function:", function)

                    new_entry = {
                        "name": impl_tool.name,
                        "provider": impl_tool.provider,
                        "description": impl_tool.description,
                        "tool_type": "custom",
                        "path": "",  # NOTE: Implement later?
                        "code": impl_tool.source_code,
                        "clearances": ["general"],
                        "permissions": [],
                        "type": "user" if tool["name"] in user_tools else "attacker",
                    }

                    f.write(str(new_entry) + "\n")
