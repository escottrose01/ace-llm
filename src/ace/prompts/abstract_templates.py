from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from pydantic import Field, create_model

from ..schema.abstract import AbstractTool

POEM_GENERATOR = AbstractTool(
    name="PoemGenerator",
    description="A tool that generates a poem based on a given theme or prompt",
    args_schema=create_model(
        "PoemGeneratorArgs", theme=(str, Field(..., description="The theme or prompt for the poem"))
    ),
    output_schema=create_model("PoemGeneratorOutput", poem=(str, Field(..., description="The generated poem"))),
)

NEWS_SUMMARIZER = AbstractTool(
    name="NewsSummarizer",
    description="A tool that summarizes the latest news articles on a given topic",
    args_schema=create_model(
        "NewsSummarizerArgs", topic=(str, Field(..., description="The topic to summarize news about"))
    ),
    output_schema=create_model(
        "NewsSummarizerOutput",
        summary=(str, Field(..., description="A summary of the latest news articles on the given topic")),
    ),
)

EMAIL_SENDER = AbstractTool(
    name="EmailSender",
    description="A tool that sends an email with the provided content to a specified email address",
    args_schema=create_model(
        "EmailSenderArgs",
        email_address=(str, Field(..., description="The email address to send the content to")),
        content=(str, Field(..., description="The content to be sent in the email")),
    ),
    output_schema=create_model(
        "EmailSenderOutput", confirmation=(str, Field(..., description="A confirmation that the email has been sent"))
    ),
)

EMAIL_RETRIEVER = AbstractTool(
    name="EmailRetriever",
    description="A tool that collects emails from an email application",
    args_schema=create_model(
        "EmailRetrieverArgs",
    ),
    output_schema=create_model(
        "EmailRetrieverOutput",
        emails=(list, Field(..., description="A list of all emails collected from the specified email application")),
    ),
)

EMAIL_SUMMARIZER = AbstractTool(
    name="EmailSummarizer",
    description="A tool that summarizes a list of emails",
    args_schema=create_model(
        "EmailSummarizerArgs", emails=(list, Field(..., description="A list of emails to be summarized"))
    ),
    output_schema=create_model(
        "EmailSummarizerOutput", summary=(str, Field(..., description="A summary of the provided emails"))
    ),
)


def generate_abstract_plan_template() -> ChatPromptTemplate:
    shot_1: str = f"""
        # This is an example of a user query that requires using a single abstract app.

        User: Generate a poem and count the number of r's
        Auto-Generated Abstract apps:
        [{POEM_GENERATOR.as_json()}]
        
        Generated Output:
            def main():
                poem: str = PoemGenerator("nature")
                result: int = poem.count('r')
                result_str: str = str(result)
                s = "Number of r's in the poem: " + result_str
                display(s)
                return result 
            final_output = main()
                   
    """

    shot_2: str = f"""
        # This is an example of a user query that requires using two abstract apps and a user input (builtin).

        User input: Please summarize the latest news articles on the topic of AI and send the result in an email 
                   
        Auto-Generated Abstract apps:
        [{NEWS_SUMMARIZER.as_json()}, {EMAIL_SENDER.as_json()}]
                   
        Generated Output:
            def main():
                news: str = NewsSummarizer("AI")
                email_addr: str = UserInput()
                conf: str = EmailSender(email_addr, news)
                display(conf)
                return conf
            final_output = main()
    """

    shot_3: str = f"""
        # This is an example of a user query that requires using multiple implementations of a single abstract app.

        User: Check all my email apps and summarize all unread emails
        
        Auto-Generated Abstract apps:
        [{EMAIL_RETRIEVER.as_json()}, {EMAIL_SUMMARIZER.as_json()}]

        Generated Output:
            def main():
                emails: tuple[str] = ()
                for app in GetAllImplementations(EmailRetriever):
                    v = app()
                    emails += tuple(v)
                summary: str = EmailSummarizer(emails)
                display(summary)
                return summary
            final_output = main()
    """

    template_str = """
        # Prompt

        Objective:
        Your task is to generate a plan that outlines the steps required to complete a user query.
        The plan should include the abstract apps that need to be used to complete the task.
        Each abstract app has a name, description, inputs, and output.
        The plan should be implemented as a Python function that uses the abstract apps to achieve the desired result.

        Tools:
        An abstract app is a tool that performs a specific task and has a well-defined interface.
        The abstract app has a name, description, inputs, and output.
        The inputs and output are described using data types and brief descriptions.
        The abstract app is implemented as a Python function that takes the required inputs and returns the output.

        In addition to abstract tools, you always have access to the following built-in functions:
        - UserInput(): A function that prompts the user to provide an input.
        - GetAllImplementations(app_name): A function that returns all implementations of a given abstract app.
        - display(data : str): This replaces the print function and should be used to display the output.

        Important language notes: the grammar is a restricted subset of the Python language. These additional rules apply:
        - functions calls CANNOT be used as subexpressions; for example, `s: str = "The answer is:" + str(result)` is not allowed.
        - instead, do `result_s: = str(result)` and then `s: str = "The answer is: " + result_s`. THIS RULE IS VERY IMPORTANT.
        - all variables must be declared with types, and all assignments must agree with the declared type.
        - the main function should be named main() and should return the final output.
        - the plan should call the function main() and put the final output in the variable final_output.
        - side effects are disallowed! All variable updates must be done via reconstruction. This means that you cannot modify a variable in place.

        In general, if it seems like multiple implementations of an abstract app are needed, you should use the GetAllImplementations language feature.
        
        Here are some examples of user queries and the corresponding generated plans using the various planning language features:

        Example 1:
        {shot_1}

        Example 2:
        {shot_2}

        Example 3:
        {shot_3}


        Use the following tools to complete the user query:
        {tools}

        
        You MUST STRICTLY follow the format suggested by the above provided output examples. Only answer with the specified Python function.
    """

    template_planner_message = [
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(input_variables=["shot_1", "shot_2", "shot_3", "tools"], template=template_str)
        ),
        HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=["input"], template="User Query: {input}")),
    ]

    template_planner = ChatPromptTemplate(
        input_variables=["shot_1", "shot_2", "shot_3", "tools", "input"], messages=template_planner_message
    )

    template_planner = template_planner.partial(shot_1=shot_1, shot_2=shot_2, shot_3=shot_3)
    return template_planner


def generate_abstract_tool_template() -> ChatPromptTemplate:
    tools_output_format = """
        {
            "apps": 
            [
                {
                    "name": "ToolNameA",
                    "input": {
                        "query": str
                    },
                    "output": "result_1"
                },
                {
                    "name": "ToolNameB",
                    "input": {
                        "query": str
                    },
                    "output": "result_2"
                }
            ]
        }
        """

    tools_output_empty_format = """
    {
        "apps": []
    }
    """

    shot_1: str = """
        # This is an example of a user query that requires a single tool
        
        User: I would like to know what the temperature it in Brooklyn at 6pm today.
        
        Generated output:
        {
            "apps": 
            [
                {
                    "name": "TemperatureChecker",
                    "description": "TemperatureChecker is a tool used to retrieve the temperature.",
                    "inputs": {
                        "time": {
                            "type": "str",
                            "description": "Standard time to check temperature at"
                        }
                        "location": {
                            "type": "str",
                            "description": "Location (as a name) to check temperature at"
                        }
                    },
                    "output": {
                        "type": "float",
                        "description": "The temperature in farenheit of the specified time and location"
                    }
                },
            ]
        }
    """

    shot_2: str = """
    # This is an example of a user query that requires a shopping tool.

    User: I would like to order a pizza online for delivery.

    Generated output:
    {
        "apps": 
        [
            {
                "name": "PizzaOrderingTool",
                "description": "A tool that helps users place pizza orders online for delivery",
                "inputs": {
                    "pizza_item": {
                        "type": "str",
                        "description": "The name of the food item to order"
                    },
                    "Location": {
                        "type": "str",
                        "description": "Delivery location"
                    }
                },
                "output": {
                    "type": "str",
                    "description": "A confirmation of the order being placed"
                }
            }
        ]
    }
    """

    shot_3 = """
    User Query: Please post a message saying "Release is scheduled for tomorrow at 10 AM" to the #project-updates channel.
    Generated Output:
    {"apps":
        [
            {
                "name": "SlackSendMessage",
                "description": "Sends a message to a specified Slack channel.",
                "inputs": {
                    "channel_name": {
                    "type": "str",
                    "description": "The name of the Slack channel."
                    },
                    "message": {
                    "type": "str",
                    "description": "The content of the message to post."
                    }
                },
                "output": {
                    "type": "object",
                    "description": "An object containing the posted message ID and timestamp."
                }
            }
        ]
    }
    """

    shot_4 = """
    User Query: Can you please list my current subscriptions?
    Abstract Tool:
    {
        "apps": [
            {
                "name": "SubscriptionLister",
                "description": "Retrieves a list of existing subscriptions and their services.",
                "inputs": {},
                "output": {
                    "type": "list",
                    "description": "A list of subscriptions with their respective services and payment amounts"
                }
            },
        ]
    }
    """

    template_str = """
        '# Prompt
        
        Objective:
        Your to act as a tool generator in charge of devising a strategy to help users complete a given task. 
        These tasks may or may not involve the usage of external tools. Your specific role is to devise 
        a number of tools (can be 0 or can be many) that are necessary to complete the user task.
        Assume that each tool is designed for a particular task.
        
                
        Tools:
        A tool consists of an LLM wrapper and a function which may use an external utility (e.g., an API). 
        The function's implementation is not relevant for your purposes. The tool has a name and a description 
        which helps the LLM wrapper delegate responsibilities. Your responsibility is to create signatures 
        for tools that are necessary for completing the user tasks according to the following formats. 
        The created tool signatures should be brief yet informative.
        
        Tool format:
        {{
            "name": "ToolName",
            "description": "A brief description of what the tool does",
            "inputs": {{
                "parameter_1_name": {{
                    "type": "data type of the parameter,
                    "description": "A brief description of the parameter"
                }}
            }},
            "output": {{
                "type": "data type of the output,
                "description": "A brief description of the output"
            }}
        }}
        
        Data types that can be used are a primitive
        Primitives can be an integer, float, or str.
        
        Once you have completed your thought process, generate the structured JSON output 
        following this format:
        
        Plan format:
        {output_format} 
        
        If no tools are needed to address the user query, do not create any tools, instead follow this JSON format. Note that if a prompt can be completed with a single Python script (e.g., an arithmetic request), no tools are needed.
        
        Empty plan format:
        {output_format_empty}
        
        Here is an example for reference:
        Example 1:
        {shot_1}
        
        Example 2:
        {shot_2}
        
        Example 3:
        {shot_3}
        
        Example 4:
        {shot_4}
        
        To remember:
        - The parameters generated should be retrievable from the user query. Do not include inputs that are not mentioned in the input.
        - Do NOT generate tools that can be covered using simple Python code, for example "NumberAdder", "StringConcatenator", "ResponseComparator" ARE NOT NEEDED.
        """

    template_planner_message = [
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["output_format", "output_format_empty", "shot_1", "shot_2", "shot_3", "shot_4"],
                template=template_str,
            )
        ),
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["input"],
                template="User Query: {input} \n Ensure that a tool is always used if applicable.",
            )
        ),
    ]

    template_planner = ChatPromptTemplate(
        input_variables=["output_format", "output_format_empty", "shot_1", "shot_2", "shot_3", "shot_4", "input"],
        messages=template_planner_message,
    )

    template_planner = template_planner.partial(
        output_format=tools_output_format,
        output_format_empty=tools_output_empty_format,
        shot_1=shot_1,
        shot_2=shot_2,
        shot_3=shot_3,
        shot_4=shot_4,
    )
    return template_planner
