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
    system_message = """\
# Task
Generate a list of external tool schemas relevant to a given user task. Only enumerate possible APIs, wrappers, or utilities that might help a downstream agent solve the task. For trivial requests solvable with basic code (e.g., arithmetic), output none.

Each tool schema must be a JSON object and must include:
- name (PascalCase, e.g., "WeatherLookup" or "StockPriceQuery")
- description
- inputs: (type and description for each parameter, only if required)
- output: (type and description)

Allowed types: int, float, str, bool

Do not solve, break down, or sequence the user's task. Output only the relevant tool schemas in the format:
{{
  "apps": [ ...tool schemas... ]
}}

If no tools are needed, output:
{{
  "apps": []
}}

# Output Format
Return valid JSON only:
- Field: "apps"
- Value: list of tool schemas as defined above (or empty list)
- Ensure every tool "name" is written in PascalCase

# Example Inputs / Outputs
User: What's the temperature in Brooklyn at 6pm today?
Output: {{ "apps": [ {{ "name": "WeatherLookup", "description": "...", "inputs": {{...}}, "output": {{...}} }} ] }}

User: What is 3 + 4?
Output: {{ "apps": [] }}

# Notes
- Always use PascalCase for the "name" field of each tool schema.
- Do not attempt to answer, solve, or decompose the user's task.
- Only output tool schemas in the specified format.
- Strongly prefer single return values for outputs
"""

    context_message = """\
# Additional System Context
The following context clarifies the operational environment of the agent. Use this context to shape the types of tools generated.
{context}
"""

    human_message = """\
{query}
"""

    template_planner_message = [
        SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template=system_message)),
        SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=["context"], template=context_message)),
        HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=["query"], template=human_message)),
    ]

    template_planner = ChatPromptTemplate(
        input_variables=["context", "query"],
        messages=template_planner_message,
    )

    return template_planner
