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
    shot_1: str = f"""\
## Example: User query using a single abstract app with structured output.
Query: Generate a poem and count the number of r's
<Available Tools>
{POEM_GENERATOR.as_json()}
</Available Tools>

Generated Output:
```python
def main():
    poem_result: PoemGenerator = PoemGenerator(theme="nature")
    poem: str = poem_result.poem
    r_count: int = 0
    poem_length: int = len(poem)

    for i in range(poem_length):
        eq: bool = poem[i] == 'r'
        if eq:
            r_count += 1

    r_count_str: str = str(r_count)
    s: str = "Number of r's in the poem: " + r_count_str
    display(s)
    return r_count
final_output = main()
```
"""

    shot_2: str = f"""\
## Example: User query using two abstract apps, with structured outputs.
Query: Please summarize the latest news articles on the topic of AI and send the result in an email to john@gmail.com
<Available Tools>
{NEWS_SUMMARIZER.as_json()}
{EMAIL_SENDER.as_json()}
</Available Tools>

Generated Output:
```python
def main():
    news_result: NewsSummarizer = NewsSummarizer(topic="AI")
    news_summary: str = news_result.summary
    email_addr: str = "john@gmail.com"
    email_result: EmailSender = EmailSender(email_address=email_addr, content=news_summary)
    confirmation: str = email_result.confirmation
    display(confirmation)
    return confirmation
final_output = main()
```
"""

    shot_3: str = """\
## Example: User query using control flow.
User: Pick a string, and count the number of 'r's in it. if the number is odd, display "Hello".
Auto-Generated Abstract apps:
<Available Tools>
</Available Tools>

Generated Output:
```python
def main():
    magic_string: str = "abracadabra"
    r_count: int = 0
    length: int = len(magic_string)
    for i in range(length):
        eq: bool = magic_string[i] == 'r'
        if eq:
            r_count += 1
    is_odd: bool = r_count % 2 == 1
    if is_odd:
        display("Hello")
    return r_count
final_output = main()
```
"""

    system_message = """\
# Task
Generate a Python function plan to solve the user query using only the provided abstract apps and permitted built-in functionality.

Key Requirements:
- Each plan should be a Python function that combines the necessary abstract apps, returning the final result.
- Use the provided abstract apps, each with a name, description, and defined inputs and outputs.
- Follow all programming constraints and output format examples given below.

Allowed Built-ins:
- abs, bool, float, int, str, frozenset, all, any, len, pow, round, sum

Blacklisted Built-ins (not allowed):
- open, exec, eval, compile, __import__, input, globals, locals, vars, dir, help, exit, quit, getattr, setattr, delattr, super, memoryview

Import Restrictions:
- Only the "math" module may be imported. Do not use any other imports.

Special Tool Provided:
- display(data: str): use this function to display output (do not use print).

Programming Language and Style Rules:
- Do not use function calls as subexpressions; assign results before continuing.
- Do not use functions that are not built-in or provided abstract apps.
- Declare all variables with types, and ensure assignments match these types.
- Assign outputs of tools to a variable that has the correct type annotation.
- Extract fields from tool outputs into variables with accurate types before using them.
- The main function must be named main(), and it should return the computed output.
- Assign the return value of main() to final_output.
- Avoid side effects or in-place updates to variables; instead, recompute values when necessary.

Instructions:
- Follow the syntactic and semantic rules above.
- Output only the required code without additional commentary, markdown code blocks, explanations, or text.
- Use only the tools, built-ins, and modules specified above.

# Output Format

Directly output the Python function plan as specified, without markdown code blocks or explanatory text. The output must match the structure shown in the examples.

# Notes
- When displaying data, use only the display function provided.
- Any use of blacklisted functionality or deviations from the specified style will result in invalid output.

# Examples
<WARNING> Do not use the tools used in the below examples. </WARNING>
<Examples>
{shot_1}
{shot_2}
{shot_3}
</Examples>
"""

    tools_message = """\
# Tools
You may use the following *below* abstract apps. Any apps listed prior to this message are not available to you, and you *should not* use them.
<Available Tools>
{tools}
</Available Tools>
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
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(input_variables=["shot_1", "shot_2", "shot_3"], template=system_message)
        ),
        SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=["tools"], template=tools_message)),
        SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=["context"], template=context_message)),
        HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=["query"], template=human_message)),
    ]

    template_planner = ChatPromptTemplate(
        input_variables=["shot_1", "shot_2", "shot_3", "tools", "context", "query"], messages=template_planner_message
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
