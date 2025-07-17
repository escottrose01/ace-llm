from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from pydantic import Field, create_model

from ..schema.abstract import AbstractTool

# We generate the schemas dynamically in case any changes are made to the interfaces


# EXAMPLE 1 (valid): Weather Checker Tool
WEATHER_ABS_TOOL = AbstractTool(
    name="WeatherChecker",
    description="A tool to retrieve weather information for a given city at a specific time",
    args_schema=create_model(
        "WeatherCheckerArgs",
        city=(str, Field(..., description="Name of the city")),
        time=(str, Field(..., description="Time for the weather check")),
    ),
    output_schema=create_model(
        "WeatherCheckerOutput", temperature=(float, Field(..., description="Temperature in Celsius"))
    ),
).as_json()

WEATHER_CONCRETE_TOOL = AbstractTool(
    name="LadybugWeather",
    description="Retrieves the weather forecast for a given location",
    args_schema=create_model(
        "LadybugWeatherArgs",
        location=(str, Field(..., description="City name for weather lookup")),
        timestamp=(str, Field(..., description="Time at which to check the weather")),
    ),
    output_schema=create_model(
        "LadybugWeatherOutput", temp=(float, Field(..., description="Reported temperature in Fahrenheit"))
    ),
).as_json()

WEATHER_INPUT_MAPPING = "def input_mapping(city, time): return {'location': city, 'timestamp': time}"
WEATHER_OUTPUT_MAPPING = "def output_mapping(result): return {'temperature': 5/9 * (result.temp - 32)}"

# EXAMPLE 2 (valid): Currency Converter
CURRENCY_ABS_TOOL = AbstractTool(
    name="CurrencyConverter",
    description="A tool to convert an amount from one currency to another",
    args_schema=create_model(
        "CurrencyConverterArgs",
        amount=(float, Field(..., description="Amount to convert")),
        source_currency=(
            str,
            Field(..., description="Currency code to convert from (e.g., USD, EUR)", min_length=3, max_length=3),
        ),
        target_currency=(
            str,
            Field(..., description="Currency code to convert to (e.g., USD, EUR)", min_length=3, max_length=3),
        ),
    ),
    output_schema=create_model(
        "CurrencyConverterOutput", converted_amount=(float, Field(..., description="Amount in the target currency"))
    ),
).as_json()

CURRENCY_CONCRETE_TOOL = AbstractTool(
    name="MoneyMoneyMoney",
    description="Converts an amount from one currency to another",
    args_schema=create_model(
        "MoneyMoneyMoneyArgs",
        amount=(float, Field(..., description="Amount to convert")),
        source=(str, Field(..., description="Currency code to convert from")),
        target=(str, Field(..., description="Currency code to convert to")),
    ),
    output_schema=create_model(
        "MoneyMoneyMoneyOutput", result=(float, Field(..., description="Amount in the target currency"))
    ),
).as_json()

CURRENCY_INPUT_MAPPING = "def input_mapping(amount, source_currency, target_currency): return {'amount': amount, 'source': source_currency, 'target': target_currency}"
CURRENCY_OUTPUT_MAPPING = "def output_mapping(result): return {'result': result.result}"

# EXAMPLE 3 (invalid): Flight Tracker vs Flight Booker
FLIGHT_TRACKER_ABS_TOOL = AbstractTool(
    name="FlightTracker",
    description="A tool to track the status of a flight",
    args_schema=create_model(
        "FlightTrackerArgs", flight_number=(str, Field(..., description="Flight number to track"))
    ),
    output_schema=create_model(
        "FlightTrackerOutput", status=(str, Field(..., description="Current status of the flight"))
    ),
).as_json()

FLIGHT_BOOKER_CONCRETE_TOOL = AbstractTool(
    name="FlightBooker",
    description="Books a flight for a given itinerary",
    args_schema=create_model(
        "FlightBookerArgs",
        origin=(str, Field(..., description="Departure airport code")),
        destination=(str, Field(..., description="Arrival airport code")),
        date=(str, Field(..., description="Date of the flight")),
    ),
    output_schema=create_model(
        "FlightBookerOutput", booking_reference=(str, Field(..., description="Reference number for the booking"))
    ),
).as_json()

# EXAMPLE 4 (invalid): Poem writer
POEM_ABS_TOOL = AbstractTool(
    name="PoemWriter",
    description="A tool to generate a poem based on a given theme",
    args_schema=create_model("PoemWriterArgs", theme=(str, Field(..., description="Theme for the poem"))),
    output_schema=create_model("PoemWriterOutput", poem=(str, Field(..., description="Generated poem"))),
).as_json()

POEM_CONCRETE_TOOL = AbstractTool(
    name="PoemWriterPro",
    description="Creates a poem of the given type",
    args_schema=create_model(
        "PoemWriterProArgs", type=(str, Field(..., description="Type of the given poem (e.g. Haiku, Sonnet, Limerick)"))
    ),
    output_schema=create_model("PoemWriterProOutput", poem=(str, Field(..., description="Generated poem"))),
).as_json()

# Example 5: MUST INCLUDE UMBRELLA TERM (e.g TERM)

MESSAGE_ABSTRACT_TOOL = AbstractTool(
    name="MessageSender",
    description="A tool to send messages to a specified user",
    args_schema=create_model("MessageSenderArgs", username=(str, Field(..., description="The reciepients username"))),
    output_schema=create_model(
        "MessageSenderOutput", confirmation=(str, Field(..., description="A confirmation of the sent message"))
    ),
).as_json()

MESSAGE_CONCRETE_TOOL = AbstractTool(
    name="WhatsAppSender",
    description="Sends messages to a user on WhatsApp",
    args_schema=create_model(
        "WhatsAppSenderArgs", recipient_id=(str, Field(..., description="Unique Identifier of the recepient"))
    ),
    output_schema=create_model(
        "WhatsAppSenderOutput", result=(str, Field(..., description="A message detailing the result of the sent text"))
    ),
).as_json()

MESSAGE_INPUT_MAPPING = "def input_mapping(username): return {'username': recipient_id}"
MESSAGE_OUTPUT_MAPPING = "def output_mapping(result): return {'result': result.result}"


def generate_tool_compatibility_json_template() -> ChatPromptTemplate:
    # Shot 1: Valid Compatibility Mapping for a Weather Checking Tool
    shot_1 = f"""\
# Example 1: Valid Compatibility Mapping for a Weather Checking Tool

abstract_tool:
{WEATHER_ABS_TOOL}

concrete_tool:
{WEATHER_CONCRETE_TOOL}

Expected JSON output:
{{
"status": "success",
"input_mapping": "{WEATHER_INPUT_MAPPING}",
"output_mapping": "{WEATHER_OUTPUT_MAPPING}"
}}
    """

    # Shot 2: Valid Compatibility Mapping for Currency Conversion
    shot_2 = f"""\
# Example 2: Mapping for Currency Conversion

abstract_tool:
{CURRENCY_ABS_TOOL}

concrete_tool:
{CURRENCY_CONCRETE_TOOL}

Expected JSON output:
{{
"status": "success",
"input_mapping": "{CURRENCY_INPUT_MAPPING}",
"output_mapping": "{CURRENCY_OUTPUT_MAPPING}"
}}
    """

    # Shot 3: Invalid Compatibility Mapping for a Flight Tracker vs Flight Booker
    shot_3 = f"""\
# Example 3: Invalid Compatibility Mapping for a Flight Tracker vs Flight Booker

abstract_tool:
{FLIGHT_TRACKER_ABS_TOOL}

concrete_tool:
{FLIGHT_BOOKER_CONCRETE_TOOL}

Expected JSON output:
{{
"status": "failure",
"error": "Concrete tool description does not match abstract tool description"
}}
    """

    # Shot 4: Invalid Compatibility Mapping for a Poem Writer
    shot_4 = f"""\
# Example 4: Invalid Compatibility Mapping for a Poem Writer

abstract_tool:
{POEM_ABS_TOOL}

concrete_tool:
{POEM_CONCRETE_TOOL}

Expected JSON output:
{{
"status": "failure",
"error": "Input parameter names do not correspond appropriately to the abstract tool's inputs"
}}
    """

    shot_5 = f"""\
# Example 5: Valid Compatibility Mapping for a Message Sender

abstract_tool:
{MESSAGE_ABSTRACT_TOOL}

concrete_tool:
{MESSAGE_CONCRETE_TOOL}

Expected JSON output:
{{
    "status": "success",
        "input_mapping": "{MESSAGE_INPUT_MAPPING}",
    "output_mapping": "{MESSAGE_OUTPUT_MAPPING}"
}}
    """

    template_str = """\
### Prompt

Objective:
You are given two JSON objects:
- "abstract_tool": a JSON object describing the abstract tool, including its name, description, inputs, and output.
- "concrete_tool": a JSON object describing the concrete tool, including its name, description, inputs, and output.


Your job:
- Compare these two tools to determine compatibility.

Instructions:
- If the concrete tool is compatible with the abstract tool, output "status": "success", and include two additional fields:
    - "input_mapping": a string containing the Python code for a function named 'input_mapping'
    - "output_mapping": a string containing the Python code for a function named 'output_mapping'
- The mapping functions should convert the concrete tool's input parameters to match those expected by the abstract tool.

- If the concrete tool is incompatible (e.g., if its input parameter names do not correspond appropriately to the abstract tool's inputs), output "status": "failure" and an "error" field with an appropriate message.
- In addition to checking input and output schemas, you should verify that the tool descriptions appropriately match each other. If the concrete tool's description does not "implement" the abstract tool's description, the mapping is considered invalid, and you should output a failure condition.
- Assume the tool has exactly one output. When you write the output mapping function, return a single value of the type indicated in the abstract tool's output schema.
- In general, please try to match the tools if at all possible. It is acceptable to "massage" the inputs and outputs so they conform with the schemas.
- Tools should only be considered incompatable if their descriptions describe completely different purposes. If the descriptions are different yet describe the same overall purpose/funcitonality, then those tools should be considered compatable.
- Do not expect/enforce percision in the names of parameters. If two parameters refer to the same general idea, then they are compatible. For example, if one abstract tool has the parameter 'name', and the concrete tool has a paramter 'id', then they can be considered compatible
- output_mapping should only take one argument, do not expand the arguments if multiple are available.
- ***Do not use f-strings, instead use concatonation***

Examples:

{shot_1}

{shot_2}

{shot_3}

{shot_4}

{shot_5}

Remember:
    - **No** code or text outside a single JSON object.
    - If there is a mismatch in descriptions or fields that cannot be reconciled, output failure.
    - Ensure that all items being returned in "input_mapping" are valid parameters in the concrete tool
    - Vague terms such as "keyword", "term" or "id" are very flexible and should be treated as umbrella terms, where they can be compatible with other terms such as "name", "user", "email", etc

Before finalizing your output, check all variable names/references in output_mapping to confirm that they are present in the concrete tool's code. This means that 'output_mapping' should be able to run with the direct output of the source code as a parameter without an errors.
    """

    system_msg = SystemMessagePromptTemplate(
        prompt=PromptTemplate(input_variables=["shot_1", "shot_2", "shot_3", "shot_4", "shot_5"], template=template_str)
    )

    human_msg = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["abstract_tool", "concrete_tool"],
            template="Abstract Tool: {abstract_tool}\nConcrete Tool: {concrete_tool}",
        )
    )

    template = ChatPromptTemplate(
        input_variables=["shot_1", "shot_2", "shot_3", "shot_4", "shot_5", "abstract_tool", "concrete_tool"],
        messages=[system_msg, human_msg],
    )
    template = template.partial(shot_1=shot_1, shot_2=shot_2, shot_3=shot_3, shot_4=shot_4, shot_5=shot_5)
    return template
