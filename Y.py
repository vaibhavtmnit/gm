import os
from typing import TypedDict, Optional, Any, List

# --- Core LangChain/LangGraph Imports ---
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages, MessageGraph
from langgraph.prebuilt import ToolNode
from langchain.tools import DynamicTool
from dateutil.parser import parse as date_parse

# --- 1. Agent State Definition ---
# This is the shared memory that passes between the nodes of our graph.

class ValidatorState(TypedDict):
    # Inputs from the user
    input_value: Any
    output_value: Any
    processing_logic: str

    # Values modified during the run
    is_transform_needed: bool
    value_to_compare: Any  # The transformed input, or the original if no transform
    identified_type: str  # e.g., 'date', 'iso8601_timestamp', 'integer'

    # State for dynamic tool handling
    tool_name_to_use: Optional[str]
    new_tool_code: Optional[str]  # Holds LLM-generated python code for a new tool

    # Final result
    final_decision: str  # "Match", "Mismatch", or "Error"
    explanation: str  # Justification for the decision
    
    # LangGraph message state
    messages: List

# --- 2. Configuration & Initialization ---
# USER: Fill in your Azure OpenAI details here.
# It's best practice to use environment variables.

os.environ["AZURE_OPENAI_API_KEY"] = "YOUR_AZURE_OPENAI_API_KEY"
os.environ["AZURE_OPENAI_ENDPOINT"] = "YOUR_AZURE_OPENAI_ENDPOINT"
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-01"
os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "your-deployment-name" # e.g., gpt-4o-mini deployment

# Initialize the LLM
llm = AzureChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
)

# --- 3. Initial Tools ---
# These are the validation functions the agent knows about at the start.

@tool
def compare_datetimes(val1: str, val2: str) -> bool:
    """Compares two string values by parsing them as datetimes. Handles any ISO format."""
    try:
        return date_parse(val1) == date_parse(val2)
    except (ValueError, TypeError):
        return False

@tool
def compare_numbers(val1: float | int, val2: float | int) -> bool:
    """Compares two numbers (integers or floats)."""
    try:
        return float(val1) == float(val2)
    except (ValueError, TypeError):
        return False

# The ToolExecutor holds a list of our validation tools.
# This list will grow as the agent generates new tools.
initial_tools = [compare_datetimes, compare_numbers]
tool_executor = ToolNode(initial_tools)


# --- 4. Node Logic (The "Agents") ---

# Pydantic model for structured output from the first agent
class Analysis(BaseModel):
    is_transform_needed: bool = Field(description="Set to true if the processing logic implies a change to the input value before comparison.")
    identified_type: str = Field(description="A short, snake_case string identifying the data type for comparison, e.g., 'iso_timestamp', 'uk_postcode', 'positive_integer'.")

def analyze_logic(state: ValidatorState):
    """Agent 1: Analyzes logic to determine the required workflow."""
    print("--- AGENT: Analyzing Logic ---")
    prompt = f"""
    You are a data validation expert. Analyze the following inputs to determine the validation plan.
    - Input Value: `{state['input_value']}`
    - Output Value: `{state['output_value']}`
    - Processing Logic: `{state['processing_logic']}`
    Based on the processing logic, do you need to transform the input value before comparing it to the output value?
    What is the specific data type we should use for the comparison?
    """
    structured_llm = llm.with_structured_output(Analysis)
    analysis_result = structured_llm.invoke(prompt)
    print(f"Analysis complete: Transform needed = {analysis_result.is_transform_needed}, Type = {analysis_result.identified_type}")

    state['is_transform_needed'] = analysis_result.is_transform_needed
    state['identified_type'] = analysis_result.identified_type
    
    if not analysis_result.is_transform_needed:
        state['value_to_compare'] = state['input_value']
    
    return state

def transform_value(state: ValidatorState):
    """Agent 2: Generates and executes code to transform the input value."""
    print("--- AGENT: Transforming Value ---")
    prompt = f"""
    You are a Python code generation expert. Your task is to write a single line of Python code that transforms a value based on a given logic.
    - Value to transform: `{state['input_value']}` (available in the `value` variable)
    - Transformation logic: `{state['processing_logic']}`
    Only output the raw Python expression. For example, if the logic is "append 'Z' to make it UTC", you would output `f"{{value}}Z"`.
    """
    code_expression = llm.invoke(prompt).content
    print(f"Generated transformation code: {code_expression}")

    # !! SECURITY WARNING !! Using eval() is risky. Sandbox this in production.
    try:
        transformed_value = eval(code_expression, {"value": state['input_value']})
        state['value_to_compare'] = transformed_value
        print(f"Transformation successful. New value: {transformed_value}")
    except Exception as e:
        print(f"Error during transformation: {e}")
        state['final_decision'] = "Error"
        state['explanation'] = f"Failed to execute transformation code: {code_expression}. Error: {e}"
    
    return state

def select_tool(state: ValidatorState):
    """Agent 3: Selects the appropriate comparison tool from the available list."""
    print("--- AGENT: Selecting Tool ---")
    identified_type = state['identified_type']
    found_tool = None
    for tool in tool_executor.tools:
        if identified_type in tool.name:
            found_tool = tool.name
            break
            
    if found_tool:
        print(f"Found suitable tool: '{found_tool}'")
        state['tool_name_to_use'] = found_tool
    else:
        print(f"No tool found for type '{identified_type}'. Flagging for generation.")
        state['tool_name_to_use'] = None
        
    return state

def generate_tool(state: ValidatorState):
    """Agent 4: Generates Python code for a new comparison tool."""
    print("--- AGENT: Generating New Tool ---")
    prompt = f"""
    You are an expert Python programmer specializing in data validation. Write a single Python function named `compare_{state['identified_type']}`.
    Function Requirements:
    1. It MUST accept two arguments: `val1` and `val2`.
    2. It MUST include a `try/except` block to handle errors gracefully.
    3. It MUST return `True` if the values are a match, and `False` otherwise.
    4. DO NOT include the `@tool` decorator or any other text. Output only raw Python code.
    """
    generated_code = llm.invoke(prompt).content
    print(f"Generated new tool code:\n{generated_code}")
    state['new_tool_code'] = generated_code
    return state

def register_new_tool(state: ValidatorState):
    """Utility Node: Registers the newly generated tool with the ToolExecutor."""
    print("--- AGENT: Registering New Tool ---")
    code_string = state['new_tool_code']
    
    # !! SECURITY WARNING !! Using exec() is risky. Sandbox this in production.
    local_namespace = {}
    exec(code_string, {}, local_namespace)
    
    tool_name = list(local_namespace.keys())[0]
    new_function = local_namespace[tool_name]
    
    new_tool = DynamicTool(name=tool_name, func=new_function, description=f"Compares two {state['identified_type']} values.")
    tool_executor.tools.append(new_tool)
    print(f"Successfully registered new tool: '{tool_name}'")
    
    return state

def prepare_tool_call(state: ValidatorState):
    """Prepares the state for the ToolNode by creating a tool call message."""
    tool_name = state['tool_name_to_use']
    val1 = state['value_to_compare']
    val2 = state['output_value']
    
    # We create a message to represent the tool call
    tool_call_message = ("human", f"Please call the tool '{tool_name}' with val1='{val1}' and val2='{val2}'.")
    state['messages'] = [tool_call_message]
    return state

def make_final_decision(state: ValidatorState):
    """Agent 5: Formulates the final response after a tool has been executed."""
    print("--- AGENT: Finalizing Decision ---")
    tool_output = state['messages'][-1]
    
    if "true" in tool_output.content.lower():
        decision = "Match"
    elif "false" in tool_output.content.lower():
        decision = "Mismatch"
    else:
        decision = "Error"
    
    state['final_decision'] = decision
    state['explanation'] = f"Comparison tool '{state['tool_name_to_use']}' returned '{tool_output.content}'."
    print(f"Final Decision: {state['final_decision']}")
    return state


# --- 5. Graph Conditional Logic ---

def should_transform(state: ValidatorState) -> str:
    return "transform_value" if state["is_transform_needed"] else "select_tool"

def tool_exists(state: ValidatorState) -> str:
    return "prepare_tool_call" if state["tool_name_to_use"] else "generate_tool"


# --- 6. Graph Assembly ---

workflow = StateGraph(ValidatorState)

workflow.add_node("analyze_logic", analyze_logic)
workflow.add_node("transform_value", transform_value)
workflow.add_node("select_tool", select_tool)
workflow.add_node("generate_tool", generate_tool)
workflow.add_node("register_new_tool", register_new_tool)
workflow.add_node("prepare_tool_call", prepare_tool_call)
workflow.add_node("execute_tool", tool_executor)
workflow.add_node("make_final_decision", make_final_decision)

workflow.set_entry_point("analyze_logic")

workflow.add_conditional_edges(
    "analyze_logic",
    should_transform,
    {"transform_value": "transform_value", "select_tool": "select_tool"}
)

workflow.add_edge("transform_value", "select_tool")

workflow.add_conditional_edges(
    "select_tool",
    tool_exists,
    {"generate_tool": "generate_tool", "prepare_tool_call": "prepare_tool_call"}
)

workflow.add_edge("generate_tool", "register_new_tool")
workflow.add_edge("register_new_tool", "select_tool") # Loop back to re-select
workflow.add_edge("prepare_tool_call", "execute_tool")
workflow.add_edge("execute_tool", "make_final_decision")
workflow.add_edge("make_final_decision", END)

# Compile the graph into a runnable app
app = workflow.compile()


# --- 7. Example Usage ---

if __name__ == "__main__":
    print("🚀 Starting Validator Agent...\n")

    # --- Scenario 1: Simple match, no transformation needed ---
    print("\n--- SCENARIO 1: Simple Date Match ---")
    inputs_1 = {
        "input_value": "2025-09-29T09:22:00",
        "output_value": "2025-09-29T09:22:00",
        "processing_logic": "No change, compare as timestamps",
        "messages": []
    }
    result_1 = app.invoke(inputs_1)
    print(f"\n✅ Result 1: {result_1['final_decision']} - {result_1['explanation']}")
    
    # --- Scenario 2: Transformation required ---
    print("\n\n--- SCENARIO 2: Date Transformation and Match ---")
    inputs_2 = {
        "input_value": "2025-09-29T10:00:00",
        "output_value": "2025-09-29T10:00:00Z",
        "processing_logic": "The input is missing timezone info, assume UTC and add 'Z' to the end.",
        "messages": []
    }
    result_2 = app.invoke(inputs_2)
    print(f"\n✅ Result 2: {result_2['final_decision']} - {result_2['explanation']}")

    # --- Scenario 3: Dynamic tool generation ---
    print("\n\n--- SCENARIO 3: Generate a New Tool for UK Postcodes ---")
    inputs_3 = {
        "input_value": "sw1a 0aa",
        "output_value": "SW1A 0AA",
        "processing_logic": "Compare as UK postcodes. They should match regardless of case and spacing.",
        "messages": []
    }
    result_3 = app.invoke(inputs_3)
    print(f"\n✅ Result 3: {result_3['final_decision']} - {result_3['explanation']}")
    
    # --- Scenario 4: Using the newly generated tool again ---
    print("\n\n--- SCENARIO 4: Re-using the Generated Postcode Tool ---")
    # Note: The tool 'compare_uk_postcode' now exists in our tool_executor for this session.
    inputs_4 = {
        "input_value": "ec1v 4pw",
        "output_value": "EC1V 4PW",
        "processing_logic": "Compare as UK postcodes.",
        "messages": []
    }
    result_4 = app.invoke(inputs_4)
    print(f"\n✅ Result 4: {result_4['final_decision']} - {result_4['explanation']}")
