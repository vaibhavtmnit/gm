from typing import TypedDict, List, Optional, Any
from langgraph.graph.message import add_messages

# The dictionary that will be passed between nodes
class ValidatorState(TypedDict):
    # Inputs
    input_value: Any
    output_value: Any
    processing_logic: str

    # Intermediate values
    is_transform_needed: bool
    value_to_compare: Any # This will be the transformed input, or the original if no transform
    identified_type: str # e.g., 'date', 'iso8601_timestamp', 'integer'
    
    # Tool-related state
    tool_name_to_use: Optional[str]
    new_tool_code: Optional[str] # Holds LLM-generated python code for a new tool
    
    # Final output
    final_decision: str # "Match", "Mismatch", or "Error"
    explanation: str # Justification for the decision



from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import ToolExecutor
from langchain_core.tools import tool

# --- Initial Tools (You'll add more dynamically) ---

# A robust date/timestamp parser is crucial
from dateutil.parser import parse as date_parse

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
    return float(val1) == float(val2)

# --- LLM and Tool Setup ---

# You already have instances, so this is for context
llm = AzureChatOpenAI(
    model="gpt-4o-mini", # Corresponds to your "o3-mini"
    temperature=0,
    azure_deployment="your-deployment-name"
)

# The ToolExecutor holds a list of your validation tools
# We start with a few, but the 'generate_tool' node will add to this list
initial_tools = [compare_datetimes, compare_numbers]
tool_executor = ToolExecutor(initial_tools)


# Placeholder logic for your nodes - you will implement the details

def analyze_logic(state: ValidatorState):
    """Agent 1: Analyzes the processing logic to decide if a transformation is needed."""
    # LLM call to interpret state['processing_logic']
    # Sets state['is_transform_needed'] to True or False
    # Also identifies the data type and stores in state['identified_type']
    print("---ANALYZING LOGIC---")
    # ... your implementation ...
    return state

def transform_value(state: ValidatorState):
    """Agent 2: Transforms the input value based on the logic. (Runs only if needed)."""
    # LLM call to generate Python code for the transformation
    # Executes the code to transform state['input_value']
    # Stores result in state['value_to_compare']
    print("---TRANSFORMING VALUE---")
    # ... your implementation ...
    return state

def select_tool(state: ValidatorState):
    """Agent 3: Identifies the correct comparison tool from the available list."""
    # Logic to match state['identified_type'] with a tool name in tool_executor.tools
    # If found, sets state['tool_name_to_use']
    # If not found, signals that a tool must be generated
    print("---SELECTING TOOL---")
    # ... your implementation ...
    return state
    
def generate_tool(state: ValidatorState):
    """Agent 4: Generates Python code for a new comparison tool. (Runs only if needed)."""
    # LLM call with a strong prompt: "Write a Python function that compares two values of type..."
    # IMPORTANT: Include robust error handling in the generated code.
    # Stores the generated Python code string in state['new_tool_code']
    print("---GENERATING NEW TOOL---")
    # ... your implementation ...
    return state

def register_new_tool(state: ValidatorState):
    """A simple utility node to update the tool_executor with the new function."""
    # Takes state['new_tool_code'], executes it to define the function,
    # then creates a new @tool from it and adds it to the tool_executor.
    # WARNING: Using exec() with LLM-generated code is a security risk.
    # Sanitize and validate thoroughly in a production environment.
    print("---REGISTERING NEW TOOL---")
    # ... your implementation ...
    return state

def make_final_decision(state: ValidatorState):
    """Agent 5: Formulates the final response after a tool has been executed."""
    # The result of the tool call will be in the state. This node just formats it.
    print("---FINALIZING DECISION---")
    # ... your implementation ...
    return state


def should_transform(state: ValidatorState) -> str:
    """Decides whether to run the transformation node or skip to tool selection."""
    return "transform_value" if state["is_transform_needed"] else "select_tool"

def tool_exists(state: ValidatorState) -> str:
    """Checks if a suitable tool was found or if one needs to be generated."""
    return "execute_tool" if state["tool_name_to_use"] else "generate_tool"



from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Create the graph instance
workflow = StateGraph(ValidatorState)

# Add all the nodes
workflow.add_node("analyze_logic", analyze_logic)
workflow.add_node("transform_value", transform_value)
workflow.add_node("select_tool", select_tool)
workflow.add_node("generate_tool", generate_tool)
workflow.add_node("register_new_tool", register_new_tool)
workflow.add_node("execute_tool", ToolNode(tools=[tool_executor])) # The special node for running tools
workflow.add_node("make_final_decision", make_final_decision)

# Define the graph's flow (the edges)
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
    {"generate_tool": "generate_tool", "execute_tool": "execute_tool"}
)

workflow.add_edge("generate_tool", "register_new_tool")
workflow.add_edge("register_new_tool", "select_tool") # Loop back to re-select after creating the tool
workflow.add_edge("execute_tool", "make_final_decision")
workflow.add_edge("make_final_decision", END)

# Compile the graph
app = workflow.compile()





