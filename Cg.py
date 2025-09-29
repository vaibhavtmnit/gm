from typing import Any, Dict, Callable
from datetime import datetime
from dateutil import parser as dateparser
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, ChatAgentExecutor

# ------------------------------------------------------
# Base comparison tools
# ------------------------------------------------------
def compare_numbers(a: Any, b: Any) -> bool:
    try:
        return float(a) == float(b)
    except Exception:
        return False

def compare_strings(a: Any, b: Any) -> bool:
    return str(a).strip() == str(b).strip()

def compare_dates(a: Any, b: Any) -> bool:
    try:
        da = dateparser.parse(str(a))
        db = dateparser.parse(str(b))
        return da == db
    except Exception:
        return False

# Registry of tools (can grow dynamically)
TOOL_REGISTRY: Dict[str, Callable] = {
    "compare_numbers": compare_numbers,
    "compare_strings": compare_strings,
    "compare_dates": compare_dates,
}

# ------------------------------------------------------
# State definition for LangGraph
# ------------------------------------------------------
class ValidatorState(Dict):
    """State passed between agents"""
    input_value: Any
    output_value: Any
    processing_logic: str
    transform_flag: bool
    result: str

# ------------------------------------------------------
# Agent: Logic Interpreter
# ------------------------------------------------------
def logic_interpreter(state: ValidatorState) -> ValidatorState:
    """
    Uses the LLM to decide if a transformation is implied by `processing_logic`.
    For example: 'convert to uppercase', 'round to 2 decimals', etc.
    """
    llm = AzureChatOpenAI(
        deployment_name="o1-preview",  # or "o3-mini"
        model="gpt-4o",
        temperature=0,
    )
    prompt = f"""
    You are a logic interpreter.
    Input: {state['input_value']}
    Output: {state['output_value']}
    Processing Logic: {state['processing_logic']}
    Transformation Flag: {state['transform_flag']}

    Task:
    - If transformation_flag is False, return "no_change".
    - If transformation_flag is True but logic implies no transformation, return "no_change".
    - Otherwise, return a Python transformation function as code that, when applied
      to input_value, should produce output_value.
    Only output either "no_change" or Python function code.
    """
    resp = llm.invoke(prompt)
    decision = resp.content.strip()

    state["transformation_code"] = None
    if decision != "no_change" and "def " in decision:
        state["transformation_code"] = decision
    return state

# ------------------------------------------------------
# Agent: Comparer
# ------------------------------------------------------
def comparer(state: ValidatorState) -> ValidatorState:
    a = state["input_value"]
    b = state["output_value"]

    # If transformation code exists, apply it to input first
    if state.get("transformation_code"):
        try:
            local_env = {}
            exec(state["transformation_code"], {}, local_env)
            func = [v for v in local_env.values() if callable(v)][0]
            a = func(a)
        except Exception as e:
            state["result"] = f"Transformation failed: {e}"
            return state

    # Pick comparison tool
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        tool = TOOL_REGISTRY["compare_numbers"]
    elif "date" in str(type(a)).lower() or "date" in str(type(b)).lower():
        tool = TOOL_REGISTRY["compare_dates"]
    elif isinstance(a, str) or isinstance(b, str):
        # Try date comparison first
        if compare_dates(a, b):
            tool = TOOL_REGISTRY["compare_dates"]
        else:
            tool = TOOL_REGISTRY["compare_strings"]
    else:
        # Dynamically create a tool for unknown types
        tool_name = f"compare_{type(a).__name__}"
        if tool_name not in TOOL_REGISTRY:
            def dynamic_tool(x, y):
                return x == y
            TOOL_REGISTRY[tool_name] = dynamic_tool
        tool = TOOL_REGISTRY[tool_name]

    try:
        match = tool(a, b)
        state["result"] = f"Match: {match}"
    except Exception as e:
        state["result"] = f"Comparison failed: {e}"

    return state

# ------------------------------------------------------
# Build LangGraph
# ------------------------------------------------------
graph = StateGraph(ValidatorState)
graph.add_node("logic_interpreter", logic_interpreter)
graph.add_node("comparer", comparer)

graph.set_entry_point("logic_interpreter")
graph.add_edge("logic_interpreter", "comparer")
graph.add_edge("comparer", END)

validator_app = graph.compile()

# ------------------------------------------------------
# Example Run
# ------------------------------------------------------
if __name__ == "__main__":
    state = ValidatorState(
        input_value="2023-01-01",
        output_value="01/01/2023",
        processing_logic="Convert to UK date format",
        transform_flag=True,
        result=None,
    )
    final = validator_app.invoke(state)
    print(final["result"])
