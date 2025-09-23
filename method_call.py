import json
import textwrap
from typing import List, Literal, TypedDict

# ==============================================================================
# 1. TYPE DEFINITIONS (using TypedDict)
# ==============================================================================

class AnalysisInput(TypedDict):
    """Input for the analysis utility."""
    object_name: str
    java_code: str
    code_line: int
    analytical_chain: str

class ChildObject(TypedDict):
    """Standard output format for a found child."""
    name_of_the_child: str
    type_of_the_child: str
    code_snippet: str
    code_block: str
    further_expand: bool
    found_in: Literal["original_code", "processed_code", "both"]

# ==============================================================================
# 2. NEW DIVERSE JAVA CODE EXAMPLE
# ==============================================================================

DIVERSE_JAVA_CODE = """
package com.analysis.example;

import java.util.List;
import java.util.stream.Collectors;

// A simple logger utility for demonstration
class Logger {
    public static void info(String message) {
        System.out.println("INFO: " + message);
    }
}

// A builder pattern example
class DataObject {
    private String name;
    private int value;

    public static class Builder {
        private String name;
        private int value;

        public Builder withName(String name) {
            this.name = name;
            return this;
        }

        public Builder withValue(int value) {
            this.value = value;
            return this;
        }

        public DataObject build() {
            return new DataObject(this);
        }
    }

    private DataObject(Builder builder) {
        this.name = builder.name;
        this.value = builder.value;
    }
}

// Main class for analysis
public class AdvancedProcessor {

    public DataObject processData(String inputName) {
        Logger.info("Starting data processing...");
        validate(inputName);

        DataObject.Builder builder = new DataObject.Builder();
        DataObject finalObject = builder.withName(inputName)
                                        .withValue(100)
                                        .build();

        processList(List.of("one", "two"));

        return finalObject;
    }

    private List<String> processList(List<String> data) {
        return data.stream()
                   .map(item -> item.toUpperCase())
                   .collect(Collectors.toList());
    }

    private void validate(String input) {
        // This method has no outgoing calls to analyze.
        if (input == null || input.isEmpty()) {
            throw new IllegalArgumentException("Input cannot be null or empty.");
        }
    }
}
"""

# ==============================================================================
# 3. MOCK LLM AND PROMPT DEFINITIONS
# ==============================================================================

# This mock class simulates the behavior of an AzureOpenAI instance
# to make the code runnable without API keys.
class MockAzureOpenAI:
    def __init__(self, responses):
        self.responses = responses
        self.history = []

    class Completions:
        def __init__(self, parent):
            self.parent = parent
        
        def create(self, messages, **kwargs):
            content = messages[-1]['content']
            self.parent.history.append(content)
            
            response_key = "default"
            if "Translate the following Java code" in content:
                response_key = "code_to_nl"
            elif "analyze the natural language description" in content:
                response_key = "nl_analysis"
            elif "find the immediate next method calls" in content:
                response_key = "direct_analysis"

            class MockChoice:
                def __init__(self, content):
                    self.message = type('obj', (object,), {'content': content})

            class MockResponse:
                def __init__(self, content):
                    self.choices = [MockChoice(content)]

            return MockResponse(self.parent.responses.get(response_key, "[]"))

    @property
    def chat(self):
        return type('obj', (object,), {'completions': self.Completions(self)})


# --- Prompts ---

PROMPT_CODE_TO_NL = textwrap.dedent("""
    Translate the following Java code into a line-by-line, simple natural language description. Be precise.

    Java Code:
    ```java
    {java_code}
    ```
    """)

PROMPT_DIRECT_ANALYSIS = textwrap.dedent("""
    You are an expert Java static analyzer. Your explicit goal is to find the immediate next method calls that are children of a specific focus object.

    ## Rules:
    1.  **Method Scope**: If the focus is a method, find all direct calls within it (static, instance, constructor).
    2.  **Chained Calls**: If the focus is a variable, find ONLY the first call in a chain. E.g., for `a.b().c()`, if the focus is `a`, the only child is `b`.
    3.  **Lambdas**: Look inside lambda expressions for method calls.
    4.  **Direct Link Only**: The call must be directly on the object. If an intermediate variable is used, it's not a direct child.
    5.  Return an empty list `[]` if no direct calls are found.

    ## Analysis History:
    {analytical_chain}

    ## Full Java Code:
    ```java
    {java_code}
    ```
    
    ## Focus Object:
    - Name: `{object_name}`
    - Defined on line: `{code_line}`

    ## Few-Shot Examples:

    ### Example 1
    - Focus Object: `processData` on line 51
    - Expected Output: A list containing calls like `info`, `validate`, `Builder`, `withName`, `processList`.

    ### Example 2
    - Focus Object: `builder` on line 55
    - Expected Output: `[ { "name_of_the_child": "withName", "type_of_the_child": "CallOnObject", ... } ]`

    ### Example 3
    - Focus Object: result of `builder.withName(inputName)` on line 56
    - Expected Output: `[ { "name_of_the_child": "withValue", "type_of_the_child": "ChainedNextCall", ... } ]`
    
    ### Example 4
    - Focus Object: `item` on line 66
    - Expected Output: `[ { "name_of_the_child": "toUpperCase", "type_of_the_child": "CallOnObject", ... } ]`

    ### Example 5
    - Focus Object: `validate` on line 71
    - Expected Output: `[]`

    Based on the rules and examples, perform the analysis on the focus object. Return ONLY the JSON list of children.
    """)

PROMPT_NL_ANALYSIS = textwrap.dedent("""
    You are a language comprehension expert. Your explicit goal is to find the immediate next actions (method calls) related to a focus object, based on a natural language description of code.

    ## Analysis History:
    {analytical_chain}

    ## Natural Language Code Description:
    {nl_code_description}
    
    ## Focus Object:
    - Name: `{object_name}`

    Based on the description, what are the immediate method calls or actions involving the focus object? Return a JSON list of children in the same format as the direct analysis.
    """)

# ==============================================================================
# 4. CORE UTILITY FUNCTION
# ==============================================================================

def find_method_calls(llm_instance: MockAzureOpenAI, input_data: AnalysisInput) -> List[ChildObject]:
    """
    Finds method call children using a dual-analysis workflow.
    """
    print(f"--- Analyzing Focus Object: '{input_data['object_name']}' ---")
    
    # --- Run 1: Direct Code Analysis ---
    print("  > Running Direct Code Analysis...")
    prompt1 = PROMPT_DIRECT_ANALYSIS.format(
        java_code=input_data["java_code"],
        object_name=input_data["object_name"],
        code_line=input_data["code_line"],
        analytical_chain=input_data["analytical_chain"]
    )
    response1 = llm_instance.chat.completions.create(messages=[{"role": "user", "content": prompt1}])
    try:
        results1 = json.loads(response1.choices[0].message.content)
        print(f"    - Found {len(results1)} potential children.")
    except json.JSONDecodeError:
        results1 = []
        print("    - Error decoding JSON from direct analysis.")

    # --- Run 2: Natural Language Analysis ---
    print("  > Running Natural Language Analysis...")
    # Step A: Convert code to natural language
    prompt2a = PROMPT_CODE_TO_NL.format(java_code=input_data["java_code"])
    response2a = llm_instance.chat.completions.create(messages=[{"role": "user", "content": prompt2a}])
    nl_description = response2a.choices[0].message.content
    
    # Step B: Analyze the natural language description
    prompt2b = PROMPT_NL_ANALYSIS.format(
        nl_code_description=nl_description,
        object_name=input_data["object_name"],
        analytical_chain=input_data["analytical_chain"]
    )
    response2b = llm_instance.chat.completions.create(messages=[{"role": "user", "content": prompt2b}])
    try:
        results2 = json.loads(response2b.choices[0].message.content)
        print(f"    - Found {len(results2)} potential children from NL.")
    except json.JSONDecodeError:
        results2 = []
        print("    - Error decoding JSON from NL analysis.")

    # --- Step 3: Merge and Compare Results ---
    print("  > Merging results...")
    merged_children = {}

    for child in results1:
        # Create a unique key for each child
        key = (child["name_of_the_child"], child["code_snippet"])
        child["found_in"] = "original_code"
        merged_children[key] = child

    for child in results2:
        key = (child["name_of_the_child"], child["code_snippet"])
        if key in merged_children:
            merged_children[key]["found_in"] = "both"
        else:
            child["found_in"] = "processed_code"
            merged_children[key] = child
            
    final_list = list(merged_children.values())
    print(f"--- Analysis Complete. Found {len(final_list)} unique children. ---\n")
    return final_list

# ==============================================================================
# 5. RUNNABLE EXAMPLE
# ==============================================================================

if __name__ == "__main__":
    # Define mock responses for the LLM
    MOCK_RESPONSES = {
        "code_to_nl": "Line 51: A method 'processData' is defined...",
        "direct_analysis": json.dumps([
            {"name_of_the_child": "info", "type_of_the_child": "StaticFactoryCall", "code_snippet": "Logger.info(...)", "code_block": "Logger.info(...);", "further_expand": False},
            {"name_of_the_child": "validate", "type_of_the_child": "MethodCall", "code_snippet": "validate(inputName)", "code_block": "validate(inputName);", "further_expand": True},
            {"name_of_the_child": "withName", "type_of_the_child": "CallOnObject", "code_snippet": "builder.withName(inputName)", "code_block": "builder.withName(inputName)...", "further_expand": True},
        ]),
        "nl_analysis": json.dumps([
            {"name_of_the_child": "info", "type_of_the_child": "StaticFactoryCall", "code_snippet": "Logger.info(...)", "code_block": "Logger.info(...);", "further_expand": False},
            {"name_of_the_child": "processList", "type_of_the_child": "MethodCall", "code_snippet": "processList(...)", "code_block": "processList(...);", "further_expand": True},
        ])
    }

    # Instantiate the mock LLM
    mock_llm = MockAzureOpenAI(responses=MOCK_RESPONSES)

    # Define the input for the analysis
    analysis_input: AnalysisInput = {
        "object_name": "processData",
        "java_code": DIVERSE_JAVA_CODE,
        "code_line": 51,
        "analytical_chain": ""
    }
    
    # Run the utility
    found_children = find_method_calls(mock_llm, analysis_input)

    # Print the final, merged results
    import pprint
    pprint.pprint(found_children)
