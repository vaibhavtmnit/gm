"""
Validator Multi-Agent (LangGraph) — 3-input interface with LLM supervision.

Inputs (only these three):
- input_value: Any
- output_value: Any
- transformation_info: str   # free text containing both:
                             #   - whether transformation is expected (true/false)
                             #   - optional transformation logic (e.g., "round to 2dp", "uppercase",
                             #     "format date as dd/mm/yyyy", "convert UTC->Europe/London")

Flow:
1) supervisor_parse_info (LLM):
      -> parses transformation_info → (transform_expected: bool, transform_logic: Optional[str])
2) router:
      A) transform_expected == False → direct_compare
      B) transform_expected == True and transform_logic is empty → check_transformed_no_logic
      C) transform_expected == True and transform_logic present → apply_logic_and_compare (LLM)
3) supervisor_explainer (LLM):
      -> writes a concise explanation into state["explanation"]

Notes:
- Deterministic comparators for numbers (tolerant), strings (trim/case rules), and datetimes
  (treat same instant equal across formats).
- Heuristic detect_transformation for scenario (B).
- LLM transformer returns a JSON with {"transformed_value": ..., "explanation": "..."}.
- If an LLM is not injected, safe fallbacks try best-effort heuristics so graph still runs.

You can inject your AzureChatOpenAI instances (o1-preview / o3-mini) at graph build time.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, TypedDict, Literal, Tuple
from dataclasses import dataclass
from datetime import datetime
import math
import re
import json

# Robust date parsing if available
try:
    from dateutil import parser as dateparser  # type: ignore
except Exception:  # pragma: no cover
    dateparser = None

# --- LangGraph
from langgraph.graph import StateGraph, END

# If you want to wire LLMs here, import your instances in your app:
# from langchain_openai import AzureChatOpenAI


# =============================================================================
# Deterministic helpers (types + comparisons)
# =============================================================================

_NUMERIC_RE = re.compile(r"^\s*[-+]?((\d+(\.\d*)?)|(\.\d+))([eE][-+]?\d+)?\s*$")

def _is_number_like(x: Any) -> bool:
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return True
    if isinstance(x, str):
        return bool(_NUMERIC_RE.match(x))
    return False

def _to_float(x: Any) -> Optional[float]:
    try:
        if isinstance(x, bool):
            return None
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            return float(x.strip())
    except Exception:
        return None
    return None

def _parse_datetime_any(x: Any) -> Optional[datetime]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    if dateparser is not None:
        try:
            return dateparser.parse(s)
        except Exception:
            pass
    fmts = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%f%z", "%d-%b-%Y", "%d %b %Y",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None

DEFAULT_NUMBER_REL_TOL = 1e-9
DEFAULT_NUMBER_ABS_TOL = 1e-9

def compare_numbers(a: Any, b: Any, *, rel_tol: float = DEFAULT_NUMBER_REL_TOL, abs_tol: float = DEFAULT_NUMBER_ABS_TOL) -> bool:
    fa = _to_float(a); fb = _to_float(b)
    if fa is None or fb is None:
        return False
    return math.isclose(fa, fb, rel_tol=rel_tol, abs_tol=abs_tol)

def compare_strings(a: Any, b: Any, *, case_sensitive: bool = False, ignore_whitespace: bool = True) -> bool:
    sa, sb = str(a), str(b)
    if ignore_whitespace:
        sa, sb = sa.strip(), sb.strip()
    if not case_sensitive:
        sa, sb = sa.lower(), sb.lower()
    return sa == sb

def compare_datetimes(a: Any, b: Any) -> bool:
    da = _parse_datetime_any(a); db = _parse_datetime_any(b)
    if da is None or db is None:
        return False
    return da == db

def best_compare(a: Any, b: Any) -> bool:
    """
    Type-aware comparison:
    - numbers: tolerant
    - datetimes: same instant equal despite format
    - strings: case-insensitive, trimmed
    - cross-kind (string vs number/datetime) handled if parsable
    """
    # try num
    if _is_number_like(a) and _is_number_like(b):
        return compare_numbers(a, b)
    # try datetime
    da = _parse_datetime_any(a); db = _parse_datetime_any(b)
    if da is not None and db is not None:
        return da == db
    # cross: string<->datetime
    if isinstance(a, str) and db is not None:
        return compare_datetimes(a, b)
    if isinstance(b, str) and da is not None:
        return compare_datetimes(a, b)
    # cross: string<->number
    if isinstance(a, str) and _is_number_like(b):
        return compare_numbers(a, b)
    if isinstance(b, str) and _is_number_like(a):
        return compare_numbers(a, b)
    # default string compare
    if isinstance(a, str) or isinstance(b, str):
        return compare_strings(a, b)
    # fallback
    return a == b


# =============================================================================
# Transformation detection (Scenario B)
# =============================================================================

@dataclass
class TransformObservation:
    happened: bool
    reasons: list[str]
    details: Dict[str, Any]

def detect_transformation(a: Any, b: Any) -> TransformObservation:
    reasons: list[str] = []
    details: Dict[str, Any] = {}

    raw_equal = (str(a) == str(b))
    normalized_equal = best_compare(a, b)

    # Strings: case/whitespace
    if isinstance(a, str) or isinstance(b, str):
        sa, sb = str(a), str(b)
        if sa.strip() != sb.strip() and sa.lower().strip() == sb.lower().strip():
            reasons.append("case_change_and/or_whitespace")
        elif sa != sb and sa.strip() == sb.strip():
            reasons.append("whitespace_trim")
        elif sa.lower() != sb.lower() and sa.strip().lower() == sb.strip().lower():
            reasons.append("case_change")

    # Dates: format & tz representation
    da = _parse_datetime_any(a)
    db = _parse_datetime_any(b)
    if da and db:
        if da == db and str(a).strip() != str(b).strip():
            reasons.append("date_format_change")
        if (da.tzinfo is None) != (db.tzinfo is None):
            reasons.append("timezone_representation_change")

    # Numbers: rounding / tiny adj
    if _is_number_like(a) and _is_number_like(b):
        fa = _to_float(a); fb = _to_float(b)
        if fa is not None and fb is not None:
            if not math.isclose(fa, fb, rel_tol=0, abs_tol=0) and math.isclose(fa, fb, rel_tol=0, abs_tol=1e-9):
                reasons.append("tiny_float_adjustment")
            if round(fa, 2) == fb or round(fa, 3) == fb:
                reasons.append("rounding")

    if not raw_equal and normalized_equal:
        reasons.append("format_change")

    happened = (len(reasons) > 0) or (not raw_equal)
    details["raw_equal"] = raw_equal
    details["normalized_equal"] = normalized_equal
    return TransformObservation(happened=happened, reasons=sorted(set(reasons)), details=details)


# =============================================================================
# LLM Supervisor roles
# =============================================================================

class ParseResult(TypedDict, total=False):
    transform_expected: bool
    transform_logic: Optional[str]

class TransformerResult(TypedDict, total=False):
    transformed_value: Any
    transformer_explanation: str

class ValidatorState(TypedDict, total=False):
    # Inputs
    input_value: Any
    output_value: Any
    transformation_info: str

    # Parsed by supervisor
    transform_expected: bool
    transform_logic: Optional[str]

    # Scenario B detection
    observation: TransformObservation

    # Scenario C transformed candidate
    transformed_input: Any

    # Verdict
    verdict: Literal["CORRECT", "INCORRECT"]
    rationale: str
    explanation: str  # final human explanation (LLM)


# ---- LLM wrappers ------------------------------------------------------------

class SupervisorParser:
    """
    Parses free-flow transformation_info → {transform_expected: bool, transform_logic: Optional[str]}
    Provide your AzureChatOpenAI via constructor. Must support .invoke(prompt) → object with .content (str).
    """
    def __init__(self, llm=None) -> None:
        self.llm = llm

    def parse(self, transformation_info: str) -> ParseResult:
        if not transformation_info or not transformation_info.strip():
            return {"transform_expected": False, "transform_logic": None}

        if not self.llm:
            # Fallback heuristics (very simple)
            text = transformation_info.lower()
            expected = "true" in text or "yes" in text or "expected" in text and "not" not in text
            # crude "no transform" detector
            if any(k in text for k in ["no transform", "no change", "unchanged", "identity", "false", "not expected"]):
                expected = False
            # try to pluck a logic-like tail after keywords
            logic = None
            for key in ["logic:", "rule:", "transform:", "apply:", "operation:", "op:", "change:"]:
                if key in text:
                    logic = transformation_info[transformation_info.lower().index(key)+len(key):].strip()
                    break
            return {"transform_expected": bool(expected), "transform_logic": logic}

        prompt = f"""
You are given a free-text description of transformation expectations and (optionally) logic.
Extract:
- transform_expected: boolean (true if a transformation is expected)
- transform_logic: string or null (the described transformation, concise; null if not specified)

Return ONLY JSON with those keys.

TEXT:
\"\"\"{transformation_info.strip()}\"\"\"
"""
        try:
            resp = self.llm.invoke(prompt)
            data = json.loads(getattr(resp, "content", "{}") or "{}")
            # Normalize
            return {
                "transform_expected": bool(data.get("transform_expected", False)),
                "transform_logic": (data.get("transform_logic") or None),
            }
        except Exception:
            # Heuristic fallback
            return {"transform_expected": True, "transform_logic": None}


class LogicTransformer:
    """
    Applies the provided transform_logic to input_value using an LLM.
    Returns {"transformed_value": ..., "transformer_explanation": "..."}.
    """
    def __init__(self, llm=None) -> None:
        self.llm = llm

    def transform(self, input_value: Any, transform_logic: str) -> TransformerResult:
        if not self.llm or not transform_logic:
            # Heuristic fallback for common cases
            try:
                logic = transform_logic.lower() if transform_logic else ""
                if "upper" in logic:
                    return {"transformed_value": str(input_value).upper(),
                            "transformer_explanation": "Applied uppercase to input."}
                if "lower" in logic:
                    return {"transformed_value": str(input_value).lower(),
                            "transformer_explanation": "Applied lowercase to input."}
                m = re.search(r"round\s+to\s+(\d+)\s*dp", logic) or re.search(r"round\s*\(?\s*(\d+)\s*\)?", logic)
                if m and _is_number_like(input_value):
                    ndp = int(m.group(1))
                    f = _to_float(input_value)
                    return {"transformed_value": f if f is None else round(f, ndp),
                            "transformer_explanation": f"Rounded to {ndp} decimal places."}
                if "trim" in logic or "strip" in logic or "whitespace" in logic:
                    return {"transformed_value": str(input_value).strip(),
                            "transformer_explanation": "Trimmed surrounding whitespace."}
                if "dd/mm/yyyy" in logic or "date format" in logic or "format date" in logic:
                    dt = _parse_datetime_any(input_value)
                    if dt:
                        return {"transformed_value": dt.strftime("%d/%m/%Y"),
                                "transformer_explanation": "Formatted date as dd/mm/yyyy."}
            except Exception:
                pass
            # Default: return unchanged with note
            return {"transformed_value": input_value, "transformer_explanation": "No LLM available; returned input."}

        # LLM JSON-only protocol to keep things robust
        summary = {
            "input_value": input_value,
            "transform_logic": transform_logic,
            "guidance": "If logic is ambiguous, pick the most common interpretation (e.g., 'round to 2dp'). "
                        "For dates/times, accept ISO forms and timezone conversions."
        }
        prompt = f"""
You transform an input according to a short logic description and return strict JSON.

DATA:
{json.dumps(summary, default=str)}

Return ONLY:
{{
  "transformed_value": <value>,     // keep type natural (number stays number, dates can be string)
  "explanation": "<one concise sentence about what you did>"
}}
"""
        try:
            resp = self.llm.invoke(prompt)
            data = json.loads(getattr(resp, "content", "{}") or "{}")
            return {
                "transformed_value": data.get("transformed_value"),
                "transformer_explanation": data.get("explanation", ""),
            }
        except Exception as e:
            return {"transformed_value": input_value, "transformer_explanation": f"LLM transform failed: {e}"}


class SupervisorExplainer:
    """
    Writes a concise human explanation into state['explanation'] describing the verdict and why.
    """
    def __init__(self, llm=None) -> None:
        self.llm = llm

    def explain(self, state: ValidatorState) -> str:
        if not self.llm:
            return f"Verdict: {state.get('verdict','UNKNOWN')}. Reason: {state.get('rationale','')}."
        compact = {
            "input_value": state.get("input_value"),
            "output_value": state.get("output_value"),
            "transformation_info": state.get("transformation_info"),
            "transform_expected": state.get("transform_expected"),
            "transform_logic": state.get("transform_logic"),
            "observation": getattr(state.get("observation", None), "__dict__", None),
            "transformed_input": state.get("transformed_input"),
            "verdict": state.get("verdict"),
            "rationale": state.get("rationale"),
        }
        prompt = f"""
Write a short (2–5 sentences) professional explanation for a validation result.
Be specific and auditable; do not reveal hidden reasoning.

DATA (JSON):
{json.dumps(compact, default=str)}

Return only the explanation paragraph.
"""
        try:
            resp = self.llm.invoke(prompt)
            return getattr(resp, "content", "").strip()
        except Exception:
            return f"Verdict: {state.get('verdict','UNKNOWN')}. Reason: {state.get('rationale','')}"


# =============================================================================
# Graph nodes
# =============================================================================

def node_supervisor_parse_info(state: ValidatorState, parser: SupervisorParser) -> ValidatorState:
    parsed = parser.parse(state.get("transformation_info", "") or "")
    state["transform_expected"] = bool(parsed.get("transform_expected", False))
    logic = parsed.get("transform_logic")
    state["transform_logic"] = logic.strip() if isinstance(logic, str) and logic.strip() else None
    return state

def node_direct_compare(state: ValidatorState) -> ValidatorState:
    a, b = state["input_value"], state["output_value"]
    match = best_compare(a, b)
    state["verdict"] = "CORRECT" if match else "INCORRECT"
    state["rationale"] = (
        "No transformation expected; values match under type-aware comparison."
        if match else
        "No transformation expected; values do not match under type-aware comparison."
    )
    return state

def node_check_transformed_no_logic(state: ValidatorState) -> ValidatorState:
    a, b = state["input_value"], state["output_value"]
    obs = detect_transformation(a, b)
    state["observation"] = obs
    # Your rule: if a change happened, it's good
    changed = not best_compare(a, b)
    if changed or obs.happened:
        state["verdict"] = "CORRECT"
        state["rationale"] = "Transformation expected; change detected between input and output."
    else:
        state["verdict"] = "INCORRECT"
        state["rationale"] = "Transformation expected; no effective change detected."
    return state

def node_apply_logic_and_compare(state: ValidatorState, transformer: LogicTransformer) -> ValidatorState:
    a, b = state["input_value"], state["output_value"]
    logic = state.get("transform_logic") or ""
    result = transformer.transform(a, logic)
    transformed = result.get("transformed_value", a)
    state["transformed_input"] = transformed  # for audit
    # Compare transformed input to output
    match = best_compare(transformed, b)
    state["verdict"] = "CORRECT" if match else "INCORRECT"
    if match:
        state["rationale"] = "Transformation applied per logic and transformed input matches output."
    else:
        state["rationale"] = "Transformation applied per logic but transformed input does not match output."
    return state

def node_supervisor_explainer(state: ValidatorState, explainer: SupervisorExplainer) -> ValidatorState:
    state["explanation"] = explainer.explain(state)
    return state


# =============================================================================
# Graph wiring
# =============================================================================

def build_validator_graph(
    parser_llm: Optional[Any] = None,        # AzureChatOpenAI for parsing
    transformer_llm: Optional[Any] = None,   # AzureChatOpenAI for transforming
    explainer_llm: Optional[Any] = None      # AzureChatOpenAI for explanation
):
    """
    Build the LangGraph with provided LLMs. All are optional; fallbacks keep it runnable.
    """
    parser = SupervisorParser(parser_llm)
    transformer = LogicTransformer(transformer_llm)
    explainer = SupervisorExplainer(explainer_llm)

    g = StateGraph(ValidatorState)

    # Wrap nodes to bind LLM helpers
    g.add_node("supervisor_parse_info", lambda s: node_supervisor_parse_info(s, parser))
    g.add_node("direct_compare", node_direct_compare)
    g.add_node("check_transformed_no_logic", node_check_transformed_no_logic)
    g.add_node("apply_logic_and_compare", lambda s: node_apply_logic_and_compare(s, transformer))
    g.add_node("supervisor_explainer", lambda s: node_supervisor_explainer(s, explainer))

    g.set_entry_point("supervisor_parse_info")

    def route(state: ValidatorState) -> Literal["direct", "no_logic", "with_logic"]:
        tf = bool(state.get("transform_expected", False))
        logic = (state.get("transform_logic") or "").strip()
        if not tf:
            return "direct"
        if tf and not logic:
            return "no_logic"
        return "with_logic"

    g.add_conditional_edges(
        "supervisor_parse_info",
        route,
        {
            "direct": "direct_compare",
            "no_logic": "check_transformed_no_logic",
            "with_logic": "apply_logic_and_compare",
        },
    )

    # Terminal → explainer → END
    g.add_edge("direct_compare", "supervisor_explainer")
    g.add_edge("check_transformed_no_logic", "supervisor_explainer")
    g.add_edge("apply_logic_and_compare", "supervisor_explainer")
    g.add_edge("supervisor_explainer", END)

    return g


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    # If you have Azure Chat LLMs, wire them in like this:
    #
    # from langchain_openai import AzureChatOpenAI
    # parser_model = AzureChatOpenAI(deployment_name="o3-mini", temperature=0)
    # transformer_model = AzureChatOpenAI(deployment_name="o3-mini", temperature=0)
    # explainer_model = AzureChatOpenAI(deployment_name="o1-preview", temperature=0)
    # app = build_validator_graph(parser_model, transformer_model, explainer_model).compile()
    #
    # Without LLMs, fallbacks are used (still runs, but explanations are basic):

    app = build_validator_graph().compile()

    tests = [
        # A) transform_expected = False → direct compare
        dict(
            input_value="2023-01-02",
            output_value="02/01/2023",  # same instant but different format → still equal
            transformation_info="transformation expected: false"
        ),
        # B) transform_expected = True, no logic → changed is OK
        dict(
            input_value="hello",
            output_value="HELLO",
            transformation_info="Yes, transformation expected. No logic provided."
        ),
        # C) transform_expected = True, with logic → LLM transforms and compares
        dict(
            input_value="3.14159",
            output_value="3.14",
            transformation_info="Transformation is expected. Logic: round to 2dp."
        ),
        # C) dates
        dict(
            input_value="2023-09-29T09:15:00Z",
            output_value="29/09/2023 10:15",
            transformation_info="Transformation expected; logic: convert UTC to Europe/London and format as dd/mm/yyyy HH:MM"
        ),
    ]

    for i, state in enumerate(tests, 1):
        out = app.invoke(state)  # type: ignore
        print(f"Case {i}: {out['verdict']} — {out['rationale']}")
        print("Explanation:", out.get("explanation","<none>"))
        print("-"*80)
