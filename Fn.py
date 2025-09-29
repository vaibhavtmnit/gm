"""
Validator Multi-Agent System (LangGraph) with LLM Supervisor

What’s new vs previous version:
- Adds `match_instruction` (free text). The LLM interprets it into a structured policy
  (tolerances, case sensitivity, whitespace handling, date handling).
- Adds `supervisor_explainer` LLM node that generates `state["explanation"]` explaining
  the final verdict in clear, concise prose (no chain-of-thought leakage).
- Keeps all deterministic checks intact.

Flow:
router → policy_interpreter → (branch) → verdict agent → supervisor_explainer → END
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Callable, TypedDict, Literal, Tuple
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

# If you want to wire LLMs here, import your instances in your app
# from langchain_openai import AzureChatOpenAI


# =============================================================================
# Tool Registry (unchanged)
# =============================================================================

class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Callable[[Any, Any], bool]] = {}

    def register(self, name: str, fn: Callable[[Any, Any], bool]) -> None:
        self._tools[name] = fn

    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Callable[[Any, Any], bool]:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not registered")
        return self._tools[name]

    def get_or_register(self, name: str, default_fn: Callable[[Any, Any], bool]) -> Callable[[Any, Any], bool]:
        if not self.has(name):
            self.register(name, default_fn)
        return self.get(name)


TOOLS = ToolRegistry()


# =============================================================================
# Type helpers & comparisons (augmented with policy support)
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


# ---- Policy (from LLM) -------------------------------------------------------

class Policy(TypedDict, total=False):
    # Numbers
    number_rel_tol: float
    number_abs_tol: float
    # Strings
    case_sensitive: bool
    ignore_whitespace: bool
    # Dates
    treat_dates_as_equal_if_same_instant: bool
    # General
    require_format_change_when_transformation_expected: bool


DEFAULT_POLICY: Policy = {
    "number_rel_tol": 1e-9,
    "number_abs_tol": 1e-9,
    "case_sensitive": False,
    "ignore_whitespace": True,
    "treat_dates_as_equal_if_same_instant": True,
    "require_format_change_when_transformation_expected": False,
}


# ---- Comparators (use policy) ------------------------------------------------

def _prep_string(s: Any, policy: Policy) -> str:
    t = str(s)
    if policy.get("ignore_whitespace", True):
        t = t.strip()
    if not policy.get("case_sensitive", False):
        t = t.lower()
    return t

def compare_numbers(a: Any, b: Any, *, rel_tol: float, abs_tol: float) -> bool:
    fa = _to_float(a); fb = _to_float(b)
    if fa is None or fb is None:
        return False
    return math.isclose(fa, fb, rel_tol=rel_tol, abs_tol=abs_tol)

def compare_strings(a: Any, b: Any, policy: Policy) -> bool:
    return _prep_string(a, policy) == _prep_string(b, policy)

def compare_datetimes(a: Any, b: Any, policy: Policy) -> bool:
    da = _parse_datetime_any(a); db = _parse_datetime_any(b)
    if da is None or db is None:
        return False
    if policy.get("treat_dates_as_equal_if_same_instant", True):
        return da == db
    # else fall back to exact string after normalization rules
    return compare_strings(a, b, policy)


def best_compare(a: Any, b: Any, policy: Policy) -> bool:
    # Normalize kinds
    def kind_of(x: Any) -> str:
        if _is_number_like(x): return "number"
        if _parse_datetime_any(x) is not None: return "datetime"
        if isinstance(x, str): return "string"
        return "other"

    ka, kb = kind_of(a), kind_of(b)

    if ka == "number" and kb == "number":
        rel = policy.get("number_rel_tol", DEFAULT_POLICY["number_rel_tol"])  # type: ignore
        abs_ = policy.get("number_abs_tol", DEFAULT_POLICY["number_abs_tol"])  # type: ignore
        return compare_numbers(a, b, rel_tol=rel, abs_tol=abs_)
    if ka == "datetime" and kb == "datetime":
        return compare_datetimes(a, b, policy)
    if ka == "string" and kb == "string":
        # If both strings parse to same instant, accept
        if compare_datetimes(a, b, policy):
            return True
        return compare_strings(a, b, policy)

    # Cross-kind helpful handling
    if {ka, kb} == {"string", "datetime"}:
        return compare_datetimes(a, b, policy)
    if {ka, kb} == {"string", "number"}:
        rel = policy.get("number_rel_tol", DEFAULT_POLICY["number_rel_tol"])  # type: ignore
        abs_ = policy.get("number_abs_tol", DEFAULT_POLICY["number_abs_tol"])  # type: ignore
        return compare_numbers(a, b, rel_tol=rel, abs_tol=abs_)

    # Fallback
    return a == b


# Register default tools (names kept for extensibility)
TOOLS.register("compare_numbers", lambda a, b: compare_numbers(a, b, rel_tol=DEFAULT_POLICY["number_rel_tol"], abs_tol=DEFAULT_POLICY["number_abs_tol"]))  # type: ignore
TOOLS.register("compare_strings", lambda a, b: compare_strings(a, b, DEFAULT_POLICY))
TOOLS.register("compare_datetimes", lambda a, b: compare_datetimes(a, b, DEFAULT_POLICY))


# =============================================================================
# Transformation detection & alignment (unchanged logic)
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
    normalized_equal = best_compare(a, b, DEFAULT_POLICY)

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


def aligns_with_processing_logic(observation: TransformObservation, processing_logic: str) -> bool:
    if not processing_logic:
        return False
    text = processing_logic.lower()

    want_round = any(k in text for k in ["round", "decimal", "dp", "2dp", "precision"])
    want_upper = any(k in text for k in ["upper", "uppercase"])
    want_lower = any(k in text for k in ["lower", "lowercase"])
    want_trim  = any(k in text for k in ["trim", "strip", "whitespace"])
    want_date_fmt = any(k in text for k in ["date format", "format", "yyyy", "dd/mm", "iso"])
    want_tz = any(k in text for k in ["timezone", "tz", "utc", "z", "offset"])

    reasons = set(observation.reasons)

    if want_round and ("rounding" in reasons or "tiny_float_adjustment" in reasons):
        return True
    if (want_upper or want_lower) and ("case_change" in reasons or "case_change_and/or_whitespace" in reasons):
        return True
    if want_trim and ("whitespace_trim" in reasons or "case_change_and/or_whitespace" in reasons):
        return True
    if want_date_fmt and ("date_format_change" in reasons or "format_change" in reasons):
        return True
    if want_tz and ("timezone_representation_change" in reasons or "format_change" in reasons):
        return True
    if observation.details.get("normalized_equal") and "format" in text:
        return True
    return False


# =============================================================================
# LangGraph State & Agents (now with LLM policy + explanation)
# =============================================================================

class Verdict(TypedDict):
    status: Literal["CORRECT", "INCORRECT"]
    rationale: str

class ValidatorState(TypedDict, total=False):
    input_value: Any
    output_value: Any
    processing_logic: Optional[str]
    transform_flag: bool
    match_instruction: Optional[str]  # Free-text instruction for matching, parsed by LLM

    # Derived
    policy: Policy
    observation: TransformObservation
    alignment: Optional[bool]
    direct_match: Optional[bool]
    verdict: Verdict
    explanation: str  # Written by LLM supervisor


# ---- Router (same) -----------------------------------------------------------

def router_analyze(state: ValidatorState) -> ValidatorState:
    return state


# ---- LLM: Policy Interpreter -------------------------------------------------

class PolicyInterpreter:
    """
    Wraps an LLM to convert free-text match instructions into a Policy dict.
    Provide your AzureChatOpenAI (o1-preview / o3-mini) via constructor.
    """
    def __init__(self, llm=None) -> None:
        self.llm = llm  # e.g., AzureChatOpenAI(deployment_name="o3-mini", temperature=0)

    def run(self, instruction: Optional[str]) -> Policy:
        if not self.llm or not instruction or not instruction.strip():
            return dict(DEFAULT_POLICY)  # copy

        prompt = f"""
You convert a human instruction into a strict JSON policy for value comparison.

Instruction:
\"\"\"{instruction.strip()}\"\"\"

Return ONLY a JSON object with keys (omit any you don't set):
- number_rel_tol: float
- number_abs_tol: float
- case_sensitive: boolean
- ignore_whitespace: boolean
- treat_dates_as_equal_if_same_instant: boolean
- require_format_change_when_transformation_expected: boolean
"""
        try:
            resp = self.llm.invoke(prompt)  # expects .content with JSON
            text = getattr(resp, "content", "") if resp else ""
            policy = json.loads(text)
            # Merge onto defaults
            merged: Policy = dict(DEFAULT_POLICY)
            merged.update({k: v for k, v in policy.items() if k in DEFAULT_POLICY})
            return merged
        except Exception:
            return dict(DEFAULT_POLICY)


def agent_policy_interpreter(state: ValidatorState, policy_llm: Optional[PolicyInterpreter] = None) -> ValidatorState:
    interpreter = policy_llm or PolicyInterpreter(llm=None)
    state["policy"] = interpreter.run(state.get("match_instruction"))
    return state


# ---- Branch agents (unchanged decisions, but use policy in comparisons) ------

def agent_direct_compare(state: ValidatorState) -> ValidatorState:
    a = state["input_value"]; b = state["output_value"]
    policy = state.get("policy", DEFAULT_POLICY)
    match = best_compare(a, b, policy)
    state["direct_match"] = match
    state["verdict"] = {
        "status": "CORRECT" if match else "INCORRECT",
        "rationale": (
            "No transformation expected; values match under policy."
            if match else
            "No transformation expected; values do not match under policy."
        )
    }
    return state

def agent_check_transformed_no_logic(state: ValidatorState) -> ValidatorState:
    a = state["input_value"]; b = state["output_value"]
    policy = state.get("policy", DEFAULT_POLICY)
    equal = best_compare(a, b, policy)
    if equal:
        state["verdict"] = {
            "status": "INCORRECT",
            "rationale": "Transformation was expected but values are effectively equal under policy."
        }
    else:
        # Optional policy: require a visible format change to accept
        if state["policy"].get("require_format_change_when_transformation_expected", False):
            obs = detect_transformation(a, b)
            if not obs.happened:
                state["verdict"] = {
                    "status": "INCORRECT",
                    "rationale": "Transformation expected; no observable change was detected."
                }
                return state
        state["verdict"] = {
            "status": "CORRECT",
            "rationale": "Transformation expected and values differ under policy; accepted as transformed."
        }
    return state

def agent_detect_and_align(state: ValidatorState) -> ValidatorState:
    a = state["input_value"]; b = state["output_value"]
    obs = detect_transformation(a, b)
    state["observation"] = obs
    align = aligns_with_processing_logic(obs, state.get("processing_logic") or "")
    state["alignment"] = align

    if obs.happened and align:
        norm_equal = best_compare(a, b, state.get("policy", DEFAULT_POLICY))
        bits = [
            "Transformation detected: " + ", ".join(obs.reasons) if obs.reasons else "Transformation detected",
            "Processing logic aligns",
            f"Normalized equality under policy: {norm_equal}"
        ]
        state["verdict"] = {"status": "CORRECT", "rationale": "; ".join(bits)}
    else:
        missing = []
        if not obs.happened:
            missing.append("no transformation observed")
        if not align:
            missing.append("does not align with processing logic")
        state["verdict"] = {"status": "INCORRECT", "rationale": " / ".join(missing) if missing else "Failed alignment."}
    return state


# ---- LLM: Supervisor Explainer ----------------------------------------------

class SupervisorExplainer:
    """
    Uses an LLM to produce a concise explanation of the result and why.
    Provide your AzureChatOpenAI via constructor.
    """
    def __init__(self, llm=None) -> None:
        self.llm = llm  # e.g., AzureChatOpenAI(deployment_name="o1-preview", temperature=0)

    def explain(self, state: ValidatorState) -> str:
        if not self.llm:
            # Deterministic fallback
            v = state.get("verdict", {})
            return (
                f"Verdict: {v.get('status','UNKNOWN')}. "
                f"Reason: {v.get('rationale','')} "
                f"(policy: {json.dumps(state.get('policy', DEFAULT_POLICY))})."
            )

        # Build a compact, auditable summary (no chain-of-thought, just conclusions)
        summary = {
            "input_value": state.get("input_value"),
            "output_value": state.get("output_value"),
            "transform_flag": state.get("transform_flag"),
            "processing_logic": state.get("processing_logic"),
            "match_instruction": state.get("match_instruction"),
            "policy": state.get("policy", DEFAULT_POLICY),
            "observation": getattr(state.get("observation", None), "__dict__", None),
            "alignment": state.get("alignment"),
            "verdict": state.get("verdict"),
        }
        prompt = f"""
You are an auditor. Given the data below, write a short, professional explanation (2–6 sentences)
that justifies the verdict. Be specific but concise. Do not include hidden reasoning.

DATA (JSON):
{json.dumps(summary, default=str)}

Write only the explanation paragraph.
"""
        try:
            resp = self.llm.invoke(prompt)
            return getattr(resp, "content", "").strip()
        except Exception:
            v = state.get("verdict", {})
            return f"Verdict: {v.get('status','UNKNOWN')}. Reason: {v.get('rationale','')}"


def agent_supervisor_explainer(state: ValidatorState, explainer: Optional[SupervisorExplainer] = None) -> ValidatorState:
    sup = explainer or SupervisorExplainer(llm=None)
    state["explanation"] = sup.explain(state)
    return state


# =============================================================================
# Graph wiring
# =============================================================================

def build_validator_graph(policy_llm: Optional[PolicyInterpreter] = None,
                          supervisor_llm: Optional[SupervisorExplainer] = None) -> StateGraph:
    g = StateGraph(ValidatorState)

    # Nodes
    g.add_node("router", router_analyze)
    # LLM policy interpreter
    g.add_node("policy_interpreter", lambda s: agent_policy_interpreter(s, policy_llm))
    # Branch nodes
    g.add_node("direct_compare", agent_direct_compare)
    g.add_node("check_transformed_no_logic", agent_check_transformed_no_logic)
    g.add_node("detect_and_align", agent_detect_and_align)
    # LLM explainer
    g.add_node("supervisor_explainer", lambda s: agent_supervisor_explainer(s, supervisor_llm))

    g.set_entry_point("router")
    g.add_edge("router", "policy_interpreter")

    def route_decision(state: ValidatorState) -> Literal["direct", "no_logic", "with_logic"]:
        tf = bool(state.get("transform_flag", False))
        logic = (state.get("processing_logic") or "").strip()
        if not tf:
            return "direct"
        if tf and not logic:
            return "no_logic"
        return "with_logic"

    g.add_conditional_edges(
        "policy_interpreter",
        route_decision,
        {"direct": "direct_compare", "no_logic": "check_transformed_no_logic", "with_logic": "detect_and_align"},
    )

    # All verdicts → supervisor explainer → END
    g.add_edge("direct_compare", "supervisor_explainer")
    g.add_edge("check_transformed_no_logic", "supervisor_explainer")
    g.add_edge("detect_and_align", "supervisor_explainer")
    g.add_edge("supervisor_explainer", END)
    return g


# Create a compiled app without bound LLMs (deterministic). In your app,
# build with your LLM instances and keep the same API.
VALIDATOR_APP = build_validator_graph().compile()


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    # If you have LLMs, pass them in like:
    # policy_llm = PolicyInterpreter(llm=AzureChatOpenAI(deployment_name="o3-mini", temperature=0))
    # sup_llm = SupervisorExplainer(llm=AzureChatOpenAI(deployment_name="o1-preview", temperature=0))
    # app = build_validator_graph(policy_llm, sup_llm).compile()

    app = VALIDATOR_APP  # deterministic fallback

    examples = [
        dict(
            input_value="3.1400",
            output_value=3.14,
            processing_logic=None,
            transform_flag=False,
            match_instruction="Numeric equality with 1e-6 tolerance; ignore whitespace; case-insensitive."
        ),
        dict(
            input_value="Hello",
            output_value="HELLO",
            processing_logic="",
            transform_flag=True,
            match_instruction="Case-insensitive match is fine."
        ),
        dict(
            input_value="hello world",
            output_value="HELLO WORLD",
            processing_logic="convert to UPPERCASE",
            transform_flag=True,
            match_instruction="Strings should be matched case-insensitively; trimming allowed."
        ),
        dict(
            input_value="3.14159",
            output_value="3.14",
            processing_logic="round to 2 decimals",
            transform_flag=True,
            match_instruction="Numbers: allow rounding to 2dp."
        ),
        dict(
            input_value="2023-01-02",
            output_value="02/01/2023",
            processing_logic="format date as dd/mm/yyyy",
            transform_flag=True,
            match_instruction="Dates represent the same instant even if formatted differently."
        ),
        dict(
            input_value=" foo ",
            output_value="foo",
            processing_logic="trim whitespace",
            transform_flag=True,
            match_instruction="Ignore whitespace differences."
        ),
    ]

    for i, state in enumerate(examples, 1):
        out = app.invoke(state)  # type: ignore
        print(f"Case {i}: {out['verdict']['status']} — {out['verdict']['rationale']}")
        print("Explanation:", out.get("explanation","<none>"))
        print("-" * 80)
