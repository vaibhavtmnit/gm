"""
Validator Multi-Agent System (LangGraph)

Scenarios enforced:
1) transform_flag == False
   -> Directly compare input_value vs output_value with best available tool.

2) transform_flag == True and processing_logic is empty/None
   -> Assume "some transformation happened".
   -> If input_value != output_value (after type-aware normalization), verdict = CORRECT.
      Else verdict = INCORRECT (flag says transformed but values are same).

3) transform_flag == True and processing_logic provided
   -> Detect whether a transformation actually happened (heuristics).
   -> Check alignment between observed transformation and processing_logic (rules/keywords).
   -> Verdict = CORRECT iff (transformation happened) AND (alignment == True).
   -> Also perform type-aware comparison post-transformation to ensure sensible equality.

Types covered: numbers, strings, dates/timestamps (ISO/loose formats).
Dynamic registry: if an appropriate comparator is unavailable, it's created and cached.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Callable, TypedDict, Literal, Tuple
from dataclasses import dataclass
from datetime import datetime, date
import math
import re

# You can keep these imports if you already have them in your env.
# dateutil parser is very robust—falls back to manual formats if absent.
try:
    from dateutil import parser as dateparser  # type: ignore
except Exception:  # pragma: no cover
    dateparser = None

# --- LangGraph ---
from langgraph.graph import StateGraph, END
# (No LLM calls are made in this file; you can wire your AzureChatOpenAI in if desired.)


# =============================================================================
# Utility: Dynamic Tool Registry
# =============================================================================

class ToolRegistry:
    """Simple registry for comparator tools; grows dynamically as needed."""
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
# Type helpers & comparisons
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

def _safe_strip(s: Any) -> str:
    return str(s).strip()

def _parse_datetime_any(x: Any) -> Optional[datetime]:
    """
    Try to parse as datetime using dateutil if present; else fall back to common formats.
    Accepts date-only (converted to 00:00).
    """
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None

    # Prefer dateutil for robustness
    if dateparser is not None:
        try:
            return dateparser.parse(s)
        except Exception:
            pass

    # Fallback common patterns
    fmts = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%d-%b-%Y",
        "%d %b %Y",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            return dt
        except Exception:
            continue
    return None

def _normalize_for_compare(x: Any) -> Tuple[str, Any]:
    """
    Returns (kind, normalized_value)
    kind in {"number","datetime","string","other"}
    """
    if _is_number_like(x):
        f = _to_float(x)
        if f is not None and math.isfinite(f):
            return "number", f
    dt = _parse_datetime_any(x)
    if dt is not None:
        return "datetime", dt
    if isinstance(x, str):
        return "string", x
    return "other", x


# ---- Default comparator tools ----

def compare_numbers(a: Any, b: Any, *, rel_tol: float = 1e-9, abs_tol: float = 1e-9) -> bool:
    fa = _to_float(a)
    fb = _to_float(b)
    if fa is None or fb is None:
        return False
    return math.isclose(fa, fb, rel_tol=rel_tol, abs_tol=abs_tol)

def compare_strings(a: Any, b: Any) -> bool:
    return _safe_strip(a) == _safe_strip(b)

def compare_datetimes(a: Any, b: Any) -> bool:
    da = _parse_datetime_any(a)
    db = _parse_datetime_any(b)
    if da is None or db is None:
        return False
    return da == db

# Register defaults
TOOLS.register("compare_numbers", compare_numbers)
TOOLS.register("compare_strings", compare_strings)
TOOLS.register("compare_datetimes", compare_datetimes)


def best_compare(a: Any, b: Any) -> bool:
    """
    Type-aware comparison choosing the best available tool.
    """
    ka, na = _normalize_for_compare(a)
    kb, nb = _normalize_for_compare(b)

    # If the kinds match, use the dedicated tool
    if ka == "number" and kb == "number":
        return TOOLS.get("compare_numbers")(na, nb)
    if ka == "datetime" and kb == "datetime":
        return TOOLS.get("compare_datetimes")(na, nb)
    if ka == "string" and kb == "string":
        # Special case: strings might actually be datetimes in different formats
        if compare_datetimes(a, b):
            return True
        return TOOLS.get("compare_strings")(na, nb)

    # Cross-kind graceful handling (e.g., string ↔ datetime/number)
    if (ka == "string" and kb == "datetime") or (ka == "datetime" and kb == "string"):
        return compare_datetimes(a, b)
    if (ka == "string" and kb == "number") or (ka == "number" and kb == "string"):
        return compare_numbers(a, b)

    # Fallback: Python equality
    name = f"compare_{ka}_to_{kb}"
    def _fallback(x: Any, y: Any) -> bool:
        return x == y
    return TOOLS.get_or_register(name, _fallback)(a, b)


# =============================================================================
# Transformation detection + logic alignment
# =============================================================================

@dataclass
class TransformObservation:
    happened: bool
    reasons: list[str]  # e.g., ["whitespace_trim", "case_change", "format_change", "rounding", "tz_shift"]
    details: Dict[str, Any]

def detect_transformation(a: Any, b: Any) -> TransformObservation:
    """
    Heuristic detector to infer whether a transformation occurred from input->output.
    """
    reasons: list[str] = []
    details: Dict[str, Any] = {}

    # Baseline equality
    raw_equal = (str(a) == str(b))
    normalized_equal = best_compare(a, b)

    # Case changes / whitespace
    if isinstance(a, str) or isinstance(b, str):
        sa, sb = str(a), str(b)
        if sa.strip() != sb.strip() and sa.lower().strip() == sb.lower().strip():
            reasons.append("case_change_and/or_whitespace")
        elif sa != sb and sa.strip() == sb.strip():
            reasons.append("whitespace_trim")
        elif sa.lower() != sb.lower() and sa.strip().lower() == sb.strip().lower():
            reasons.append("case_change")

    # Date format change
    da = _parse_datetime_any(a)
    db = _parse_datetime_any(b)
    if da and db:
        if da == db and str(a).strip() != str(b).strip():
            reasons.append("date_format_change")

        # Timezone shift (very crude — only if naive vs aware or different tz offsets)
        if (da.tzinfo is None) != (db.tzinfo is None):
            reasons.append("timezone_representation_change")

    # Numeric rounding or formatting
    if _is_number_like(a) and _is_number_like(b):
        fa = _to_float(a); fb = _to_float(b)
        if fa is not None and fb is not None:
            if not math.isclose(fa, fb, rel_tol=0, abs_tol=0) and math.isclose(fa, fb, rel_tol=0, abs_tol=1e-9):
                reasons.append("tiny_float_adjustment")
            # Rounding: e.g., 3.14159 -> 3.14
            if round(fa, 2) == fb or round(fa, 3) == fb:
                reasons.append("rounding")

    # General formatting change (as a catch-all if normalized_equal but raw different)
    if not raw_equal and normalized_equal:
        reasons.append("format_change")

    happened = (len(reasons) > 0) or (not raw_equal)
    details["raw_equal"] = raw_equal
    details["normalized_equal"] = normalized_equal
    return TransformObservation(happened=happened, reasons=sorted(set(reasons)), details=details)


def aligns_with_processing_logic(observation: TransformObservation, processing_logic: str) -> bool:
    """
    Very simple keyword-based alignment between observed transformation and textual logic.
    Extend the keyword maps as needed.
    """
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

    # Numbers / rounding alignment
    if want_round and ("rounding" in reasons or "tiny_float_adjustment" in reasons):
        return True

    # Case alignment
    if want_upper or want_lower:
        if "case_change" in reasons or "case_change_and/or_whitespace" in reasons:
            return True

    # Trim alignment
    if want_trim and ("whitespace_trim" in reasons or "case_change_and/or_whitespace" in reasons):
        return True

    # Date/time format alignment
    if want_date_fmt and ("date_format_change" in reasons or "format_change" in reasons):
        return True

    # Timezone representation change
    if want_tz and ("timezone_representation_change" in reasons or "format_change" in reasons):
        return True

    # Generic format change request
    if "format" in text and ("format_change" in reasons or "date_format_change" in reasons):
        return True

    # As a conservative default, if normalized_equal is True and logic is vague about formatting
    if observation.details.get("normalized_equal") and "format" in text:
        return True

    return False


# =============================================================================
# LangGraph State & Agents
# =============================================================================

class Verdict(TypedDict):
    status: Literal["CORRECT", "INCORRECT"]
    rationale: str

class ValidatorState(TypedDict, total=False):
    input_value: Any
    output_value: Any
    processing_logic: Optional[str]
    transform_flag: bool

    # Derived / scratch
    observation: TransformObservation
    alignment: Optional[bool]
    direct_match: Optional[bool]

    verdict: Verdict


# --- Agents (pure functions; you can replace with LLM calls if desired) ---

def router_analyze(state: ValidatorState) -> ValidatorState:
    """
    Decides the next path purely based on transform_flag and presence of logic.
    """
    # no-op; routing is done by conditional edges.
    return state

def agent_direct_compare(state: ValidatorState) -> ValidatorState:
    """Compare input vs output when no transformation expected."""
    a = state["input_value"]; b = state["output_value"]
    match = best_compare(a, b)
    state["direct_match"] = match
    state["verdict"] = {
        "status": "CORRECT" if match else "INCORRECT",
        "rationale": (
            "No transformation expected; values match with appropriate comparator."
            if match else
            "No transformation expected; values do not match."
        )
    }
    return state

def agent_check_transformed_no_logic(state: ValidatorState) -> ValidatorState:
    """
    Transform flag true, but no processing logic provided:
    - If input != output (by best_compare), we accept that 'some transformation happened' → CORRECT.
    - Else → INCORRECT (flag says transformed but values are same).
    """
    a = state["input_value"]; b = state["output_value"]
    equal = best_compare(a, b)
    if equal:
        state["verdict"] = {
            "status": "INCORRECT",
            "rationale": "Transformation was expected but values are effectively equal."
        }
    else:
        state["verdict"] = {
            "status": "CORRECT",
            "rationale": "Transformation was expected and values differ; accepted as transformed."
        }
    return state

def agent_detect_and_align(state: ValidatorState) -> ValidatorState:
    """
    Transform flag true and processing logic provided:
    - Detect transformation.
    - Verify alignment with processing logic.
    - If both true → CORRECT; else → INCORRECT.
    """
    a = state["input_value"]; b = state["output_value"]
    obs = detect_transformation(a, b)
    state["observation"] = obs
    align = aligns_with_processing_logic(obs, state.get("processing_logic") or "")
    state["alignment"] = align

    if obs.happened and align:
        # Optional: also ensure normalized equality if logic implies equality post-transform
        # Many format/case transforms keep value equivalence under normalized comparator
        norm_equal = best_compare(a, b)
        rationale_bits = [
            "Transformation detected: " + ", ".join(obs.reasons) if obs.reasons else "Transformation detected",
            "Processing logic aligns",
            f"Normalized equality: {norm_equal}"
        ]
        state["verdict"] = {
            "status": "CORRECT",
            "rationale": "; ".join(rationale_bits)
        }
    else:
        missing = []
        if not obs.happened:
            missing.append("no transformation observed")
        if not align:
            missing.append("does not align with processing logic")
        state["verdict"] = {
            "status": "INCORRECT",
            "rationale": " / ".join(missing) if missing else "Failed alignment."
        }
    return state


# =============================================================================
# Graph wiring
# =============================================================================

def build_validator_graph() -> StateGraph:
    g = StateGraph(ValidatorState)

    # Nodes
    g.add_node("router", router_analyze)
    g.add_node("direct_compare", agent_direct_compare)
    g.add_node("check_transformed_no_logic", agent_check_transformed_no_logic)
    g.add_node("detect_and_align", agent_detect_and_align)

    g.set_entry_point("router")

    # Conditional routing
    def route_decision(state: ValidatorState) -> Literal["direct", "no_logic", "with_logic"]:
        tf = bool(state.get("transform_flag", False))
        logic = (state.get("processing_logic") or "").strip()
        if not tf:
            return "direct"
        if tf and not logic:
            return "no_logic"
        return "with_logic"

    g.add_conditional_edges(
        "router",
        route_decision,
        {
            "direct": "direct_compare",
            "no_logic": "check_transformed_no_logic",
            "with_logic": "detect_and_align",
        },
    )

    # All terminal nodes go to END
    g.add_edge("direct_compare", END)
    g.add_edge("check_transformed_no_logic", END)
    g.add_edge("detect_and_align", END)
    return g


VALIDATOR_APP = build_validator_graph().compile()


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    examples = [
        # 1) No transform expected (numbers equal)
        dict(
            input_value="3.1400",
            output_value=3.14,
            processing_logic=None,
            transform_flag=False
        ),
        # 2) Transform expected, no logic; values differ -> accept as transformed
        dict(
            input_value="Hello",
            output_value="HELLO",
            processing_logic="",
            transform_flag=True
        ),
        # 3) Transform expected, logic provided; case change aligned
        dict(
            input_value="hello world",
            output_value="HELLO WORLD",
            processing_logic="convert to UPPERCASE",
            transform_flag=True
        ),
        # 4) Transform expected, logic provided; round to 2dp aligned
        dict(
            input_value="3.14159",
            output_value="3.14",
            processing_logic="round to 2 decimals",
            transform_flag=True
        ),
        # 5) Transform expected, logic provided; date format change aligned
        dict(
            input_value="2023-01-02",
            output_value="02/01/2023",
            processing_logic="format date as dd/mm/yyyy",
            transform_flag=True
        ),
        # 6) Transform expected, logic provided; but no change observed
        dict(
            input_value="foo",
            output_value="foo",
            processing_logic="trim whitespace",
            transform_flag=True
        ),
    ]

    for i, state in enumerate(examples, 1):
        out = VALIDATOR_APP.invoke(state)  # type: ignore
        print(f"Case {i}: {out['verdict']['status']} — {out['verdict']['rationale']}")
