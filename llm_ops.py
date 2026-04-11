"""Gemini calls, per-rule extraction passes, and token accounting."""

import json

from functions import extract_clauses_locally, safe_parse

MODEL_NAME = "gemini-2.5-flash"
API_DISABLED = False

# "single" → one Gemini call per chunk (all rules in one prompt; fewest API calls).
# "per_rule" → one call for initial JSON + one call per CLAUSE_EXTRACTION_RULES entry.
EXTRACTION_API_MODE = "single"

# Extraction policy rules. In "per_rule" mode, each entry is its own API round-trip.
CLAUSE_EXTRACTION_RULES = [
    (
        "separate_numbering",
        "Separate top-level clauses (1, 2, 3) and subclauses (i, ii, iii). "
        "Each distinct numbered segment must be its own JSON object with accurate clause_number.",
    ),
    (
        "no_merge",
        "Do NOT merge multiple distinct clauses or subclauses into a single object.",
    ),
    (
        "full_meaning",
        "Preserve full semantic meaning; do not summarize away obligations, amounts, or dates.",
    ),
    (
        "formatting",
        "Fix OCR or line-break artifacts while preserving legal meaning.",
    ),
]


def _rules_bullet_list():
    lines = []
    for i, (_rid, instruction) in enumerate(CLAUSE_EXTRACTION_RULES, start=1):
        lines.append(f"{i}. ({_rid}) {instruction}")
    return "\n".join(lines)


def extract_clauses_for_chunk(client, chunk):
    """Extract clauses via Gemini; API count depends on EXTRACTION_API_MODE."""
    if EXTRACTION_API_MODE == "per_rule":
        return _extract_clauses_per_rule_passes(client, chunk)
    return _extract_clauses_single_call(client, chunk)


def _extract_clauses_single_call(client, chunk):
    global API_DISABLED

    if API_DISABLED:
        return extract_clauses_locally(chunk)

    rules_block = _rules_bullet_list()
    prompt = f"""You are a legal AI system. Extract ALL clauses from the contract into a JSON array.
Each object must have: clause_number, clause_title, clause_text.

Apply ALL of these rules together:
{rules_block}

Return ONLY a valid JSON array. No markdown, no explanation.

CONTRACT TEXT:
{chunk}
"""

    try:
        response = tracked_generate(
            client, prompt, label="clause extraction (single call, all rules)"
        )
        raw = response.text
    except Exception as e:
        msg = str(e)
        print("Clause extraction failed:", msg)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            API_DISABLED = True
            print("Disabling further API calls for this run due to quota limit.")
        local = extract_clauses_locally(chunk)
        print(f"Using local fallback extractor: found {len(local)} clauses in this chunk.")
        return local

    clauses = safe_parse(raw)
    if clauses is None or not isinstance(clauses, list):
        local = extract_clauses_locally(chunk)
        print("JSON parse failed; using local fallback for this chunk.")
        return local
    return clauses


def _extract_clauses_per_rule_passes(client, chunk):
    """Initial extraction plus one LLM call per policy rule in CLAUSE_EXTRACTION_RULES."""
    global API_DISABLED

    if API_DISABLED:
        return extract_clauses_locally(chunk)

    initial_prompt = f"""You are a legal AI system. Extract ALL clauses from the contract into a JSON array.
Each object must have: clause_number, clause_title, clause_text.

Return ONLY a valid JSON array. No markdown, no explanation.

CONTRACT TEXT:
{chunk}
"""

    try:
        response = tracked_generate(
            client, initial_prompt, label="initial clause extraction"
        )
        raw = response.text
    except Exception as e:
        msg = str(e)
        print("Clause extraction failed:", msg)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            API_DISABLED = True
            print("Disabling further API calls for this run due to quota limit.")
        local = extract_clauses_locally(chunk)
        print(f"Using local fallback extractor: found {len(local)} clauses in this chunk.")
        return local

    clauses = safe_parse(raw)
    if clauses is None or not isinstance(clauses, list):
        local = extract_clauses_locally(chunk)
        print("Initial JSON parse failed; using local fallback for this chunk.")
        return local

    for rule_id, rule_instruction in CLAUSE_EXTRACTION_RULES:
        rule_prompt = f"""You refine clause extraction JSON under ONE rule only.

RULE_ID: {rule_id}
RULE: {rule_instruction}

You MUST return ONLY a valid JSON array of objects with keys:
clause_number, clause_title, clause_text.

CONTRACT TEXT (for reference):
{chunk}

CURRENT_CLAUSES_JSON:
{json.dumps(clauses, ensure_ascii=False)}

Apply ONLY this rule; keep all clauses that still belong. Return ONLY the JSON array."""

        try:
            r = tracked_generate(
                client, rule_prompt, label=f"rule pass ({rule_id})"
            )
            parsed = safe_parse(r.text)
            if isinstance(parsed, list):
                clauses = parsed
        except Exception as e:
            print(f"Rule pass {rule_id} failed:", e)
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                API_DISABLED = True
            break

    return clauses


class TokenTracker:
    def __init__(self):
        self.prompt_tokens = 0
        self.candidates_tokens = 0
        self.total_tokens = 0
        self.calls = 0

    def reset(self):
        self.__init__()

    def add(self, response):
        um = getattr(response, "usage_metadata", None)
        if um is None:
            return
        self.calls += 1
        if um.prompt_token_count is not None:
            self.prompt_tokens += um.prompt_token_count
        if um.candidates_token_count is not None:
            self.candidates_tokens += um.candidates_token_count
        if um.total_token_count is not None:
            self.total_tokens += um.total_token_count


token_tracker = TokenTracker()


def reset_token_tracker():
    token_tracker.reset()


def tracked_generate(client, contents, config=None, model=MODEL_NAME, label=""):
    if label:
        print(f"  → LLM request: {label} …", flush=True)
    kwargs = {"model": model, "contents": contents}
    if config is not None:
        kwargs["config"] = config
    response = client.models.generate_content(**kwargs)
    token_tracker.add(response)
    if label:
        print(f"  ← Done: {label}", flush=True)
    return response


def format_token_report():
    t = token_tracker
    lines = [
        "",
        "=== LLM token usage (cumulative) ===",
        f"API calls counted: {t.calls}",
        f"Prompt tokens (sum): {t.prompt_tokens}",
        f"Candidates tokens (sum): {t.candidates_tokens}",
        f"Total tokens (sum of per-response totals): {t.total_tokens}",
        "====================================",
    ]
    return "\n".join(lines)


def print_token_report():
    print(format_token_report())
