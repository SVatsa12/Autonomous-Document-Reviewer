"""Contract pipeline: PDF → clauses (LLM + per-rule passes) → rent/deposit analysis."""

import env_load  # noqa: F401 — loads GOOGLE_API_KEY from .env when python-dotenv is installed

import json
import sys
import time

# Show prints immediately in terminals/IDEs that fully buffer stdout.
try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, OSError):
    pass

from functions import (
    analyze_rent_clause,
    clean_clauses,
    clean_text,
    extract_rent_and_deposit,
    extract_rent_with_regex,
    extract_text,
    check_deposit_fairness,
    is_rent_clause,
    split_text,
)
from llm_ops import (
    CLAUSE_EXTRACTION_RULES,
    EXTRACTION_API_MODE,
    MODEL_NAME,
    extract_clauses_for_chunk,
    print_token_report,
)

# Smaller chunks → shorter per-request prompts (often faster each call); more chunks if the PDF is long.
CHUNK_MAX_CHARS = 8000
# Pause between chunks to ease rate limits (was 12s).
CHUNK_SLEEP_SEC = 2


def run_pipeline(pdf_path: str = "rent2.pdf", client=None):
    """Full processing used by the LLM tool and for direct runs."""
    if client is None:
        from google import genai

        client = genai.Client()

    print("Extracting text...", flush=True)
    text = extract_text(pdf_path)
    text = clean_text(text)

    print(
        f"Splitting text (max {CHUNK_MAX_CHARS} chars per chunk)...",
        flush=True,
    )
    chunks = split_text(text, max_chars=CHUNK_MAX_CHARS)

    all_clauses = []

    for i, chunk in enumerate(chunks):
        if EXTRACTION_API_MODE == "per_rule":
            call_note = (
                f"1 initial + {len(CLAUSE_EXTRACTION_RULES)} rule LLM calls per chunk"
            )
        else:
            call_note = "1 LLM call per chunk (all rules in one prompt)"
        print(
            f"Processing chunk {i + 1}/{len(chunks)} (model={MODEL_NAME}, "
            f"extraction={EXTRACTION_API_MODE!r}); ~{len(chunk)} chars — {call_note}.",
            flush=True,
        )

        parsed = extract_clauses_for_chunk(client, chunk)

        if isinstance(parsed, list):
            all_clauses.extend(parsed)
        else:
            print("Unexpected clause payload from extractor")

        if i < len(chunks) - 1:
            time.sleep(CHUNK_SLEEP_SEC)

    all_clauses = clean_clauses(all_clauses)

    print(f"\nExtracted {len(all_clauses)} clauses\n", flush=True)
    print("=" * 50, flush=True)

    global_rent = None
    for c in all_clauses:
        if is_rent_clause(c):
            rent = extract_rent_with_regex(c.get("clause_text"))
            c["rent"] = rent
            if rent is not None and global_rent is None:
                global_rent = rent

    print("\n RENT ANALYSIS\n")
    print("=" * 50)

    for c in all_clauses:
        result = analyze_rent_clause(c)

        if result:
            print(f"\nClause: {result['clause_number']}")
            print(f"Rent: ₹{result['rent']}")
            print(f"Status: {result['status']}")
            print(f"Message: {result['message']}")
            print("-" * 40)

    print("\n DEPOSIT ANALYSIS\n")
    print("=" * 50)
    found_deposit = None
    for c in all_clauses:
        text = c.get("clause_text", "")
        _, deposit = extract_rent_and_deposit(text)
        if deposit is not None:
            found_deposit = deposit
            break

    if found_deposit is not None:
        print(f"FOUND DEPOSIT: {found_deposit}")

    for c in all_clauses:
        text = c.get("clause_text")
        _, deposit = extract_rent_and_deposit(text)

        if deposit is not None:
            rent = global_rent
            result = check_deposit_fairness(rent, deposit)

            print(f"\nClause: {c['clause_number']}")
            print(f"Rent: ₹{rent}")
            print(f"Deposit: ₹{deposit}")
            print(f"Status: {result['status']}")
            print(f"Message: {result['message']}")
            print("-" * 40)

    out_path = "clauses_output.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_clauses, f, indent=2, ensure_ascii=False)

    print(f"\n Saved to {out_path}")
    print_token_report()


if __name__ == "__main__":
    run_pipeline()
