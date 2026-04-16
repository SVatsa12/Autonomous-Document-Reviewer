"""Contract pipeline: PDF → clauses (LLM + per-rule passes) → rent/deposit analysis."""

import env_load 
import json
import sys
import time


try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
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
)
from vector_db import ClauseVectorDB


CHUNK_MAX_CHARS = 8000
# Pause between chunks to ease rate limits (was 12s).
CHUNK_SLEEP_SEC = 2
VECTOR_DB_PATH = "clause_vectors.json"


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

    print("\nBuilding vector database for semantic clause search...", flush=True)
    vector_db = ClauseVectorDB(persist_path=VECTOR_DB_PATH)
    vector_db.add_clauses(client, all_clauses)
    vector_db.save()
    print(f"Vector DB saved to {VECTOR_DB_PATH}", flush=True)

    global_rent = None
    rent_matches = vector_db.search(
        client,
        "monthly rent amount payable by tenant",
        top_k=5,
    )
    rent_candidates = [r["clause"] for r in rent_matches] if rent_matches else all_clauses
    for c in rent_candidates:
        if is_rent_clause(c):
            rent = extract_rent_with_regex(c.get("clause_text"))
            c["rent"] = rent
            if rent is not None and global_rent is None:
                global_rent = rent

    print("\n RENT ANALYSIS\n")
    print("=" * 50)

    for c in rent_candidates:
        result = analyze_rent_clause(c)

        if result:
            print(f"\nClause: {result['clause_number']}")
            print(f"Rent: Rs {result['rent']}")
            print(f"Status: {result['status']}")
            print(f"Message: {result['message']}")
            print("-" * 40)

    print("\n DEPOSIT ANALYSIS\n")
    print("=" * 50)
    found_deposit = None
    deposit_matches = vector_db.search(
        client,
        "security deposit refundable amount and terms",
        top_k=5,
    )
    deposit_candidates = (
        [r["clause"] for r in deposit_matches] if deposit_matches else all_clauses
    )
    for c in deposit_candidates:
        text = c.get("clause_text", "")
        _, deposit = extract_rent_and_deposit(text)
        if deposit is not None:
            found_deposit = deposit
            break

    if found_deposit is not None:
        print(f"FOUND DEPOSIT: {found_deposit}")

    for c in deposit_candidates:
        text = c.get("clause_text")
        _, deposit = extract_rent_and_deposit(text)

        if deposit is not None:
            rent = global_rent
            result = check_deposit_fairness(rent, deposit)

            print(f"\nClause: {c['clause_number']}")
            print(f"Rent: Rs {rent}")
            print(f"Deposit: Rs {deposit}")
            print(f"Status: {result['status']}")
            print(f"Message: {result['message']}")
            print("-" * 40)

    out_path = "clauses_output.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_clauses, f, indent=2, ensure_ascii=False)

    print(f"\n Saved to {out_path}")


if __name__ == "__main__":
    run_pipeline()
