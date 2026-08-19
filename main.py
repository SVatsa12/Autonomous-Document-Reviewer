"""Contract pipeline: PDF → clauses (LLM + per-rule passes) → rent/deposit analysis."""

import env_load 
import json
import os
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
from enhanced_extraction import extract_rent_enhanced, extract_deposit_enhanced, analyze_deposit_fairness_enhanced
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
        from groq_client import GroqClient

        client = GroqClient(api_key=os.environ.get("GROQ_API_KEY", ""))

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

    # Add contract clauses
    vector_db.add_clauses(client, all_clauses)

    # Load and add regulations
    print("Loading legal regulations...", flush=True)
    import json as json_module
    try:
        regs_path = "regulations.json"
        with open(regs_path, "r", encoding="utf-8") as f:
            regulations = json_module.load(f)
        vector_db.add_regulations(client, regulations)
        print(f"Loaded {len(regulations)} regulations into vector DB", flush=True)
    except FileNotFoundError:
        print("Warning: regulations.json not found. RAG queries about legal standards will use only contract clauses.", flush=True)
    except Exception as e:
        print(f"Warning: Failed to load regulations: {e}", flush=True)

    vector_db.save()
    print(f"Vector DB saved to {VECTOR_DB_PATH}", flush=True)

    # ============================================
    # ENHANCED RENT EXTRACTION
    # ============================================
    print("\n" + "=" * 50)
    print(" RENT ANALYSIS (Enhanced Extraction)")
    print("=" * 50)

    # Use comprehensive extraction from ALL clauses
    global_rent, rent_clause = extract_rent_enhanced(all_clauses)

    if global_rent:
        print(f"\n✓ Found rent: Rs {global_rent:,}")
        if rent_clause:
            print(f"  Source: Clause {rent_clause.get('clause_number')}: {rent_clause.get('clause_text', '')[:100]}...")
    else:
        print("\n✗ No rent amount found in contract")

    # Analyze all rent-related clauses
    rent_found = False
    for clause in all_clauses:
        if is_rent_clause(clause):
            rent_val = extract_rent_with_regex(clause.get("clause_text", ""))
            clause["rent"] = rent_val
            if rent_val and not rent_found:
                result = analyze_rent_clause(clause)
                if result:
                    print(f"\nClause: {result['clause_number']}")
                    print(f"Rent: Rs {result['rent']:,}")
                    print(f"Status: {result['status']}")
                    print(f"Message: {result['message']}")
                    print("-" * 40)
                    rent_found = True

    # ============================================
    # ENHANCED DEPOSIT EXTRACTION
    # ============================================
    print("\n" + "=" * 50)
    print(" DEPOSIT ANALYSIS (Enhanced Extraction)")
    print("=" * 50)

    # Use comprehensive extraction from ALL clauses
    found_deposit, deposit_clause = extract_deposit_enhanced(all_clauses)

    if found_deposit:
        print(f"\n✓ Found security deposit: Rs {found_deposit:,}")
        if deposit_clause:
            print(f"  Source: Clause {deposit_clause.get('clause_number')}: {deposit_clause.get('clause_text', '')[:100]}...")
    else:
        print("\n✗ No security deposit amount found in contract")

    # Analyze deposit fairness if we have both rent and deposit
    if found_deposit and global_rent:
        result = analyze_deposit_fairness_enhanced(global_rent, found_deposit)
        print(f"\n--- Deposit Fairness Analysis ---")
        print(f"Rent: Rs {global_rent:,}")
        print(f"Deposit: Rs {found_deposit:,}")
        print(f"Ratio: {found_deposit / global_rent:.2f}x")
        print(f"Status: {result['status']}")
        print(f"Message: {result['message']}")
        if 'legal_note' in result:
            print(f"Legal Note: {result['legal_note']}")
        print("-" * 40)
    elif found_deposit and not global_rent:
        print("\n⚠ Cannot evaluate fairness: rent amount not found")
    elif global_rent and not found_deposit:
        print("\n⚠ Cannot evaluate fairness: deposit amount not found")

    out_path = "clauses_output.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_clauses, f, indent=2, ensure_ascii=False)

    print(f"\n Saved to {out_path}")


if __name__ == "__main__":
    run_pipeline()
