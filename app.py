"""
Entry point: run contract pipeline once, then summarize results.
Set GOOGLE_API_KEY in .env or the environment before running.
"""

import env_load  

import os

from google import genai

import llm_ops
import main

MODEL_NAME = llm_ops.MODEL_NAME


_active_client = None


def run_contract_pipeline(pdf_path: str = "rent2.pdf") -> str:
    """Run full clause extraction (with per-rule LLM passes) and rent/deposit analysis.

    Call this tool to process the rental agreement PDF when the user wants analysis.
    """
    global _active_client
    if _active_client is None:
        raise RuntimeError("LLM client not initialized for tool execution.")
    main.run_pipeline(pdf_path, client=_active_client)
    return f"Pipeline completed for {pdf_path}."


def run_orchestrator(pdf_path: str = "rent2.pdf"):
    """Run pipeline exactly once, then optionally ask model for a short completion note."""
    global _active_client

    if not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit(
            "Missing GOOGLE_API_KEY. Set it in your environment before running app.py."
        )

    llm_ops.reset_token_tracker()
    _active_client = genai.Client()

    # Avoid repeated automatic tool invocations by executing the pipeline directly once.
    tool_result = run_contract_pipeline(pdf_path=pdf_path)

    response = _active_client.models.generate_content(
        model=MODEL_NAME,
        contents=(
            "Summarize this processing result in one short sentence for the user: "
            f"{tool_result}"
        ),
    )

    llm_ops.token_tracker.add(response)
    if response.text:
        print("Orchestrator model message:", response.text)
    llm_ops.print_token_report()


if __name__ == "__main__":
    run_orchestrator()
