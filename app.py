"""
Entry point: an LLM orchestrator uses tool calling to run `main.run_pipeline`.
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
    """Ask the model to invoke the pipeline via function calling, then print token usage."""
    global _active_client

    if not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit(
            "Missing GOOGLE_API_KEY. Set it in your environment before running app.py."
        )

    llm_ops.reset_token_tracker()
    _active_client = genai.Client()

    instruction = (
        "The user wants to process a rental contract PDF and produce clause extraction "
        "plus rent and deposit analysis. "
        f"You MUST call the tool run_contract_pipeline with pdf_path={pdf_path!r}. "
        "Do not answer with plain text only; use the tool."
    )

    response = _active_client.models.generate_content(
        model=MODEL_NAME,
        contents=instruction,
        config={
            "tools": [run_contract_pipeline],
            "automatic_function_calling": {"ignore_call_history": True},
        },
    )

    llm_ops.token_tracker.add(response)
    if response.text:
        print("Orchestrator model message:", response.text)
    llm_ops.print_token_report()


if __name__ == "__main__":
    run_orchestrator()
