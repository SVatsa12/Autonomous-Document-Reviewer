# Contract clause pipeline (LLM-operable)

This repository is structured so an **LLM agent** can load these instructions, choose actions, and drive execution without guessing file roles.

## Machine-readable project map

| Path | Role |
|------|------|
| `app.py` | **Orchestrator entry.** Calls Gemini with **function calling**; the model must invoke `run_contract_pipeline`, which runs `main.run_pipeline`. |
| `main.py` | **Pipeline implementation.** PDF → text → chunks → clause extraction (via `llm_ops`) → rent/deposit analysis → `clauses_output.json`. Runnable with `python main.py` (skips orchestrator). |
| `llm_ops.py` | **Gemini clause extraction.** Set **`EXTRACTION_API_MODE`**: `"single"` (default) = **one API call per chunk** with all rules in one prompt; `"per_rule"` = initial call + **one call per** `CLAUSE_EXTRACTION_RULES` entry. Tracks **token usage** in `token_tracker`. |
| `functions.py` | **No LLM.** PDF/text utilities, regex helpers, `check_rent_limit`, `check_deposit_fairness`, and related pure functions. |
| `env_load.py` | Loads **`.env`** from the project folder (via `python-dotenv`) so `GOOGLE_API_KEY` is set before `genai.Client()`. |
| `clauses_output.json` | **Output artifact** produced by the pipeline. |

## Environment

- **`.env`** (recommended): Create or edit `.env` in the project root with `GOOGLE_API_KEY=your_key`. See `.env.example`. The file is gitignored.
- **`GOOGLE_API_KEY`**: Still works if set in the shell instead of `.env`.
- Install deps: `pip install -r requirements.txt` (includes `python-dotenv`).

Do not commit real API keys into tracked files.

## How an LLM should run the system

1. **Preferred (orchestrated):** Execute `python app.py` from the project root.  
   - The orchestrator sends one user-equivalent instruction to Gemini with tools.  
   - Gemini calls `run_contract_pipeline` → `main.run_pipeline` runs.  
   - Token counts include the orchestrator turn **and** every extraction/rule pass inside `llm_ops`.

2. **Direct pipeline (no orchestrator tool call):** Execute `python main.py`.  
   - Same processing except the initial “please call the tool” Gemini turn is skipped.  
   - Token counts include only extraction/rule passes.

## Clause extraction API modes (`llm_ops.EXTRACTION_API_MODE`)

- **`single` (default):** One Gemini request per chunk; the prompt lists every rule in `CLAUSE_EXTRACTION_RULES` and asks for one JSON array. **Fewest API calls.**
- **`per_rule`:** One call for an initial extraction, then **one call per rule** to refine the JSON (matches “each rule = one LLM call” assignments). **More API calls, slower.**

If the API is unavailable or quota is hit, the code sets `API_DISABLED` and falls back to **local regex extraction** (no tokens for those steps).

## Token reporting

After `main.run_pipeline` finishes, the process prints a **cumulative** summary:

- `API calls counted` — responses that included `usage_metadata`
- `Prompt tokens (sum)`
- `Candidates tokens (sum)`
- `Total tokens (sum of per-response totals)`

When using `app.py`, the orchestrator response is included in the same tracker before the report.

## Contract assumptions for agents

- Default PDF path is `rent2.pdf` in the working directory unless the tool argument overrides it.
- Rent limit for policy check is **80000** (see `functions.check_rent_limit`).
- Chunking uses **`CHUNK_MAX_CHARS = 8000`** in `main.py` (tune there). Sleep between chunks is **`CHUNK_SLEEP_SEC = 2`** to reduce rate-limit risk; increase if you hit 429 errors.

## Dependencies

- `google-genai` (Gemini client)
- `pypdf`

Install as needed for the active Python environment.
