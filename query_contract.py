"""Command-line interface for querying contract analysis using RAG."""

import sys
import json
import argparse
from pathlib import Path

try:
    from groq_client import GroqClient
except ImportError:
    print("Error: groq not installed. Run: pip install groq")
    sys.exit(1)

from vector_db import ClauseVectorDB
from rag_engine import RAGQueryEngine


def load_api_key():
    """Load GROQ_API_KEY from .env file or environment."""
    import os
    from dotenv import load_dotenv

    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found. Set it in .env file or environment.")
        sys.exit(1)
    return api_key


def run_query(args):
    """Run a single query or interactive mode."""
    api_key = load_api_key()
    has_llm = True
    try:
        client = GroqClient(api_key=api_key)
    except Exception as e:
        print(f"Warning: Could not initialize LLM client: {e}")
        print("Will run in OFFLINE mode (template-based answers only).")
        client = None
        has_llm = False

    # Load vector DB
    db_path = args.db_path or "clause_vectors.json"
    print(f"Loading vector database from {db_path}...")
    vector_db = ClauseVectorDB(persist_path=db_path)
    vector_db.load()
    print(f"Loaded {len(vector_db.documents)} documents")

    # Create RAG engine
    rag = RAGQueryEngine(vector_db)

    if args.query:
        # Single query mode
        try:
            result = rag.query(
                client if has_llm else None,
                args.query,
                top_k_contract=args.contract,
                top_k_regulations=args.regulations,
                answer_style=args.style,
                include_sources=not args.no_sources,
            )
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                print("\n[INFO] API quota exceeded. Switching to OFFLINE mode...")
                print("(Run 'python offline_query.py' for interactive offline queries)\n")
                # Fallback to offline
                from offline_query import offline_query
                answer = offline_query(vector_db, args.query)
                print("\n" + "=" * 60)
                print("QUERY:", args.query)
                print("=" * 60)
                print("\nOFFLINE ANSWER:\n")
                print(answer)
                print("\n" + "-" * 60)
                print("Sources: Using vector search only (no LLM)")
                return
            else:
                raise

        print("\n" + "=" * 60)
        print("QUERY:", result["query"])
        print("=" * 60)
        print("\nANSWER:\n")
        print(result["answer"])
        print("\n" + "-" * 60)
        print("SOURCES:")
        print(f"  Contract clauses: {result['stats']['contract_clauses_found']}")
        print(f"  Regulations: {result['stats']['regulations_found']}")

        if args.json_output:
            import json
            print(json.dumps(result, indent=2))

    else:
        # Interactive mode
        print("\n" + "=" * 60)
        print("Interactive RAG Query Mode")
        print("=" * 60)
        print("Commands:")
        print("  /quit, /exit - Exit the program")
        print("  /help       - Show this help")
        print("  /style X    - Change answer style (comprehensive, concise, legal_analysis)")
        print("  /contract N - Set number of contract clauses (default 5)")
        print("  /regulations N - Set number of regulations (default 3)")
        print("  /json on/off - Toggle JSON output")
        print("  /offline     - Toggle offline mode (no API calls)")
        print("=" * 60 + "\n")

        style = args.style
        top_k_contract = args.contract
        top_k_regulations = args.regulations
        json_output = args.json_output
        offline_mode = False

        while True:
            try:
                query = input("\nQuery> ").strip()

                if not query:
                    continue

                # Handle commands
                if query.startswith("/"):
                    cmd_parts = query.split()
                    cmd = cmd_parts[0].lower()

                    if cmd in ["/quit", "/exit"]:
                        print("Exiting.")
                        break
                    elif cmd == "/help":
                        print("Commands:")
                        print("  /quit, /exit - Exit")
                        print("  /help - Show help")
                        print("  /style X - Change answer style")
                        print("  /contract N - Set contract clauses count")
                        print("  /regulations N - Set regulations count")
                        print("  /json on/off - Toggle JSON output")
                        print("  /offline - Toggle offline mode")
                        continue
                    elif cmd == "/offline":
                        offline_mode = not offline_mode
                        print(f"Offline mode: {'ON' if offline_mode else 'OFF'}")
                        if offline_mode:
                            print("(Using template-based answers, no LLM)")
                        else:
                            print("(Using Groq LLM for answers)")
                        continue
                    elif cmd == "/style" and len(cmd_parts) >= 2:
                        style = cmd_parts[1]
                        print(f"Answer style set to: {style}")
                        continue
                    elif cmd == "/contract" and len(cmd_parts) >= 2:
                        top_k_contract = int(cmd_parts[1])
                        print(f"Contract clauses set to: {top_k_contract}")
                        continue
                    elif cmd == "/regulations" and len(cmd_parts) >= 2:
                        top_k_regulations = int(cmd_parts[1])
                        print(f"Regulations set to: {top_k_regulations}")
                        continue
                    elif cmd == "/json" and len(cmd_parts) >= 2:
                        json_output = cmd_parts[1].lower() == "on"
                        print(f"JSON output: {json_output}")
                        continue
                    else:
                        print("Unknown command. Type /help for list.")
                        continue

                # Execute query
                if offline_mode:
                    from offline_query import offline_query
                    print("\n" + "-" * 70)
                    answer = offline_query(vector_db, query)
                    print(answer)
                else:
                    try:
                        result = rag.query(
                            client if has_llm else None,
                            query,
                            top_k_contract=top_k_contract,
                            top_k_regulations=top_k_regulations,
                            answer_style=style,
                            include_sources=not json_output,
                        )
                        print("\n" + "-" * 70)
                        print("ANSWER:\n")
                        print(result["answer"])
                        if not json_output:
                            print(f"\nStats: {result['stats']['contract_clauses_found']} clauses, {result['stats']['regulations_found']} regulations")
                    except Exception as e:
                        if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                            print("\n[WARNING] API quota exceeded. Use /offline to switch to offline mode.")
                        else:
                            print(f"Error: {e}")

            except KeyboardInterrupt:
                print("\n\nExiting.")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Query rental contract with semantic search and legal regulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ask a single question
  python query_contract.py -q "What is the security deposit amount?"

  # Interactive mode
  python query_contract.py

  # Change retrieval settings
  python query_contract.py -q "Can the landlord increase rent?" --contract 8 --regulations 5

  # Get JSON output
  python query_contract.py -q "What are my maintenance responsibilities?" --json
        """,
    )

    parser.add_argument(
        "-q", "--query",
        help="Single query to execute (without this, enters interactive mode)",
    )
    parser.add_argument(
        "--db-path",
        help="Path to vector database (default: clause_vectors.json)",
    )
    parser.add_argument(
        "--contract", type=int, default=5,
        help="Number of contract clauses to retrieve (default: 5)",
    )
    parser.add_argument(
        "--regulations", type=int, default=3,
        help="Number of regulations to retrieve (default: 3)",
    )
    parser.add_argument(
        "--style", default="comprehensive",
        choices=["comprehensive", "concise", "legal_analysis"],
        help="Answer style (default: comprehensive)",
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Don't include source citations in output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output raw JSON with sources",
    )

    args = parser.parse_args()
    run_query(args)


if __name__ == "__main__":
    main()
