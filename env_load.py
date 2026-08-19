"""Load GROQ_API_KEY from a .env file next to this package."""

from pathlib import Path


def load_env():
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path(__file__).resolve().parent / ".env")


load_env()
