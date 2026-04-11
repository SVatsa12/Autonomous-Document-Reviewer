"""Load GOOGLE_API_KEY from a .env file next to this package (optional if python-dotenv is installed)."""

from pathlib import Path


def load_env():
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path(__file__).resolve().parent / ".env")


load_env()
