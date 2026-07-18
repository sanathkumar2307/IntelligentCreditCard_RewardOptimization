"""Cloud entrypoint for the Streamlit RAG app."""

from pathlib import Path
import sys

# Ensure the Streamlit app directory is importable from project root.
APP_DIR = Path(__file__).resolve().parent / "scripts" / "cc_details_insertion"
sys.path.insert(0, str(APP_DIR))

# Importing this module runs the Streamlit app code.
import chatbot_app  # noqa: F401
