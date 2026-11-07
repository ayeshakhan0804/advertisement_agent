"""Run: python check_setup.py  - tests your API key and model in ~5 seconds."""
import os
from dotenv import load_dotenv
load_dotenv()
from agent import MODEL, llm
print("Model:", MODEL, "| key found:", bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")))
try:
    print("Reply:", llm(0).invoke("Say OK").content)
except Exception as e:
    print("FAILED:", type(e).__name__, e)