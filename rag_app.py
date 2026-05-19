import os
import sys
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY is not set.")
    sys.exit(1)

print("✓ GEMINI_API_KEY loaded successfully.")

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Server is running."}
