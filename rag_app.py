import os
import sys
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY is not set.")
    print("Make sure your .env file exists and contains: GEMINI_API_KEY=your_key_here")
    sys.exit(1)

print("✓ GEMINI_API_KEY loaded successfully.")

genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Server is running."}

@app.get("/test-gemini")
async def test_gemini():
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(
        "Explain what a large language model is in one paragraph."
    )
    return {"response": response.text}
