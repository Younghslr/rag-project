import os
import sys
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY is not set.")
    print("Make sure your .env file exists and contains: OPENAI_API_KEY=your_key_here")
    sys.exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

class QueryRequest(BaseModel):
    question: str


# --- Helper ---

def call_openai(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# --- Validation functions ---

def validate_user_input(text: str):
    if text is None or text.strip() == "":
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if len(text) < 5:
        raise HTTPException(status_code=400, detail="Question is too short")
    if len(text) > 500:
        raise HTTPException(status_code=400, detail="Question is too long")


def validate_model_output(text: str):
    if text is None or text.strip() == "":
        raise HTTPException(status_code=500, detail="AI returned an empty response")
    if len(text) < 10:
        raise HTTPException(status_code=500, detail="AI response is too short")


def review_model_output(original_answer: str):
    review_prompt = f"""
You are reviewing an AI-generated response.

Your job:
- If the response is unclear, incomplete, or poorly written, improve it.
- If the response is already good, return it unchanged.

AI response to review:
{original_answer}
"""
    return call_openai(review_prompt)


# --- Endpoints ---

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Server is running."}


@app.get("/test-gemini")
async def test_gemini():
    # Step 1: Generate an outline
    outline = call_openai(
        "Create a short 3-point outline explaining how Retrieval-Augmented Generation (RAG) works. "
        "Format it as a numbered list with one sentence per point."
    )

    # Step 2: Expand the outline into a full explanation
    final = call_openai(
        f"Using this outline:\n\n{outline}\n\n"
        "Write a clear, beginner-friendly explanation of RAG based on those points. "
        "Keep it to three short paragraphs."
    )

    return {
        "step_1_outline": outline,
        "step_2_response": final,
    }


@app.post("/query")
def query_ai(request: QueryRequest):
    validate_user_input(request.question)

    raw_answer = call_openai(request.question)

    validate_model_output(raw_answer)

    reviewed_answer = review_model_output(raw_answer)

    return {
        "question": request.question,
        "answer": reviewed_answer
    }
