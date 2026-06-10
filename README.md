RAG Project
This repository contains my Retrieval-Augmented Generation (RAG) project for the GenAI Secure Coding course.
This project will be built incrementally each week.

Week 5 — First Backend API + Gemini Call
What was built
Added a FastAPI backend with two endpoints:

GET /health — Returns a simple status check confirming the server is running.
GET /test-gemini — Calls the Google Gemini API and returns a one-paragraph explanation of what a large language model is.

Where the Gemini call lives
The Gemini API call is inside the test_gemini() function in rag_app.py. It uses the google-generativeai SDK, initializes the gemini-2.5-flash model, and calls generate_content() with a prompt. The response text is returned as JSON.
How to run

Create a .env file in the project root with your Gemini API key: GEMINI_API_KEY=your_key_here
Activate your virtual environment and install dependencies: pip install fastapi uvicorn google-generativeai python-dotenv
Start the server: uvicorn rag_app:app --reload
Visit http://127.0.0.1:8000/test-gemini in your browser.

What I learned

How to build a REST API with FastAPI
How to call the Gemini API using the google-generativeai SDK
How to keep API keys secure using .env and .gitignore so they never get committed to GitHub


Week 6 — Multi-Step Execution

What was built
The /test-gemini endpoint now performs two sequential Gemini API calls instead of one.

Step 1: Ask Gemini to generate a short 3-point outline on how RAG works.
Step 2: Pass that outline back to Gemini as context and ask it to expand it into a full beginner-friendly explanation.

Why the steps are separated
Separating the steps gives each call a single, clear responsibility. The outline step forces the model to plan before writing, which produces more structured and coherent output. This mirrors how real AI systems use multi-step flows to improve reasoning and control the final result.

What the endpoint returns
step_1_outline — the raw outline from the first call
step_2_response — the full explanation generated from that outline

Challenges
The main challenge was understanding that the output of the first call (a plain string) can simply be embedded into the second prompt using an f-string. No special framework needed — just variables.

Week 7 — Validating User Input and AI Output

What was built
Added a new POST /query endpoint that validates input, calls Gemini twice, and returns a reviewed answer.

Why input validation exists
Users can send empty, too-short, too-long, or malicious input. Validating before the model call prevents wasted API calls and protects the system from bad data.

Why output validation exists
AI models can return empty or very short responses. Validating the output before returning it ensures we never send a useless answer back to the user.

Why a second AI model call is used
The first model answers the question. The second model reviews that answer and improves it if needed. This pattern makes AI output more reliable and is commonly used in production GenAI systems.

What the endpoint does
1. Validates the user's question (not empty, between 5–500 characters)
2. Calls Gemini to generate an answer
3. Validates the answer (not empty, at least 10 characters)
4. Calls Gemini again to review and improve the answer
5. Returns the final reviewed answer

Git Commands Used So Far

git clone
git status
git add
git commit
git commit --amendssss
git push
git rm --cached