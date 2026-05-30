# RAG Project

This repository contains my Retrieval-Augmented Generation (RAG) project for the GenAI Secure Coding course.

This project will be built incrementally each week.

---

## Week 5 — First Backend API + Gemini Call

### What was built
Added a FastAPI backend with two endpoints:

- \`GET /health\` — Returns a simple status check confirming the server is running.
- \`GET /test-gemini\` — Calls the Google Gemini API and returns a one-paragraph explanation of what a large language model is.

### Where the Gemini call lives
The Gemini API call is inside the \`test_gemini()\` function in \`rag_app.py\`. It uses the \`google-generativeai\` SDK, initializes the \`gemini-2.5-flash\` model, and calls \`generate_content()\` with a prompt. The response text is returned as JSON.

### How to run
1. Create a \`.env\` file in the project root with your Gemini API key: \`GEMINI_API_KEY=your_key_here\`
2. Activate your virtual environment and install dependencies: \`pip install fastapi uvicorn google-generativeai python-dotenv\`
3. Start the server: \`uvicorn rag_app:app --reload\`
4. Visit \`http://127.0.0.1:8000/test-gemini\` in your browser.

### What I learned
- How to build a REST API with FastAPI
- How to call the Gemini API using the \`google-generativeai\` SDK
- How to keep API keys secure using \`.env\` and \`.gitignore\` so they never get committed to GitHub

---

## Git Commands Used So Far

- \`git clone\`
- \`git status\`
- \`git add\`
- \`git commit\`
- \`git commit --amend\`
- \`git push\`
- \`git rm --cached\`
