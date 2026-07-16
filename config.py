# config.py
# ---------
# Central configuration file for the RAG project.
# Keeping all settings in one place makes it easy to change them
# without hunting through multiple files.

import os
from dotenv import load_dotenv

# Load environment variables from a .env file in the project root.
# This is how we keep secrets (like API keys) out of our code.
load_dotenv()

# --- API Keys ---
# Your OpenAI API key, loaded from the .env file.
# Never put your actual key directly in code — it could get shared by accident.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Model Settings ---
# The name of the AI model we use for generating answers.
OPENAI_MODEL = "gpt-4o-mini"

# --- Vector Store Settings ---
# The name of our ChromaDB collection (like a table in a database).
COLLECTION_NAME = "tech_docs"

# How many documents to retrieve from the vector store per query.
# Retrieving top 3 gives the LLM enough context without overwhelming it.
TOP_K_RESULTS = 3

# The similarity threshold for filtering results.
# ChromaDB returns L2 distances: 0 = identical, 2 = completely different.
# We keep results where the distance is below this number.
# 1.0 is a reasonable default — anything further is probably not relevant.
SIMILARITY_THRESHOLD = 1.0

# --- Generation Settings ---
# Temperature controls how "creative" the AI is when generating answers.
# 0.2 is low — answers will be more factual and consistent.
TEMPERATURE = 0.2

# --- Conversation Settings ---
# How many recent messages to include in the conversation history.
# Keeping only the last 10 turns prevents the context from growing too large.
MAX_HISTORY_TURNS = 10
