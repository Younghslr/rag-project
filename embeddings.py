# embeddings.py
# -------------
# This file handles converting text into embeddings (vectors).
#
# What is an embedding?
# An embedding is a list of numbers (a vector) that represents the
# meaning of a piece of text. Similar texts produce similar vectors,
# which is what makes semantic search possible.
#
# For example:
#   "Python is a programming language" → [0.12, -0.34, 0.87, ...]
#   "Python helps you write code"      → [0.11, -0.35, 0.85, ...]  (very similar!)
#   "I love pizza"                     → [-0.72, 0.54, -0.12, ...]  (very different!)

from openai import OpenAI
from config import OPENAI_API_KEY

_client = OpenAI(api_key=OPENAI_API_KEY)
EMBEDDING_MODEL = "text-embedding-3-small"


def embed_text(text):
    """
    Convert a single piece of text into a vector embedding.

    Args:
        text: A string of text to embed.

    Returns:
        A list of floating point numbers (the embedding vector).
    """
    response = _client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def embed_documents(documents):
    """
    Convert a list of documents into a list of embeddings.

    Args:
        documents: A list of strings to embed.

    Returns:
        A list of embedding vectors (a list of lists of floats).
    """
    response = _client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=documents
    )
    return [item.embedding for item in response.data]
