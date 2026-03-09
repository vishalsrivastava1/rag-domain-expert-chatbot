import os, logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

from src.retriever import HybridRetriever, get_retriever
from src.memory import ConversationMemory

load_dotenv()  # loads your .env file

logger = logging.getLogger(__name__)

# ── Settings ──────────────────────────────────────────────────────
TOP_K       = 5       # number of chunks to retrieve
LLM_MODEL   = "gpt-4o-mini"
TEMPERATURE = 0.2     # lower = more factual answers
MAX_TOKENS  = 1024

SYSTEM_PROMPT = """You are NASABot, an expert on NASA space research.
Answer questions using ONLY the provided context from NASA documents.
Rules:
- Always cite sources using [Source N] notation
- Be precise with numbers, dates, and mission names
- If context is insufficient, say so clearly
- Keep answers 2-4 paragraphs
- Never make up facts"""


def format_context(chunks):
    """Format retrieved chunks into a readable context string."""
    context_parts = []
    sources = []
    for i, chunk in enumerate(chunks, start=1):
        title = chunk.get("title", chunk["source"])
        context_parts.append(
            f"[Source {i}] {title} (Page {chunk['page']})"
            f"\n{chunk['text']}"
        )
        sources.append({
            "num": i,
            "title": title,
            "source": chunk["source"],
            "page": chunk["page"],
            "total_pages": chunk["total_pages"],
            "score": chunk.get("score", 0),
        })
    return "\n\n---\n\n".join(context_parts), sources


class RAGChain:
    def __init__(self):
        self.retriever = get_retriever()
        self.memory = ConversationMemory(max_exchanges=7)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in .env!")
        self.client = OpenAI(api_key=api_key)
        logger.info("RAGChain ready")

    def answer(self, query: str, session_memory=None):
        """
        Main function: takes a question, returns answer + sources.
        Returns: { answer, sources, query, expanded_query, num_chunks }
        """
        mem = session_memory or self.memory

        # Expand follow-up questions with context
        expanded = mem.get_expanded_query(query)

        # Retrieve relevant chunks
        chunks = self.retriever.retrieve(expanded, top_k=TOP_K)

        if not chunks:
            msg = ("I couldn't find relevant information for that question. "
                   "Try rephrasing or ask about a NASA mission or topic.")
            mem.add_user(query)
            mem.add_assistant(msg)
            return {"answer": msg, "sources": [], "query": query,
                    "expanded_query": expanded, "num_chunks": 0}

        # Format context and get source list
        context, sources = format_context(chunks)

        # Build the full prompt including conversation history
        history = mem.get_history_str()
        history_section = f"CONVERSATION HISTORY:\n{history}\n\n" if history else ""

        user_prompt = (
            f"{history_section}"
            f"CONTEXT FROM NASA DOCUMENTS:\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            f"Answer the question using the context. Cite sources as [Source N]."
        )

        # Call GPT-4o-mini
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"Error generating answer: {e}"

        # Save to memory
        mem.add_user(query)
        mem.add_assistant(answer, sources=sources)

        return {
            "answer": answer,
            "sources": sources,
            "query": query,
            "expanded_query": expanded,
            "num_chunks": len(chunks),
        }


# Singleton
_chain = None

def get_chain():
    global _chain
    if _chain is None:
        _chain = RAGChain()
    return _chain
