from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Message:
    role: str      # "user" or "assistant"
    content: str
    sources: List[dict] = field(default_factory=list)


class ConversationMemory:
    """
    Remembers the last N conversation exchanges.
    Also detects follow-up questions and expands them.
    """

    def __init__(self, max_exchanges: int = 7):
        self.max_exchanges = max_exchanges
        self.messages: List[Message] = []

    def add_user(self, content: str):
        self.messages.append(Message(role="user", content=content))
        self._trim()

    def add_assistant(self, content: str, sources: list = None):
        self.messages.append(Message(
            role="assistant", content=content, sources=sources or []
        ))
        self._trim()

    def _trim(self):
        """Keep only last max_exchanges pairs."""
        limit = self.max_exchanges * 2
        if len(self.messages) > limit:
            self.messages = self.messages[-limit:]

    def get_history_str(self) -> str:
        """Returns a readable string of the conversation so far."""
        lines = []
        for m in self.messages:
            role = "User" if m.role == "user" else "Assistant"
            lines.append(f"{role}: {m.content}")
        return "\n".join(lines)

    def is_followup(self, query: str) -> bool:
        """Detect if the query references previous context."""
        if not self.messages: return False
        signals = [
            "it", "this", "that", "they", "them", "mentioned",
            "above", "what about", "more about", "tell me more",
            "explain more", "how so", "why is that", "the mission",
        ]
        return any(s in query.lower() for s in signals)

    def get_expanded_query(self, query: str) -> str:
        """For follow-ups, add context from last user message."""
        if not self.is_followup(query): return query
        for m in reversed(self.messages):
            if m.role == "user" and m.content != query:
                # extract capitalized words as context hints
                words = [w for w in m.content.split() if len(w) > 4 and w[0].isupper()]
                if words:
                    return f"{" ".join(words[:3])} {query}"
        return query

    def clear(self):
        self.messages = []

    def __len__(self):
        return len(self.messages)
