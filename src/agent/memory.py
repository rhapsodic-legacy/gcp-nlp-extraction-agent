"""Session memory backends for the Customer Insight Agent.

Provides two interchangeable backends -- LocalMemory (in-memory dict for
development) and FirestoreMemory (persistent, scalable production store).
Both expose the same interface for sessions, messages, and context.
"""

import json
from datetime import datetime, timezone
from typing import Optional


class LocalMemory:
    """In-memory session store for development and testing.

    Data is not persisted across process restarts.
    """

    def __init__(self):
        self.sessions: dict[str, dict] = {}

    def create_session(self, session_id: str) -> dict:
        """Create a new session with the given ID."""
        session = {
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "messages": [],
            "context": {},
        }
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[dict]:
        """Return the session dict, or None if not found."""
        return self.sessions.get(session_id)

    def add_message(self, session_id: str, role: str, content: str):
        """Append a message to the session's conversation history.

        Auto-creates the session if it does not already exist.
        """
        if session_id not in self.sessions:
            self.create_session(session_id)
        self.sessions[session_id]["messages"].append(
            {"role": role, "content": content, "timestamp": datetime.now(timezone.utc).isoformat()}
        )

    def set_context(self, session_id: str, key: str, value):
        """Store an arbitrary key-value pair in the session context."""
        if session_id not in self.sessions:
            self.create_session(session_id)
        self.sessions[session_id]["context"][key] = value

    def get_context(self, session_id: str, key: str, default=None):
        """Retrieve a context value, returning *default* if not found."""
        session = self.sessions.get(session_id, {})
        return session.get("context", {}).get(key, default)


class FirestoreMemory:
    """Persistent session store backed by Google Cloud Firestore.

    Sessions are stored as Firestore documents with messages in a
    subcollection, allowing efficient partial reads.
    """

    def __init__(self, collection_name: str = "agent_sessions"):
        from google.cloud import firestore

        self.db = firestore.Client()
        self.collection = collection_name

    def create_session(self, session_id: str) -> dict:
        """Create a new session document in Firestore."""
        session = {
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "context": {},
        }
        self.db.collection(self.collection).document(session_id).set(session)
        return session

    def get_session(self, session_id: str) -> Optional[dict]:
        """Fetch a session document, or return None if not found."""
        doc = self.db.collection(self.collection).document(session_id).get()
        if doc.exists:
            return doc.to_dict()
        return None

    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to the session's message subcollection."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.db.collection(self.collection).document(session_id).collection("messages").add(msg)

    def get_messages(self, session_id: str, limit: int = 50) -> list[dict]:
        """Retrieve recent messages from a session, ordered by time."""
        msgs_ref = (
            self.db.collection(self.collection)
            .document(session_id)
            .collection("messages")
            .order_by("timestamp")
            .limit(limit)
        )
        return [doc.to_dict() for doc in msgs_ref.stream()]

    def set_context(self, session_id: str, key: str, value):
        """Update a context field in the session document.

        Uses Firestore's dot-notation update so we only modify the
        specific key, not overwrite the entire context object.
        """
        self.db.collection(self.collection).document(session_id).update({f"context.{key}": value})

    def get_context(self, session_id: str, key: str, default=None):
        """Read a context value from the session."""
        session = self.get_session(session_id)
        if session:
            return session.get("context", {}).get(key, default)
        return default
