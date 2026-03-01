"""Session memory for the Customer Insight Agent.

Every good computer needs memory, and an agent is no different. Without
memory, every query starts from zero — no context, no continuity, no
ability to say "like that thing we found earlier." That's not how a good
research assistant works.

This module provides two memory backends with the same interface:

1. LocalMemory: An in-memory Python dict. Fast, zero setup, perfect for
   development and testing. Disappears when the process exits, but that's
   fine for prototyping — you're iterating fast, not building for the ages.

2. FirestoreMemory: Google Cloud Firestore. Persistent, scalable, real-time.
   Each session is a Firestore document, messages are a subcollection.
   This is the production path — sessions survive restarts and can be
   queried by other services.

Both backends support the same operations: create sessions, store messages,
and get/set arbitrary context (like "last searched topic" or "documents
already analyzed"). The agent doesn't know which backend it's talking to,
and that's exactly how it should be.
"""

import json
from datetime import datetime, timezone
from typing import Optional


class LocalMemory:
    """In-memory session store — the development/prototyping backend.

    It's a dict. It works. It's fast. It doesn't need credentials.
    For an MVP, that's everything you want. Think of it as breadboarding
    a circuit before you commit to a PCB layout.
    """

    def __init__(self):
        self.sessions: dict[str, dict] = {}

    def create_session(self, session_id: str) -> dict:
        """Spin up a new session. Clean slate, ready to go."""
        session = {
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "messages": [],
            "context": {},
        }
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[dict]:
        """Look up a session by ID. Returns None if it doesn't exist."""
        return self.sessions.get(session_id)

    def add_message(self, session_id: str, role: str, content: str):
        """Append a message to the session's conversation history.

        Auto-creates the session if it doesn't exist yet — because
        the agent shouldn't have to worry about session lifecycle
        during a hot reasoning loop.
        """
        if session_id not in self.sessions:
            self.create_session(session_id)
        self.sessions[session_id]["messages"].append(
            {"role": role, "content": content, "timestamp": datetime.now(timezone.utc).isoformat()}
        )

    def set_context(self, session_id: str, key: str, value):
        """Store arbitrary context data in the session.

        The agent can stash anything here — search results it wants
        to reference later, extracted entities, intermediate findings.
        It's like scratch paper for the reasoning process.
        """
        if session_id not in self.sessions:
            self.create_session(session_id)
        self.sessions[session_id]["context"][key] = value

    def get_context(self, session_id: str, key: str, default=None):
        """Retrieve a context value. Returns the default if not found."""
        session = self.sessions.get(session_id, {})
        return session.get("context", {}).get(key, default)


class FirestoreMemory:
    """Persistent session store using Google Cloud Firestore.

    This is the real deal — sessions survive process restarts, can be
    accessed from multiple services, and scale without you thinking about it.
    Each session is a Firestore document with messages as a subcollection,
    which means you can query messages efficiently without loading the
    entire session into memory.

    Firestore's real-time capabilities also open the door to fun things
    like live dashboards showing agent activity — but that's a future
    iteration. For now, it's just reliable persistent storage.
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
        """Fetch a session document. Returns None if it doesn't exist."""
        doc = self.db.collection(self.collection).document(session_id).get()
        if doc.exists:
            return doc.to_dict()
        return None

    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to the session's message subcollection.

        Subcollections are great here — you can fetch the last N messages
        without loading the entire session document. Efficient reads
        mean lower latency and lower cost. Two things I always care about.
        """
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
