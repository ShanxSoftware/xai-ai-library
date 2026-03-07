# src/xaihandler/memorystore.py (new, ~120 LOC, zero extra deps)
import sqlite3
import json
import hashlib
import datetime
import uuid
from typing import Dict, List, Optional, Any

class MemoryStore:
    def __init__(self, db_path: str = "assistant.db"):
        self.db_path = db_path
        self._init_tables()
        self._verify_fk_enforcement

    def _init_tables(self):  # update existing method with full usage_logs
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                PRAGMA foreign_keys = ON;   -- Enable enforcement (required)
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY, 
                    title TEXT NOT NULL, 
                    created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    session_id TEXT NOT NULL, 
                    role TEXT NOT NULL, 
                    content TEXT NOT NULL,
                    response_id TEXT, 
                    timestamp TEXT NOT NULL,
                    display INTEGER DEFAULT 1,
                    cached_prompt_tokens INT,
                    prompt_tokens INT,
                    reasoning_tokens INT,
                    completion_tokens INT, 
                    server_side_tools_used INT,
                    total_tokens INT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE);
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
                CREATE TABLE IF NOT EXISTS global_context (
                    key TEXT PRIMARY KEY, 
                    value TEXT, 
                    tags TEXT, 
                    updated_at TEXT);  -- tags = JSON array
                CREATE TABLE IF NOT EXISTS usage_summary (
                    summary_date TEXT PRIMARY KEY,          -- 'YYYY-MM-DD'
                    total_tokens INTEGER DEFAULT 0,
                    prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    reasoning_tokens INTEGER DEFAULT 0,
                    call_count INTEGER DEFAULT 0,
                    last_updated TEXT,                      -- ISO timestamp of last aggregation
                    rollover_from_previous INTEGER DEFAULT 0
                );

            """)

    def _verify_fk_enforcement(self):
        """Call once after init — fails fast if something disabled FKs."""
        with sqlite3.connect(self.db_path) as conn:
            enabled = conn.execute("PRAGMA foreign_keys").fetchone()[0]
            if not enabled:
                raise RuntimeError("Foreign key enforcement is disabled in MemoryStore")

    def get_current_month_total(self) -> int:
        """Returns total tokens used this calendar month from messages table only."""
        now = datetime.datetime.now()
        first_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        first_str = first_of_month.isoformat()

        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute(
                """
                SELECT COALESCE(SUM(total_tokens), 0)
                FROM messages
                WHERE timestamp >= ?
                """,
                (first_str,)
            ).fetchone()[0]
        return total
    
    def today_usage(self) -> int:
        """Used for daily soft cap — still needed separately."""
        today_start = datetime.date.today().isoformat() + "T00:00:00"
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute(
                """
                SELECT COALESCE(SUM(total_tokens), 0)
                FROM messages
                WHERE timestamp >= ?
                """,
                (today_start,)
            ).fetchone()[0]
        return total
    
    # Core methods that directly support both client workflows
    def start_session(self, title: str) -> str:
        """
        Creates a new Session ID for tracking a conversation
        Args:
            title: str the session title for later searching
        Returns: 
            str - unique session id
        """
        session_id = uuid.uuid4().hex
        with sqlite3.connect(self.db_path) as conn: 
            conn.execute("INSERT INTO sessions (session_id, title, created_at) VALUES (?, ?, ?)", (session_id, title, datetime.datetime.now().isoformat()))
        return session_id

    def add_message(self, 
                    session_id: str, 
                    role: str, 
                    content: str,                    
                    display: bool,
                    response_id: Optional[str]="",  
                    cached_prompt_tokens: Optional[int]=0, 
                    prompt_tokens: Optional[int]=0, 
                    reasoning_tokens: Optional[int]=0, 
                    completion_tokens: Optional[int]=0,
                    server_side_tools_used: Optional[int]=0, 
                    total_tokens: Optional[int]=0, 
                    title: Optional[str] = None):
        """
        Adds a message and the message's usage data to the database. 
        Args: 
            session_id: str unique identifier for the session
            role: str message role (SYSTEM, USER, ASSISTANT, TOOL_RESULT)
            content: str the message content
            display: bool lets the UI know if a message should be displayed or hidden
            response_id: str - The xai response_id for future conversations
            cached_tokens: int cached prompt tokens used
            prompt_tokens: int tokens the prompt used to be understood
            reasoning_tokens: int tokens that the model used in reasoning the final response
            server_tokens: int tokens associated with server-side tool calls
            total_tokens: int total_tokens associated with this message
            title: Optional[str] update the session title if approriate. 
        """
        # TODO: Update to include citations and reasoning in the database. 
        with sqlite3.connect(self.db_path) as conn:
            if title:
                conn.execute("UPDATE sessions SET title = ? WHERE session_id = ?", (title, session_id))
                
            conn.execute("""
                INSERT INTO messages 
                    (session_id, 
                    role, 
                    content, 
                    response_id,
                    timestamp, 
                    display, 
                    cached_prompt_tokens,
                    prompt_tokens,
                    reasoning_tokens,
                    completion_tokens,
                    server_side_tools_used,
                    total_tokens)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (session_id, 
                 role, 
                 content, 
                 response_id, 
                 datetime.datetime.now().isoformat(), 
                 1 if display else 0, 
                 cached_prompt_tokens, 
                 prompt_tokens, 
                 reasoning_tokens, 
                 completion_tokens, 
                 server_side_tools_used, 
                 total_tokens))

    def get_context(self, session_id: str, max_tokens: int = 120000) -> List[Dict[str, str]]:
        """Returns ordered messages + any matching global context (simple keyword match for now)."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC LIMIT 50",
                (session_id,)
            ).fetchall()
        messages = [{"role": r, "content": c} for r, c in rows]

        # Global context stub (expand later with xAI structured filter if needed)
        global_rows = conn.execute("SELECT value FROM global_context").fetchall()
        if global_rows:
            messages.insert(0, {"role": "system", "content": "Global context: " + " | ".join([r[0] for r in global_rows])})
        return messages

    def check_and_update_budget(self, estimated_tokens: int = 1000, daily_limit: int = 50000) -> bool:
        """Pre-call guard using today's usage_logs (lightweight, no external tokenizer)."""
        today = datetime.datetime.now().date().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute(
                "SELECT COALESCE(SUM(total_tokens), 0) FROM usage_logs WHERE timestamp >= ?",
                (today + "T00:00:00",)
            ).fetchone()[0]
        if total + estimated_tokens > daily_limit:
            return False
        return True

    def get_session_id_from_response_id(self, response_id: str) -> str:
        """
        Use the xAI generated resposne_id to find the appropriate session_id

        Args: 
            response_id: str response_id generated by xai and stored in the message table. 

        Returns: 
            session_id: UUID string for session management

        Raises:
            ValueError: if the response_id is not found (e.g. retention expiry or data loss).
        """
        
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT session_id FROM messages WHERE response_id = ?", (response_id,)).fetchone()
        if row is None: 
            raise ValueError(f"response_id '{response_id}' not found in any session")
        return row[0]
            
    def upsert_global(self, key: str, value: str, tags: List[str] = None):
        """
        Updates global_context table with new or updated information. 

        Args: 
            key: str string for indexing items
            value: str value for use in future conversations
            tags: List[str] - JSON list of tags. 
        """
        tags_json = json.dumps(tags or [])
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO global_context 
                   (key, value, tags, updated_at) 
                   VALUES (?, ?, ?, ?)""",
                (key, value, tags_json, datetime.datetime.now().isoformat())
            )
            conn.commit()

    def search_global(self, query: str) -> List[Dict]:
        """
        Searches the global_context table. 

        Args: 
            query: str search query

        Returns: 
            List[Dict] List of key, value, tags that came from the search
        """
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT key, value, tags 
                   FROM global_context 
                   WHERE key LIKE ? OR value LIKE ? OR tags LIKE ?""",
                (f"%{query}%", f"%{query}%", f"%{query}%")
            ).fetchall()
        return [{"key": k, "value": v, "tags": json.loads(t)} for k, v, t in rows]
    
    def list_global_keys(self) -> List[str]:
        """
        List the entire global_context table. 

        Returns: 
            List[str]: Each row of the global_context table
        """
        with sqlite3.connect(self.db_path) as conn:
            return [row[0] for row in conn.execute("SELECT key, value, tags FROM global_context")]