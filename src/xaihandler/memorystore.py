# src/xaihandler/memorystore.py (new, ~120 LOC, zero extra deps)
import sqlite3
import json
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from .definitions import BatchStatus, JobCard, AutonomousOutput, Job, JOB_STATUS
class MemoryStore:
    def __init__(self, db_path: str = "assistant.db"):
        self.db_path = db_path
        self._init_tables()
        self._verify_fk_enforcement

    def _init_tables(self):  # update existing method with full usage_logs
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                PRAGMA foreign_keys = ON;   -- Enable enforcement (required)
                PRAGMA journal_mode=WAL;
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
                CREATE TABLE IF NOT EXISTS job_list (
                    job_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT, -- ISO Format datetime string
                    updated_at TEXT, -- ISO Format datetime string
                    estimated_tokens INT DEFAULT 0, 
                    parent_job_id TEXT, 
                    description TEXT,
                    priority INT,
                    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'in progress', 'blocked', 'completed')), -- Ensure they match the ENUM in definitions.py
                    session_id TEXT,
                    progress REAL DEFAULT 0.0, --(0-1)
                    clarification_needed INT DEFAULT 0, 
                    job_card TEXT, -- JSON Job card for the agent to start work.
                    client_tool_round_count INT DEFAULT 0, 
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
                    FOREIGN KEY (parent_job_id) REFERENCES job_list(job_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_job_list_title ON job_list(title);
                CREATE TABLE IF NOT EXISTS batch_list (
                    batch_id TEXT PRIMARY KEY,
                    session_id TEXT, 
                    batch_send TEXT,
                    incomplete INT DEFAULT 1,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
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
        now = datetime.now()
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
            conn.execute("INSERT INTO sessions (session_id, title, created_at) VALUES (?, ?, ?)", (session_id, title, datetime.now().isoformat(),))
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
                conn.execute("UPDATE sessions SET title = ? WHERE session_id = ?", (title, session_id,))
                
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
                 datetime.now().isoformat(), 
                 1 if display else 0, 
                 cached_prompt_tokens, 
                 prompt_tokens, 
                 reasoning_tokens, 
                 completion_tokens, 
                 server_side_tools_used, 
                 total_tokens,))

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
        today = datetime.now().date().isoformat()
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
                (key, value, tags_json, datetime.now().isoformat(),)
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
        
    # Methods that help with autonomous workflow
    def get_jobs(self) -> List[Job]:
        """
        Retrieves the job information for the next incomplete job that requires work. 
        """
        with sqlite3.connect(self.db_path) as conn: 
            rows = conn.execute("""
                SELECT job_id, title, estimated_tokens, parent_job_id, priority, status, session_id, progress, clarification_needed, job_card
                FROM job_list
                WHERE progress < 3 AND clarification_needed = 0
                ORDER BY priority ASC, created_at ASC
            """).fetchall()
        jobs = []
        for row in rows:
            jobs.append(Job(
                job_id=row[0],
                 title= row[1],
                estimated_tokens=row[2],
                parent_job_id=row[3],
                priority=row[4],
                status=row[5],
                session_id=row[6],
                progress=row[7],
                clarification_needed=row[8],
                job_card=JobCard.model_validate_json(row[9])
            ))
        
        return jobs
        
    def increment_job_tool(self, job_id: str): 
        with sqlite3.connect(self.db_path) as conn: 
            conn.execute("UPDATE job_list SET client_tool_round_count = client_tool_round_count + 1 WHERE job_id = ?", (job_id,))
        
    def reset_job_tool(self, job_id: str): 
        with sqlite3.connect(self.db_path) as conn: 
            conn.execute("UPDATE job_list SET client_tool_round_count = 0 WHERE job_id = ?", (job_id,))

    def get_job_tool(self, job_id: str) -> int: 
        with sqlite3.connect(self.db_path) as conn: 
            tool_count = conn.execute("SELECT client_tool_round_count FROM job_list WHERE job_id = ?", (job_id,)).fetchone()[0]
            return tool_count
        
    def get_response_id(self, session_id: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn: 
            row = conn.execute("SELECT response_id FROM messages WHERE session_id = ? ORDER BY timestamp DESC", (session_id,)).fetchone()
            if row and row[0]: 
                return row[0]
            
    def get_session_id_from_job_id(self, job_id: str) -> str:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT session_id FROM job_list WHERE job_id = ?", (job_id,)).fetchone()
            return row[0] if row else None
        
    def update_job(self, job_id: str, job_output: AutonomousOutput):
        """
        Updates a job with the progress made. 
        """
        with sqlite3.connect(self.db_path) as conn: 
            conn.execute("""
                UPDATE job_list
                SET updated_at = ?, status = ?, progress = ?, clarification_needed = ?, job_card = ?
                WHERE job_id = ?""",
                (datetime.now().isoformat(),
                job_output.status, 
                job_output.progress,
                1 if job_output.clarification_needed else 0,
                job_output.job_card.model_dump_json(),
                job_id,)
            )
    def add_job(self, title: str, job_card: JobCard):
        job_id = uuid.uuid4().hex
        with sqlite3.connect(self.db_path) as conn: 
            conn.execute("""
            INSERT INTO job_list (
                job_id, title, created_at, updated_at, status, progress, clarification_needed, job_card
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?
                )""",
                (job_id, title, datetime.now().isoformat(), datetime.now().isoformat(), "pending", 0.0, 0, job_card.model_dump_json(),)
            )

    def get_batch_send(self) -> datetime: 
        """Gets the last batch_send time or returns now-30s"""
        with sqlite3.connect(self.db_path) as conn: 
            batch_send = conn.execute("SELECT MAX(batch_send) FROM batch_list").fetchone() # Does MAX() work on strings? 
            if batch_send and batch_send[0]:
                return datetime.fromisoformat(batch_send[0])
        
        return datetime.now() - timedelta(seconds=30)
    
    def get_batch(self) -> BatchStatus:
        """
        Retrieves the oldest incomplete batch and returns the BatchStatus object. 

        Returns: 
            BatchStatus: A batchstatus object for maintaining the state of a batch of API calls. 
        """
        with sqlite3.connect(self.db_path) as conn:
            batch = conn.execute("""
                SELECT *
                FROM batch_list
                WHERE incomplete = 1
                ORDER BY batch_send ASC
            """).fetchone()
            if batch is None: 
                return None
            else: 
                return BatchStatus(batch_id=batch["batch_id"],
                    session_id=batch["session_id"],
                    batch_send=datetime.fromisoformat(batch["batch_send"]),
                    incomplete=True if batch["incomplete"] == 1 else False
                )
    
    def upsert_batch(self, batch_status: BatchStatus):
        """
        Update or insert batch status for state monitoring

        Args: 
            batch_status: BatchStatus contains the current state of the batch. 
        """
        with sqlite3.connect(self.db_path) as conn: 
            conn.execute("""
                INSERT OR REPLACE INTO batch_list
                (batch_id, session_id, batch_send, incomplete) VALUES (?, ?, ?, ?)""",
                (batch_status.batch_id, 
                batch_status.session_id, 
                batch_status.batch_send.isoformat(), 
                1 if batch_status.incomplete else 0,)
            )