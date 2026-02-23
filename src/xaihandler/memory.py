import sqlite3
import json
import numpy as np
from typing import Dict, List, Optional, Any
from sentence_transformers import SentenceTransformer
from datetime import datetime 
from xai_sdk import Client
from xai_sdk.chat import system

class StatefulMemory: 
    def __init__(self, xai_api_key: str, db_path: str = "memory_vault.db", embed_model: str = "all-MiniLM-L6-v2"):
        """
        Constructor - Initilizes class for use. 

        Args: 
            xai_api_key: API key for calls to the xAI API
            db_path: file path to the memory database
            embed_model: SentenceTransformer embed model
        """
        self.db_path = db_path
        self.embedder = SentenceTransformer(embed_model)
        self._init_db()
        self.short_term: List[Dict[str, str]] = [] # [{"role: "user", "content": "..."}, ...]
        self.proc_log: Dict[str, Dict] = {} # Loaded on init
        self.xai_api_key = xai_api_key

    def _init_db(self):
        """Initialize Database"""
        conn = sqlite3.connect(self.db_path)
        # Semantic Memory Table. 
        conn.execute("""
            CREATE TABLE IF NOT EXISTS semantic (
                id INTEGER PRIMARY KEY,
                cue TEXT,
                summary TEXT, 
                embedding BLOB,
                timestamp TEXT,
                relevance REAL
            ) 
        """)
        # Usage Tracking Table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS usage_logs (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,     -- ISO format for easy querying
                method TEXT,        -- e.g., 'chat', 'agent_chat', 'structured_intent' 
                workflow_context TEXT, -- JSON-serialized: e.g., {'intent_type': 'get_weather', 'tool_called': 'get_weather'}
                prompt TEXT,        -- Full prompt/message sent
                response TEXT,      -- Full response content receieved 
                prompt_tokens INTEGER, 
                cached_prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER, 
                reasoning_tokens INTEGER DEFAULT 0, 
                total_tokens INTEGER
            )
        """)
        conn.commit()
        conn.close()

    def add_exchange(self, exchange: List[Dict[str, str]]):
        """
        Append to short-term; summarize if buffer full.
        
        Args: 
            exchange: list of user/ai exchanges
        """
        self.short_term.extend(exchange)
        if len(self.short_term) > 10: # Tune N
            summary = self._generate_summary(self.short_term[-5:]) # Last chunk
            self._store_semantic(cue="recent_session", summary=summary)
            self.short_term = self.short_term[-5:] # Keep tail

    def _generate_summary(self, exchanges: List[Dict]) -> str: 
        """
        Generate summary of exchanges
        
        Args: 
            exchanges: list of user/ai exchanges
        
        Returns: 
            str: summary of a list of user/AI exchanges created using xAI-Grok-3
        """
        # Call xAI to summarize
        history_str = "\n".join([f"{e['role']}: {e['content']}" for e in exchanges])
        prompt = f"Summarize key facts, user prefs, and outcomes from: \n{history_str}\nKeep under 150 words."
        client = Client(self.xai_api_key, timeout=3600)
        
        chat = client.chat.create(
            model="grok-3", 
            messages=[
                system(prompt) # Will the response actually be produced without a user role prompt? 
            ]
        )
        response = chat.sample()
        return response.content
    
    def _store_semantic(self, cue: str, summary: str): 
        """
        Store a summary in semantic memory
        
        Args: 
            cue: cue used to recall the summary
            summary: summary of user/ai exchanges to be recalled with cue
        """
        embedding = self.embedder.encode(cue + " " + summary).tobytes()
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO semantic (cue, summary, embedding, timestamp, relevance) VALUES (?, ?, ?, ?, ?)",
            (cue, summary, embedding, datetime.now().isoformat(), 1.0) # Innitial rel=1, decay later
        )
        conn.commit()
        conn.close()

    def retrieve_relevant(self, current_prompt: str, top_k: int = 3) -> List[str]: 
        """
        Embed prompt, fetch top matches.
        
        Args: 
            current_prompt: prompt used as a base for locating previous interactions
            top_k: The number of summaries to select
        
        Returns: 
            List: List of summaries that are similar to the current prompt
        """
        prompt_emb = self.embedder.encode(current_prompt)
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT cue, summary, embedding FROM semantic").fetchall()
        conn.close()

        similarities = []
        for row in rows:
            emb = np.frombuffer(row[2], dtype=np.float32)
            sim = np.dot(prompt_emb, emb) / (np.linalg.norm(prompt_emb) * np.linalg.norm(emb))
            similarities.append((sim, row[1])) # (score, summary)
        
        top_summaries = [s for _, s in sorted(similarities, reverse=True)[:top_k]]
        return top_summaries

    def update_procedural(self, tool_name: str, cue: str, success: bool, optimized_prompt: str = ""):
        """
        Log tool outcomes for reuse.
        Args: 
            tool_name: name of tool used
            cue: memory cue to select tools
            success: Did the tool solve the problem? 
            optimized_prompt: best prompt to select the tool. 
        """
        if cue not in self.proc_log: 
            self.proc_log[cue] = {"uses": 0, "successes": 0, "best_prompt": ""}
            self.proc_log[cue]["successes"] += 1
            self.proc_log[cue]["best_prompt"] = optimized_prompt
        # Persist to file/DB as needed

    def get_context_for_call(self, current_prompt: str) -> str: 
        """
        Assemble context: short-term + relevant long-term + procedural hints.
        
        Args: 
            current_prompt: prompt used to find relevent contexts. 
            
        Returns: 
            str: a context prompt that can be used passed to an LLM for context spanning sessions.
        """
        short_str = "\n".join([f"{e['role']}: {e['content']}" for e in self.short_term])
        long_str = "\n".join(self.retrieve_relevant(current_prompt=current_prompt))
        proc_hint = self._get_proc_hint(current_prompt) # e.g., "Try rpc_analyze_pdf first"
        return f"Recent: {short_str}\nMemories: {long_str}\nSkills: {proc_hint}"
    
    def _get_proc_hint(self, prompt: str) -> str: 
        """
        Get procedural hint from past tool usage patterns.
        Args: 
            prompt: 
            
        
        Returns: 
            str: 
        """

        if not self.proc_log: # Handle empty case
            return "No strong patterns yet."
        
        # Simple fuzzy match; upgrade to embeddings
        best_match = max(self.proc_log.items(), key=lambda x: 1 if x[0] in prompt else 0)
        if best_match[1]["successes"] / max(1, best_match[1]["uses"]) > 0.7:
            return f"Reuse: {best_match[0]} with prompt '{best_match[1]['bestprompt'][:50]}...'"
        return "No strong patterns yet."
    
    def log_usage(
        self, 
        method: str, 
        prompt: str, 
        response: str, 
        usage: Dict[str, int], # From response.usage: e.g., {'prompt_tokens': 100, 'completion_tokens': 50, ...}
        workflow_context: Optional[Dict[str, Any]] = None # Extra context like intent or tool
    ):
        """
        Log a single API Usage event. 

        Args: 
            method: The handler method that triggered the call (e.g., 'chat').
            prompt: The input prompt/message.
            response: The output resposne content. 
            usage: Dict from response.usage (prompt_tokens, completions_tokens, etc.).
            workflow_context: Optional dict for workflow details (e.e., {'intent': 'chat', 'iteration': 2}). 
        """
        timestamp = datetime.now().isoformat()
        workflow_json = json.dumps(workflow_context or {})
        
        # SAFE USAGE EXTRACTION - THIS FIXES THE ERROR
        def safe_get_usage_value(usage_obj, key, default=0):
            """Extract value from dict, namedtuple, or object."""
            if isinstance(usage_obj, dict):
                return usage_obj.get(key, default)
            elif hasattr(usage_obj, key):
                return getattr(usage_obj, key, default)
            elif hasattr(usage_obj, '_asdict'):  # NamedTuple
                return usage_obj._asdict().get(key, default)
            else:
                self.logger.warning(f"Unknown usage format: {type(usage_obj)}")
                return default
            
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT INTO usage_logs (
                timestamp, method, workflow_context, prompt, response, prompt_tokens, cached_prompt_tokens, 
                completion_tokens, reasoning_tokens, total_tokens
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, 
            (
                timestamp, method, workflow_json, prompt, response,
                getattr(usage, 'prompt_tokens', 0),
                getattr(usage, 'cached_prompt_tokens', 0),
                getattr(usage, 'completion_tokens', 0),
                getattr(usage, 'reasoning_tokens', 0),
                getattr(usage, 'total_tokens', 0)
            )
        )
        conn.commit()
        conn.close()

    def query_usage(
        self,
        start_date: Optional[str] = None,  # ISO format, e.g., '2025-09-01'
        end_date: Optional[str] = None,
        method: Optional[str] = None,
        min_tokens: Optional[int] = None
    ) -> List[Dict]:
        """
        Query usage logs with filters. Returns list of dicts for analysis.

        Args:
            start_date: Filter logs >= this date.
            end_date: Filter logs <= this date.
            method: Filter by method (e.g., 'chat').
            min_tokens: Filter logs where total_tokens >= this.

        Returns:
            List[Dict]: Each dict is a row (e.g., {'timestamp': '...', 'total_tokens': 100, ...}).
        """
        conn = sqlite3.connect(self.db_path)
        query = "SELECT * FROM usage_logs WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        if method:
            query += " AND method = ?"
            params.append(method)
        if min_tokens:
            query += " AND total_tokens >= ?"
            params.append(min_tokens)
        
        rows = conn.execute(query, params).fetchall()
        conn.close()
        
        columns = ['id', 'timestamp', 'method', 'workflow_context', 'prompt', 'response',
                   'prompt_tokens', 'cached_prompt_tokens', 'completion_tokens', 'reasoning_tokens', 'total_tokens']
        return [dict(zip(columns, row)) for row in rows]

    def get_usage_summary(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict:
        """
        Aggregate summary: Total tokens, avg per call, rate (tokens per hour), high-consumption prompts.

        Returns:
            Dict: e.g., {'total_tokens': 5000, 'avg_tokens_per_call': 100, 'tokens_per_hour': 200, ...}
        """
        logs = self.query_usage(start_date, end_date)
        if not logs:
            return {'message': 'No logs found'}
        
        total_calls = len(logs)
        total_tokens = sum(log['total_tokens'] for log in logs)
        avg_tokens = total_tokens / total_calls if total_calls else 0
        
        # Rate: Tokens per hour (assuming uniform distribution; refine as needed)
        timestamps = sorted(datetime.fromisoformat(log['timestamp']) for log in logs)
        time_span_hours = (timestamps[-1] - timestamps[0]).total_seconds() / 3600 if len(timestamps) > 1 else 0
        tokens_per_hour = total_tokens / time_span_hours if time_span_hours else 0
        
        # Top 3 high-consumption prompts (for optimization insights)
        high_consumption = sorted(logs, key=lambda x: x['total_tokens'], reverse=True)[:3]
        high_prompts = [(log['prompt'][:100] + '...', log['total_tokens']) for log in high_consumption]
        
        return {
            'total_calls': total_calls,
            'total_tokens': total_tokens,
            'avg_tokens_per_call': avg_tokens,
            'tokens_per_hour': tokens_per_hour,
            'high_consumption_prompts': high_prompts,
            # Add more aggregates as needed (e.g., by method)
        }
