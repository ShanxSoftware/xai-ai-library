import pytest
import sqlite3
from pathlib import Path
from src.memory import StatefulMemory

@pytest.fixture
def temp_memory(tmp_path):
    """Create temporary memory instance."""
    db_path = tmp_path / "test_memory.db"
    memory = StatefulMemory("fake_key", str(db_path))
    return memory 

def test_memory_initialization(temp_memory):
    """Test memory initializes correctly."""
    assert len(temp_memory.short_term) == 0
    assert isinstance(temp_memory.proc_log, dict)

    # Verify DB tables created 
    conn = sqlite3.connect(temp_memory.db_path)
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    assert ('semantic',) in tables
    assert ('usage_logs',) in tables
    conn.close()

def test_add_exchange(temp_memory): 
    """Test adding conversation exchanges."""
    exchange = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]

    temp_memory.add_exchange(exchange)

    assert len(temp_memory.short_term) == 2
    assert temp_memory.short_term[0]["content"] == "Hello"

def test_usage_logging(temp_memory, tmp_path): 
    """Test usage logging to database."""
    usage_data = {
        "method": "test_chat",
        "prompt": "Test prompt", 
        "response": "Test response",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        "workflow_context": None # Explicitly pass None
    }

    temp_memory.log_usage(**usage_data)

    # Debug: Print actual row structure
    # Verify logged to DB
    conn = sqlite3.connect(temp_memory.db_path)
    row = conn.execute("SELECT timestamp, method, workflow_context, prompt, response, prompt_tokens FROM usage_logs WHERE method=?", (usage_data["method"],)).fetchone()
    print(f"DEBUG: Row columns: {[i for i in range(len(row))]}")
    print(f"DEBUG: Row values: {row}")
    print(f"DEBUG: Schema: {conn.execute('PRAGMA table_info(usage_logs)').fetchall()}")
    assert row is not None
    assert row[3] == "Test prompt" # prompt field
    assert row[5] == 10 # prompt_tokens
    conn.close()