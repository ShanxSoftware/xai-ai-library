import os
from typing import Optional, Dict
from pathlib import Path
from dataclasses import dataclass

from personality import AgentPersonality
@dataclass
class HandlerConfig:
    api_key: str
    model: str = "grok-4"
    personality: Optional[AgentPersonality] = None
    tools: Optional[Dict[str, tuple]] = None  # (func, schema, desc)
    memory_db: str = "memory_vault.db"
    timeout: int = 3600
    
    @classmethod
    def from_env(cls) -> 'HandlerConfig':
        """Load from environment variables."""
        return cls(
            api_key=os.getenv("XAI_API_KEY"),
            model=os.getenv("XAI_MODEL", "grok-4"),
            timeout=int(os.getenv("XAI_TIMEOUT", "3600"))
        )
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> 'HandlerConfig':
        """Load from YAML config file."""
        import yaml
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        
        return cls(
            api_key=config_data['api_key'],
            model=config_data.get('model', 'grok-4'),
            personality=AgentPersonality(**config_data.get('personality', {})) if config_data.get('personality') else None,
            timeout=config_data.get('timeout', 3600)
        )