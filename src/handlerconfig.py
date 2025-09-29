import os
import logging
from pydantic import BaseModel
from typing import Optional, Dict, Callable, Any
from pathlib import Path
from dataclasses import dataclass, field
from personality import AgentPersonality

@dataclass
class HandlerConfig:
    api_key: str
    model: str = "grok-4"
    personality: Optional[AgentPersonality] = None
    tools: Dict[str, tuple[Callable, Dict[str, Any], str]] = field(default_factory=dict)
    memory_db: str = "memory_vault.db"
    timeout: int = 3600
    logger = logging.Logger(name="HandlerConfig_Log", level=logging.WARNING)

    def __init__(self, api_key: str, model: str = "grok-4", personality: Optional[AgentPersonality] = None,
        tools: Optional[Dict[str, tuple[Callable, type[BaseModel], str]]] = None, 
        memory_db: str = "memory_vault.db", timeout: int = 3600): 
        """
        Initiates the class

        Args:
            api_key: xAI API Key
            model: xAI Model String
            personality: AgentPersonality 
            tools: Tool definitions 
            memory_db: path to database
            timeout: API Call Timeout
        """
        self.api_key = api_key
        self.model = model
        self.personality = personality
        self.memory_db = memory_db
        self.timeout = timeout
        # Convert tools schemas to JSON
        self.tools = {}
        if tools: 
            for tool_name, (tool_func, param_model, description) in tools.items(): 
                param_schema = (param_model.model_json_schema() if isinstance(param_model, type) and issubclass(param_model, BaseModel)
                                else param_model or {"type": "object", "properties": {}})
                self.tools[tool_name] = (tool_func, param_schema, description)

    @classmethod
    def from_env(cls) -> 'HandlerConfig':
        """Load from environment variables."""
        tools= {} # Assume tools come from env or a seperate config for now
        # Example: Parse tools from XAI_TOOLS env var if defined
        tools_env = os.getenv("XAI_TOOLS")
        if tools_env: 
            import json
            tools_data = json.loads(tools_env)
            tools = {k: (lambda x: x, v.get("schema", {}), v.get("desc", "")) for k, v in tools_data.items()} # Placeholder func
        return cls(
            api_key=os.getenv("XAI_API_KEY"),
            model=os.getenv("XAI_MODEL", "grok-4"),
            personality=None,
            tools=tools, 
            timeout=int(os.getenv("XAI_TIMEOUT", "3600"))
        )
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> 'HandlerConfig':
        """Load from YAML config file."""
        import yaml
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        
        tools = {}
        if config_data.get('tools'): 
            for tool_name, tool_data in config_data['tools'].items():
                param_model = tool_data.get('param_model')
                param_schema = (param_model.model_json_schema() if isinstance(param_model, type) and issubclass(param_model, BaseModel)
                                else param_model or {"type": "object", "properties": {}})
                tools[tool_name] = (lambda x: x, param_schema, tool_data.get('description', ""))
                cls.logger.warning(f"No function defined for tool '{tool_name} in YAML; manual assignment required.") if cls.logger else None 
        return cls(
            api_key=config_data['api_key'],
            model=config_data.get('model', 'grok-4'),
            personality=AgentPersonality(**config_data.get('personality', {})) if config_data.get('personality') else None,
            timeout=config_data.get('timeout', 3600)
        )
    
    def add_tool(self, tool_name: str, tool_func: Callable, param_model: Optional[type[BaseModel]], description: str):
        """
        Add tool to toolbox

        Args: 
            tool_name: Name of the new tool
            tool_func: pointer to tool function
            param_model: Pydantic model for the tool parameters. 
            description: Tool desciption
        """        
        param_schema = param_model.model_json_schema() if param_model else {"type": "object", "properties": {}}
        self.tools[tool_name] = (tool_func, param_schema, description)

    def remove_tool(self, tool_name: str): 
        """
        Remove tool from toolbox

        Args: 
            tool_name: Name of tool to remove
        """
        try: 
            self.tools.pop(tool_name)
            self.logger.info(f"Removed tool: {tool_name}")
        except KeyError: 
            self.logger.warning(f"Tool '{tool_name}' not found.")