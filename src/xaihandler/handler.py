import os
from dotenv import load_dotenv
from xai_sdk import Client
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

from .personality import AgentPersonality
from .memory import StatefulMemory
from .handlerconfig import HandlerConfig

load_dotenv()

class xAI_Handler:
    def __init__(self, api_key: Optional[str] = None, validate_connection: bool = False):
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        if not self.api_key:
            raise ValueError("XAI_API_KEY missing in .env")
        self.client = Client(api_key=self.api_key)
        self.config = HandlerConfig(api_key=self.api_key)
        self.personality: Optional[AgentPersonality] = None
        self.memory = StatefulMemory(xai_api_key=self.api_key)
        self.tools: Dict[str, tuple] = {}

    def set_personality(self, personality: AgentPersonality):
        self.personality = personality

    def add_tool(self, name: str, func: callable, param_model: BaseModel, desc: str):
        self.tools[name] = (func, param_model, desc)

    def remove_tool(self, name: str):
        self.tools.pop(name, None)

    def chat(self, message: str, session_id: str = "default", 
             context_mode: str = "auto", max_tool_rounds: int = 5) -> Dict[str, Any]:
        # Auto-detect mode for autonomous agents
        if context_mode == "auto":
            context_mode = "analysis" if any(k in message.lower() for k in ["pdf", "analyze", "research", "crawl"]) else "conversational"

        system = self.personality.get_system_prompt() if self.personality else ""
        
        # Build context per mode
        if context_mode == "conversational":
            history = self.memory.get_history(session_id)[-20:]  # last 20 turns
        elif context_mode == "analysis":
            history = self.memory.get_history(session_id)[-5:]   # minimal + RAG later
        else:  # autonomous
            history = []  # task-only; use RAG or task queue

        messages = [
            {"role": "system", "content": system},
            *history,
            {"role": "user", "content": message}
        ]
        
        print(f"DEBUG context_mode={context_mode} | history_len={len(history)}")  # verify
        # TODO: insert RAG chunks here for PDFs
        # TODO: token count + auto-summarize if >1.4M (use xai_sdk tokenizer)

        tool_defs = [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": model.model_json_schema()
                }
            }
            for name, (_, model, desc) in self.tools.items()
        ] or None

        for _ in range(max_tool_rounds):
            resp = self.client.chat.create(
                model="grok-4",
                messages=messages,
                tools=tool_defs,
                tool_choice="auto"
            )
            msg = resp.choices[0].message
            messages.append(msg.model_dump())

            if not msg.tool_calls:
                content = msg.content
                break

            # Execute tools
            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                args = tool_call.function.arguments
                if name in self.tools:
                    func, model, _ = self.tools[name]
                    params = model.model_validate_json(args) if args else {}
                    result = func(**params.model_dump() if isinstance(params, BaseModel) else params)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)
                    })

        self.memory.add(session_id, "user", message)
        self.memory.add(session_id, "assistant", content)
        return {"content": content}