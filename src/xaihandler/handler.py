import os
import json
import datetime
import calendar
from dotenv import load_dotenv
from xai_sdk import Client
from xai_sdk.chat import (
    system,
    user, 
    assistant,
    tool, 
    tool_result
)
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

from .personality import AgentPersonality
from .personality import Archetype
# from .memory import StatefulMemory ## DEPRECIATED
from .memorystore import MemoryStore
from .handlerconfig import HandlerConfig
from .definitions import BudgetExceeded, DailySoftBudgetExceeded, CONTEXT_MODE

load_dotenv()
class xAI_Handler:
    def set_budget(self, total_tokens: int):
        """
        Set the monthly token budget for this agent
        args: 
            total_tokens: int max_number of tokens for the month.
        """
        if total_tokens is not None and total_tokens <= 0:
            raise ValueError("token_budget must be positive")
        self.total_token_budget = total_tokens

    def get_budget(self) -> int: 
        return self.total_token_budget

    def set_personality(self, personality: AgentPersonality):
        """
        Set the personality for this agent

        Args: 
            personality: AgentPersonality defining DISC personality weights for this agent. 
        """
        self.personality = personality

    def add_tool(self, name: str, description: str, parameters: BaseModel, func: callable):
        """
        Add a new tool to the registry

        Args: 
            name: str name of the tool
            description: str description of the tool
            parameters Pydantic JSON schema of the tool. 
        """
        self.tools.append(tool(name=name, description=description, parameters=parameters.model_json_schema()))
        self.tool_map[name] = func

    def _register_memory_tools(self):
        class UpsertContext(BaseModel):
            key: str = Field(..., description="Unique identifier for the context item")
            value: str = Field(..., description="The information to store")
            tags: List[str] = Field(default_factory=list, description="Optional categorization tags")

        class SearchContext(BaseModel):
            query: str = Field(..., description="Keyword or phrase to search for")

        self.add_tool(
            name="store_global_context",
            description="Store or update persistent information useful across all conversations (location, preferences, facts, dimensions, etc.)",
            parameters=UpsertContext,
            func=lambda key, value, tags: self.memory.upsert_global(key, value, tags)
        )

        self.add_tool(
            name="search_global_context",
            description="Search previously stored global context for relevant information",
            parameters=SearchContext,
            func=lambda query: self.memory.search_global(query)
        )

    def remove_tool(self, name: str): #JAKOB: Grok, are we actually going to remove tools during runtime? 
        """
        Removes a tool from the registry

        Args: 
            name: str name of the tool to remove. 
        """
        """
        TODO: implement function or delete function. Leaning towards delete as I can not see a use case 
        for removing tools, if tools are being changed it's likely that the whole program needs to 
        restart anyway which would mean the new tool version will load then.
        """
        ...

    def _precall_check(self, message: str) -> bool:
        """
        Guard that enforces both monthly hard limit and configurable daily soft cap with carry-over.
        Returns True if call is allowed.
        Raises BudgetExceeded if monthly limit would be breached.
        """
        # 1. Get real token count for user message
        try:
            token_seq = self.client.tokenize.tokenize_text(text=message, model=self.model)
            user_tokens = len(token_seq)
        except Exception:
            user_tokens = len(message) // 4 + 50  # fallback

        # Conservative estimate: user + system + tools + reasoning buffer + 20% margin
        estimated_cost = user_tokens + 1200 + int(user_tokens * 0.2)

        monthly_used = self.memory.get_current_month_total()

        if monthly_used + estimated_cost > self.total_token_budget:
            remaining = self.total_token_budget - monthly_used
            raise BudgetExceeded(
                f"Monthly budget would be exceeded. "
                f"Used: {monthly_used:,} / {self.total_token_budget:,}. "
                f"Estimated cost: {estimated_cost:,} > remaining {remaining:,}. "
                f"Resets on the 1st of next month."
            )
        now = datetime.datetime.now()
        # Daily soft cap (now enforced)
        today_used = self.memory.today_usage()
        daily_avg = self.total_token_budget / calendar.monthrange(now.year, now.month)[1]
        previous_days = now.day - 1
        expected_so_far = daily_avg * previous_days
        previous_usage = monthly_used - today_used   
        carry_over = max(0, expected_so_far - previous_usage)

        daily_allowance = daily_avg + carry_over
        daily_soft_cap = daily_allowance * 1.5

        if today_used + estimated_cost > daily_soft_cap:
            raise DailySoftBudgetExceeded(  # define this new exception in definitions.py
                f"Daily soft limit would be exceeded. "
                f"Today: {today_used:,} / ~{daily_soft_cap:,.0f} "
                f"(base allowance {daily_allowance:,.0f}). "
                f"Est. cost {estimated_cost:,} > remaining {daily_soft_cap - today_used:,.0f}."
            )
            #print(msg)  # or log
            # For now: warn only (still allow if monthly headroom exists)
            # TODO: Later: raise SoftBudgetWarning(msg) if you want to enforce hard daily cap

        return True

    def chat(self, message: str, 
             session_id: str = "default",
             previous_response_id: Optional[str] = None,
             context_mode: CONTEXT_MODE = CONTEXT_MODE.CONVERSATIONAL) -> Dict[str, Any]:
        """Thin per-turn primitive. Contains client-tool mini-loop only."""

        chat = None
        if previous_response_id is None:
            # New conversation
            session_id = self.memory.start_session("message")
            system_prompt = self.personality.system_prompt if self.personality else "You are a helpful assistant."
            system_prompt = system_prompt + """\n\n
You have access to persistent global context — information that remains available across all conversations with this user.

Use the store_global_context tool to save or update useful, reusable facts (user preferences, locations, family details, recurring instructions, object properties, private knowledge not on the internet, etc.).  
Use search_global_context to retrieve previously stored items when relevant.

Guidelines:
- Store only concise, high-value information — never full conversation transcripts.
- Prefer stable, predictable keys (e.g. "user_location", "preferred_unit_system").
- When the user says "remember", "note", "update", "don't forget" or similar, consider storing or updating an entry.
- If uncertain whether something is new or an update, you may ask for clarification — but trust context when clear.
            """
            chat = self.client.chat.create(
                model="grok-4-1-fast-reasoning", #self.model,
                tools=self.tools,  # your list of xai_sdk.chat.tool(...) objects
                store_messages=True
            )
            chat.append(system(system_prompt))
            
        else:
            session_id = self.memory.get_session_id_from_response_id(response_id=previous_response_id)
            chat = self.client.chat.create(
                model="grok-4-1-fast-reasoning", #self.model,
                tools=self.tools,  # your list of xai_sdk.chat.tool(...) objects
                store_messages=True,
                previous_response_id=previous_response_id
            )

        chat.append(user(message))
        self.memory.add_message(session_id=session_id, role="user", content=message, display=True)
        try:
            self._precall_check(message=message)
        except BudgetExceeded as e:
            rejection_text = str(e)  # or custom formatting
            if self.personality:
                # apply DISC variation to str(e) if desired
                pass
            self.memory.add_message(
                session_id=session_id,
                role="assistant",
                content=rejection_text,
                display=True,
                response_id="budget-rejected-monthly",
                total_tokens=0
            )
            return {
                "content": rejection_text,
                "response_id": "budget-rejected-monthly",
                "usage": {"total_tokens": 0},
                "citations": None,
                "reasoning": None
            }
        except DailySoftBudgetExceeded as e:
            rejection_text = str(e)
            # optionally softer wording for interactive sessions later
            self.memory.add_message(
                session_id=session_id,
                role="assistant",
                content=rejection_text,
                display=True,
                response_id="budget-rejected-daily",
                total_tokens=0
            )
            return {
                "content": rejection_text,
                "response_id": "budget-rejected-daily",
                "usage": {"total_tokens": 0},
                "citations": None,
                "reasoning": None
            }

        final_content = ""
        final_response = None
        client_tool_count = 0
        while client_tool_count < self.max_client_tool_calls: # Prevents an endless loop of tool calls, at some point we have to produce a response
            response = chat.sample()             
            self.memory.add_message(
                session_id=session_id, 
                role=response.role, 
                content=response.content, 
                display=len(response.tool_calls) == 0,
                response_id=response.id, 
                cached_prompt_tokens=getattr(response.usage, 'cached_prompt_tokens', 0),
                prompt_tokens=getattr(response.usage, 'prompt_tokens', 0),
                reasoning_tokens=getattr(response.usage, 'reasoning_tokens', 0),
                completion_tokens=getattr(response.usage, 'completion_tokens', 0),
                server_side_tools_used=len(getattr(response.usage, 'server_side_tools_used', 0)),  
                total_tokens=getattr(response.usage, 'total_tokens', 0)
            )
            final_response = response
            final_content = response.content
            
            # TODO: Update to include citations and reasoning in the database. 
            if not response.tool_calls:
                break

            # Client-tool mini-loop (SDK resets max_turns automatically)
            
            chat.append(response)  # send tool-call message back
            for tc in response.tool_calls:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                func_name = tc.function.name
                try:
                    if func_name not in self.tool_map: 
                        # TODO: Should some message be passed to the LLM for clean handling of failed tool call? 
                        raise ValueError(f"Tool '{func_name}' not registered")

                    result = self.tool_map[func_name](**args)  # your execute
                    chat.append(tool_result(json.dumps(result) if not isinstance(result, str) else result))
                except Exception as e:
                    chat.append(tool_result(f"Tool error: {e}"))
            client_tool_count = client_tool_count + 1
            if client_tool_count == self.max_client_tool_calls: # Let the model know that no more client tools will be run this round
                chat.append(user("This is the last round of client side tool calls, if further client side calls are desired say 'if you would like me to continue this analysis say yes' otherwise make this your final resposne"))

        return {
            "content": final_content,
            "response_id": final_response.id,
            "usage": final_response.usage,
            "citations": getattr(final_response, "citations", None),
            "reasoning": getattr(final_response, "encrypted_content", None)
        }
    
    def __init__(self, api_key: Optional[str] = None, 
                 model: Optional[str] = None, 
                 timeout: Optional[int] = None, 
                 validate_connection: bool = False,
                 max_client_tool_calls: Optional[int] = 5,
                 token_budget: Optional[int] = 1500000):
        """
        Agent Constructor

        Args: 
            api_key: optional Key for access to the xAI API, if not set look in .env
            model: optional default xAI model, if not set look in .env
            timeout: optional xAI API timeout, if not set look in .env
            validate_connection: default False, ensure connection is valid before sending traffic. 
        """
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        if not self.api_key:
            raise ValueError("XAI_API_KEY missing in .env")
        
        self.model = model or os.getenv("XAI_MODEL")
        if not self.model: 
            raise ValueError("XAI_MODEL missing in .env")
        
        self.timeout = timeout or os.getenv("XAI_TIMEOUT")
        if not self.timeout: 
            raise ValueError("XAI_TIMEOUT missing in .env")
        
        self.client = Client(api_key=self.api_key)
        self.config = HandlerConfig(api_key=self.api_key)
        self.memory = MemoryStore()
        if token_budget is not None and token_budget <= 0:
            raise ValueError("token_budget must be positive")
        self.total_token_budget = token_budget
        
        self.tools = []
        self.tool_map = {}
        self.personality: Optional[AgentPersonality] = None
        self.validate_connection = validate_connection
        self.max_client_tool_calls = max_client_tool_calls
        self._register_memory_tools()