import os
import base64
import json
import numpy as np
import logging 
import fitz # PyMuPDF for PDF text extraction
import uuid
from personality import Archetype, Trait, AgentTrait, AgentPersonality
from memory import StatefulMemory
from handlerconfig import HandlerConfig
from typing import Dict, Callable, Tuple, Any, List, Optional, Literal, Union, Type
from pydantic import BaseModel, Field, model_validator
from tenacity import retry, stop_after_attempt, wait_exponential
from pathlib import Path
from dataclasses import dataclass 
from xai_sdk import Client
from xai_sdk.chat import BaseChat, Response, user, system, assistant, image, tool, tool_result
from xai_sdk.search import SearchParameters

class UserIntent(BaseModel): 
    """Schema for intent detection + params."""
    type: Literal['get_weather', 'query_zotero', 'book_flight', 'chat'] = Field(..., description="User's intent")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the intent")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model's confidence in this intent") 

class xAI_Handler: 
    """
        TODO: 
            - Unit Tests
    """

    def __init__ (
            self, 
            config: Optional[HandlerConfig] = None,
            api_key: Optional[str] = None, 
            model: Optional[str] = None, 
            timeout: Optional[int] = None, 
            logger: Optional[logging.Logger] = None, 
            validate_connection: bool = True, 
            tools: Optional[Dict[str, Tuple[Callable, type[BaseModel], str]]] = None, # New: {name: (proc_func, ParamModel, desc)}
            personality: Optional[AgentPersonality] = None, 
            **kwargs
        ):
        if config: 
            self.config = config
        else:
            self.config = HandlerConfig(
                api_key=api_key or os.getenv("XAI_API_KEY"),
                model=model or os.getenv("XAI_MODEL") or "grok-4",
                personality=personality or AgentPersonality(
                    name="Alex",
                    gender="female",
                    primary_archetype=Archetype.AMIABLE,
                    primary_weight=0.6,
                    secondary_archetype=Archetype.EXPRESSIVE,
                    job_description="Personal assistant",
                    traits={
                        "empathy": AgentTrait(trait=Trait.EMPATHY, intensity=60), 
                        "curiosity": AgentTrait(trait=Trait.CURIOSITY, intensity=50),
                        "precision": AgentTrait(trait=Trait.PRECISION, intensity=90)
                    }
                ),
                tools=tools or {},
                memory_db="memory_vault.db",
                timeout=timeout or int(os.getenv("XAI_TIMEOUT", "3600"))
            )
            
        self.tools: Dict[str, Tuple[Callable, Optional[type[BaseModel]], str]] = self.config.tools or {}
        self.personality = self.config.personality or AgentPersonality(
                name="Alex",
                gender="female",
                primary_archetype=Archetype.AMIABLE,
                primary_weight=0.6,
                secondary_archetype=Archetype.EXPRESSIVE,
                job_description="Personal assistant",
                traits={
                    "empathy": AgentTrait(trait=Trait.EMPATHY, intensity=60), 
                    "curiosity": AgentTrait(trait=Trait.CURIOSITY, intensity=50),
                    "precision": AgentTrait(trait=Trait.PRECISION, intensity=90)
                }
            )
        self.api_key = self.config.api_key
        self.model = self.config.model
        self.timeout = self.config.timeout
        self.logger = logger or logging.getLogger(__name__)
        self.memory = StatefulMemory(xai_api_key=self.api_key)
        self.tool_definitions: List[Any] = []  # SDK tool objects
        self.tools_map: Dict[str, Callable] = {}  # Name -> func for execution
        self._rebuild_tool_definitions() # xAI-style list of tool objects
        
        self.intent_schema = UserIntent.model_json_schema()
        self._active_chats: Dict[str, BaseChat] = {}
        self._chat_states: Dict[str, List[Dict]] = {}
        self.parallel_calling = True # Grok told me to add this, not sure why yet. 
        self._default_session_id = str(uuid.uuid4())
        self.client = Client(
            api_key=self.api_key,
            timeout=self.timeout
        )
        
        if validate_connection: 
            if self._validate_connection():
                logger.info(
                    "xAI API handler initialized.", extra={
                        "component": "xai-handler", 
                        "personality": personality.name,
                        "status": "ready", 
                        "message": f"{personality.name} reporting for duty."
                    }
                )   
    def _get_session_id(self, session_id: Optional[str] = None) -> str: 
        """"Smart session ID resolution."""
        if session_id is None: 
            # Return default or create new based on strategy
            return getattr(self, '_default_session_id', 'default')
        return session_id 
    
    def _get_or_create_chat(self, session_id: str = "default", tool_choice: str = "auto") -> BaseChat:
        """
        Get existing chat or create new one from the stored state.
        """
        session_id = self._get_session_id(session_id)
        
        if session_id not in self._active_chats:
            if session_id in self._chat_states and self._chat_states[session_id]:
                # Reload from stored messages (skip system prompt duplication)
                messages = self._chat_states[session_id].copy()
                # Ensure system prompt is always first (in case it was missing)
                if not messages or messages[0].get("role") != "system":
                    messages.insert(0, system(self.personality.to_system_prompt()))
                else:
                    # Update existing system prompt
                    messages[0] = system(self.personality.to_system_prompt())
            else:
                # Fresh chat
                messages = [system(self.personality.to_system_prompt())]
            
            tools_config = {}
            if self.tool_definitions: # Check if tools are configurd
                tools_config = {
                    "tools": self.tool_definitions,
                    "tool_choice": tool_choice
                }

            chat = self.client.chat.create(
                model=self.model,
                messages=messages,
                **tools_config
            )
            
            self._active_chats[session_id] = chat
        
        return self._active_chats[session_id]

    def _sync_chat(self, session_id: str):
        """
        Save chat state and clear active object to free memory.
        """
        session_id = self._get_session_id(session_id)
        if session_id in self._active_chats:
            chat = self._active_chats[session_id]
            # Strip system prompt from stored state to avoid duplication on reload
            stored_messages = [msg for msg in chat.messages if msg.get("role") != "system"]
            self._chat_states[session_id] = stored_messages
            del self._active_chats[session_id]
        """
        Suggestions from Grok for improvement. 
        - In _sync_chat, use msg.model_dump() if messages are Pydantic (SDK might use them).
        - Add self._chat_states persistence (e.g., JSON dump on shutdown) for cross-run sessions.
        """

    def _validate_connection(self): 
        """Verify API connectivity and credentials."""
        try: 
            # Minimal connectivity test
            chat = self.client.chat.create(
                model=self.model,
                messages=[
                    system(self.personality.to_system_prompt()),
                    user("Please confirm that you are working? Say 'Yes, I'm operational!' if you can receive this.")
                ],
                max_tokens=50, # Keep it short for validation
                temperature=0.1, # Deterministic response
            )
            response = chat.sample() 

            # xAI SDK Response validation(no HTTP status codes)
            if response and hasattr(response, 'content') and response.content: 
                if "operational" in response.content.lower() or "yes" in response.content.lower(): 
                    self.logger.debug(
                        "API connection validated successfully", 
                        extra={
                            "xAI Response": response.content
                        }
                    )
                    return True
                else: 
                    # Got a response but not the expected one
                    # Grok said this should be logged before being raised. However, upon further investigation, it's caught in the except statement, and logged just below. 
                    raise ValueError(f"Unexpected response: {response.content}")
            else: 
                
                raise ValueError("Empty or invalid resposne from xAI API")
            
        except Exception as e: 
            self.logger.error(
                f"API connection validation failed: {str(e)}", 
                extra={
                    "error_type": type(e).__name__, 
                    "model": self.model
                }
            )
            raise ConnectionError(f"xAI API connection failed: {str(e)}") from e

    def _rebuild_tool_definitions(self):
        """
        Rebuild tool_definitions from current self.tools.
        Call this after add/remove/clear.
        """
        self.tool_definitions = []
        self.tools_map = {}  # Reset map
        
        for tool_name, (tool_func, param_model, description) in self.tools.items():
            # Generate schema from Pydantic model (docs best practice)
            if param_model and issubclass(param_model, BaseModel):
                parameters = {
                    "type": "object",
                    "properties": param_model.model_json_schema().get("properties", {}),
                    "required": list(param_model.model_json_schema().get("required", [])), 
                    "additionalProperties": False # Grok suggested adding this for strict schemas to prevent hallucinations)
                }
            else:
                # Fallback: Raw dict schema
                parameters = param_model or {"type": "object", "properties": {}}
            
            # Create SDK tool object
            sdk_tool = tool(
                name=tool_name,
                description=description,
                parameters=parameters
            )
            self.tool_definitions.append(sdk_tool)
            
            # Map function for execution
            self.tools_map[tool_name] = tool_func
        
        self.logger.debug(f"Rebuilt {len(self.tool_definitions)} tools: {list(self.tools_map.keys())}")

    def add_tool(self, tool_name: str, tool_func: Callable, param_model: Optional[type[BaseModel]], description: str):
        """
        Dynamically add a tool. Rebuilds definitions automatically.
        
        Args:
            tool_name: Unique name for the tool.
            tool_func: The callable to execute (e.g., def get_weather(city: str) -> str: ...).
            param_model: Pydantic model for args (or None for no params).
            description: Human-readable description for the model.
        """
        if tool_name in self.tools:
            self.logger.warning(f"Tool '{tool_name}' already exists; overwriting.")
        
        self.tools[tool_name] = (tool_func, param_model, description)
        self._rebuild_tool_definitions()
        self.logger.info(f"Added tool: {tool_name}")

    def remove_tool(self, tool_name: str):
        """Remove a tool by name. Rebuilds definitions."""
        if tool_name in self.tools:
            del self.tools[tool_name]
            self._rebuild_tool_definitions()
            self.logger.info(f"Removed tool: {tool_name}")
        else:
            self.logger.warning(f"Tool '{tool_name}' not found.")

    def clear_tools(self):
        """Clear all tools. Rebuilds definitions."""
        self.tools = {}
        self._rebuild_tool_definitions()
        self.logger.info("Cleared all tools.")

    def check_tools(self) -> Dict[str, Any]:
        """Return current tools status for debugging."""
        return {
            "tools_count": len(self.tools),
            "tool_names": list(self.tools.keys()),
            "definitions_count": len(self.tool_definitions),
            "parallel_calling": True  # Default per docs
        }
    
    def get_tool(tool_name: str) -> Dict: # Grok suggested implementing this method. 
        """
        Get's tool information for inspection. 

        Args: 
            tool_name: Name of tool to inspect. 

        Returns: 
            Dict containing the data for the tool. 
        """
        pass # TODO: Implement this method. 

    def _detect_tool_need(self, message: str) -> bool:
        """Heuristic: Keywords or intent check."""
        tool_keywords = ["check", "fetch", "search", "calculate", "calendar", "weather"]  # GROK: This needs to either pull from the tool list or be configurable. Different agents will have different tool boxes. 
        if any(kw in message.lower() for kw in tool_keywords):
            return True
        # Or use structured_intent (faster than full chat)
        try:
            intent = self.structured_intent(message, quick=True)  # GROK: structured_intent isn't implemented yet. But I've had good success with the previous API version selecting the correct intent based on instructions in the system prompt. We'll have to work on this later.
            return intent.type != "chat"
        except:
            return False

    def _get_chat_history(self, session_id: str, summarize: bool = True) -> List[Dict]:
        """Get/summarize history for forking."""
        chat = self._get_or_create_chat(session_id)
        history = chat.messages  # Or serialized
        if summarize and len(history) > 5:  # Threshold
            summary_prompt = "Summarize this conversation history concisely:"
            summary = self.chat(summary_prompt + json.dumps(history), session_id="summary_temp")['content']
            return [system(summary)] # GROK: I don't think this is correct. Is this supposed to be [assistant(summary)]? Or is the second system prompt giving the LLM context without impacting current conversation? 
        return history  # Raw for short

    def clear_session(self, session_id: str):
        # GROK: This was never implemented.         
        pass
    
    def chat_with_tools(self, message: str, session_id: str = "default", **kwargs) -> Dict: # GROK: Is this better than the new chat method below? 
        """
        User-facing: Seamless chat that auto-detects and handles tools.
        """
        # 1. Quick intent check - does this need tools?
        needs_tools = self._detect_tool_need(message)
        
        if needs_tools:
            # 2. Fork sub-session for clean tool execution
            result = self._execute_tool_task(message, session_id, **kwargs)
            
            # 3. Integrate result back to main session
            main_result = self._integrate_tool_result(result, session_id)
            return main_result
        else:
            # 4. Normal chat flow
            return self.chat(message, session_id=session_id, handle_tools=False, **kwargs)
    
    def _execute_tool_task(self, message: str, main_session_id: str, **kwargs) -> Dict: # GROK: Should the chat function below use this? 
        """Fork clean session for tool execution."""
        tool_session_id = f"{main_session_id}_tool_{uuid.uuid4().hex[:8]}"
        
        # Clone relevant context (summarize if long)
        context_summary = self._summarize_context(main_session_id, max_messages=3)
        
        # Create focused tool session
        tool_chat = self._get_or_create_chat(tool_session_id, tool_choice="required")
        tool_chat.messages = [
            system("You are a tool execution specialist. Execute the requested tools and return only the results. Be precise and complete."),
            *context_summary,  # Minimal relevant context
            user(f"Execute: {message}")
        ]
        
        # Run tool loop (existing logic)
        result = self._run_tool_conversation(tool_session_id, **kwargs)
        
        # Cleanup
        self.clear_session(tool_session_id)
        return result
    
    def _integrate_tool_result(self, tool_result: Dict, main_session_id: str) -> Dict: # GROK: Should the chat function below use this? 
        """Append tool result to main conversation naturally."""
        main_chat = self._get_or_create_chat(main_session_id)
        
        # Natural integration - format result as assistant response
        integrated_response = self._format_tool_result(tool_result)
        if integrated_response:
            main_chat.messages.append(assistant(integrated_response))
        
        # Optional: Add tool metadata for transparency
        if tool_result.get('tool_calls'):
            main_chat.messages.append(assistant(f"[Used tools: {', '.join(tc.function.name for tc in tool_result['tool_calls'])}]"))
        
        return {
            'content': integrated_response,
            'tool_calls': tool_result.get('tool_calls', []),
            'session_id': main_session_id,
            'usage': tool_result.get('usage', {})
        }

    def automate_workflow(self, task: str, workflow_id: str = None, 
                         clean_session: bool = False, **kwargs) -> Dict:
        """
        Automated tool workflows - clean or continuous mode.
        """
        session_id = workflow_id or f"workflow_{uuid.uuid4().hex[:8]}"
        
        if clean_session:
            # Pattern A: Fresh execution
            return self._run_clean_automation(task, session_id, **kwargs)
        else:
            # Pattern B: Continuous workflow
            return self._run_continuous_workflow(task, session_id, **kwargs)
    
    def _run_clean_automation(self, task: str, session_id: str, **kwargs) -> Dict:
        """One-shot automation with clean slate."""
        chat = self._get_or_create_chat(session_id, tool_choice="required")
        chat.messages = [
            system("Execute this automation task completely using available tools. Return structured results."),
            user(task)
        ]
        
        # Run full tool loop until complete
        result = self._run_tool_conversation(session_id, max_rounds=10, **kwargs)
        
        # Return structured output
        return {
            'task': task,
            'result': result['content'],
            'tool_calls': result['tool_calls'],
            'status': 'completed',
            'session_id': session_id
        }
    
    def _run_continuous_workflow(self, task: str, session_id: str, **kwargs) -> Dict:
        """Stateful workflow building on previous results."""
        chat = self._get_or_create_chat(session_id, tool_choice="auto")
        
        # Append to existing workflow
        chat.messages.append(user(f"Next step: {task}"))
        
        # Allow multiple rounds, maintain state
        result = self._run_tool_conversation(session_id, max_rounds=5, **kwargs)
        
        return {
            'task': task,
            'result': result['content'],
            'tool_calls': result['tool_calls'],
            'workflow_state': self._summarize_session(session_id),
            'session_id': session_id
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def chat(self, message: str, session_id: Optional[str] = None, 
            handle_tools: bool = True, tool_choice: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Send message to conversation, handle tool calls, return structured response.
        
        Args:
            message: User input
            session_id: Conversation identifier (None uses default)
            handle_tools: If True, execute tools and continue conversation
            tool_choice: Pass to xai-sdk.chat, leave default. 
            **kwargs: Passed to chat.sample() (temperature, max_tokens, etc.)
            
        Returns:
            Dict: {
                'content': str,  # Final assistant response
                'tool_calls': List[Dict],  # Any tool calls made
                'session_id': str,  # Actual session used
                'usage': Dict  # Token usage
            }
        """

        """
        Grok suggests exposing parallel_function_calling as kwarg to chat.create() - this needs more investigation. 
        """
        session_id = self._get_session_id(session_id)

        # NEW: Detect if tools likely needed (heuristic or intent)
        needs_tools = self._detect_tool_need(message) if handle_tools else False

        if needs_tools:
            # Fork sub-session for tools
            tool_session_id = f"{session_id}_tool_{uuid.uuid4().hex[:8]}"
            tool_chat = self._get_or_create_chat(tool_session_id, tool_choice=tool_choice or "required") # GROK: I don't think this is going to work, I predict it will create an endless loop re-running the same tool. 
            
            # Clone main history (summarize if long)
            main_history = self._get_chat_history(session_id)  # Implement: return summarized messages
            tool_chat.messages.extend(main_history)  # Add context # GROK: Doesn't this eliminate the advantage of a clean chat?
            
            # Append query + tool prompt
            tool_chat.messages.append(user(message))
            tool_chat.messages.insert(0, system("Use tools for this query if needed. Return only the result."))
            
            # Run tool loop (existing code)
            # DETERMINE TOOL BEHAVIOR
            has_tools = bool(self.tool_definitions)
            effective_tool_handling = handle_tools and has_tools

            # Only pass tool_choice if we have tools
            effective_tool_choice = tool_choice or ("auto" if effective_tool_handling else "none")
        
            #chat = self._get_or_create_chat(session_id, tool_choice=effective_tool_choice)
            #chat.messages.append(user(message))
            
            tool_calls = []
            final_content = ""
            
            if effective_tool_handling:
                # Tool loop: Handle multiple rounds of tool calls
                max_tool_rounds = kwargs.pop('max_tool_rounds', 5)
                prev_result = []
                for round_num in range(max_tool_rounds):
                    # Get assistant response (may contain tool calls)
                    try:
                        response = tool_chat.sample(**kwargs)
                    except Exception as e: 
                        if "429" in str(e): # Rate limit
                            self.logger.warning("Rate limited, backing off...")
                        raise e
                                    
                    # Check for tool calls in response
                    if hasattr(response, 'tool_calls') and response.tool_calls:
                        tool_calls.extend(response.tool_calls)
                        
                        # Execute tools and add results
                        for tool_call in response.tool_calls:
                            tool_name = tool_call.function.name
                            if tool_name in self.tools_map:
                                try:
                                    # Parse arguments
                                    args = json.loads(tool_call.function.arguments)
                                    # Execute tool
                                    result = self.tools_map[tool_name](**args)
                                    if result not in prev_result:
                                        # Add tool result to conversation
                                        tool_chat.messages.append(tool_result(
                                            result=json.dumps(result)
                                        ))
                                        self.logger.debug(f"Tool '{tool_name}' executed: {result}")
                                        prev_result.append(result)
                                    else: 
                                        break #exit loop because of duplicate/redudnante tool call. 
                                except Exception as e:
                                    # Tool failed - add error result
                                    tool_chat.messages.append(tool_result(
                                        result=json.dumps({"error": str(e)})
                                    ))
                                    self.logger.error(f"Tool '{tool_name}' failed: {e}")
                                    if effective_tool_choice == "required":
                                        self.logger.warning("Tool failed with 'required' choice—model may hallucinate next.")
                            else:
                                # Unknown tool
                                tool_chat.messages.append(tool_result(
                                    tool_call_id=tool_call.id,
                                    content=json.dumps({"error": f"Unknown tool: {tool_name}"})
                                ))
                        
                        # Continue conversation with tool results
                        continue
                    elif round_num == max_tool_rounds -1 or not (hasattr(response, 'tool_calls') and response.tool_calls): # force a final response and ignore further tool call requests. 
                        # No more tool calls - this is the final response
                        if response.content:
                            final_content = response.content
                            tool_chat.messages.append(assistant(final_content))  # Append for history
                            usage = getattr(response, 'usage', {})
                            break
            
            # Integrate back to main
            main_chat = self._get_or_create_chat(session_id)
            if final_content: 
                main_chat.messages.append(assistant(final_content))  # Or formatted summary
            
            self.clear_session(tool_session_id)  # Cleanup sub
            
            # return { 'content': final_content, ... }  # Existing return

        else:
            # Simple chat without tools
            main_chat = self._get_or_create_chat(session_id)
            try: 
                response = main_chat.sample(**kwargs)
            except Exception as e: 
                if "429" in str(e): # Rate limit
                     self.logger.warning("Rate limited, backing off...")
                raise

            if response.content: 
                main_chat.messages.append(assistant(response.content))
            final_content = response.content
            usage = getattr(response, 'usage', {})
        
        # Sync state
        # self._sync_chat(session_id) - commented out for debugging purposes. 
        
        # Log to memory
        self.memory.add_exchange([
            {"role": "user", "content": message},
            {"role": "assistant", "content": final_content}
        ])
        
        # FIXED LOGGING: Handle SDK ToolCall objects
        def is_tool_executed(tool_call):
            """Check if a tool call was successfully executed."""
            if not hasattr(tool_call, 'function'):
                return False
            # Check if result was added (we append tool_result after execution)
            # For now, assume executed if call exists (since we handle all calls)
            return True
        
        tools_executed_count = sum(1 for tc in tool_calls if is_tool_executed(tc))
        
        self.memory.log_usage(
            method='chat',
            prompt=message,
            response=final_content,
            usage=usage,
            workflow_context={
                'session_id': session_id,
                'tools_available': has_tools,
                'tools_handled': effective_tool_handling,
                'tool_calls': len(tool_calls),
                'tools_executed': tools_executed_count  # Fixed!
            }
        )
        
        return {
            'content': final_content,
            'tool_calls': tool_calls,
            'session_id': session_id,
            'usage': usage
        }
    
    async def achat(self, message: str, session_id: str = "default", **kwargs) -> str:
        """
        Async version of chat(). Requires xAI SDK async support. - Revisit this based on doc.x.ai patterns
        """
        # Assuming SDK has async methods - adjust based on actual API
        chat = self._get_or_create_chat(session_id)
        chat.messages.append(user(message))
        
        # Hypothetical async sample - check SDK docs
        response = await chat.asample(**kwargs)  # or similar
        
        self._sync_chat(session_id)
        
        # Async logging (fire-and-forget or await as needed)
        # self.memory.add_exchange([...])  # Consider async-safe memory
        
        return response.content

    def _execute_tool(self, tool_call: Response.tool_calls) -> Dict: # GROK: I changed to Response.tool_calls because that was what intellisense suggested. 
        """
        Execute a single tool call and format result for xAI.

        Arguments: 
            tool_call: 

        Returns: 
            Dict
        """
        name = tool_call.function.name # GROK: Intellisense didn't like the .function.name parts. 
        if name not in self.tools_map: 
            return {"error": f"Tool '{name}' not found"}
        
        proc, ParamModel, _ = self.tools[name]
        try: 
            args = json.loads(tool_call.function.arguments)
            param_instance = ParamModel(**args) # Validate/parse args via Pydantic
            result = proc(param_instance) # Call the procedure with model instance
            return result # Assume proc returns serializable dict/str
        except Exception as e: 
            return {"error": str(e)}

    def _serialize_prompt(self, messages: List[Dict], max_length: int = 2000) -> str: 
        """
        Serialize conversation history for loging. Truncates if too long. 

        Args:
            messages: Full chat.messages list
            max_length: Max characters before truncation
        
        Returns: 
            JSON string of conversation, truncated if needed 
        """
        # Filter out system messages for cleaner logs (optional)
        user_assistant_only = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
            if msg["role"] in ["user", "assistant", "tool"] # Include tool results                                    
        ]

        serialized = json.dumps(user_assistant_only, ensure_ascii=False)

        if len(serialized) > max_length: 
            # Smart truncation: Keep first/last exchanges, summarize middle
            head = user_assistant_only[:3] # First 3 exchanges
            tail = user_assistant_only[-3:] # Last 3 exchanges
            middle_count = len(user_assistant_only) - 6

            truncated = {
                "structure": f"[Total: {len(user_assistant_only)} exchanges; Middle {middle_count} omitted]",
                "beginning": head, 
                "end": tail
            }
            serialized = json.dumps(truncated, ensure_ascii=False)
        
        return serialized
    
    def encode_image(image_path): 
        """
        Source: https://docs.x.ai/docs/guides/image-understanding
        Convert image file to string for sending to grok.
        """
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string
    
    def _detect_visual_content(self, document: Path, max_pages: int = 20) -> Dict[str, Any]:
        """
        Scan PDF for images, drawings, and tables. Returns page numbers and content types.
        
        Args:
            document: Path to PDF
            max_pages: Limit scanning to first N pages for performance
        
        Returns:
            Dict with detected visuals: {'images': [page_nums], 'drawings': [page_nums], 'table_areas': [...]}
        """
        if not document.suffix.lower() =='.pdf': 
            return {'images': [], 'drawings': [], 'table_areas': []}
        
        pdf_doc = fitz.open(document)
        visual_pages = {'images': [], 'drawings': [], 'table_areas': []}

        for page_num in range(min(len(pdf_doc), max_pages)):
            page = pdf_doc.load_page(page_num)

            # 1. Detect Images (figures, charts, photos)
            images = page.get_images(full=True) # Returns list of image dicts
            if images: 
                visual_pages['images'].append({
                    'page': page_num,
                    'count': len(images),
                    'image_refs': [img[0] for img in images] # xref IDs for later extraction
                })
            
            # 2. Detect Drawings (vector graphics, diagrams, flowcharts)
            drawings = page.get_drawings()
            if drawings: 
                visual_pages['drawings'].append({
                    'page': page_num, 
                    'count': len(drawings), 
                    'types': [d.get('type', 'unknown') for d in drawings]
                })

            # 3. Detect Tables (via layout analysis - more complex)
            # Get page blocks to identify potential table regions
            blocks = page.get_text("dict")['blocks']
            table_candidates = []
            for block in blocks: 
                if 'lines' in block: # Text block with structured lines 
                    line_count = len(block['lines'])
                    if line_count > 3 and 'I' in page.get_text("text", clip=block['bbox']): # Rough table detection
                        table_candidates.append({
                            'page': page_num,
                            'bbox': block['bbox'], # Coordinates for extraction 
                            'line_count': line_count
                        })
            if table_candidates: 
                visual_pages['table_areas'].extend(table_candidates)
        
        pdf_doc.close()
        return visual_pages
    
    def _extract_visual_context(self, document: Path, visual_pages: Dict[str, Any], 
                           extract_images: bool = True, max_images: int = 3) -> str:
        """
        Extract visual content and return context for the prompt.
        
        Args:
            document: PDF path
            visual_pages: From _detect_visual_content()
            extract_images: Whether to actually extract image data
            max_images: Limit number of images to extract
        
        Returns:
            Context string: "Found 2 images on pages 3,7; 1 table on page 5"
        """
        context_parts = []
        # Image context
        if visual_pages['images']:
            image_pages = [v['page'] for v in visual_pages['images'][:max_images]]
            if extract_images: 
                # Actually extract images as base64 for vision analysis
                pdf_doc = fitz.open(document)
                image_context = []
                for visual_info in visual_pages['images'][:max_images]:
                    page_num = visual_info['page']
                    page = pdf_doc.load_page(page_num)
                    # Extract first image on this page
                    img_list = page.get_images(full=True)
                    if img_list: 
                        xref = img_list[0][0]
                        base_image = pdf_doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        base64_img = base64.b64encode(image_bytes).decode('utf-8')
                        image_context.append({
                            'page': page_num, 
                            'base64': base64_img,
                            'description': f"Image on page {page_num + 1}"
                        })
                pdf_doc.close()
                # Store base64 images for later use in messages
                self._temp_images = image_context # Instance variable for access in main method
            else:
                image_context = []

            context_parts.append(f"Found {len(visual_pages['images'])} images on pages {image_pages}")
        
        # Drawing context
        if visual_pages['drawings']: 
            drawing_pages = [v['page'] for v in visual_pages['drawings']]
            context_parts.append(f"Found {len(visual_pages['drawings'])} diagrams/drawings on pages {drawing_pages}")

        # Table context 
        if visual_pages['table_areas']: 
            table_pages = [t['page'] for t in visual_pages['table_areas']]
            context_parts.append(f"Found {len(visual_pages['table_areas'])} potential tables on pages {table_pages}")
        
        return "Visual content detected: " + "; ".join(context_parts) if context_parts else "No visual content detected."
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def structured_document_analysis(
            self, 
            instruction_prompt: str, 
            document: Union[str, Path], 
            output_schema: Type[BaseModel],
            detect_visuals: bool = True,
            use_vision: bool = False,
            max_visual_pages: int = 5,
            **kwargs
    ) -> BaseModel: 
        """
        Enhanced document analysis with automatic visual detection. 

        Args: 
            instruction_prompt: System instructions for analysis (e.g., "Extract title, authors, abstract, and key findings.").
            document: Raw text string or Path to pdf/file. 
            output_schema: Pydantic BaseModel subclass defining the output structure.
            detect_visuals: Scan for images/tables (default: True)
            use_vision: Send detected images to vision model (requires grok-2-vision)
            max_visual_pages: Limit visual scanning to first N pages
            **kwargs: Passed to chat.create (e.g., temperature=0.1 for determinism).
        
        Returns: 
            Instance of output_schema, parsed from the API response. 

        Raises: 
            ValueError: If schema unsupported or parsing fails. 
            FileNotFoundError: if document path invalid.
        """
        visual_pages = []
        # Extract base text content
        if isinstance(document, Path) and document.suffix.lower() == '.pdf':
            if not document.exists():
                raise FileNotFoundError(f"Document not found: {document}")
            
            pdf_doc = fitz.open(document)
            document_content = ""
            for page_num in range(len(pdf_doc)):
                page = pdf_doc.load_page(page_num)
                page_text = page.get_text("text")  # Correct API call!
                document_content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            pdf_doc.close()
            
            # Detect visuals if requested
            visual_context = ""
            self._temp_images = []  # Reset
            if detect_visuals:
                visual_pages = self._detect_visual_content(document, max_visual_pages)
                visual_context = self._extract_visual_context(document, visual_pages, 
                                                            extract_images=use_vision, 
                                                            max_images=max_visual_pages)
                instruction_prompt += f"\n\n{visual_context}"  # Add to prompt
        else:
            # Handle text input
            document_content = str(document)
            visual_context = ""

        if not document_content.strip():
            raise ValueError("Document content is empty.")
        
        # Build messages with optional vision content
        messages = [
            {"role": "system", "content": f"{self.personality.to_system_prompt()}\n\n{instruction_prompt}"},
            {"role": "user", "content": f"Analyze this document:\n\n{document_content}"}
        ]

        # Add images if using vision
        if use_vision and hasattr(self, '_temp_images') and self._temp_images:
            for img_info in self._temp_images:
                messages[-1]['content'] += f"\n\n{img_info['description']}:"
                # Append image using xAI SDK format (adjust based on actual API)
                messages.append({"type": "image", "image_url": {"url": f"data:image/png;base64,{img_info['base64']}"}})

        # Response format for structured output
        schema_name = output_schema.__name__
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": output_schema.model_json_schema(),
                "strict": True # Enforces schema compliance
            }
        }

        # Create and sample chat
        chat = self.client.chat.create(
            model=self.model,
            messages=messages,
            response_format=response_format,
            **kwargs
        )
        try: 
            response = chat.sample()
        except Exception as e: 
            if "429" in str(e): # Rate limit
                self.logger.warning("Rate limited, backing off...")
                raise
            raise 

        # Log usage (prompt is instruction + document for context)
        prompt_for_log = f"{instruction_prompt}\n\nDocument preview: {document_content[:500]}..." # Truncate for storage
        usage_dict = getattr(response, 'usage', {}) if hasattr(response, 'usage') else {}
        self.memory.log_usage(
            method='structured_document_analysis',
            prompt=prompt_for_log, 
            response=response.content,
            usage=usage_dict,
            workflow_context={
                'schema': schema_name,
                'document_type': 'pdf' if isinstance(document, Path) and document.suffix.lower() == '.pdf' else 'text',
                'document_length_chars': len(document_content)
            }
        )

        # Add to memory (simplified exchange)
        exchange = [
            {"role": "user", "content": f"Analyze: {document_content[:200]}..."}, 
            {"role": "assistant", "content": response.content}
        ]
        self.memory.add_exchange(exchange)

        # Parse structured output (guaranteed valid due to strict mode)
        try: 
            structured_result = output_schema.model_validate_json(response.content)
            return structured_result
        except ValueError as e: 
            self.logger.error(f"Structured parsing failed: {e}")
            raise ValueError(f"API resposne did not match schema '{schema_name}': {response.content}") from e

    def structured_intent(self, message: str, system_prompt: str = "Detect the user's intent and extract relevant parameters.", **kwargs) -> UserIntent:
        """
        Detect intent via structured output. Returns pydantic model, not raw JSON. 

        Args: 
            message: User input
            system_prompt: Task-focused prompt (no schema details needed) 

        Returns: 
            UserIntent: Type-safe parsed result
        """

        payload = {
            "model": self.model, 
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ], 
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "user_intent",
                    "schema": self.intent_schema,
                    "strict": True # Enfoces exact schema match
                }
            },
            **kwargs
        }

        response = self.chat_completion(**payload)
        json_content = response["choices"][0]["message"]["content"]

        # Parse to Pydantic (guaranteed valid)
        intent = UserIntent.model_validate_json(json_content)

        # Log usage
        usage_dict = response.get('usage', {})  # Assuming dict format here (adjust per SDK)
        self.memory.log_usage(
            method='structured_intent',
            prompt=message,
            response=json_content,
            usage=usage_dict,
            workflow_context={'detected_intent': intent.type, 'confidence': intent.confidence}
        )

        return intent
    
    def intent_agent(self, message: str, system_prompt: str = None, **kwargs) -> Dict: 
        """
        Full flow: Detect intent -> Execute tool -> Return result. 
        """
        # Step 1: Structured intent detection
        intent = self.structured_intent(message, system_prompt or "You are a helpful assistant. Detect what the user wants and extract parameters.")
        
        print(f"Detected: {intent.type} (confidence: {intent.confidence})")

        # Step 2: Route to tool or direct response
        if intent.type == "chat": 
            return self.simple_generate(message, **kwargs)
        elif intent.type in self.tools: 
            try: 
                # Extract params, execute tool
                result = self.tools[intent.type](**intent.params)
                return {"intent": intent.type, "result": result, "success": True}
            except Exception as e: 
                return {"intent": intent.type, "error": str(e), "success": False}
        else: 
            return {"intent": intent.type, "error": "Unkown intent", "success": False}