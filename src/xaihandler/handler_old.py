import os
import base64
import json
import numpy as np
import logging 
import fitz # PyMuPDF for PDF text extraction
import uuid
import grpc 
import re 
import inspect
from .personality import Archetype, Trait, AgentTrait, AgentPersonality
from .memory import StatefulMemory
from .handlerconfig import HandlerConfig
from typing import Dict, Callable, Tuple, Any, List, Optional, Literal, Union, Type
from pydantic import BaseModel, Field, model_validator
from tenacity import retry, stop_after_attempt, wait_exponential
from pathlib import Path
from xai_sdk import Client
from xai_sdk.chat import BaseChat, Response, user, system, assistant, image, tool, tool_result
from xai_sdk.proto.v5.chat_pb2 import MessageRole, Message
from xai_sdk.sync.chat import Chat
from xai_sdk.search import SearchParameters

class UserIntent(BaseModel): 
    """Schema for intent detection + params."""
    type: Literal['get_weather', 'query_zotero', 'book_flight', 'chat'] = Field(..., description="User's intent")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the intent")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model's confidence in this intent") 

# Assuming a role mapper; adjust based on your enum (inspect MessageRole to confirm values)
ROLE_MAP = {
    MessageRole.ROLE_SYSTEM: "system",
    MessageRole.ROLE_USER: "user",
    MessageRole.ROLE_ASSISTANT: "assistant",
    MessageRole.ROLE_TOOL: "tool_result",
    # Add others if needed
}

class xAI_Handler:
    """
    This class manages conversation context and handles xAI-SDK and xAI-API calls for AI powered assistants. 
    """ 
    def __init__ (
            self, 
            config: Optional[HandlerConfig] = None,
            api_key: Optional[str] = None, 
            model: Optional[str] = None, 
            timeout: Optional[float] = None, 
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
                    traits=[
                        AgentTrait(trait=Trait.EMPATHY, intensity=60),
                        AgentTrait(trait=Trait.CURIOSITY, intensity=50), 
                        AgentTrait(trait=Trait.PRECISION, intensity=90)  
                    ]
                ),
                tools=tools or {},
                memory_db="memory_vault.db",
                timeout=timeout or int(os.getenv("XAI_TIMEOUT", "3600"))
            )
        
        self.logger = logger or logging.getLogger(__name__)
        self.tool_definitions: List[Any] = []  # SDK tool objects
        self.tools_map: Dict[str, Callable] = {}  # Name -> func for execution
        self.rebuild_tool_definitions() # xAI-style list of tool objects
        
        self.intent_schema = UserIntent.model_json_schema()
        self._active_chats: Dict[str, BaseChat] = {}
        self._chat_states: Dict[str, List[Dict]] = {}
        # self.parallel_calling = True # Uncomment this when you begin work on async functions 
        self._default_session_id = str(uuid.uuid4())
        self.client = Client(
            api_key=self.config.api_key,
            timeout=self.config.timeout
        )
        
        if validate_connection and self._validate_connection():
            self.logger.info(
                "xAI API handler initialized.", extra={
                    "component": "xai-handler", 
                    "personality": personality.name,
                    "status": "ready", 
                    "message": f"{personality.name} reporting for duty."
                }
            ) 
        else: 
            self.logger.warning("API Connection not validated, calls to xAI API may fail.")

    def _validate_connection(self): 
        """Verify API connectivity and credentials."""
        try: 
            # Minimal connectivity test
            chat = self.client.chat.create(
                model=self.config.model,
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
        except grpc.RpcError as e: # TODO: Create an error handler function to standardise error handling throughout the file. 
            self.logger.error(f"API error in chat {e.code().name} - {e.details()}")
            return False
        except Exception as e: 
            self.logger.error(
                f"API connection validation failed: {str(e)}", 
                extra={
                    "error_type": type(e).__name__, 
                    "model": self.model
                }
            )
            return False
    
    @property 
    def memory(self): 
        if not hasattr(self, '_memory'): 
            self._memory = StatefulMemory(xai_api_key=self.config.api_key, db_path=self.config.memory_db)
        return self._memory

    def _get_session_id(self, session_id: Optional[str] = None) -> str: 
        """"Smart session ID resolution."""
        if session_id is None: 
            # Return default or create new based on strategy
            return getattr(self, '_default_session_id', 'default')
        return session_id 
    
    def _get_or_create_chat(self, session_id: str = "default", tool_choice: str = "auto") -> tuple[str, Chat]:
        """
        Get existing chat or create new one from the stored state.
        """
        session_id = self._get_session_id(session_id)
        
        if session_id not in self._active_chats:
            messages = [system(self.config.personality.system_prompt)] # TODO: Remove this and append the system message below. 
             
            if self.tool_definitions: # Check if tools are configured
                tools_config = {
                    "tools": self.tool_definitions,
                    "tool_choice": tool_choice
                }
                chat = self.client.chat.create(
                    model=self.config.model,
                    messages=messages,
                    **tools_config
                )
            else:
                chat = self.client.chat.create(
                    model=self.config.model,
                    messages=messages
                )
                
            self._active_chats[session_id] = chat            
        
        return [session_id, self._active_chats[session_id]] # Updated so that session_id is passed to calling function when no session_id was specified. 

    def _sync_chat(self, session_id: str):
        """
        Save chat state and clear active object to free memory.
        TODO: Implement this method per below description. 

        This method should index the chat history of an active chat that is slated for deletion.
        The indexed chat history should be incorporated into StatefulMemory. 
        TODO: Ensure StatefulMemory can accept a chat history.
        TODO: As per Grok suggestion, pair the del command with either weakref or context manager. 
        TODO: Experiement with chat.messages.dump as a more efficient method of transfering the entire message history
        """
        """ session_id = self._get_session_id(session_id)
        if session_id in self._active_chats:
            chat = self._active_chats[session_id]
            self._chat_states[session_id] = chat.messages
            del self._active_chats[session_id] """
        """
        Suggestions from Grok for improvement. 
        - In _sync_chat, use msg.model_dump() if messages are Pydantic (SDK might use them).
        - Add self._chat_states persistence (e.g., JSON dump on shutdown) for cross-run sessions.
        """

    def rebuild_tool_definitions(self):
        """
        Rebuild tool_definitions from current self.tools.
        Call this after add/remove/clear.
        """
        self.tool_definitions = []
        self.tools_map = {}  # Reset map
        
        for tool_name, (tool_func, param_schema, description) in self.config.tools.items():         
            # Create SDK tool object
            sdk_tool = tool(
                name=tool_name,
                description=description,
                parameters=param_schema
            )
            self.tool_definitions.append(sdk_tool)
            # Map function for execution
            self.tools_map[tool_name] = tool_func
        self.logger.info(f"Rebuilt {len(self.tool_definitions)} tools: {list(self.tools_map.keys())}")

    def add_tool(self, tool_name: str, tool_func: Callable, param_model: Optional[type[BaseModel]], description: str, rebuild_definitions: bool = True):
        """
        Dynamically add a tool. Rebuilds definitions automatically.
        
        Args:
            tool_name: Unique name for the tool.
            tool_func: The callable to execute (e.g., def get_weather(city: str) -> str: ...).
            param_model: Pydantic model for args (or None for no params).
            description: Human-readable description for the model.
            rebuild_definitions: If chaining multiple tool changes, set to false until the final change.  
        """
        if tool_name in self.config.tools:
            self.logger.warning(f"Tool '{tool_name}' already exists; overwriting.")
        
        self.config.add_tool(tool_name, tool_func, param_model, description)
        if rebuild_definitions: 
            self.rebuild_tool_definitions()
        
    def remove_tool(self, tool_name: str, rebuild_definitions: bool = True):
        """
        Remove a tool by name. Rebuilds definitions.
        
        Args: 
            tool_name: Name of tool to remove
            rebuild_definitions: If chaining multiple tool changes, set to false until the final change.
        """
        if tool_name in self.config.tools:
            self.config.remove_tool(tool_name)
            if rebuild_definitions: 
                self.rebuild_tool_definitions()
        else:
            self.logger.warning(f"Tool '{tool_name}' not found.")

    def clear_tools(self, rebuild_definitions: bool = True):
        """Clear all tools. Rebuilds definitions."""
        self.config.tools.clear()
        if rebuild_definitions: 
            self.rebuild_tool_definitions()

    def check_tools(self) -> Dict[str, Any]:
        """Return current tools status for debugging."""
        return {
            "tools_count": len(self.config.tools),
            "tool_names": list(self.config.tools.keys()),
            "definitions_count": len(self.tool_definitions)
            #"parallel_calling": True  # When this becomes a thing uncomment and pull the actual value. 
        }
    
    def get_tool(self, tool_name: str) -> Optional[Dict[str, Any]]: 
        """
        Gets tool information for inspection. 

        Args: 
            tool_name: Name of tool to inspect. 

        Returns: 
            Dict containing the data for the tool. 
        """
        if tool_name in self.config.tools:
            tool_func, param_schema, description = self.config.tools[tool_name]
            return {
                "name": tool_name,
                "description": description,
                "param_model": param_schema,
                "func_signature": inspect.signature(tool_func) if callable(tool_func) else None
            }
        self.logger.warning(f"Tool '{tool_name}' not found.")
        return None 

    def _detect_tool_need(self, message: str) -> bool:
        """Heuristic: Keywords or intent check."""
        tool_keywords = set()
        for desc in self.config.tools.values(): 
            tool_keywords.update(re.findall(r'\b\w+\b', desc[2].lower())) # Extract words from description
        if any(kw in message.lower() for kw in tool_keywords):
            return True

    def _convert_message_container_to_list_of_dict(self, message_container) -> List[Dict[str, str]]:
        for msg in message_container: 
            rv: List[Dict[str, str]] = []
            try: 
                role = ROLE_MAP[msg.role].lower()
            except Exception as e: 
                self.logger.warning(f"Detecting role failed: {e}")
            content = None
            try: 
                content = msg.content.pop().text
            except Exception as e: 
                self.logger.warning(f"Empty content element: {e}")
            if content: 
                rv.append({"role": role, "content": content})
            
        return rv
    

    def _get_chat_history(self, session_id: str, summarize: bool = True) -> List[Dict[str, str]]:
        """
        Get/summarize history for forking.
        
        Args: 
            session_id: ID of the session that needs to be summarised. 
            summarize: Option to turn off summaries if required. 
        
        Returns: 
            List: [Message] small sample of messages for use as context in other chats. 
        """
        history = []
        try: 
            chat_tupple = self._get_or_create_chat(session_id)
            if len(chat_tupple[1].messages) > 5 and summarize: 
                temp_chat = self.client.chat.create(model="grok-4-fast-reasoning") # TODO: Add summarization model to config, use less tokens then main model. 
                temp_chat.append(system("You are an efficient librarian, capable of condensing information for other chat bots."))
                temp_chat.messages.extend(chat_tupple[1].messages[:-5]) # JAKOB: Grok suggested extend instead of append to avoid a Type Error. Pylance doesn't think extend exists. 
                temp_chat.append(user("Please summarize this conversation for use as context in another chat."))
                summary = temp_chat.sample() 
                history.append({"role": "assistant", "content": summary.content})
        except Exception as e: 
            self.logger.error(f"Summarization failed: {e}") # TODO: Implement error handling, Type Error and what ever sample throws. 

        # Hmm if messages len is < 5 this creates a bunch of blank mesages that have roles. WHY? 
        last_5 = chat_tupple[1].messages[-5:]
        history.extend(self._convert_message_container_to_list_of_dict(message_container=last_5))        
        return history  

    def clear_session(self, session_id: str):
        """
        Archive a session after it has completed its useful life.
        TODO: Implement logic for indexing chat history to StatefulMemory. 
        Args: 
            session_id: Session ID to be archived. 
        """
        
        if self._active_chats[session_id]:
            del self._active_chats[session_id]
        else:
            self.logger.warning(f"Chat {session_id} is not in the active chat list.")
    
    def _execute_tool_task(self, message: str, context_messages: List[Dict] = None, **kwargs) -> Dict: # GROK: Should the chat function below use this? 
        """Fork clean session for tool execution."""
        tool_session_id = f"_tool_{uuid.uuid4().hex[:8]}"
        tool_chat = self._get_or_create_chat(tool_session_id, tool_choice="auto")[1]
        try:
            tool_chat.append(system("You are a tool execution specialist. Execute the requested tools and return only the results. Be precise and complete."))
            # tool_chat.append(context_messages) #JAKOB: Grok this produces a value error because append can only append one message
            # tool_chat.extend(context_messages) # JAKOB: Grok the extend function doesn't exist so this won't work. 

            for msg in context_messages or []: # JAKOB: Grok, the for loop will iterate through the messages. However, msg.get produces an attribute exception
                role = msg.get("role")
                content = msg.get("content")
                if role == "assistant":
                    tool_chat.append(assistant(content))
                elif role == user: 
                    tool_chat.append(user(content))
                elif role == "tool_result":
                    tool_chat.append(tool_result(content))
            tool_chat.append(user(message)) # JAKOB: This doesn't get called if the above messages fail. 
            
        except ValueError as e:
            self.logger.error(f"Value Error attempting to append message to chat. {e}")# TODO: Figure out what to do here. 
            self.clear_session(tool_session_id)

            return {
                'content': "Exception occured tool call failed. ",
                'tool_calls': 0,
                'tools_executed': 0,
                'session_id': tool_session_id,
                'usage': "N/A"
            }   
        except Exception as e: 
            self.logger.error(f"tool_chat.message.extend failed {e}")
            self.clear_session(tool_session_id)

            return {
                'content': "Exception occured tool call failed. ",
                'tool_calls': 0,
                'tools_executed': 0,
                'session_id': tool_session_id,
                'usage': "N/A"
            }   
        # Run tool loop (existing code)
        tool_calls = []
        final_content = ""
        max_tool_rounds = kwargs.pop('max_tool_rounds', 5)
        prev_result = []
        for round_num in range(max_tool_rounds):
            # Get assistant response (may contain tool calls)
            try: # TODO: Fix this for actual error handling. 
                response = tool_chat.sample(**kwargs) # JAKOB: Pylance isn't picking up sample. I think because BaseChat does not have sample method, but the derivitive chats do. 
            except grpc.RpcError as e: 
                self.logger.error(f"API error in chat {e.code().name} - {e.details()}")
                raise e # TODO: Add specific instructions for each code. 
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
                                try: 
                                    tool_chat.append(tool_result(
                                        result=json.dumps(result)
                                    ))
                                    self.logger.info(f"Tool '{tool_name}' executed: {result}")
                                    prev_result.append(result)
                                except ValueError as e:
                                    self.logger.error(f"Value Error attempting to append message to chat. {e}")
                                    pass # TODO: Figure out what to do here. 
                            else: 
                                break #exit loop because of duplicate/redudnante tool call. 
                        except Exception as e: # TODO: This is likely redundant code based on the nested exception handling
                            # Tool failed - add error result
                            tool_chat.messages.append(tool_result(
                                result=json.dumps({"error": str(e)})
                            ))
                            self.logger.error(f"Tool '{tool_name}' failed: {e}")
                    else:
                        # Unknown tool
                        try: 
                            tool_chat.append(tool_result(
                                content=json.dumps({"error": f"Unknown tool: {tool_name}"})
                            ))
                        except ValueError as e:
                            self.logger.error(f"Value Error attempting to append message to chat. {e}")
                            pass # TODO: Figure out what to do here.  
                
                # Continue conversation with tool results
                continue
            elif round_num == max_tool_rounds -1 or not (hasattr(response, 'tool_calls') and response.tool_calls): # force a final response and ignore further tool call requests. 
                # No more tool calls - this is the final response
                if response.content:
                    final_content = response.content
                    try: 
                        tool_chat.append(assistant(final_content))  # Append for history
                    except ValueError as e:
                        self.logger.error(f"Value Error attempting to append message to chat. {e}")
                        pass # TODO: Figure out what to do here. 
                    usage = getattr(response, 'usage', {})
                    break

        # FIXED LOGGING: Handle SDK ToolCall objects
        def is_tool_executed(tool_call):
            """Check if a tool call was successfully executed."""
            if not hasattr(tool_call, 'function'):
                return False
            # Check if result was added (we append tool_result after execution)
            # For now, assume executed if call exists (since we handle all calls)
            return True
        
        tools_executed_count = sum(1 for tc in tool_calls if is_tool_executed(tc))
        tool_chat_history = self._convert_message_container_to_list_of_dict (message_container=tool_chat.messages)
        # Cleanup
        self.clear_session(tool_session_id)

        return {
            'content': tool_chat_history,
            'tool_calls': tool_calls,
            'tools_executed': tools_executed_count,
            'session_id': tool_session_id,
            'usage': usage
        }
    
    def _integrate_tool_result(self, tool_messages: List[Dict[str, str]], main_session_id: str):  # GROK: Should the chat function below use this? 
        """Append tool result to main conversation naturally."""
        main_chat = self._get_or_create_chat(main_session_id)[1]
        try:
            for msg in tool_messages: 
                if msg["role"] == "assistant":
                    main_chat.append(assistant(msg["content"]))
                if msg["role"] == "tool_result": 
                    main_chat.append(tool_result(msg["content"]))

        except ValueError as e:
            self.logger.warning(f"Error integrating tool results with main chat. {e}")
            pass # TODO: Figure out what to do here. 
        #return main_chat.messages

    def automate_workflow(self, task: str, workflow_id: str = None, 
                         clean_session: bool = False, **kwargs) -> Dict:
        """
        Automated tool workflows - clean or continuous mode.
        JAKOB: Grok, this procedure was generated by a previous Grok Instance. This other instance didn't think through the entire stack
            procedures to get to this method. I think the idea behind it is good but we need to work on it. Ask me for more information if
            you see this comment and I haven't discussed the "automated routines" and "chained instructions" work flows. 
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
        chat = self._get_or_create_chat(session_id, tool_choice="required")[1]
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
        chat = self._get_or_create_chat(session_id, tool_choice="auto")[1]
        
        # Append to existing workflow
        try: 
            chat.messages.append(user(f"Next step: {task}"))
        except ValueError as e:
                self.logger.error(f"Value Error attempting to append message to chat. {e}")
                pass # TODO: Figure out what to do here. 
        
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
        temp = self._get_or_create_chat(session_id) # TODO: Return session_id as well as chat, session_id is None during the whole interaction here. 
        session_id = temp[0]
        main_chat = temp[1]
        if self._detect_tool_need(message) and handle_tools:
            # REFACTOR to _execute_tool_task. 
            context = self._get_chat_history(session_id)
            tool_result = self._execute_tool_task(message=message, context_messages=context) # If this fails remove context. 

            self._integrate_tool_result(tool_result["content"], session_id) 
            last_message = main_chat.messages.pop() if main_chat.messages else None
            final_content = last_message.content.pop().text if last_message else "" # Get last message
            usage = tool_result["usage"]
            tool_calls = tool_result.get("tool_calls")
            workflow_context={
                'session_id': session_id,
                'tools_available': self.check_tools(), 
                'tool_calls': tool_calls  
            }
            
        else:
            # Simple chat without tools
            try: 
                main_chat.append(user(message))
                response = main_chat.sample(**kwargs)
            except grpc.RpcError as e: # TODO: Add instructions for each code. 
                self.logger.error(f"API error in chat {e.code().name} - {e.details()}")
                return {
                    'content': "An error occured during API call",
                    'tool_calls': 0,
                    'session_id': session_id,
                    'usage': "none"
                }
                return False
            except Exception as e: # TODO: Correct this to proper error checking for sample. 
                if "429" in str(e): # Rate limit
                     self.logger.warning("Rate limited, backing off...")
                raise

            if response.content: 
                try: 
                    main_chat.messages.append(assistant(response.content)) # TODO: Check if chat.sample automatically appends assistant message to chat history. 
                except ValueError as e:
                    self.logger.error(f"Value Error attempting to append message to chat. {e}")
                    pass # TODO: Figure out what to do here. 
            final_content = response.content
            usage = getattr(response, 'usage', {})
            workflow_context={
                'session_id': session_id,
                'tools_available': self.check_tools(), 
                'tool_calls': 0  
            }
            tool_calls = 0
        
        # Sync state
        # self._sync_chat(session_id) - commented out for debugging purposes. 
        
        # Log to memory
        self.memory.add_exchange([
            {"role": "user", "content": message},
            {"role": "assistant", "content": final_content}
        ])

        """ self.memory.log_usage(
            method='chat',
            prompt=message,
            response=final_content,
            usage=usage,
            workflow_context=workflow_context
        ) """
        
        return {
            'content': final_content,
            'tool_calls': tool_calls,
            'session_id': session_id,
            'usage': usage
        }
    
    async def achat(self, message: str, session_id: str = "default", **kwargs) -> str:
        #TODO: Uncomment this line in _init_ when you're ready to implement this function self.parallel_calling = True
        """
        Async version of chat(). Requires xAI SDK async support. - Revisit this based on doc.x.ai patterns
        """
        # Assuming SDK has async methods - adjust based on actual API
        chat = self._get_or_create_chat(session_id)
        try: 
            chat.messages.append(user(message))
        except ValueError as e:
            self.logger.error(f"Value Error attempting to append message to chat. {e}")
            pass # TODO: Figure out what to do here. 
        
        # Hypothetical async sample - check SDK docs
        response = await chat.asample(**kwargs)  # or similar
        
        self._sync_chat(session_id)
        
        # Async logging (fire-and-forget or await as needed)
        # self.memory.add_exchange([...])  # Consider async-safe memory
        
        return response.content

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
                messages[-1]['content'] += f"\n\n{img_info['description']}:" # TODO: Double check if this is acceptable. 
                # Append image using xAI SDK format (adjust based on actual API)
                try: 
                    messages.append({"type": "image", "image_url": {"url": f"data:image/png;base64,{img_info['base64']}"}})
                except ValueError as e:
                    self.logger.error(f"Value Error attempting to append message to chat. {e}")
                    pass # TODO: Figure out what to do here. 

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
            model=self.config.model,
            messages=messages,
            response_format=response_format,
            **kwargs
        )
        try: 
            response = chat.sample()
        except grpc.RpcError as e: # TODO: Add instructions for each code. 
                self.logger.error(f"API error in chat {e.code().name} - {e.details()}")
                return False
        except Exception as e: # TODO: Fix this for correct error handling. 
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
                result = self.config.tools[intent.type](**intent.params)
                return {"intent": intent.type, "result": result, "success": True}
            except Exception as e: 
                return {"intent": intent.type, "error": str(e), "success": False}
        else: 
            return {"intent": intent.type, "error": "Unkown intent", "success": False}