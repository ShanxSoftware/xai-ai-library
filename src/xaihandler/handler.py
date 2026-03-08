import os
import json
from datetime import datetime, timedelta
import calendar
import threading
import time
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
from .definitions import BudgetExceeded, DailySoftBudgetExceeded, CONTEXT_MODE, AutonomousOutput, BatchStatus, JobCard

load_dotenv()
class xAI_Handler:  
    def set_budget(self, total_tokens: int):
        """
        Set the monthly token budget for this agent
        args: 
            total_tokens: int max_number of tokens for the month.
        """
        with self._lock:
            if total_tokens is not None and total_tokens <= 0:
                raise ValueError("token_budget must be positive")
            self.total_token_budget = total_tokens

    def get_budget(self) -> int: 
        with self._lock:
            return self.total_token_budget

    def set_execute(self, execute: bool): 
        """
        Sets the execute variable value so that the execute loop knows if it should continue running or not.
        Args: 
            execute: bool - the new value of the execute loop control variable. 
        """
        with self._lock:
            self._execute = execute

    def get_execute(self) -> bool:
        """
        Gets the current value for the execute loop's control variable.

        Returns: bool
        """
        with self._lock:
            return self._execute

    def set_personality(self, personality: AgentPersonality):
        """
        Set the personality for this agent

        Args: 
            personality: AgentPersonality defining DISC personality weights for this agent. 
        """
        with self._lock:
            self.personality = personality

    def add_tool(self, name: str, description: str, parameters: BaseModel, func: callable):
        """
        Add a new tool to the registry

        Args: 
            name: str name of the tool
            description: str description of the tool
            parameters Pydantic JSON schema of the tool. 
        """
        with self._lock:
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
        with self._lock:
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
            now = datetime.now()
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
        with self._lock:
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
                    chat.append(user("This is the last round of client side tool calls, if further client side calls are desired say 'if you would like me to continue this analysis say yes' otherwise make this your final response"))

            return {
                "content": final_content,
                "response_id": final_response.id,
                "usage": final_response.usage,
                "citations": getattr(final_response, "citations", None),
                "reasoning": getattr(final_response, "encrypted_content", None)
            }
    
    def execution_loop(self, test_mode: Optional[bool] = False):
        """
        Execution Loop - performs autonomous AI Agent tasks 
        """        
        with self._lock:
                        ## Copied from _precall_check() - Rather than have a precall check, we will estimate the next requests tokens compare the sum of batch_tokens and request_tokens with our daily and monthly limits, if we're under add the request to the batch, otherwise stop procesing requests and execute the batch. 
            monthly_used = self.memory.get_current_month_total()
            self._execute = True
            loop_counter = 0
            batch_requests = []
            batch_tokens = 0

            while self._execute:
                # Check for existing batch
                batch_status = self.memory.get_batch() 
                if batch_status is not None:
                    # Process Results
                    while batch_status.incomplete:
                        time.sleep(5)
                        batch_results = self.client.batch.get(batch_id=batch_status.batch_id)
                        print(f"Batch state: pending={batch_results.state.num_pending}, success={batch_results.state.num_success}, error={batch_results.state.num_error}")
                        batch_status.incomplete = batch_results.state.num_pending > 0 # If there are more results pending then the loop must continue. 
                        self.memory.add_message(session_id=batch_status.batch_session_id, role="user", content=f"Completed Requests: {batch_results.state.num_success}, Errors: {batch_results.state.num_error}, Pending: {batch_results.state.num_pending}", display=False)
                        
                        all_succeeded = []
                        all_failed = []
                        pagination_token = None
                        while True: 
                            page = self.client.batch.list_batch_results(
                                batch_id=batch_status.batch_id,
                                limit=100,
                                pagination_token=pagination_token
                            )
                            all_succeeded.extend(page.succeeded)
                            all_failed.extend(page.failed)
                            if page.pagination_token is None: 
                                break
                            pagination_token = page.pagination_token
                        print(f"Processing {len(all_succeeded)} succeeded results")
                        for result in all_succeeded:
                            
                            print(f"  Result for {result.batch_request_id}: tool_calls={len(result.response.tool_calls)}")
                            job_chat = self.client.chat.create(model=self.model, 
                                store_messages=True,
                                tools=self.tools,
                                batch_request_id=result.batch_request_id,
                                response_format=AutonomousOutput,
                                previous_response_id=result.response.response_id)
                            self.memory.add_message(session_id=self.memory.get_session_id_from_job_id(job_id=result.batch_request_id), # TODO: Create Method.
                                role=result.response.role,
                                content=result.response.content,
                                display=False,
                                cached_prompt_tokens=getattr(result.response.usage, 'cached_prompt_tokens', 0),
                                prompt_tokens=getattr(result.response.usage, 'prompt_tokens', 0),
                                reasoning_tokens=getattr(result.response.usage, 'reasoning_tokens', 0),
                                completion_tokens=getattr(result.response.usage, 'completion_tokens', 0),
                                server_side_tools_used=len(getattr(result.response.usage, 'server_side_tools_used', 0)),  
                                total_tokens=getattr(result.response.usage, 'total_tokens', 0)
                            )
                            # execute pending client side tool calls - There is no maximum number of rounds for autonomous mode
                            if len(result.response.tool_calls) > 0:
                                current_tool_round = self.memory.get_job_tool(job_id=result.batch_request_id)
                                if current_tool_round < 5:
                                    self.memory.increment_job_tool(job_id=result.batch_request_id)
                                    payload = 0
                                    for tc in result.response.tool_calls:
                                        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                                        func_name = tc.function.name
                                        try:
                                            if func_name not in self.tool_map: 
                                                job_chat.append(tool_result(f"Tool error: {func_name} is not mapped."))
                                            else: 
                                                result = self.tool_map[func_name](**args)  # your execute
                                                if payload + len(f"""\"tool_result": \"{result}\"""") < 24000000: 
                                                    payload = payload + len(f"""\"tool_result": \"{result}\"""")
                                                    batch_tokens = batch_tokens + self.client.tokenize.tokenize_text(text="result", model=self.model) # add next token estimate
                                                    job_chat.append(tool_result(json.dumps(result) if not isinstance(result, str) else result))
                                                else: 
                                                    result = f"tool_call_id: {tc.id} - {func_name}, result was too big to include in this turn. Please request it again. If you have three consequtive requests return the same error then report the error in your final output"
                                                    payload = payload + len(f"""\"tool_result\": \"{result}\"""")
                                                    batch_tokens = batch_tokens + self.client.tokenize.tokenize_text(text=result)
                                                    job_chat.append(tool_result(result=result, tool_call_id=tc.id))
                                        except Exception as e:
                                            job_chat.append(tool_result(f"Tool error: {e}"))
                                    
                                else:
                                    job_chat.append(user("You have exceeded 5 client side tool rounds. If an error has not occured make sure that your output reflects the necessity of further tool_calls next turn."))
                                job_tokens = 0
                                job_payload = 0
                                for message in job_chat.messages: 
                                    job_tokens += len(self.client.tokenize.tokenize_text(text=message.content, model=self.model))
                                    job_payload = len(message.content)
                                if job_payload > self.MAX_PAYLOAD: 
                                    # TODO: ADD MESSAGE 
                                    break
                                est_tokens =job_tokens * self.COMPLETION_MULTIPLIER + self.RESPONSE_BUFFER
                                if batch_tokens + est_tokens > (self.total_token_budget - monthly_used) * 0.9:
                                    break # TODO: ADD MESSAGE
                                batch_tokens += est_tokens
                                batch_requests.append(job_chat) # In the interest of rapid development, I did not add another budget check here. A budget check partway through a job would necessitate code that pauses job execution. For now the risk of a budget overrun is acceptable because of checks else where in the program. TODO: Add budget checks and task pauses 
                            else: 
                                # update job record
                                try: 
                                    job_output = AutonomousOutput.model_validate_json(result.response.content)
                                    print("Parsed successfully")
                                    self.memory.update_job(job_id=result.batch_request_id, job_output=job_output) 
                                except Exception as e:
                                    print(f"Parse failed: {str(e)}")
                                    self.memory.add_message(
                                        session_id=...,
                                        role="system",
                                        content=f"Structured output parse failed: {str(e)[:200]}\nRaw: {result.response.content[:300]}",
                                        display=False
                                    )
                                    self.memory.update_job(result.batch_request_id, status="blocked", clarification_needed=1) 

                        # TODO: Implement procedure for failed requests. 
                        
                        self.memory.upsert_batch(batch_status=batch_status)

                # Get more Jobs!
                loop_counter = loop_counter + 1
                # 1. Setup Agent
                system_prompt = self.personality.system_prompt + """
                    \n\n In this manifestation, you are an autonomous agent, diligently completing your assigned work.
                    \n - You will be given a JSON job card that describes the task and how to complete it. 
                    \n - It is not expected that you complete the entire job-card in one pass. As part of your response, you will update the job card with your progress so that the next call will know where to pick up from. 
                    \n - Your output will be a structured JSON response. Part of the updated-job-card field will give you descretional use for building your own structured output that will help future iterations complete the job. 
                    \n - You will also have an opportunity, with your final response, to provide feedback on how the task efficiency can be improved in the future.
                    \n\n Additionally, you have access to a global_context database table (key, value, tags) for facts across user interaction sessions and autonomous job completions. 
                    \n - Use the store_global_context tool to save or update useful facts and associtaions that would be useful in the future. 
                    \n - use the search_global_context tool to retrieve relevent stored items. 
                    \n - Store only concise, high-value information - never full transcripts.
                    \n - Prefer stable, predictable keys (e.g. "user_location", "preferred_unit_system").
                    \n - When storing data in the global_context, use the JSON tags to provide meaningful search terms and links to other facts.  
                    """
                job_data = self.memory.get_jobs()
                if len(job_data) > 0:
                    for job in job_data:
                        if len(batch_requests) < 100: # xai batch limit
                            payload = 0
                            if job.session_id is None or job.session_id == "": # session_id - if this breaks, then change put row_factory in memorystore and use column names. 
                                job.session_id = self.memory.start_session(title=job.title)
                            previous_response_id = self.memory.get_response_id(job.session_id)
                            if previous_response_id is not None: 
                                job_chat = self.client.chat.create(model=self.model, 
                                    store_messages=True, 
                                    tools=self.tools, 
                                    batch_request_id=job.job_id,
                                    response_format=AutonomousOutput,
                                    previous_response_id=previous_response_id)
                                job_chat.append(system(system_prompt))
                                payload = payload + len(f"""system: "{system_prompt}" """)
                                batch_requests = batch_requests + self.client.tokenize.tokenize_text(text=system_prompt, model=self.model)
                            else: 
                                job_chat = self.client.chat.create(model=self.model, 
                                    store_messages=True, 
                                    tools=self.tools,
                                    response_format=AutonomousOutput,
                                    batch_request_id=job.job_id)
                            if payload + len(f"""user: "{job.job_id}" """) < self.MAX_PAYLOAD: # Keep Payloads less than 24MB xai limit is 25MB this provides a 1MB buffer.
                                payload = payload + len(f"""user: "{job.job_card.model_dump_json}" """)
                                est_tokens = len(self.client.tokenize.tokenize_text(job.job_card.model_dump_json, self.model))
                                if previous_response_id is None:
                                    est_tokens += len(self.client.tokenize.tokenize_text(system_prompt, self.model))
                                est_tokens = int(est_tokens * self.COMPLETION_MULTIPLIER) + self.RESPONSE_BUFFER  # buffer for tools + response
                                if batch_tokens + est_tokens > (self.total_token_budget - monthly_used) * 0.9:
                                    break
                                batch_tokens += est_tokens
                                
                            else: 
                                # TODO: Add code that tells the user to reduce the size of the job card
                                self.memory.add_message(session_id=job.session_id, role="user", content="Current jobcard is to large to process", display=False)
                        else: 
                            break # If max batch requests are added stop processing jobs. 
                
                # Execute the batch
                if len(batch_requests) > 0:
                    last_send = self.memory.get_batch_send()
                    elapsed = (datetime.now() - last_send).total_seconds()
                    if elapsed < 30:
                        time.sleep(30 - elapsed) # Respect xai usage limits. 
                    
                    batch_session_id = self.memory.start_session(f"Batch Requests: {datetime.now().isoformat()} - Loop Count: {loop_counter}")
                    batch_id = self.client.batch.create(batch_name=f"{self.personality.name}'s Auto-Job Batch")
                    self.client.batch.add(batch_id=batch_id, batch_requests=batch_requests)            
                    self.memory.add_message(session_id=batch_session_id, role="user", content=f"Added {len(batch_requests)} requests for an estimated {batch_tokens} tokens to batch processing", display=False)
                    self.memory.upsert_batch(BatchStatus(batch_id=batch_id, session_id=batch_session_id))
                else: 
                    if test_mode:
                        time.sleep(10)
                        self._execute=False
                    else:
                        time.sleep(30) # Sleep 30s then look for new jobs. 
                    

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
        self._lock = threading.RLock()
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
    
        # CONSTANTS
        self.COMPLETION_MULTIPLIER = 1.5
        self.RESPONSE_BUFFER = 800
        self.MAX_PAYLOAD = 24000000