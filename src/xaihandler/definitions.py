# definitions.py (full replacement)
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Dict, Any, Callable, List, Optional

#CHANGE LOG: Removed the tool definitions as it duplicates functionality that is in xai_sdk

class BudgetExceeded(Exception):
    """Raised when monthly token budget would be exceeded."""
    pass
class DailySoftBudgetExceeded(Exception): 
    """Raised when daily token budget would be exceed"""
    pass 

class CONTEXT_MODE(Enum):
    CONVERSATIONAL = "conversational"
    ANALYSIS = "analysis"
    AUTONOMOUS = "autonomous"

class JobCard(BaseModel): 
    job_title: str = Field(..., description="Title of the Job")
    # Task Collection
    # Resource Collection - for instance list of documents for parsing

class JOB_STATUS(Enum): # CHANGES HERE NEED TO BE REFLECTED IN THE DATABASE ENUM
    PENDING = "pending"
    IN_PROGRESS = "in progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
class Job(BaseModel):
    """To replace the base SQL Tuples in get_jobs"""
    job_id: str = Field(..., description="Job Id") 
    title: str = Field(..., description="Title of the Job")
    estimated_tokens: Optional[int] = Field(..., description="Estimated Job Completion Tokens")
    parent_job_id: Optional[str] = Field(..., description="job_id of the parent job")
    priority: Optional[int] = Field(..., description="Job Priority")
    status: Optional[JOB_STATUS] = Field(..., description="Current Job Status")
    session_id: Optional[str] = Field(..., description="session_id for tracking in messages and other locaitons")
    progress: Optional[float] = Field(..., description="Progress from 0.0 to 1.0"), 
    clarification_needed: Optional[bool] = Field(..., description="If the user has to clarify something before execution can continue")
    job_card: JobCard = Field(..., description="Current Job Card for the next round of execution")
class AutonomousOutput(BaseModel):
    action: str = Field(..., description="list of actions from the job_card that was just completed.")
    next_task: str = Field(..., description="The next task on the job_card to be completed")
    status: JOB_STATUS = Field(..., description="Current status of the job")
    progress: float = Field(..., description="Current progress from 0.0 to 1.0")
    user_message: str = Field(..., description="Questions to the user for clarification")
    reasoning_summary: str = Field(..., description="Any reasoning that lead to this result")
    result: str = Field(..., description="Summary of the work completed this round")
    clarification_needed: bool = Field(default=False, description="Set User Interaction Flag and begin process for getting user input")
    job_card: JobCard = Field(..., description="Updated Job Card for the next round")

class BatchStatus(BaseModel):
    batch_id: str = Field(..., description="batch_id issued by xai")
    session_id: str = Field(..., description="session_id for tracking in memorystore")
    batch_send: datetime = Field(..., description="datetime the batch was sent to xai for processing, to prevent sending multiple batches too close together")
    incomplete: bool = Field(..., description="if the batch has not returned all it's results yet this should be true.")
        