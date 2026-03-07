# definitions.py (full replacement)
from enum import Enum
from pydantic import BaseModel
from typing import Dict, Any, Callable, List

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