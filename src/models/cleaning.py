from pydantic import BaseModel
from enum import Enum
from typing import Optional


class CleaningActionType(str, Enum):
    STANDARDIZE_VALUES = "standardize_values"
    FILL_NULLS = "fill_nulls"
    REMOVE_DUPLICATES = "remove_duplicates"
    FIX_TYPES = "fix_types"
    SEPARATE_COLUMN = "separate_column"   # e.g. split returns from sales
    RENAME_COLUMN = "rename_column"


class CleaningAction(BaseModel):
    action_type: CleaningActionType
    target_column: str
    description: str                     # human-readable explanation
    is_destructive: bool                 # if True, requires user confirmation
    params: dict = {}                    # action-specific params
    rows_affected: int = 0
    reversible: bool = True


class CleaningPlan(BaseModel):
    table_name: str
    actions: list[CleaningAction]
    estimated_quality_improvement: float  # expected new quality score
    requires_user_approval: bool          # True if any destructive action


class CleaningLog(BaseModel):
    table_name: str
    original_view: str                    # name of original table/view
    cleaned_view: str                     # name of cleaned view
    actions_applied: list[CleaningAction]
    rows_before: int
    rows_after: int
