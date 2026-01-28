from datetime import datetime
from typing import Annotated
from typing import Any
from typing import Literal
from typing import NotRequired

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from pydantic import Field
from typing_extensions import TypedDict


class PlanStep(BaseModel):
    step_number: int
    title: str
    description: str = ""
    status: Literal["pending", "in_progress", "completed"] = "pending"


class PlanDraft(BaseModel):
    title: str
    steps: list[PlanStep] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class Plan(BaseModel):
    title: str
    steps: list[PlanStep] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    version: int = 1
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class PlanVersion(BaseModel):
    plan: Plan
    timestamp: datetime = Field(default_factory=datetime.now)
    change_summary: str = ""


class PreservedContext(BaseModel):
    original_requirements: str = ""
    key_decisions: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    rejected_options: list[str] = Field(default_factory=list)
    clarifications_given: list[str] = Field(default_factory=list)
    important_context: list[str] = Field(default_factory=list)


class PlanningState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_plan: NotRequired[Plan | None]
    plan_versions: NotRequired[list[PlanVersion]]
    user_preferences: NotRequired[dict[str, Any]]
    conversation_summary: NotRequired[str]
    preserved_context: NotRequired[PreservedContext]


STATE_DEFAULTS: dict[str, Any] = {
    "current_plan": None,
    "plan_versions": [],
    "user_preferences": {},
    "conversation_summary": "",
    "preserved_context": PreservedContext(),
}


def get_state_value(state: PlanningState, key: str) -> Any:
    """Get state value with proper default. Single source of truth."""
    if key == "messages":
        return state.get("messages", [])

    default = STATE_DEFAULTS.get(key)
    value = state.get(key)

    if value is None and default is not None:
        # For mutable defaults, return a fresh copy
        if isinstance(default, list):
            return []
        if isinstance(default, dict):
            return {}
        if isinstance(default, PreservedContext):
            return PreservedContext()
        return default
    return value


class AgentResponse(BaseModel):
    message: str
    plan: PlanDraft | None = None
    clarifying_questions: list[str] = Field(default_factory=list)
    extracted_preferences: dict[str, Any] = Field(default_factory=dict)
    extracted_constraints: list[str] = Field(default_factory=list)
    extracted_decisions: list[str] = Field(default_factory=list)
