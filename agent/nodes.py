import json
from datetime import datetime
from typing import Literal

from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import RemoveMessage

from .prompts import COMPRESSION_PROMPT
from .prompts import SUMMARY_PROMPT
from .prompts import SYSTEM_PROMPT
from .state import AgentResponse
from .state import get_state_value
from .state import Plan
from .state import PlanningState
from .state import PlanVersion
from .state import PreservedContext
from utils.diff_generator import generate_plan_diff
from utils.token_counter import count_tokens
from utils.token_counter import should_compress


def get_llm():
    return ChatOpenAI(
        model="gpt-oss-120b", base_url="https://api.cerebras.ai/v1", temperature=0.7
    )


def format_plan_for_prompt(plan: Plan | None) -> str:
    if plan is None:
        return "No plan created yet."

    lines = [f"Title: {plan.title}", f"Version: {plan.version}", "Steps:"]
    for step in plan.steps:
        status_icon = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}
        lines.append(
            f"  {status_icon.get(step.status, '[ ]')} {step.step_number}. {step.title}"
        )
        if step.description:
            lines.append(f"      {step.description}")
    return "\n".join(lines)


def format_context_for_prompt(ctx: PreservedContext) -> str:
    parts = []
    if ctx.original_requirements:
        parts.append(f"Goal: {ctx.original_requirements}")
    if ctx.key_decisions:
        parts.append(f"Decisions: {', '.join(ctx.key_decisions)}")
    if ctx.constraints:
        parts.append(f"Constraints: {', '.join(ctx.constraints)}")
    if ctx.rejected_options:
        parts.append(f"Rejected: {', '.join(ctx.rejected_options)}")
    if ctx.clarifications_given:
        parts.append(f"Clarifications: {', '.join(ctx.clarifications_given)}")
    if ctx.important_context:
        parts.append(f"Important: {', '.join(ctx.important_context)}")
    return "\n".join(parts) if parts else "No context yet."


def compress_context_node(state: PlanningState) -> dict:
    messages = list(get_state_value(state, "messages"))
    preserved = get_state_value(state, "preserved_context")

    if len(messages) <= 6:
        return {}

    old_messages = messages[:-4]
    recent_messages = messages[-4:]

    conversation_text = "\n".join(
        [
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in old_messages
        ]
    )

    llm = get_llm()
    prompt = COMPRESSION_PROMPT.format(
        conversation=conversation_text,
        preserved_context=preserved.model_dump_json() if preserved else "{}",
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        data = json.loads(content.strip())

        new_preserved = PreservedContext(
            original_requirements=data.get("original_requirements")
            or preserved.original_requirements,
            key_decisions=list(
                dict.fromkeys(preserved.key_decisions + data.get("key_decisions", []))
            ),
            constraints=list(
                dict.fromkeys(preserved.constraints + data.get("constraints", []))
            ),
            rejected_options=list(
                dict.fromkeys(
                    preserved.rejected_options + data.get("rejected_options", [])
                )
            ),
            clarifications_given=list(
                dict.fromkeys(
                    preserved.clarifications_given
                    + data.get("clarifications_given", [])
                )
            ),
            important_context=list(
                dict.fromkeys(
                    preserved.important_context + data.get("important_context", [])
                )
            ),
        )

        summary = data.get("summary", "Previous conversation summarized.")
        summary_msg = SystemMessage(
            content=f"[Previous conversation summary: {summary}]"
        )

        # To preserve message order we must remove all messages, then add summary first + recent messages
        remove_all = [RemoveMessage(id=m.id) for m in messages]

        return {
            "messages": remove_all + [summary_msg] + recent_messages,
            "preserved_context": new_preserved,
            "conversation_summary": summary,
        }
    except Exception:
        # Fallback: remove oldest messages, keep recent 6
        if len(messages) > 6:
            remove_ops = [RemoveMessage(id=m.id) for m in messages[:-6]]
            return {"messages": remove_ops}
        return {}


def planning_agent_node(state: PlanningState) -> dict:
    messages = get_state_value(state, "messages")
    current_plan = get_state_value(state, "current_plan")
    preserved = get_state_value(state, "preserved_context")
    plan_versions = list(get_state_value(state, "plan_versions"))
    user_prefs = dict(get_state_value(state, "user_preferences"))
    conversation_summary = get_state_value(state, "conversation_summary")

    system_content = SYSTEM_PROMPT.format(
        context=format_context_for_prompt(preserved),
        current_plan=format_plan_for_prompt(current_plan),
        conversation_summary=conversation_summary or "None yet.",
    )

    llm = get_llm()

    schema_instruction = """
Respond with JSON in this format:
{
    "message": "your response to the user",
    "plan": null or {"title": "...", "steps": [{"step_number": 1, "title": "...", "description": "...", "status": "pending"}], "metadata": {}},
    "clarifying_questions": ["list any clarifying questions here"],
    "extracted_preferences": {},
    "extracted_constraints": [],
    "extracted_decisions": []
}
"""

    full_messages = [
        SystemMessage(content=system_content + "\n\n" + schema_instruction)
    ] + list(messages)

    response = llm.invoke(full_messages)
    content = response.content

    try:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        data = json.loads(content.strip())
        agent_response = AgentResponse(**data)
    except Exception:
        agent_response = AgentResponse(message=response.content)

    message_content = agent_response.message
    if agent_response.clarifying_questions:
        questions = "\n".join([f"- {q}" for q in agent_response.clarifying_questions])
        message_content = f"{message_content}\n\n{questions}"

    result: dict = {"messages": [AIMessage(content=message_content)]}

    if agent_response.plan:
        new_plan = Plan(
            title=agent_response.plan.title,
            steps=agent_response.plan.steps,
            metadata=agent_response.plan.metadata or {},
            version=(current_plan.version + 1) if current_plan else 1,
            created_at=current_plan.created_at if current_plan else datetime.now(),
            updated_at=datetime.now(),
        )
        if current_plan:
            old_dict = (
                current_plan.model_dump()
                if hasattr(current_plan, "model_dump")
                else current_plan
            )
            new_dict = new_plan.model_dump()
            changes = generate_plan_diff(old_dict, new_dict)

            plan_versions.append(
                PlanVersion(plan=current_plan, change_summary="\n".join(changes))
            )

        result["current_plan"] = new_plan
        result["plan_versions"] = plan_versions

    if agent_response.extracted_preferences:
        user_prefs.update(agent_response.extracted_preferences)
        result["user_preferences"] = user_prefs

    if agent_response.extracted_constraints or agent_response.extracted_decisions:
        new_preserved = PreservedContext(
            original_requirements=preserved.original_requirements,
            key_decisions=list(
                dict.fromkeys(
                    preserved.key_decisions + agent_response.extracted_decisions
                )
            ),
            constraints=list(
                dict.fromkeys(
                    preserved.constraints + agent_response.extracted_constraints
                )
            ),
            rejected_options=list(preserved.rejected_options),
            clarifications_given=list(preserved.clarifications_given),
            important_context=list(preserved.important_context),
        )
        result["preserved_context"] = new_preserved
    return result


def should_compress_check(state: PlanningState) -> Literal["compress", "agent"]:
    messages = get_state_value(state, "messages")
    plan = get_state_value(state, "current_plan")
    ctx = get_state_value(state, "preserved_context")

    plan_tokens = count_tokens(format_plan_for_prompt(plan)) if plan else 0
    ctx_tokens = count_tokens(format_context_for_prompt(ctx))

    if should_compress(messages, plan_tokens, ctx_tokens):
        return "compress"
    return "agent"


def generate_executive_summary(state: PlanningState) -> str:
    """Generate an executive summary of the entire planning conversation."""
    messages = get_state_value(state, "messages")
    current_plan = get_state_value(state, "current_plan")
    preserved = get_state_value(state, "preserved_context")

    # Only format last 6 messages for context
    recent = messages[-6:] if len(messages) > 6 else messages
    recent_text = "\n".join(
        [
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in recent
        ]
    )

    prompt = SUMMARY_PROMPT.format(
        preserved_context=format_context_for_prompt(preserved),
        current_plan=format_plan_for_prompt(current_plan),
        recent_messages=recent_text,
    )

    llm = get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content
