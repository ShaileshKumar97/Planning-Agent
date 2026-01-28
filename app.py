import uuid

import streamlit as st
from dotenv import load_dotenv

from agent.graph import create_graph
from agent.graph import get_conversation_state
from agent.graph import get_response
from agent.nodes import generate_executive_summary
from agent.state import Plan
from utils.diff_generator import generate_plan_diff
from utils.token_counter import get_token_usage

load_dotenv()

st.set_page_config(page_title="Planning Agent", layout="wide")


def init_session():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "graph" not in st.session_state:
        st.session_state.graph = create_graph()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_plan" not in st.session_state:
        st.session_state.current_plan = None
    if "recent_changes" not in st.session_state:
        st.session_state.recent_changes = []
    if "preserved_context" not in st.session_state:
        st.session_state.preserved_context = None
    if "executive_summary" not in st.session_state:
        st.session_state.executive_summary = ""


def format_plan(plan: Plan) -> str:
    if not plan:
        return "No plan created yet."

    lines = [f"**{plan.title}**", f"*Version {plan.version}*", ""]
    for step in plan.steps:
        status = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}
        lines.append(
            f"{status.get(step.status, '[ ]')} **{step.step_number}. {step.title}**"
        )
        if step.description:
            lines.append(f"   {step.description}")
        lines.append("")
    return "\n".join(lines)


def sidebar():
    with st.sidebar:
        st.header("Current Plan")

        if st.session_state.current_plan:
            plan = st.session_state.current_plan
            st.markdown(format_plan(plan))
        else:
            st.info("No plan created yet. Start a conversation to create one.")

        st.divider()

        # recent changes
        st.subheader("Recent Changes")
        if st.session_state.recent_changes:
            for change in st.session_state.recent_changes[-5:]:
                st.text(change)
        else:
            st.text("No changes yet")

        st.divider()

        # session info
        st.subheader("Session Info")
        msg_count = len(st.session_state.messages)
        st.text(f"Turns: {msg_count // 2}")

        # token usage
        state = get_conversation_state(
            st.session_state.graph, st.session_state.thread_id
        )
        if state and state.get("messages"):
            usage = get_token_usage(state["messages"])
            pct = (usage["total"] / usage["limit"]) * 100
            st.text(f"Context: {usage['total']:,} / {usage['limit']:,}")
            st.progress(min(pct / 100, 1.0))

        st.divider()

        # preserved context
        with st.expander("Preserved Context"):
            ctx = st.session_state.preserved_context
            if ctx:
                if ctx.original_requirements:
                    st.text(f"Goal: {ctx.original_requirements}")
                if ctx.key_decisions:
                    st.text("Decisions:")
                    for d in ctx.key_decisions:
                        st.text(f"  - {d}")
                if ctx.constraints:
                    st.text("Constraints:")
                    for c in ctx.constraints:
                        st.text(f"  - {c}")
                if ctx.rejected_options:
                    st.text("Rejected:")
                    for r in ctx.rejected_options:
                        st.text(f"  - {r}")
            else:
                st.text("No context preserved yet")

        st.divider()

        # executive summary
        if st.button("Generate Executive Summary"):
            if len(st.session_state.messages) > 0:
                with st.spinner("Generating summary..."):
                    state = get_conversation_state(
                        st.session_state.graph, st.session_state.thread_id
                    )
                    if state:
                        summary = generate_executive_summary(state)
                        st.session_state.executive_summary = summary
            else:
                st.warning("Start a conversation first!")

        if st.session_state.executive_summary:
            with st.expander("Executive Summary", expanded=False):
                st.markdown(st.session_state.executive_summary)

        st.divider()

        if st.button("New Conversation"):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.current_plan = None
            st.session_state.recent_changes = []
            st.session_state.preserved_context = None
            st.session_state.executive_summary = ""
            st.rerun()


def chat():
    st.header("Planning Agent")

    # show messages
    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role):
            st.markdown(msg["content"])

    # input
    if prompt := st.chat_input("What would you like to plan?"):
        # add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = get_response(
                    st.session_state.graph, prompt, st.session_state.thread_id
                )

                # extract response
                messages = result.get("messages", [])
                if messages:
                    response_content = messages[-1].content
                else:
                    response_content = "I'm sorry, I couldn't process that request."

                st.markdown(response_content)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response_content}
                )

                # update plan if changed
                new_plan = result.get("current_plan")
                if new_plan:
                    old_plan = st.session_state.current_plan
                    if old_plan:
                        changes = generate_plan_diff(
                            old_plan.model_dump(), new_plan.model_dump()
                        )
                        st.session_state.recent_changes.extend(changes)
                    st.session_state.current_plan = new_plan

                # update preserved context
                ctx = result.get("preserved_context")
                if ctx:
                    st.session_state.preserved_context = ctx

        st.rerun()


def main():
    init_session()

    col1, col2 = st.columns([2, 1])

    with col1:
        chat()

    with col2:
        sidebar()


if __name__ == "__main__":
    main()
