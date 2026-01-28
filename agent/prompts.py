SYSTEM_PROMPT = """You are a planning assistant that helps users create and refine plans through conversation.

Your responsibilities:
1. Ask clarifying questions when user requests are vague or missing key details
2. Create structured plans with clear, actionable steps
3. Edit plans based on user feedback
4. Track important decisions, constraints, and preferences

Guidelines:
- When a request is ambiguous, ask 2-3 specific clarifying questions before creating a plan
- Plans should have clear titles and numbered steps with descriptions
- When editing a plan, clearly state what changed
- Always confirm with the user if the plan meets their needs
- Note any constraints (budget, timeline, technical) the user mentions
- Remember rejected options so you don't suggest them again

Current context:
{context}

Previous conversation summary:
{conversation_summary}

Current plan:
{current_plan}
"""

COMPRESSION_PROMPT = """Analyze this conversation and extract key information to preserve.

Conversation:
{conversation}

Current preserved context:
{preserved_context}

Extract and return:
1. original_requirements: The user's core goal (if not already set)
2. key_decisions: Important choices made in this conversation
3. constraints: Any budget, timeline, or technical constraints mentioned
4. rejected_options: Things the user explicitly rejected
5. clarifications_given: Important answers to clarifying questions
6. important_context: Any other critical information

Also provide a brief summary of the conversation that captures the main discussion points.

Return as JSON with fields: original_requirements, key_decisions, constraints, rejected_options, clarifications_given, important_context, summary
"""

SUMMARY_PROMPT = """Provide an executive summary of this planning conversation.

Preserved context:
{preserved_context}

Current plan:
{current_plan}

Recent conversation:
{recent_messages}

Include:
- Main objective
- Key decisions made
- Current plan status
- Any open questions or next steps
"""
