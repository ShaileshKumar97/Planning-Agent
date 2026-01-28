import tiktoken
from langchain_core.messages import BaseMessage

ENCODING = tiktoken.get_encoding("cl100k_base")

# token budget constants
SYSTEM_PROMPT_BUFFER = 800
PLAN_BUFFER = 500
PRESERVED_CONTEXT_BUFFER = 500
RESPONSE_RESERVE = 1500
TOTAL_LIMIT = 8000
COMPRESSION_THRESHOLD = (
    TOTAL_LIMIT
    - SYSTEM_PROMPT_BUFFER
    - PLAN_BUFFER
    - PRESERVED_CONTEXT_BUFFER
    - RESPONSE_RESERVE
)


def count_tokens(text: str) -> int:
    """Count tokens in a string."""
    return len(ENCODING.encode(text))


def estimate_tokens(messages: list[BaseMessage]) -> int:
    """Estimate token count for a list of messages."""
    total = 0
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        total += count_tokens(content)
        total += 4  # overhead per message
    return total


def should_compress(
    messages: list[BaseMessage], plan_tokens: int = 0, context_tokens: int = 0
) -> bool:
    """Check if context compression is needed."""
    message_tokens = estimate_tokens(messages)
    total = message_tokens + plan_tokens + context_tokens + SYSTEM_PROMPT_BUFFER
    return total > COMPRESSION_THRESHOLD


def get_token_usage(
    messages: list[BaseMessage], plan_tokens: int = 0, context_tokens: int = 0
) -> dict:
    """Get current token usage breakdown."""
    message_tokens = estimate_tokens(messages)
    return {
        "messages": message_tokens,
        "plan": plan_tokens,
        "context": context_tokens,
        "system": SYSTEM_PROMPT_BUFFER,
        "total": message_tokens + plan_tokens + context_tokens + SYSTEM_PROMPT_BUFFER,
        "limit": TOTAL_LIMIT,
        "threshold": COMPRESSION_THRESHOLD,
    }
