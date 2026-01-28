# Planning Agent

A conversational planning agent built with LangGraph and Streamlit that helps users create and refine plans through natural dialogue.

## Features

- Multi-turn conversations with context preservation
- Asks clarifying questions when requests are not clear
- Structured plan creation and editing
- Plan versioning and change tracking
- Context compression when approaching token limit (taking 8k as limit)
- Preservation of key details (requirements, decisions, constraints)
- Executive summary generation
- Diff visibility for plan changes

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create a .env file in the project root
echo "CEREBRAS_API_KEY=your_api_key_here" > .env
```

4. Run the application:
```bash
streamlit run app.py
```

## Project Structure

```
planning-agent/
├── app.py              # Streamlit UI
├── agent/
│   ├── graph.py        # LangGraph definition
│   ├── state.py        # State + Pydantic models
│   ├── nodes.py        # Graph nodes (compress, agent)
│   └── prompts.py      # System and compression prompts
├── utils/
│   ├── token_counter.py
│   └── diff_generator.py
├── requirements.txt
└── README.md
```

## Design Decisions

### Graph Flow (Single Agent Architecture)
```
START → conditional check → compress (if needed) → agent → END
```
Checks token count at the start and routes to compression node only when threshold (4700 tokens is exceeded. This avoids unnecessary LLM calls for summarization.

### State Management
- `MemorySaver` checkpointer for conversation persistence
- Thread-based isolation for multiple conversations
- `add_messages` reducer with [`RemoveMessage`](https://docs.langchain.com/oss/javascript/langchain/short-term-memory#delete-messages) for proper message handling
- Pydantic models for structured plan data

### Context Compression
Simulates 8K token limit. When threshold is hit:
1. Keep last 4 messages (2 turns) for continuity
2. Summarize older messages via LLM
3. Extract key info (requirements, decisions, constraints) to `PreservedContext`
4. Remove old messages and rebuild: summary first, then recent messages to maintain message order

Token budget breakdown: This is an estimate
- System prompt: 800, Plan: 500, Context: 500, Response: 1500
- Compression threshold: 4700 tokens

### Plan Versioning
Each edit increments version and saves old plan with change summary. Simple diff generator produces human-readable output (`+ Added step 3`).

### Executive Summary
Button in sidebar generates summary of entire conversation using preserved context, current plan, and recent messages when needed.

## Usage

1. Start a conversation by describing what you want to plan
2. The agent will ask clarifying questions if needed
3. Once requirements are clear, a plan is created
4. Review plan in sidebar, request changes via chat
5. Track diffs in "Recent Changes" section
6. Click "Generate Executive Summary" for conversation overview
7. View preserved context, token usage and session info in the sidebar

## Limitations:
- Free LLM (Cerebras: gpt-oss-120b) sometimes returns malformed JSON, breaking structured output parsing
- Diffs shown in the sidebar after edits complete, not in real-time
- UI can be updated with more user friendly format

## Dependencies

- langgraph: Graph-based agent framework
- langchain-openai: OpenAI Style API integration
- streamlit: Web UI
- pydantic: Data validation
- tiktoken: Token counting
