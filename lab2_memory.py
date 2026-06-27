"""
Lab 2: Stateful Agent with Memory
----------------------------------
The agent has two memory tools: remember() and recall().
It decides when to save facts and when to look them up.
Run: python lab2_memory.py
"""
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

# ── MEMORY STORE ──────────────────────────────────────────────────────────────
# Simple dict. In production: use Redis, DynamoDB, or a vector store.
# The key insight: the memory is EXTERNAL to Claude.
# Claude's context window is temporary. The memory store is persistent.
memory_store: dict = {}

def remember(key: str, value: str) -> str:
    memory_store[key] = value
    print(f"  [MEMORY] Saved: {key} = {value}")
    return f"Remembered: {key} = {value}"

def recall(key: str) -> str:
    value = memory_store.get(key)
    print(f"  [MEMORY] Recalled: {key} = {value}")
    if value:
        return f"{key}: {value}"
    return f"Nothing stored for '{key}'"

def list_memories() -> str:
    if not memory_store:
        return "Memory is empty."
    return "\n".join([f"{k}: {v}" for k, v in memory_store.items()])

TOOLS = [
    {"name": "remember", "description": "Save a fact to memory by key.",
     "input_schema": {"type": "object", "properties": {
         "key": {"type": "string"}, "value": {"type": "string"}},
         "required": ["key", "value"]}},
    {"name": "recall", "description": "Look up a fact from memory by key.",
     "input_schema": {"type": "object", "properties": {
         "key": {"type": "string"}}, "required": ["key"]}},
    {"name": "list_memories", "description": "List all stored memories.",
     "input_schema": {"type": "object", "properties": {}}}
]

def execute_tool(name: str, inputs: dict) -> str:
    if name == "remember": return remember(**inputs)
    elif name == "recall": return recall(**inputs)
    elif name == "list_memories": return list_memories()
    return "Unknown tool"

def agent_turn(messages: list, user_input: str) -> str:
    """Single turn of the agent — adds the user message and runs the loop."""
    messages.append({"role": "user", "content": user_input})
    print(f"\n── User: {user_input}")

    for _ in range(8):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system="You are a helpful project manager assistant with access to a memory store. Always use the remember tool to save important facts, and the recall tool before answering questions about past information.",
            tools=TOOLS,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            answer = next((b.text for b in response.content if hasattr(b, "text")), "")
            messages.append({"role": "assistant", "content": response.content})
            print(f"── Agent: {answer}")
            return answer

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = execute_tool(block.name, block.input)
                tool_results.append({"type": "tool_result",
                    "tool_use_id": block.id, "content": result})
        messages.append({"role": "user", "content": tool_results})

    return "Max turns reached"

# ── RUN THE DEMO ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("LAB 2 — STATEFUL AGENT DEMO")
    print("=" * 60)

    conversation = []  # shared message history across all turns

    # Turn 1: give it information to remember
    agent_turn(conversation, "Project Alpha has a budget of $50,000 and is due in December.")

    # Turn 2: ask something requiring recall
    agent_turn(conversation, "What's the budget for Project Alpha?")

    # Turn 3: update the data
    agent_turn(conversation, "The budget for Project Alpha has been increased to $75,000.")

    # Turn 4: check it remembered the update
    agent_turn(conversation, "What is the full current status of Project Alpha?")
