"""
Lab 1: Your First Agent Loop
----------------------------
Claude decides which tools to call. You execute them.
The loop runs until Claude produces a final text answer.
Run: python lab1_agent.py
"""
import anthropic
from dotenv import load_dotenv

load_dotenv(override=True)
client = anthropic.Anthropic()

# ── TOOLS ────────────────────────────────────────────────────────────────────
# We define two tools and give them to Claude. Claude never calls these
# directly — it outputs a tool_use block, and WE execute the function.
# This is a critical distinction. Claude decides WHAT. We decide HOW.

def get_weather(city: str) -> str:
    # Simulated — replace with a real API if you want
    weather_data = {
        "london": "18°C, partly cloudy",
        "new york": "24°C, sunny",
        "tokyo": "29°C, humid",
    }
    return weather_data.get(city.lower(), f"No data for {city}")

def calculate(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

# Tool schemas — Claude reads these to know what tools exist and how to call them
TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city. Returns temperature and conditions.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"]
        }
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression. Example: '18 * 5'",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]
        }
    }
]

# ── TOOL EXECUTOR ─────────────────────────────────────────────────────────────
def execute_tool(name: str, inputs: dict) -> str:
    print(f"  → Executing: {name}({inputs})")
    if name == "get_weather":
        return get_weather(**inputs)
    elif name == "calculate":
        return calculate(**inputs)
    return "Unknown tool"

# ── THE AGENT LOOP ────────────────────────────────────────────────────────────
# This is the core. It's ~20 lines. That's all an agent loop is.
# 1. Call Claude with the current messages
# 2. If Claude wants a tool → execute it → append result → go back to step 1
# 3. If Claude gives a text response → that's the final answer → stop

def run_agent(user_question: str, max_turns: int = 10) -> str:
    messages = [{"role": "user", "content": user_question}]

    print(f"\n── Question: {user_question}")
    print("── Starting agent loop...\n")

    for turn in range(max_turns):
        print(f"[Turn {turn + 1}] Calling Claude...")

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages
        )

        # OBSERVE: what did Claude return?
        print(f"  Stop reason: {response.stop_reason}")

        if response.stop_reason == "end_turn":
            # Claude has a final answer — extract and return it
            for block in response.content:
                if hasattr(block, "text"):
                    print(f"\n── Final Answer: {block.text}")
                    return block.text

        elif response.stop_reason == "tool_use":
            # Claude wants to use a tool — we execute it
            # First: append Claude's response to the message history
            messages.append({"role": "assistant", "content": response.content})

            # Then: execute each tool Claude requested
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    print(f"  ← Result: {result}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            # Append tool results — Claude will observe these next turn
            messages.append({"role": "user", "content": tool_results})

    return "Max turns reached."

# ── RUN IT ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # This question requires TWO tools in sequence.
    # Claude figures out the order — you don't tell it.
    run_agent("What is the temperature in London today, and what is that number multiplied by 5?")
