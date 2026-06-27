"""
Lab 4: Multi-Agent System
--------------------------
Supervisor delegates to Research Agent and Writing Agent.
Each agent has its own system prompt and tools.
Supervisor aggregates and delivers the final output.
Run: python lab4_multiagent.py
"""
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

# ── SHARED TOOL IMPLEMENTATIONS ───────────────────────────────────────────────
def web_search(query: str) -> str:
    print(f"    [SEARCH] {query}")
    return f"""Search results for: {query}
1. AI investment hit $200B globally in 2024 (source: PitchBook)
2. GPT-4o, Claude 3.5, Gemini Ultra launched — multimodal became standard
3. Agentic AI moved from research to production deployment at scale
4. EU AI Act came into force — compliance deadline 2026
5. 78% of Fortune 500 companies now have AI deployed in production"""

def read_source(topic: str) -> str:
    print(f"    [READ] {topic}")
    data = {
        "investment": "2024 AI investment: $200B globally. US leads at $85B. Healthcare AI up 140%.",
        "models": "2024 model releases: GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro. Context windows extended to 1M tokens.",
        "regulation": "EU AI Act signed June 2024. High-risk AI systems face mandatory audits. US voluntary framework published.",
        "enterprise": "Enterprise AI deployment up 340% YoY. Main use cases: customer service, code generation, data analysis.",
    }
    for k, v in data.items():
        if k in topic.lower():
            return v
    return f"General AI trend data for {topic}: significant growth in capabilities and adoption."

# ── RESEARCH AGENT ────────────────────────────────────────────────────────────
RESEARCH_TOOLS = [
    {"name": "web_search", "description": "Search for recent facts and statistics.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}},
     "required": ["query"]}},
    {"name": "read_source", "description": "Get detailed data on a specific topic: investment, models, regulation, or enterprise.",
     "input_schema": {"type": "object", "properties": {"topic": {"type": "string"}},
     "required": ["topic"]}}
]

def run_research_agent(research_task: str) -> str:
    print("\n  [RESEARCH AGENT] Starting...")
    messages = [{"role": "user", "content": research_task}]

    for _ in range(10):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system="You are a specialist research agent. Gather comprehensive data using your tools. Return a structured summary of all facts found — include numbers, dates, and specific details.",
            tools=RESEARCH_TOOLS,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            result = next((b.text for b in response.content if hasattr(b, "text")), "")
            print(f"  [RESEARCH AGENT] Done. {len(result)} chars of data.")
            return result

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = web_search(**block.input) if block.name == "web_search" else read_source(**block.input)
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
        messages.append({"role": "user", "content": tool_results})

    return "Research incomplete."

# ── WRITING AGENT ─────────────────────────────────────────────────────────────
def run_writing_agent(topic: str, research_data: str) -> str:
    print("\n  [WRITING AGENT] Starting...")
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system="You are a specialist writing agent. Transform research data into clear, engaging, well-structured reports. Use headers, include key stats, and make it scannable. Avoid jargon.",
        messages=[{
            "role": "user",
            "content": f"Write a professional report on: {topic}\n\nResearch data:\n{research_data}\n\nFormat: executive summary, key trends (3-5), numbers + evidence, brief outlook."
        }]
    )
    result = response.content[0].text
    print(f"  [WRITING AGENT] Done. Report: {len(result)} chars.")
    return result

# ── SUPERVISOR ────────────────────────────────────────────────────────────────
def run_supervisor(user_request: str) -> str:
    print(f"\n{'='*60}")
    print("SUPERVISOR: Breaking down the task...")
    print(f"Task: {user_request}")
    print(f"{'='*60}")

    # Step 1: Supervisor plans the delegation
    plan_response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system="You are a supervisor agent. Given a task, output ONLY two lines: RESEARCH_TASK: [what to research] WRITING_TASK: [what to write]",
        messages=[{"role": "user", "content": user_request}]
    )
    plan = plan_response.content[0].text
    print(f"\nSupervisor plan:\n{plan}")

    # Parse the plan (simple split)
    research_task = "Research AI developments in 2024-2025"
    for line in plan.split("\n"):
        if line.startswith("RESEARCH_TASK:"):
            research_task = line.replace("RESEARCH_TASK:", "").strip()

    # Step 2: Delegate to Research Agent
    print(f"\nSUPERVISOR → Research Agent: '{research_task}'")
    research_data = run_research_agent(research_task)

    # Step 3: Delegate to Writing Agent with the research data
    print(f"\nSUPERVISOR → Writing Agent: Turn research into report")
    report = run_writing_agent(user_request, research_data)

    # Step 4: Supervisor reviews and delivers
    final_response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        system="You are a supervisor. Review the report and add a one-sentence quality assessment at the top, then output the full report.",
        messages=[{"role": "user", "content": f"Review and deliver:\n\n{report}"}]
    )

    return final_response.content[0].text

# ── RUN IT ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = run_supervisor("Write a concise report on the state of AI in 2024 and what it means for enterprise teams.")
    print(f"\n{'='*60}")
    print("FINAL DELIVERED REPORT:")
    print(f"{'='*60}")
    print(result)
