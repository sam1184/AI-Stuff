"""
Lab 3: Tool Chaining
---------------------
4 tools. Claude picks the order.
Simulates a research agent: search → read → summarise → save.
Run: python lab3_chaining.py
"""
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

notes_db: dict = {}

# ── TOOL IMPLEMENTATIONS ──────────────────────────────────────────────────────
def web_search(query: str) -> str:
    # Simulated. Replace with a real search API (Brave, SerpAPI, etc.)
    print(f"  [SEARCH] Query: {query}")
    return f"""Results for: {query}
1. [Climeworks] Direct air capture company. Latest round: $650M Series D.
2. [Northvolt] Battery technology. European gigafactory. 2024 challenges.
3. [Form Energy] Iron-air batteries for grid storage. Long-duration focus.
4. [Twelve] CO2-to-chemicals company. Backed by major airlines.
"""

def read_page(url_or_company: str) -> str:
    print(f"  [READ] Reading: {url_or_company}")
    data = {
        "climeworks": "Climeworks captures CO2 directly from air. Swiss company. $650M raised. Operating first commercial plant in Iceland.",
        "northvolt": "Swedish battery maker. Built gigafactory. Filed for bankruptcy protection in 2024 despite strong early funding.",
        "form energy": "Makes iron-air batteries. 100-hour storage duration. Targeting grid-scale deployment by 2025.",
        "twelve": "Converts CO2 into jet fuel and chemicals using electricity. $645M raised. United partnership signed.",
    }
    key = url_or_company.lower().replace(".", "").strip()
    for k, v in data.items():
        if k in key:
            return v
    return f"Page content for {url_or_company}: General climate tech company information."

def save_note(title: str, content: str) -> str:
    notes_db[title] = content
    print(f"  [SAVE] Saved note: '{title}'")
    return f"Saved: {title}"

TOOLS = [
    {"name": "web_search",
     "description": "Search the web for recent information on a topic.",
     "input_schema": {"type": "object",
         "properties": {"query": {"type": "string"}},
         "required": ["query"]}},
    {"name": "read_page",
     "description": "Read detailed content about a company or URL.",
     "input_schema": {"type": "object",
         "properties": {"url_or_company": {"type": "string"}},
         "required": ["url_or_company"]}},
    {"name": "save_note",
     "description": "Save a research note with a title and content.",
     "input_schema": {"type": "object",
         "properties": {
             "title": {"type": "string"},
             "content": {"type": "string"}},
         "required": ["title", "content"]}}
]

def execute_tool(name, inputs):
    if name == "web_search": return web_search(**inputs)
    if name == "read_page": return read_page(**inputs)
    if name == "save_note": return save_note(**inputs)
    return "Unknown"

def run_research_agent(task: str) -> str:
    messages = [{"role": "user", "content": task}]
    print(f"\n── Task: {task}\n")

    for turn in range(15):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system="You are a research agent. For research tasks: search first, read each result, then save a well-structured summary note. Be thorough.",
            tools=TOOLS,
            messages=messages
        )

        print(f"[Turn {turn+1}] {response.stop_reason}")

        if response.stop_reason == "end_turn":
            answer = next((b.text for b in response.content if hasattr(b, "text")), "")
            print(f"\n── Done.\n── Saved notes: {list(notes_db.keys())}")
            return answer

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = execute_tool(block.name, block.input)
                tool_results.append({"type": "tool_result",
                    "tool_use_id": block.id, "content": result})
        messages.append({"role": "user", "content": tool_results})

if __name__ == "__main__":
    run_research_agent("Research the top climate tech startups from 2024. Read about each one and save a comprehensive summary note.")
