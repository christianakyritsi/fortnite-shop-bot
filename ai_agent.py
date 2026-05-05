import os
import json
from groq import Groq
from dotenv import load_dotenv

import tools

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """
You are a Fortnite Shop Analyst, an AI assistant inside a Discord server.

You help users understand Fortnite item rarity, shop history, and today's shop.

You have access to tools that read a real shop history dataset spanning 2017 to today.
ALWAYS use the tools when answering factual questions about specific items, today's shop, or rare returns.
Never guess or make up numbers. If a tool returns no data, say so.

Keep replies short, friendly, and Discord-friendly. No markdown headings.
"""

# Tool schemas — what the LLM sees
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_item",
            "description": "Look up shop history for a Fortnite cosmetic by name. Returns rarity, type, total appearances, first/last seen dates, and days since last appearance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The cosmetic name (fuzzy match supported)."}
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_todays_shop",
            "description": "Get the list of items in today's Fortnite shop, with scarcity info (days since last seen) for each.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_rare_returns",
            "description": "Find items in today's shop that have been gone for a long time. Useful for highlighting rare returns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_days": {"type": "integer", "description": "Minimum days since last seen. Default 100."}
                },
            },
        },
    },
]

# Map tool names to actual Python functions
TOOL_REGISTRY = {
    "lookup_item": tools.lookup_item,
    "get_todays_shop": tools.get_todays_shop,
    "find_rare_returns": tools.find_rare_returns,
}


def ask_llama(user_message: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # Loop: LLM may want to call tools, possibly multiple times
    for _ in range(5):  # safety cap
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.3,
            max_tokens=600,
        )
        msg = completion.choices[0].message

        # If LLM didn't call any tools, we're done
        if not msg.tool_calls:
            return msg.content or "Sorry, I couldn't generate a response."

        # Otherwise: execute each tool call and feed results back
        messages.append(msg)  # the assistant's tool-call message
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            try:
                result = TOOL_REGISTRY[fn_name](**fn_args)
            except Exception as e:
                result = {"error": str(e)}
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

    return "I got stuck in a tool loop. Try rephrasing the question."