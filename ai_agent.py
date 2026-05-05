import os
import json
from groq import Groq
from dotenv import load_dotenv

import tools

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """
You are a Fortnite Shop Analyst inside a Discord server.

You have access to tools that read a real shop history dataset (2017–today) and a prediction model.
ALWAYS use the tools when answering factual questions. Never guess or make up numbers.

Tool guidance:
- "How rare is X?" / "When was X last seen?" → use lookup_item
- "What's in the shop today?" → use get_todays_shop
- "Anything rare returning today?" → use find_rare_returns
- "When is X coming back?" / "Will X return soon?" → use predict_return

When using predict_return, frame the probability honestly. The model has moderate accuracy
(AUC 0.63), so describe predictions as informed estimates, not certainties. Mention the
probability and contextualize it with days_since_last_seen and average_gap_days.

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
    {
        "type": "function",
        "function": {
            "name": "predict_return",
            "description": "Predict the probability that a Fortnite item returns to the shop within the next 30 days. Uses a LightGBM model with moderate accuracy (AUC 0.63). Use this when users ask 'when is X coming back?' or 'will X return?' or similar predictive questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The cosmetic name (fuzzy match supported)."}
                },
                "required": ["name"],
            },
        },
    },
]

# Map tool names to actual Python functions
TOOL_REGISTRY = {
    "lookup_item": tools.lookup_item,
    "get_todays_shop": tools.get_todays_shop,
    "find_rare_returns": tools.find_rare_returns,
    "predict_return": tools.predict_return,
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