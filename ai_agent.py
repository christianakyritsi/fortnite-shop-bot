import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

import tools as tools_module

load_dotenv()

# ============ Tools (wrap your existing functions for LangChain) ============

@tool
def lookup_item(name: str) -> dict:
    """Look up shop history for a Fortnite cosmetic by name. Returns rarity, type,
    total appearances, first/last seen dates, and days since last appearance."""
    return tools_module.lookup_item(name)

@tool
def get_todays_shop() -> dict:
    """Get the list of items in today's Fortnite shop, with scarcity info
    (days since last seen) for each."""
    return tools_module.get_todays_shop()

@tool
def find_rare_returns(min_days: int = 100) -> dict:
    """Find items in today's shop that have been gone for a long time.
    Useful for highlighting rare returns."""
    return tools_module.find_rare_returns(min_days)

@tool
def predict_return(name: str) -> dict:
    """Predict the probability that a Fortnite item returns to the shop within
    the next 30 days. Uses a LightGBM model with moderate accuracy (AUC 0.63).
    Use for 'when is X coming back?' or 'will X return?' questions."""
    return tools_module.predict_return(name)


TOOLS = [lookup_item, get_todays_shop, find_rare_returns, predict_return]

# ============ LLM ============

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=600,
).bind_tools(TOOLS)

SYSTEM_PROMPT = """You are a Fortnite Shop Analyst inside a Discord server.

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

Keep replies short, friendly, and Discord-friendly. No markdown headings."""

# ============ Graph state ============

class State(TypedDict):
    messages: Annotated[list, add_messages]

# ============ Nodes ============

def agent_node(state: State) -> dict:
    """Call the LLM with the current messages."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(TOOLS)

# ============ Conditional edge ============

def should_continue(state: State) -> str:
    """If the LLM called a tool, route to tools node. Otherwise end."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# ============ Build the graph ============

builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")  # after tools, go back to agent

graph = builder.compile()

# ============ Public function ============

def ask_llama(user_message: str) -> str:
    """Run the user message through the LangGraph agent and return final text."""
    result = graph.invoke({
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
    })
    final = result["messages"][-1]
    return final.content or "Sorry, I couldn't generate a response."