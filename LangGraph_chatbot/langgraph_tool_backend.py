from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START, END
from typing import TypedDict, Literal, Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchResults, DuckDuckGoSearchRun
from langchain_core.tools import tool

import random, requests

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant"
)

search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=AAN4HX4405HICSJE"
    r = requests.get(url)
    return r.json()

# Make tool list
tools = [get_stock_price, search_tool, calculator]

# Make the LLM tool-aware
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)  # Executes tool calls

conn = sqlite3.connect(database="database_chatbot", check_same_thread=False)

conn.execute("""
CREATE TABLE IF NOT EXISTS threads (
    thread_id TEXT PRIMARY KEY,
    title TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()

checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)

# graph structure
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
# If the LLM asked for a tool, go to ToolNode; else finish
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node") 

chatbot = graph.compile(checkpointer=checkpointer)


def save_thread_title(thread_id, title):

    conn.execute(
        """
        INSERT OR REPLACE INTO threads
        (thread_id, title)
        VALUES (?, ?)
        """,
        (str(thread_id), title)
    )

    conn.commit()


def retrive_all_threads():
    cursor = conn.execute(
        """
        SELECT thread_id, title
        FROM threads
        ORDER BY created_at DESC
        """
    )

    return {
        row[0]: {
            "title": row[1]
        }
        for row in cursor.fetchall()
    }


# print(retrive_all_threads())

# CONFIG = {"configurable" : {"thread_id" : "thread-1"}}

# response = chatbot.invoke({"messages": HumanMessage(content="what is the first question i have asked?")}, config=CONFIG)
# print(response)



# below is streaming code in langgraph
# for message_chunk, metadata in chatbot.stream({"messages": HumanMessage(content="Who is Yann lecun?")}, 
#                config = {"configurable" : {"thread_id" : "thread-1"}},
#                stream_mode="messages"):
    
#     if message_chunk.content:
#         print(message_chunk.content, end="", flush=True)