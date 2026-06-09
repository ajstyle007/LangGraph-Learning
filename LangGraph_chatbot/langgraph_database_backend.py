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

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant"
)


class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):

    #take user query from the state
    messages = state["messages"]

    #send to llm
    response = llm.invoke(messages)

    #response store state
    return {"messages" : response}


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

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

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


print(retrive_all_threads())

# CONFIG = {"configurable" : {"thread_id" : "thread-1"}}

# response = chatbot.invoke({"messages": HumanMessage(content="what is the first question i have asked?")}, config=CONFIG)
# print(response)



# below is streaming code in langgraph
# for message_chunk, metadata in chatbot.stream({"messages": HumanMessage(content="Who is Yann lecun?")}, 
#                config = {"configurable" : {"thread_id" : "thread-1"}},
#                stream_mode="messages"):
    
#     if message_chunk.content:
#         print(message_chunk.content, end="", flush=True)