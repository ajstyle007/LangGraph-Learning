from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START, END
from typing import TypedDict, Literal, Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

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


checkpointer = InMemorySaver()

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# below is streaming code in langgraph
# for message_chunk, metadata in chatbot.stream({"messages": HumanMessage(content="Who is Yann lecun?")}, 
#                config = {"configurable" : {"thread_id" : "thread-1"}},
#                stream_mode="messages"):
    
#     if message_chunk.content:
#         print(message_chunk.content, end="", flush=True)