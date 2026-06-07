import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import BaseMessage, HumanMessage


import streamlit as st

st.markdown("""
<style>
    /* Top Header Styling */
    .sticky-header-container {
        position: fixed;
        top: 15px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 999999;
        pointer-events: none;
    }

    .sticky-title {
        background-color: #0E1117; 
        padding: 10px 40px;        
        border: 1px solid #262730; 
        border-radius: 30px;       
        box-shadow: 0px 4px 15px rgba(0,0,0,0.5); 
        text-align: center;
        width: fit-content;        
        pointer-events: auto;      
    }

    /* Sirf top par space banayi hai taaki pehla message title ke peeche na chupe */
    .stMainBlockContainer {
        padding-top: 80px !important;
    }
</style>
""", unsafe_allow_html=True)

# 2. Render Top Title
st.markdown("""
<div class="sticky-header-container">
    <div class="sticky-title">
        <h1 style="margin:0; color: green; font-size: 34px; white-space: nowrap;">ChatBot using LangGraph</h1>
    </div>
</div>
""", unsafe_allow_html=True)

# st.title("Basic ChatBot using LangGraph", text_alignment="center")

CONFIG = {"configurable" : {"thread_id" : "thread-1"}}

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


# SideBar UI
st.sidebar.header("User Chats")

st.sidebar.button("New Chat")


#loading the conversation history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])
                         

user_input = st.chat_input("Type your message here")

if user_input:

    st.session_state["message_history"].append({"role" : "user", "content" : user_input})
    with st.chat_message("user"):
        st.text(user_input)


    with st.chat_message("assistant"):
        
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream({"messages": HumanMessage(content=user_input)}, 
               config = {"configurable" : {"thread_id" : "thread-1"}},
               stream_mode="messages"))
    
    st.session_state["message_history"].append({"role" : "assistant", "content" : ai_message})

        