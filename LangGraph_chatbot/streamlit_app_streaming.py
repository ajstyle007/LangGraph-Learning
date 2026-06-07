import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import BaseMessage, HumanMessage


st.markdown("""
<style>
    /* Target the sticky title container */
    .sticky-title {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        text-align: center;
        background-color: #0E1117; /* Matches Streamlit's default dark theme bg */
        z-index: 999999;          /* Ensure it stays above chat elements */
        padding: 10px 20px;
        border-bottom: 1px solid #262730; /* Optional: adds a nice separator */
        
        .stMain {
        margin-top: 70px; 
    }
    }
    
    /* Crucial: Add padding to the main body so the fixed header doesn't cover the first chat messages */
    .stMain {
        margin-top: 50px; 
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="sticky-title"><h1 style="margin:0; color : green; font-size: 34px;">Basic ChatBot using LangGraph</h1></div>', unsafe_allow_html=True)

# st.title("Basic ChatBot using LangGraph", text_alignment="center")

CONFIG = {"configurable" : {"thread_id" : "thread-1"}}

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

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

        