import streamlit as st
from langgraph_tool_backend import chatbot, retrive_all_threads, save_thread_title
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
import streamlit as st
import uuid
from langchain_groq import ChatGroq

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

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    # add_thread(st.session_state["thread_id"])
    st.session_state["message_history"] = []

def add_thread(thread_id):
    if str(thread_id) not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"][str(thread_id)] = {
            "title": None
        }

def load_conversations(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])


llm = ChatGroq(
    model="llama-3.1-8b-instant"
)

def generate_chat_title(user_message, llm):

    messages = [
        SystemMessage(
            content="""
        You generate chat titles.

        Rules:
        - 2 to 5 words only
        - No punctuation
        - No quotes
        - No explanations
        - No complete sentences
        - Return ONLY the title

        Examples:
        User: How to deploy LangGraph on AWS?
        Title: LangGraph AWS Deployment

        User: What is quantization in LLMs?
        Title: LLM Quantization Guide
        """
        ),
        HumanMessage(content=user_message)
    ]

    response = llm.invoke(messages)

    return response.content.strip()


if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrive_all_threads()

add_thread(st.session_state["thread_id"])

# st.write(type(st.session_state["chat_threads"]))

# SideBar UI
st.sidebar.header("User Chats")

if st.sidebar.button("New Chat"):
    reset_chat()

threads = list(st.session_state["chat_threads"].items())

for thread_id, metadata in reversed(threads):
    # st.write(metadata)
    # st.write(type(metadata["title"]))
    title = metadata.get("title") or "New Chat"
    if st.sidebar.button(title, key=str(thread_id)):
        
        st.session_state["thread_id"] = thread_id
        messages = load_conversations(thread_id)

        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            else:
                role = "assistant"
            temp_messages.append({"role":role, "content":msg.content})

        st.session_state["message_history"] = temp_messages

#loading the conversation history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])
                         

user_input = st.chat_input("Type your message here")

if user_input:

    if str(st.session_state["thread_id"]) not in st.session_state["chat_threads"]:
        add_thread(st.session_state["thread_id"])

    is_first_message = len(st.session_state["message_history"]) == 0

    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )

    st.session_state["message_history"].append({"role" : "user", "content" : user_input})
    with st.chat_message("user"):
        st.text(user_input)
    
    # CONFIG = {"configurable" : {"thread_id" : st.session_state["thread_id"]}}

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_turn",
    }


    # Assistant streaming block
    with st.chat_message("assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )
    
    st.session_state["message_history"].append({"role" : "assistant", "content" : ai_message})


    if is_first_message:

        title = generate_chat_title(
            f"""
            User: {user_input}

            Assistant: {ai_message}
            """,
            llm=llm
        )
    
        print("TITLE =", title)
        print("TYPE =", type(title))

        st.session_state["chat_threads"][str(st.session_state["thread_id"])]["title"] = title

        save_thread_title(st.session_state["thread_id"], title)

        