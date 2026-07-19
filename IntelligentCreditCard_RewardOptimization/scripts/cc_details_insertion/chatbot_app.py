import streamlit as st
from rag_retrieval_langchain import final_chain
import time
import traceback
import logging


logger = logging.getLogger("chatbot_app")
logger.info("Streamlit script execution started")

# Page configuration
st.set_page_config(
    page_title="Credit Card Rewards Chatbot",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("💳 Credit Card Rewards Chatbot")
st.markdown("*Your AI-powered assistant for credit card optimization and benefits*")

# Sidebar configuration
with st.sidebar:
    st.header("Settings")
    
    # Reset conversation button
    if st.button("🔄 Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_active = False
        st.rerun()
    
    st.divider()
    st.markdown("""
    ### How it works:
    1. Ask about credit card benefits
    2. Compare cards
    3. Get personalized recommendations
    4. Understand reward strategies
    
    ### Supported Queries:
    - Single transaction recommendations
    - Monthly spend optimization
    - Point transfer strategies
    - Card comparison
    - General credit card queries
    """)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_active" not in st.session_state:
    st.session_state.conversation_active = False

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Input area
st.divider()
col1, col2 = st.columns([0.95, 0.05])

with col1:
    user_input = st.chat_input(
        "Ask me about credit card benefits, rewards, or comparisons...",
        key="user_input"
    )

if user_input:
    logger.info("User submitted query", extra={"query_length": len(user_input)})
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate response using RAG chain
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # Show loading state
            with st.spinner("Thinking..."):
                logger.info("Invoking final_chain")
                start_time = time.time()
                # Invoke the RAG chain
                response = final_chain.invoke(user_input)
                elapsed = round(time.time() - start_time, 3)
                logger.info(
                    "final_chain completed",
                    extra={"duration_seconds": elapsed, "response_length": len(str(response))},
                )
                
                # Display the response
                message_placeholder.markdown(response)
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                
        except Exception as e:
            # Emit full traceback to server logs (Cloud Run) for debugging.
            traceback.print_exc()
            logger.exception("Query processing failed")
            error_message = f"⚠️ An error occurred: {str(e)}"
            message_placeholder.markdown(error_message)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message
            })

# Footer
st.divider()
st.markdown("""
<div style='text-align: center'>
    <small>Powered by OpenAI GPT-4 and LangChain RAG • Credit Card Database Integration</small>
</div>
""", unsafe_allow_html=True)
