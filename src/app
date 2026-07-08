import streamlit as st
from llm import rag_chat

st.set_page_config(
    page_title="Banking RAG Assistant",
    page_icon="🏦",
    layout="wide"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown(
    """
    <h1 style='text-align: center;'>🏦 Banking RAG Assistant</h1>
    <p style='text-align: center;'>Get Instant Answers to Banking Queries</p>
    """,
    unsafe_allow_html=True
)

for chat in st.session_state.chat_history:
    with st.chat_message("user", avatar="🏦"):
        st.write(chat["question"])

    with st.chat_message("assistant", avatar="🤖"):
        st.write(chat["answer"])

question = st.chat_input("Ask a banking question")

if question:
    with st.spinner("Searching documents..."):
        answer = rag_chat(question)

    st.session_state.chat_history.append(
        {
            "question": question,
            "answer": answer
        }
    )

    st.rerun()
