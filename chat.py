import os
import streamlit as st
import warnings
import logging

from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate

# -------------------- CONFIG --------------------

# Hide warnings and logs
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

# Load Google API key (from Streamlit secrets or env)
api_key = os.getenv("GOOGLE_API_KEY") or st.secrets["GOOGLE_API_KEY"]

# Model config
LLM_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "embedding-001"
INDEX_DIR = "faiss_index"  # Folder with index.faiss and index.pkl

# -------------------- LOAD RAG --------------------

@st.cache_resource(show_spinner="🔄 Loading AI & Vector DB…")
def load_rag_chain():
    embedding_function = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=api_key
    )
    vector_store = FAISS.load_local(
        INDEX_DIR,
        embedding_function,
        allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever()

    prompt_template = """
    Answer the question using ONLY the provided context.
    If the answer is not in the context, respond with:
    "I’m sorry, I don’t have that information right now."

    Context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0.0,
        google_api_key=api_key
    )

    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

rag_chain = load_rag_chain()

# -------------------- UI --------------------

st.set_page_config(page_title="📄 Fore Solutions Chatbot", page_icon="💬")
st.title("💬 Chat with Fore Solutions PDF")
st.markdown("Ask a question. Answers come **only from the uploaded PDF**.")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_input = st.chat_input("Type your question here...")

# Query and display result
if user_input:
    with st.spinner("Generating answer..."):
        answer = rag_chain.invoke(user_input).replace("\n", " ")
    st.session_state.chat_history.append((user_input, answer))

# Display chat messages
for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)
