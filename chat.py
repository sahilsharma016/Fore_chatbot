import os
import streamlit as st
import warnings
import logging
import google.generativeai as genai

from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

# -------------------- CONFIG --------------------
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

# Get API key from environment or Streamlit secrets
api_key = os.getenv("GOOGLE_API_KEY") or st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

LLM_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "embedding-001"
INDEX_DIR = "faiss_index"  # path to index.faiss and index.pkl

# -------------------- Embedding Logic --------------------

def embed_documents_direct(texts):
    """Embed a list of document texts for indexing or retrieval."""
    response = genai.embed_content(
        model=EMBED_MODEL,
        content=texts,
        task_type="retrieval_document"
    )
    return response['embedding']

def embed_query_direct(text):
    """Embed a single query text."""
    response = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
        task_type="retrieval_query"
    )
    return response['embedding']

class CustomGeminiEmbeddings:
    def embed_documents(self, texts):
        return [embed_documents_direct(text) for text in texts]

    def embed_query(self, text):
        return embed_query_direct(text)

# -------------------- RAG Chain Loader --------------------

@st.cache_resource(show_spinner="🔄 Loading AI & Vector DB…")
def load_rag_chain():
    embedding_function = CustomGeminiEmbeddings()
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

    from langchain_google_genai import ChatGoogleGenerativeAI
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

# -------------------- Streamlit UI --------------------

st.set_page_config(page_title="💬 Fore Solutions Chatbot", page_icon="🤖")
st.title("💬 Chat with Fore Solutions PDF")
st.markdown("Ask questions based **only on the uploaded PDF content**.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Type your question here...")

if user_input:
    with st.spinner("Thinking..."):
        try:
            answer = rag_chain.invoke(user_input).replace("\n", " ")
        except Exception as e:
            st.error("An error occurred while generating the response.")
            st.exception(e)
            answer = "⚠️ Failed to get an answer due to an internal error."
    st.session_state.chat_history.append((user_input, answer))

# Display history
for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)

