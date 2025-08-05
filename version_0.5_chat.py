import streamlit as st
import os
import logging
import warnings
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from google.oauth2.service_account import Credentials
from datetime import datetime

from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate

# --- Suppress warnings and logs ---
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

# --- Set API Key ---
#os.environ["GOOGLE_API_KEY"] = ""
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# --- Model Config ---
llm_model_name = "gemini-2.5-flash"
embedding_model_name = "embedding-001"

# --- Load Models ---
llm = ChatGoogleGenerativeAI(model=llm_model_name, temperature=0.0)

#embedding_function = GoogleGenerativeAIEmbeddings(model=embedding_model_name)

embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


# --- Load Vector Store ---
@st.cache_resource
def load_vector_store():
    return FAISS.load_local("faiss_index", embedding_function, allow_dangerous_deserialization=True)

vector_store = load_vector_store()
retriever = vector_store.as_retriever()

# --- Prompt Template ---
prompt_template = """
Answer the question using ONLY the provided context.
If the answer is not in the context, respond with:
"I’m sorry, I don’t have that information right now."

Context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# --- RAG Chain ---
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)




def log_to_gsheet(question, response):
    scope = ["https://www.googleapis.com/auth/spreadsheets"]
    
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=scope
    )

    client = gspread.authorize(creds)
    sheet = client.open("Fore Chatbot Logs").sheet1
    sheet.append_row([question, response])


# --- Chat Function ---
def chat_with_pdf(question: str) -> str:
    response = rag_chain.invoke(question)
    return response.replace('\n', ' ')

# --- Streamlit UI ---
st.set_page_config(page_title="Fore Chatbot", layout="centered")

st.title("📄 Testing - version 0.5")
#user_input = st.text_input("Ask a question:who is director of fore")
user_input = st.text_input("", placeholder="who is director of fore")

if user_input:
    response = chat_with_pdf(user_input)

    st.write(response)
    log_to_gsheet(user_input, response)

