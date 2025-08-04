import google.generativeai as genai
import os
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

api_key = ""
genai.configure(api_key=api_key)

EMBEDDING_MODEL_NAME = "embedding-001"

def embed_documents_direct(texts):
    response = genai.embed_content(
        model=EMBEDDING_MODEL_NAME,
        content=texts,
        task_type="retrieval_document"
    )
    return response['embedding']

def embed_query_direct(text):
    response = genai.embed_content(
        model=EMBEDDING_MODEL_NAME,
        content=text,
        task_type="retrieval_query"
    )
    return response['embedding']

class CustomGeminiEmbeddings:
    def embed_documents(self, texts):
        if texts and isinstance(texts[0], Document):
            content_texts = [doc.page_content for doc in texts]
            return embed_documents_direct(content_texts)
        else:
            return embed_documents_direct(texts)

    def embed_query(self, text):
        return embed_query_direct(text)

loader = PyMuPDFLoader("Fore_Solutionsdetail.pdf")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = text_splitter.split_documents(data)

embedding_function = CustomGeminiEmbeddings()

faiss_index = FAISS.from_texts(
    [doc.page_content for doc in chunks], 
    embedding=embedding_function
)
faiss_index.save_local("faiss_index")
# vector_db.persist()
print("Vector database created and persisted.")


# In[24]:


import warnings
import logging
import os

from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate

warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

# --- API Key Configuration ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyDPIjub16iX7uwgA7Nl_AjW1q0_CAqdDcs"

# --- Model Config ---
llm_model_name = "gemini-2.5-flash"
embedding_model_name = "embedding-001"

# Initialize LLM and Embedding
llm = ChatGoogleGenerativeAI(model=llm_model_name, temperature=0.0)
embedding_function = GoogleGenerativeAIEmbeddings(model=embedding_model_name)

# Load existing FAISS vector store
print("Loading vector store...")
vector_store = FAISS.load_local("faiss_index", embedding_function, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever()
print("Vector store loaded successfully.")

# --- Simple RAG Prompt ---
prompt_template = """
Answer the question using ONLY the provided context.
If the answer is not in the context, respond with:
"I’m sorry, I don’t have that information right now."

Context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# --- Build RAG Chain ---
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- Function to Query ---
def chat_with_pdf(question: str) -> str:
    response = rag_chain.invoke(question)
    cleaned_output = response.replace('\n', ' ')

    return cleaned_output







