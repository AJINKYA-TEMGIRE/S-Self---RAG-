from typing import List, TypedDict
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model = "openai/gpt-oss-120b")
emb = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

flag = True # currently

if flag == False:
    documents = (
        PyPDFLoader("./Books/book1.pdf").load()
        + PyPDFLoader("./Books/book2.pdf").load()
        + PyPDFLoader("./Books/book3.pdf").load()
    )

    chunks = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150).split_documents(documents)
    for d in chunks:
        d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")

    vectordatabase = FAISS.from_documents(chunks , emb)
    vectordatabase.save_local("faiss_index_database")


database = FAISS.load_local("faiss_index_database" , emb , allow_dangerous_deserialization=True)
retriever = database.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k" : 5}
)









