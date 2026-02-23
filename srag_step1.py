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
        PyPDFLoader("./Documents/2310.11511v1.pdf").load()
        + PyPDFLoader("./Documents/2401.15884v3.pdf").load()
    )

    chunks = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150).split_documents(documents)

    vectordatabase = FAISS.from_documents(chunks , emb)
    vectordatabase.save_local("faiss_index_database")


database = FAISS.load_local("faiss_index_database" , emb , allow_dangerous_deserialization=True)
retriever = database.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k" : 5}
)

class State(TypedDict):
    question : str
    to_retrieve : bool
    docs : List[Document]
    answer : str

def need_retrieval(state : State) -> State:

    class retrievellm(BaseModel):
        need : bool
    
    eval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict evaluator.\n"
            "You will be given one question.\n"
            "You have to check if that question's answer is in your information or not \n"
            "If the user is asking related to Self Rag or Corrective Rag related things then you have to compulsory return need:True\n"
            "Otherwise you need to return need:False\n"
            "Output JSON only.",
        ),
        ("human", "Question: {question}"),
    ])

    chain = eval_prompt | llm.with_structured_output(retrievellm)

    result = chain.invoke({"question" : state["question"]})

    return {"to_retrieve" :  result.need}


def retrieve_node(state: State) -> State:
    q = state["question"]
    return {"docs": retriever.invoke(q)}

def generate_direct(state: State) -> State:
    direct_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using only your general knowledge.\n"
            "Do NOT assume access to external documents.\n"
            "If you are unsure or the answer requires specific sources, say:\n"
            "'I don't know based on my general knowledge.'"
        ),
        ("human", "{question}"),
    ]
)

    out = llm.invoke(
        direct_generation_prompt.format_messages(
            question=state["question"]
        )
    )
    return {
        "answer": out.content
    }

def condition(state: State) -> str:
    if state["to_retrieve"]:
        return "retrieve_node"
    else:
        return "generate_direct"

graph = StateGraph(State)

graph.add_node("need_retrieval" , need_retrieval)
graph.add_node("generate_direct" , generate_direct)
graph.add_node("retrieve_node" , retrieve_node)

graph.add_edge(START , "need_retrieval")
graph.add_conditional_edges("need_retrieval" ,
                           condition,
                           {"retrieve_node" , "retrieve_node",
                            "generate_direct","generate_direct"})
graph.add_edge("generate_direct" , END)
graph.add_edge("retrieve_node" , END)

workflow = graph.compile()

answer = workflow.invoke(
    {"question" : "Tell me what is self attention?",
     "docs" : [],
     "answer" : "",
     "to_retrieve" : ""}
)

print(answer["to_retrieve"])
print(answer["docs"])
print(answer["answer"])











