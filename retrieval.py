### retrieval.py
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

def load_retriever(emb_model, persist_path="vectorstore"):
    embeddings = HuggingFaceEmbeddings(model_name=emb_model)
    vectorstore = FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore.as_retriever()