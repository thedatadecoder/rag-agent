### ingest.py
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import os

def ingest_documents(source_dir, emb_model, chunk_size, overlap, persist_path="vectorstore"):
    loader = DirectoryLoader(
        source_dir,
        glob="**/*.*",
        loader_cls=lambda path: PyPDFLoader(path) if path.endswith(".pdf") else TextLoader(path)
    )
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=emb_model)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(persist_path)
    