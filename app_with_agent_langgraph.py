
import os
import gradio as gr
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
# from langchain_community.tools import Calculator
from langchain.agents import initialize_agent, Tool
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.docstore.document import Document
from langchain_core.tools import tool

import tempfile
import shutil

@tool
def calculator_tool(expression: str) -> str:
    """Evaluate a basic math expression like '2 + 2' or '12 * (3 + 1)'."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Math Error: {e}"

# Load HF model
model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(pipeline=pipe)

# Calculator tool
calculator = calculator_tool
tools = [Tool(name="Calculator", func=calculator.run, description="Useful for math questions")]

# Temp dir for uploaded files
UPLOAD_DIR = tempfile.mkdtemp()

def load_docs(files):
    docs = []
    for file in files:
        path = file.name
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        else:
            loader = TextLoader(path)
        docs.extend(loader.load())
    return docs

def create_vector_store(files):
    docs = load_docs(files)
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_documents(docs, embeddings)

def get_retriever_store(vs):
    return vs.as_retriever(search_kwargs={"k": 3})

def query_chain(query, retriever, agent_mode):
    docs = retriever.get_relevant_documents(query)
    if agent_mode:
        agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=False)
        result = agent.run(query)
    else:
        qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Use the following pieces of context to answer the question at the end.
If you don't know the answer, say you don't know.

{context}

Question: {question}
Helpful Answer:""",
        )
        qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt)
        result = qa_chain.run(input_documents=docs, question=query)

    return [{"role": "user", "content": query}, {"role": "assistant", "content": result}]

# Gradio UI
with gr.Blocks() as demo:
    with gr.Row():
        file_upload = gr.File(file_types=[".pdf", ".txt"], file_count="multiple", label="Upload knowledge base")
        agent_mode = gr.Checkbox(label="Use Agent (with calculator)", value=False)
    chatbot = gr.Chatbot(label="Assistant", type="messages")
    with gr.Row():
        query = gr.Textbox(label="Ask a question")
        submit = gr.Button("Submit")

    vectorstore_state = gr.State()

    def upload_files(files):
        for f in files:
            shutil.copy(f.name, os.path.join(UPLOAD_DIR, os.path.basename(f.name)))
        return create_vector_store(files)

    file_upload.change(upload_files, inputs=file_upload, outputs=vectorstore_state)

    submit.click(fn=query_chain, inputs=[query, vectorstore_state, agent_mode], outputs=chatbot)

demo.launch(share=True)
