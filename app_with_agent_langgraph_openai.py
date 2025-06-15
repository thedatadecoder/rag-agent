import gradio as gr
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
# from langgraph.prebuilt import create_tool_calling_agent
from langchain.agents import AgentExecutor, create_tool_calling_agent #, tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langchain_core.tools import tool

import os
import tempfile
import shutil
import pickle
from pathlib import Path

# ---- EMBEDDING + RAG SETUP ----
def create_vectorstore_from_files(file_objs):
    temp_dir = tempfile.mkdtemp()
    for file in file_objs:
        file_path = Path(temp_dir) / file.name
        with open(file_path, "wb") as f:
            f.write(file.read())

    from langchain.document_loaders import PyPDFLoader, TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    docs = []
    for path in Path(temp_dir).rglob("*"):
        if path.suffix == ".pdf":
            docs.extend(PyPDFLoader(str(path)).load())
        elif path.suffix == ".txt":
            docs.extend(TextLoader(str(path)).load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    shutil.rmtree(temp_dir)
    return vectorstore

# ---- TOOL: CALCULATOR ----
@tool
def calculator_tool(expression: str) -> str:
    """Evaluate a basic math expression like '2 + 2' or '12 * (3 + 1)'."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Math Error: {e}"

# ---- GRADIO APP ----

llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-base",
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 256}
)

# Optional shared state
rag_chain = None
vectorstore = None

def upload_kb(files):
    global rag_chain, vectorstore
    vectorstore = create_vectorstore_from_files(files)
    retriever = vectorstore.as_retriever()
    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return "✅ Knowledge base uploaded!"

# ---- LANGGRAPH PLANNER AGENT ----
tools = [calculator_tool]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can use tools if needed."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_tool_calling_agent(llm, tools, prompt)


class AgentState(dict): pass
def get_agent_state(state): return AgentState(messages=[HumanMessage(content=state["query"])])

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)
agent_executor = workflow.compile()

# ---- QUERY HANDLER ----
def handle_query(query, agent_mode):
    if not rag_chain:
        return [{"role": "assistant", "content": "❗Please upload a knowledge base first."}]

    if agent_mode:
        state = agent_executor.invoke({"query": query})
        result = state["messages"][-1].content
    else:
        result = rag_chain.run(query)

    return [{"role": "user", "content": query}, {"role": "assistant", "content": result}]

# ---- GRADIO UI ----
with gr.Blocks() as demo:
    gr.Markdown("## 🔍 RAG + LangGraph Planner Agent with Calculator Tool")
    with gr.Row():
        kb_upload = gr.File(file_types=[".pdf", ".txt"], file_count="multiple", label="Upload Knowledge Base")
        upload_btn = gr.Button("Upload")
        upload_msg = gr.Textbox(interactive=False)
    upload_btn.click(fn=upload_kb, inputs=[kb_upload], outputs=[upload_msg])

    agent_mode = gr.Checkbox(label="🔁 Agent Mode (Tool Use Enabled)", value=False)
    chatbot = gr.Chatbot(label="Assistant", type="messages")
    user_input = gr.Textbox(label="Your Question")
    ask_btn = gr.Button("Ask")

    ask_btn.click(fn=handle_query, inputs=[user_input, agent_mode], outputs=[chatbot])

demo.launch()
