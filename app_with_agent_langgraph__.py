
import os
import shutil
import gradio as gr

from langchain.chains import RetrievalQA
# from langchain_community.tools import Calculator
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFacePipeline

from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from typing import TypedDict, Union
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage

from rag_tuning import tune_rag
from ingest import ingest_documents
from retrieval import load_retriever
from chain import build_qa_chain

from langchain_core.tools import tool

@tool
def calc(expression: str) -> str:
    """Evaluate a single-line math expression."""
    import numexpr
    try:
        result = numexpr.evaluate(expression.strip()).item()
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"

KB_DIR = "user_knowledge_base"

def clear_knowledge_base():
    if os.path.exists(KB_DIR):
        shutil.rmtree(KB_DIR)
    os.makedirs(KB_DIR, exist_ok=True)

def save_uploaded_files(files):
    clear_knowledge_base()
    for file in files:
        shutil.copy(file.name, os.path.join(KB_DIR, os.path.basename(file.name)))

# LangGraph state
class AgentState(TypedDict):
    query: str
    qa_chain: Runnable
    answer: Union[str, None]
    messages: list  # <-- This is now added

# Nodes
def retrieval_node(state: AgentState) -> AgentState:
    qa_chain = state["qa_chain"]
    query = state["query"]
    result = qa_chain.invoke(query)
    return {
        **state,
        "answer": result,
        "messages": state.get("messages", []),
    }

# Tool setup
# calc = Calculator()
calcu = calc.as_tool()  # add your math tool here
tool_node = ToolNode(tools=[calcu])

# Build LangGraph agent
def build_agent_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_conditional_edges("retrieval", tools_condition)
    workflow.add_node("tools", tool_node)
    workflow.add_edge("tools", END)
    workflow.add_edge("retrieval", END)
    workflow.set_entry_point("retrieval")
    return workflow.compile()

agent_app = build_agent_graph()

# Gradio app
def process_rag(files, sample_question, reference_answer):
    try:
        save_uploaded_files(files)

        config = tune_rag(KB_DIR, sample_question, reference_answer)

        ingest_documents(KB_DIR, config["params"]["embedding_model"], config["params"]["retriever_chunk_size"], config["params"]["retriever_overlap"])
        retriever = load_retriever(config["params"]["embedding_model"])
        qa_chain = build_qa_chain(config["model_attribs"]["pipeline_task"], retriever, config["model_attribs"]["llm_model"], config["model_kwargs"])

        return qa_chain, config
    except Exception as e:
        return None, f"Error: {str(e)}"

def query_chain(query, qa_chain, agent_mode):
    if not qa_chain:
        return [
            {"role": "user", "content": query},
            {"role": "assistant", "content": "Please upload knowledge base first."}
        ]

    if agent_mode:
        messages = [{"role": "user", "content": query}]
        final_state = agent_app.invoke({
            "messages": messages,
            "query": query,
            "qa_chain": qa_chain
        })
        answer_obj = final_state.get("answer", "")
        if isinstance(answer_obj, dict):
            response = answer_obj.get("result", "")
        else:
            response = answer_obj
    else:
        answer_obj = qa_chain.invoke(query)
        if isinstance(answer_obj, dict):
            response = answer_obj.get("result", "")
        else:
            response = str(answer_obj)

    return [
        {"role": "user", "content": query},
        {"role": "assistant", "content": response}]


# UI
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 RAG Chatbot with Agent Mode")

    with gr.Row():
        file_input = gr.File(label="Upload Knowledge Base", file_types=[".txt", ".pdf"], file_count="multiple")

    with gr.Row():
        question = gr.Textbox(label="Sample Question (optional)")
        reference_answer = gr.Textbox(label="Sample Answer (optional)")

    run_button = gr.Button("Run Tuning + Load")

    config_output = gr.Textbox(label="Best Config")
    chatbot = gr.Chatbot(label="RAG Chatbot",type='messages')
    user_query = gr.Textbox(label="Ask a Question")
    ask_button = gr.Button("Ask")

    qa_chain_state = gr.State()
    agent_toggle = gr.Checkbox(label="Use Agent Mode (with Tools)", value=False)

    run_button.click(
        fn=process_rag,
        inputs=[file_input, question, reference_answer],
        outputs=[qa_chain_state, config_output]
    )

    ask_button.click(
        fn=query_chain,
        inputs=[user_query, qa_chain_state, agent_toggle],
        outputs=[chatbot]
    )

if __name__ == "__main__":
    demo.launch(share=True)
