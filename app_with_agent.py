
import gradio as gr
import shutil
import os

from rag_tuning import tune_rag
from ingest import ingest_documents
from retrieval import load_retriever
from chain import build_qa_chain
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
# from langchain_community.tools_calculator import Calculator
from langchain.chains import RetrievalQA
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


# Temporary knowledge base directory
KB_DIR = "user_knowledge_base"

def clear_knowledge_base():
    if os.path.exists(KB_DIR):
        shutil.rmtree(KB_DIR)
    os.makedirs(KB_DIR, exist_ok=True)

def save_uploaded_files(files):
    clear_knowledge_base()
    for file in files:
        shutil.copy(file.name, os.path.join(KB_DIR, os.path.basename(file.name)))

def process_rag(files, sample_question, reference_answer, agent_mode):
    try:
        save_uploaded_files(files)

        # Tune hyperparameters if user provided prompt+answer
        config = tune_rag(
            KB_DIR,
            sample_question=sample_question,
            reference_answer=reference_answer,
        )

        ingest_documents(KB_DIR, config["params"]["embedding_model"], config["params"]["retriever_chunk_size"], config["params"]["retriever_overlap"])
        retriever = load_retriever(config["params"]["embedding_model"])

        if agent_mode:
            # RAG chain for agent
            rag_chain = build_qa_chain(
                config["model_attribs"]["pipeline_task"],
                retriever,
                config["model_attribs"]["llm_model"],
                config["model_kwargs"]
            )

            tools = [
                Tool(
                    name="KnowledgeBaseQA",
                    func=rag_chain.run,
                    description="Use this tool to answer questions from the uploaded knowledge base."
                ),
                # Calculator()
                calc.as_tool()  # add your math tool here
            ]

            agent = initialize_agent(
                tools=tools,
                llm=rag_chain.combine_documents_chain.llm_chain.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True
            )
            return agent, config, agent_mode
        else:
            qa_chain = build_qa_chain(
                config["model_attribs"]["pipeline_task"],
                retriever,
                config["model_attribs"]["llm_model"],
                config["model_kwargs"]
            )
            return qa_chain, config, agent_mode
    except Exception as e:
        return None, f"Error: {str(e)}", agent_mode

def query_chain(query, qa_chain, agent_mode, chat_history):
    if not qa_chain:
        return chat_history + [["User", "Please upload a knowledge base and run tuning first."]]

    if agent_mode:
        response = qa_chain.run(query)
    else:
        response = qa_chain.invoke(query)
        if isinstance(response, dict) and "result" in response:
            response = response["result"]

    return chat_history + [[query, response]]

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 RAG + Agent Chatbot with Auto-Tuning")

    with gr.Row():
        file_input = gr.File(label="Upload Knowledge Base (Text/PDF)", file_types=[".txt", ".pdf"], file_count="multiple")

    with gr.Row():
        question = gr.Textbox(label="(Optional) Sample Question for Tuning")
        reference_answer = gr.Textbox(label="(Optional) Reference Answer")

    agent_mode_checkbox = gr.Checkbox(label="🔧 Use Agent Mode (Enables Calculator + RAG)", value=False)
    run_button = gr.Button("🚀 Run Tuning + Load Chatbot")

    output_config = gr.Textbox(label="Best Configuration Found")
    chatbot = gr.Chatbot(label="Chat History", height=400)
    user_query = gr.Textbox(label="Your Question")
    ask_button = gr.Button("Ask")

    state_chain = gr.State()
    state_mode = gr.State()
    state_chat = gr.State([])

    run_button.click(
        fn=process_rag,
        inputs=[file_input, question, reference_answer, agent_mode_checkbox],
        outputs=[state_chain, output_config, state_mode]
    )

    ask_button.click(
        fn=query_chain,
        inputs=[user_query, state_chain, state_mode, state_chat],
        outputs=[chatbot],
        show_progress=True
    ).then(
        lambda chat, query: chat + [[query, ""]], [chatbot, user_query], state_chat
    )

if __name__ == "__main__":
    demo.launch(share=True)
