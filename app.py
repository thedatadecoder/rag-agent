import gradio as gr
import shutil
import os
from rag_tuning import tune_rag
from ingest import ingest_documents
from retrieval import load_retriever
from chain import build_qa_chain

from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType

KB_DIR = "user_knowledge_base"

def clear_knowledge_base():
    if os.path.exists(KB_DIR):
        shutil.rmtree(KB_DIR)
    os.makedirs(KB_DIR, exist_ok=True)

def save_uploaded_files(files):
    clear_knowledge_base()
    for file in files:
        shutil.copy(file.name, os.path.join(KB_DIR, os.path.basename(file.name)))

def process_rag(files, sample_question, reference_answer):
    try:
        save_uploaded_files(files)

        config = tune_rag(
            KB_DIR,
            sample_question=sample_question,
            reference_answer=reference_answer,
        )

        ingest_documents(KB_DIR,
                         config["params"]["embedding_model"],
                         config["params"]["retriever_chunk_size"],
                         config["params"]["retriever_overlap"])
        
        retriever = load_retriever(config["params"]["embedding_model"])

        qa_chain = build_qa_chain(config["model_attribs"]["pipeline_task"],
                                  retriever,
                                  config["model_attribs"]["llm_model"],
                                  config["model_kwargs"])

        # Define a LangChain tool for the retriever QA
        tools = [
            Tool(
                name="KnowledgeBaseQA",
                func=qa_chain.run,
                description="Use this for questions about the uploaded knowledge base"
            )
        ]

        # Agent initialization
        agent_executor = initialize_agent(
            tools=tools,
            llm=qa_chain.llm,  # Use same LLM as QA chain
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

        return agent_executor, str(config)

    except Exception as e:
        return None, f"Error: {str(e)}"

def chat_with_rag(message, chat_history, agent):
    if not agent:
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": "Please run tuning first."})
        return chat_history, chat_history

    try:
        answer = agent.run(message)
    except Exception as e:
        answer = f"Error: {str(e)}"

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": answer})
    return chat_history, chat_history


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 RAG Assistant with Auto-Tuning + Agentic Reasoning")

    with gr.Row():
        file_input = gr.File(label="Upload Knowledge Base (Text/PDF)", file_types=[".txt", ".pdf"], file_count="multiple")

    with gr.Row():
        question = gr.Textbox(label="(Optional) Sample Question")
        reference_answer = gr.Textbox(label="(Optional) Sample Answer")

    run_button = gr.Button("🚀 Run Tuning + Load Agent")

    output_config = gr.Textbox(label="Best Configuration")
    chatbot = gr.Chatbot(label="RAG Q&A", type="messages")

    with gr.Row():
        user_query = gr.Textbox(label="Ask a Question", placeholder="Type your question and press Ask")
        ask_button = gr.Button("Ask")

    agent_state = gr.State()
    chat_history_state = gr.State([])

    run_button.click(
        fn=process_rag,
        inputs=[file_input, question, reference_answer],
        outputs=[agent_state, output_config]
    )

    ask_button.click(
        fn=chat_with_rag,
        inputs=[user_query, chat_history_state, agent_state],
        outputs=[chatbot, chat_history_state]
    )

if __name__ == "__main__":
    demo.launch(share=True)
