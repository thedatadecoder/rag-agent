import os
import gradio as gr
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, Tool
from transformers import pipeline

# === Embeddings & Globals ===
embeddings = HuggingFaceEmbeddings()
vectorstore = None
retriever = None
qa_chain = None

# === Prompt ===
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Use the following context to answer the question.
If you don't know the answer, say you don't know.

Context: {context}
Question: {question}

Helpful answer:"""
)

# === Light LLM for text generation ===
pipe = pipeline("text-generation", model="sshleifer/tiny-gpt2", max_new_tokens=256)
llm = HuggingFacePipeline(pipeline=pipe)

# === DuckDuckGo Tool for Agent Mode ===
search = DuckDuckGoSearchRun()
tools = [Tool(name="DuckDuckGoSearch", func=search.run, description="Use for real-time internet search")]

# === QA Chain Builder ===
def build_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    return retriever, qa_chain

# === File Upload Handler ===
def process_files(file_list, sample_q=None, sample_a=None):
    global vectorstore, retriever, qa_chain
    all_docs = []

    for file in file_list:
        path = file.name
        loader = PyPDFLoader(path) if path.endswith(".pdf") else TextLoader(path)
        all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever, qa_chain = build_qa_chain(vectorstore)
    return  # No output to avoid Gradio warning

# === ReAct-style Agent Prompt ===
react_prompt_template = """Answer the following question using the tools available.

Question: {input}

You can use the following tools:
DuckDuckGoSearch: for any internet-based or current-events queries.

Thought: Let's figure this out step-by-step.
{agent_scratchpad}"""

# === Main Query Logic ===
def query_chain(query, qa_chain, agent_mode):
    if not qa_chain:
        return [{"role": "assistant", "content": "Please upload a knowledge base first."}]

    if agent_mode:
        from langchain.prompts import PromptTemplate

        prompt = PromptTemplate(
            input_variables=["input", "agent_scratchpad"],
            template=react_prompt_template
        )

        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent="zero-shot-react-description",
            verbose=True,
            handle_parsing_errors=True
        )

        response = agent.invoke({"input": query})["output"]
    else:
        docs = retriever.invoke(query)
        response = qa_chain.invoke({"input_documents": docs, "question": query})

    return [
        {"role": "user", "content": query},
        {"role": "assistant", "content": response}
    ]

# === Gradio UI ===
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 RAG Chatbot with Agentic Internet Search")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat History", render=True, type="messages")
            query_input = gr.Textbox(label="Enter your question")
            submit_btn = gr.Button("Ask")
        with gr.Column(scale=1):
            file_upload = gr.File(file_types=[".pdf", ".txt"], file_count="multiple", label="Upload Knowledge Base")
            agent_toggle = gr.Checkbox(label="Enable Agentic Internet Search", value=False)

    file_upload.change(fn=process_files, inputs=file_upload, outputs=[])
    submit_btn.click(fn=lambda q, a: query_chain(q, qa_chain, a), inputs=[query_input, agent_toggle], outputs=chatbot)

if __name__ == "__main__":
    demo.launch(share=True)
