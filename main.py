### main.py
from ingest import ingest_documents
from retrieval import load_retriever
from chain import build_qa_chain
from rag_tuning import tune_rag

def main():
    print("Tuning hyperparameters based on the knowledge base...")
    config = tune_rag(
        "knowledge_base",
        sample_question="What is LangChain?",
        reference_answer="LangChain is a framework for building LLM-powered applications with tools like agents and retrieval.",
        expected_keywords=None #["framework", "agents", "retrieval"]
    )

    print("Best configuration found:")
    print(config)

    # Now use this config to reinitialize everything
    ingest_documents("knowledge_base/", config["params"]["embedding_model"], config["params"]["retriever_chunk_size"], config["params"]["retriever_overlap"])

    retriever = load_retriever(config["params"]["embedding_model"])

    
    qa_chain = build_qa_chain(config["model_attribs"]["pipeline_task"], retriever, config["model_attribs"]["llm_model"], config["model_kwargs"])

    # Q&A loop
    while True:
        query = input("\nYour question: ")
        if query.lower() == "exit":
            break
        print("\nAnswer:", qa_chain.invoke(query))


if __name__ == "__main__":
    main()
