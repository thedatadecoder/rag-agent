# rag_tuning.py

from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from rouge_score import rouge_scorer
from bert_score import score as bertscore_score
import os
from itertools import product

# Define model-specific hyperparameter grids
MODEL_CONFIGS = {
    "google/flan-t5-base": {
        "supports_sampling": False,
        "pipeline_task": "text2text-generation",
        "grid": {
            "embedding_model": [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/paraphrase-MiniLM-L3-v2"
            ],
            "retriever_chunk_size": [500, 800],
            "retriever_overlap": [100, 150],
        },
    },
    "HuggingFaceH4/zephyr-7b-beta": {
        "supports_sampling": True,
        "pipeline_task": "text-generation",
        "grid": {
            "embedding_model": [
                "sentence-transformers/all-MiniLM-L6-v2",
                "BAAI/bge-small-en"
            ],
            "retriever_chunk_size": [500, 800],
            "retriever_overlap": [100, 150],
            "temperature": [0.3, 0.7], 
            "top_p": [0.8, 0.9],             
            "top_k": [50, 100],              
            "do_sample": [True, False]
        }
    },
}

# Generate all key-value parameter combinations
def generate_param_combinations(param_grid):
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    for combination in product(*values):
        yield dict(zip(keys, combination))

# Evaluate the chain on a simple heuristic
def evaluate_chain_(chain, question, expected_keywords=None):
    try:
        answer = chain.invoke(question)
        if expected_keywords:
            score = sum(1 for kw in expected_keywords if kw.lower() in answer.lower()) / len(expected_keywords)
            return score
        return 1.0
    except Exception as e:
        return 0.0
    


def evaluate_chain(chain, question, reference_answer=None, method="rouge"):
    try:
        generated_answer = chain.invoke(question)

        if not reference_answer:
            return 0.0

        if method == "rouge":
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            scores = scorer.score(reference_answer, generated_answer)
            return scores['rougeL'].fmeasure  # Value between 0 and 1

        elif method == "bertscore":
            P, R, F1 = bertscore_score([generated_answer], [reference_answer], lang="en", verbose=False)
            return F1[0].item()  # BERTScore F1

        else:
            raise ValueError(f"Unsupported evaluation method: {method}")

    except Exception as e:
        print(f"Evaluation failed: {e}")
        return 0.0


def tune_rag(knowledge_path, sample_question, reference_answer, expected_keywords=None):
    best_config = {}
    best_score = -1

    for model_name, config in MODEL_CONFIGS.items():
        grid = config["grid"]
        supports_sampling = config["supports_sampling"]

        for params in generate_param_combinations(grid):
            emb_model = params["embedding_model"]
            chunk_size = params["retriever_chunk_size"]
            overlap = params["retriever_overlap"]

            # Load and split documents
            loader = DirectoryLoader(
                knowledge_path,
                glob="**/*.*",
                loader_cls=lambda path: PyPDFLoader(path) if path.endswith(".pdf") else TextLoader(path)
            )
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
            chunks = splitter.split_documents(docs)

            # Embed and retrieve
            embeddings = HuggingFaceEmbeddings(model_name=emb_model)
            vectorstore = FAISS.from_documents(chunks, embeddings)
            retriever = vectorstore.as_retriever()

            # Build generation model
            model_kwargs = {}
            if supports_sampling:
                model_kwargs = {
                    "temperature": params["temperature"],
                    "top_p": params["top_p"],
                    "top_k": params["top_k"],
                    "do_sample": params["do_sample"]
                }

            pipe = pipeline(
                task=config["pipeline_task"],
                model=model_name,
                tokenizer=model_name,
                max_new_tokens=512,
                model_kwargs=model_kwargs
            )

            llm = HuggingFacePipeline(pipeline=pipe)
            qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

            # score = evaluate_chain(qa_chain, sample_question, expected_keywords)
            score = evaluate_chain(
                        chain=qa_chain,
                        question=sample_question,
                        reference_answer=reference_answer,
                        method="bertscore"  # or "rouge"
                    )

            if score > best_score:
                best_score = score
                print("Best Score:", best_score)
                best_config = params.copy()
                best_config.update({"llm_model": model_name, "score": score})

    return best_config

# This file should now be imported and called from main.py
