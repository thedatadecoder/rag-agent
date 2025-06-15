from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoConfig
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from rouge_score import rouge_scorer
from bert_score import score as bertscore_score
from itertools import product
import torch

# ----------------------------
# Model-specific configurations
# ----------------------------
MODEL_CONFIGS = {
    "google/flan-t5-base": {
        "supports_sampling": False,
        "pipeline_task": "text2text-generation",
        "model_type": "seq2seq",
        "grid": {
            "embedding_model": [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/paraphrase-MiniLM-L3-v2"
            ],
            "retriever_chunk_size": [500, 800],
            "retriever_overlap": [100, 150],
        },
    },
    # "HuggingFaceH4/zephyr-7b-beta": {
    #     "supports_sampling": True,
    #     "pipeline_task": "text-generation",
    #     "model_type": "causal",
    #     "grid": {
    #         "embedding_model": [
    #             "sentence-transformers/all-MiniLM-L6-v2",
    #             "BAAI/bge-small-en"
    #         ],
    #         "retriever_chunk_size": [500, 800],
    #         "retriever_overlap": [100, 150],
    #         "temperature": [0.3, 0.7], 
    #         "top_p": [0.8, 0.9],             
    #         "top_k": [50, 100],              
    #         "do_sample": [True, False]
    #     }
    # },
}

# ----------------------------
# Grid search combinations
# ----------------------------
def generate_param_combinations(param_grid):
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    for combination in product(*values):
        yield dict(zip(keys, combination))

# ----------------------------
# Evaluation using ROUGE or BERTScore
# ----------------------------
def evaluate_chain(chain, question, reference_answer=None, method="rouge"):
    try:
        raw_output = chain.invoke(question)

        if isinstance(raw_output, list):
            if raw_output and isinstance(raw_output[0], dict) and 'generated_text' in raw_output[0]:
                generated_answer = raw_output[0]['generated_text']
            else:
                generated_answer = ' '.join(str(x) for x in raw_output)
        elif isinstance(raw_output, dict):
            generated_answer = raw_output.get('generated_text', str(raw_output))
        else:
            generated_answer = str(raw_output)

        if not reference_answer:
            return 0.0

        if method == "rouge":
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            scores = scorer.score(reference_answer, generated_answer)
            return scores['rougeL'].fmeasure

        elif method == "bertscore":
            P, R, F1 = bertscore_score([generated_answer], [reference_answer], lang="en", verbose=False)
            return F1[0].item()

        else:
            raise ValueError(f"Unsupported evaluation method: {method}")

    except Exception as e:
        print(f"Evaluation failed: {e}")
        return 0.0

# ----------------------------
# Main tuning logic
# ----------------------------
def tune_rag(knowledge_path, sample_question, reference_answer, expected_keywords=None):
    best_config = {}
    best_score = -1

    for model_name, config in MODEL_CONFIGS.items():
        grid = config["grid"]
        supports_sampling = config["supports_sampling"]
        pipeline_task = config["pipeline_task"]
        model_type = config.get("model_type", "causal")

        for params in generate_param_combinations(grid):
            emb_model = params["embedding_model"]
            chunk_size = params["retriever_chunk_size"]
            overlap = params["retriever_overlap"]

        
            # Load and split knowledge base
            loader = DirectoryLoader(
                knowledge_path,
                glob="**/*.*",
                loader_cls=lambda path: PyPDFLoader(path) if path.endswith(".pdf") else TextLoader(path)
            )
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=min(chunk_size, 400), chunk_overlap=min(overlap, 50))

            chunks = splitter.split_documents(docs)

            # Embed
            embeddings = HuggingFaceEmbeddings(model_name=emb_model)
            vectorstore = FAISS.from_documents(chunks, embeddings)
            retriever = vectorstore.as_retriever()

            # Load tokenizer and set max token limit
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.model_max_length = 512

            # Load model based on type
            if model_type == "seq2seq":
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

            model_kwargs = {}
            if supports_sampling:
                model_kwargs = {
                    "temperature": params["temperature"],
                    "top_p": params["top_p"],
                    "top_k": params["top_k"],
                    "do_sample": params["do_sample"]
                }

            pipe = pipeline(
                task=pipeline_task,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                truncation=True,
                # return_tensors="pt",
                **model_kwargs  # unpack directly here
            )


            llm = HuggingFacePipeline(pipeline=pipe)
            qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

            score = evaluate_chain(
                chain=qa_chain,
                question=sample_question,
                reference_answer=reference_answer,
                method="bertscore"
            )

            if score > best_score:
                best_score = score
                print("Best Score:", best_score)
                best_config["params"] = params.copy()
                best_config["model_kwargs"] = model_kwargs
                best_config["model_attribs"] = {
                    "llm_model": model_name,
                    "score": score,
                    "pipeline_task": pipeline_task}

    return best_config
