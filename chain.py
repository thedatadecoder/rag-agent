### chain.py
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

# NEW:
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

def build_qa_chain(task, retriever, llm_model, model_kwargs):
    hf_pipeline = pipeline(task, model=llm_model, tokenizer=llm_model, model_kwargs=model_kwargs, max_new_tokens=512)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
