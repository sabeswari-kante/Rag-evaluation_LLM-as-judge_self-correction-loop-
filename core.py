from multiprocessing import context
import os
from typing import Any,Dict
from dotenv import load_dotenv
load_dotenv()
from evaluation import evaluate_and_correct
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain_mistralai import MistralAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.tools import tool 
from langchain_chroma import Chroma
from generic_queries import *
from securitylayer import EnhancedLLMGuardrails, CustomGuardrails


embeddings = MistralAIEmbeddings(model="mistral-embed",api_key=os.getenv("MISTRAL_API_KEY")) 
 
CHROMA_DIR       = "chroma_store"
COLLECTION_NAME  = "New_collection"
BATCH_SIZE       = 20

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
    collection_metadata={"hnsw:space": "cosine"},)


# security layer
guardrails = EnhancedLLMGuardrails()
custom_guardrails = CustomGuardrails(blocked_topics=["politics", "gambling", "weapon", "bomb", "explosive", "nuclear", "poison", "malware", "hack"])

#chatmodel

model = init_chat_model("mistral-small-latest",model_provider = 'mistralai')

@tool(response_format='content_and_artifact')
def retrieve_content(query:str):
    """
    Use semantic retrieval — fetches relevant chunks from chromadb. 
    using HNSW-indexed cosine similarity search.
    Always call this before answering any esays realted question.
    Use this Relevance retrived helps in final output 
    """
    docs_retrived= vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(query)

    #serialize docs for the model 
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source','Unknown')}\n\nContent: {doc.page_content}")
        for doc in docs_retrived
    )

    return serialized ,docs_retrived

def _invoke_agent(agent, messages: list):
    return agent.invoke({"messages": messages})
    
def run_llm(query:str, chat_history: list = None) -> Dict[str, Any]:
    """
    run the rag pipeline to andwer a query using relevance retrived documents
    Args:
        query: the user's question
    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """   
    # general queires 
    general_reply = handle_generic_query(query)
    if general_reply:
        return {
            "answer": general_reply,
            "context": [],
            "blocked": False
        }

    # checking for prompts 
    injection_check = custom_guardrails.check_prompt_injection(query)

    if injection_check['detected']:
        return {
            "answer": f"Request blocked: {injection_check['reason']}. I'm designed to help with Paul Graham's essays only. I can't assist with that kind of request. Try asking something query",
            "context": [],
            "blocked": True
        }

    # For (toxicity + harmful intent + PII masking)
    input_validation = guardrails.validate_input(query)
    if not input_validation['safe']:
        return {
            "answer": f"Request blocked: {injection_check['reason']}. I'm designed to help with Paul Graham's essays only. I can't assist with that kind of request. Try asking something query",
            "context": [],
            "blocked": True
        }

    # (PII masked)
    safe_query = input_validation['sanitized_input']

    system_prompt = (
        "You are a helpful AI assistant that answers questions based on Paul Graham's essays. "
        "You have access to a retrieval tool that fetches relevant essay chunks from vector . "
        "Always call the retrieve_content tool before answering any question. "
        "Ground your answer strictly in retrieved context. "
        "Always cite the source essay in your answer. "
        "If you cannot find the answer in the retrived documentation, say so"
    )

    agent = create_agent(model, tools=[retrieve_content], system_prompt=system_prompt)

    history = chat_history or []
    # messages list
    messages = history + [{"role": "user", "content": safe_query}]

    #invoke agent
    response = _invoke_agent(agent, messages)
    # response = agent.invoke({"messages": [{"role": "user", "content": safe_query}]})

    # extract answer from the final one
    raw_answer = response["messages"][-1].content

    answer = (" ".join(
            block["text"] if isinstance(block, dict) else str(block)
            for block in raw_answer
            if not isinstance(block, dict) or block.get("type") == "text"
        )
        if isinstance(raw_answer,list)
    else raw_answer ) 

    #check results
    output_validation = guardrails.validate_output(answer)
    if not output_validation['safe']:
        return {
            "answer": f"Response blocked: {output_validation['reason']}The generated response was flagged by my safety filter. Please rephrase your question or ask something related to LangChain documentation.",
            "context": [],
            "blocked": True
        }

    # extract context documents from toolmessage artifacts
    context_documents = []
    for message in response["messages"]:
        #check if this is a toolmessage with artifacts
        if isinstance(message,ToolMessage) and hasattr(message,"artifact"):
            context_documents.extend(message.artifact)
    
    return {"answer": output_validation['sanitized_output'],
        "context": context_documents,
        "blocked": False
    }

if __name__ == "__main__":
    result = run_llm(query="What does Paul Graham say about doing things that don't scale?")
    print(result["answer"])
    eval_result = evaluate_and_correct(
        query        = query,
        answer       = rag_result["answer"],
        context_docs = rag_result["context"],
    )

    # Always use this — corrected if needed
    final_answer = eval_result["final_answer"]
    scores       = eval_result["scores"]