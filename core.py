from multiprocessing import context
import os
from typing import Any,Dict
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain_chroma import Chroma
from langchain.tools import tool
from langchain_ollama import OllamaEmbeddings
from securitylayer import EnhancedLLMGuardrails, CustomGuardrails

# embeddings = OllamaEmbeddings(model="mxbai-embed-large")   # nomic-embed-text
embeddings = OllamaEmbeddings(model="nomic-embed-text") 
 
vectorstore = Chroma(persist_directory = 'chroma_db' ,embedding_function= embeddings)

# security layer
guardrails = EnhancedLLMGuardrails()
custom_guardrails = CustomGuardrails(blocked_topics=["politics", "gambling"])

#chatmodel
model = init_chat_model('qwen2.5',model_provider = 'ollama')

@tool(response_format='content_and_artifact')
def retrieve_content(query:str):
    """ Relevance retrived helps in final output """
    docs_retrived= vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(query)

    #serialize docs for the model 
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source','Unknown')}\n\nContent: {doc.page_content}")
        for doc in docs_retrived
    )

    return serialized ,docs_retrived

def run_llm(query:str) -> Dict[str,Any]:
    """
    run the rag pipeline to andwer a query using relevance retrived documents
    Args:
        query: the user's question
    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """   

    # checking for prompts 
    injection_check = custom_guardrails.check_prompt_injection(query)

    if injection_check['detected']:
        return {
            "answer": f"Request blocked: {injection_check['reason']}",
            "context": [],
            "blocked": True
        }

    # For (toxicity + harmful intent + PII masking)
    input_validation = guardrails.validate_input(query)
    if not input_validation['safe']:
        return {
            "answer": f" Request blocked: {input_validation['reason']}",
            "context": [],
            "blocked": True
        }

    # (PII masked)
    safe_query = input_validation['sanitized_input']

    system_prompt = (
        " You are a helpful AI assistant that answers questions about Langchain documentation"
        "You have access to a tool that retrived relevant documents"
        "Use tool to find relevant information before answering questions."
        "Always cite the sources you use in your answers."
        "If you cannot find the answer in the retrived documentation, say so"
    )

    agent = create_agent(model, tools=[retrieve_content], system_prompt=system_prompt)

    # messages list
    messages = [{"role": "user", "content": query}]

    #invoke agent
    response = agent.invoke({"messages": [{"role": "user", "content": safe_query}]})

    # extract answer from the final one
    answer = response["messages"][-1].content

    #check results
    output_validation = guardrails.validate_output(answer)
    if not output_validation['safe']:
        return {
            "answer": f"Response blocked: {output_validation['reason']}",
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
    result = run_llm(query='what are ai agents?')
    print(result["answer"])

