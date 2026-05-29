from multiprocessing import context
import os
from typing import Any, Dict
from dotenv import load_dotenv
load_dotenv()
from evaluation import evaluate_and_correct
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool
from generic_queries import *
from securitylayer import EnhancedLLMGuardrails, CustomGuardrails
import json

embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=os.getenv("MISTRAL_API_KEY"),
)

CHROMA_DIR      = "chroma_store"
COLLECTION_NAME = "New_collection"
BATCH_SIZE      = 20

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
    collection_metadata={"hnsw:space": "cosine"},
)

#  security layer 
guardrails = EnhancedLLMGuardrails()
custom_guardrails = CustomGuardrails(
    blocked_topics=[
        "politics", "gambling", "weapon", "bomb",
        "explosive", "nuclear", "poison", "malware", "hack",
    ]
)

#  chat model 
model = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")


@tool(response_format="content_and_artifact")
def retrieve_content(query: str):
    """
    Use semantic retrieval — fetches relevant chunks from ChromaDB
    using HNSW-indexed cosine similarity search.
    Always call this before answering any LangChain-related question.
    Relevant retrieved chunks are used to ground the final output.
    """
    docs_retrieved = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    ).invoke(query)

    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}"
        for doc in docs_retrieved
    )
    return serialized, docs_retrieved


def _invoke_agent(agent, messages: list):
    return agent.invoke({"messages": messages})


def run_llm(query: str, chat_history: list = None) -> Dict[str, Any]:
    """
    Runs the RAG pipeline to answer a query using retrieved LangChain documentation.

    Args:
        query:        The user's question.
        chat_history: Optional prior conversation messages.

    Returns:
        {
            "answer":  str,
            "context": List[Document],
            "blocked": bool,
        }
    """
    # generic query shortcut
    general_reply = handle_generic_query(query)
    if general_reply:
        return {"answer": general_reply, "context": [], "blocked": False}

    # prompt injection check 
    injection_check = custom_guardrails.check_prompt_injection(query)
    if injection_check["detected"]:
        return {
            "answer": (
                f"Request blocked: {injection_check['reason']}. "
                "I'm designed to help with LangChain documentation only. "
                "Please ask a question about LangChain."
            ),
            "context": [],
            "blocked": True,
        }

    #  toxicity / harmful intent / PII masking 
    input_validation = guardrails.validate_input(query)
    if not input_validation["safe"]:
        return {
            "answer": (
                f"Request blocked: {input_validation.get('reason', 'unsafe input')}. "
                "I'm designed to help with LangChain documentation only. "
                "Please ask a question about LangChain."
            ),
            "context": [],
            "blocked": True,
        }

    safe_query = input_validation["sanitized_input"]

    # system prompt
    system_prompt = (
        "You are a helpful AI assistant that answers questions strictly based on "
        "LangChain documentation. "
        "You have access to a retrieval tool that fetches relevant documentation "
        "chunks from a vector store. "
        "ALWAYS call the retrieve_content tool before answering any question. "
        "Ground your answer STRICTLY in the retrieved context. "
        "Do NOT use any pre-trained knowledge about LangChain that is not present "
        "in the retrieved chunks. "
        "If the retrieved context does not contain the answer, say: "
        "'The provided documentation does not contain this information.' "
        "Always cite the source document in your answer."
    )

    agent    = create_agent(model, tools=[retrieve_content], system_prompt=system_prompt)
    history  = chat_history or []
    messages = history + [{"role": "user", "content": safe_query}]

    response = _invoke_agent(agent, messages)

    raw_answer = response["messages"][-1].content
    answer = (
        " ".join(
            block["text"] if isinstance(block, dict) else str(block)
            for block in raw_answer
            if not isinstance(block, dict) or block.get("type") == "text"
        )
        if isinstance(raw_answer, list)
        else raw_answer
    )

    # output safety check 
    output_validation = guardrails.validate_output(answer)
    if not output_validation["safe"]:
        return {
            "answer": (
                f"Response blocked: {output_validation.get('reason', 'unsafe output')}. "
                "The generated response was flagged by the safety filter. "
                "Please rephrase your question or ask something related to LangChain documentation."
            ),
            "context": [],
            "blocked": True,
        }

    #  extract retrieved docs from ToolMessage artifacts
    context_documents = []
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            context_documents.extend(message.artifact)

    return {
        "answer":  output_validation["sanitized_output"],
        "context": context_documents,
        "blocked": False,
    }


if __name__ == "__main__":

 
    query      = "What advice does Paul Graham give about doing things that don't scale?"
    rag_result = run_llm(query=query)
 
    if rag_result["blocked"]:
        print("Blocked:", rag_result["answer"])
    else:
        eval_result  = evaluate_and_correct(
            query        = query,
            answer       = rag_result["answer"],
            context_docs = rag_result["context"],
        )
        final_answer = eval_result["final_answer"]
        confidence   = eval_result["confidence"]
 
        print("Final Answer :", final_answer)
        print("Confidence   :", confidence["level"], "—", confidence["message"])
        print("Flags        :", confidence["flags"])
        print("Log file     :", eval_result["log_file"])
        print("Scores       :", json.dumps(
            {k: v["score"] for k, v in eval_result["scores"].items()},
            indent=2,
        ))
 