

import os
import json
from typing import Any, Dict, List
from dotenv import load_dotenv
load_dotenv()
from core import run_llm
from langchain_core.documents import Document
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from ragas import evaluate, EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision

# CHeck with ragas if failed then move to llm as judge
RAGAS_LOW_THRESHOLD  = 0.4   # bad v
RAGAS_MID_THRESHOLD  = 0.75  # below this  → uncertain → LLM judge
# above 0.75 → RAGAS score trusted as-is

FAITHFULNESS_CORRECTION_THRESHOLD = 0.5   # below this → self-correction
MAX_CORRECTION_ATTEMPTS           = 2

# Mistral for both RAGAS internals + LLM judge
judge_llm = ChatMistralAI(
    model="mistral-small-latest",
    api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0.0,
)
mistral_embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=os.getenv("MISTRAL_API_KEY"),
)

def run_ragas(
    query:        str,
    answer:       str,
    context_docs: List[Document],
) -> Dict[str, float]:
    """
    Runs RAGAS faithfulness, answer_relevancy, context_precision.
    No ground truth needed for any of these three.
    Returns dict of metric → float score.
    """
    context_texts = [doc.page_content for doc in context_docs]

    dataset = EvaluationDataset.from_dict({
        "user_input":          [query],
        "response":            [answer],
        "retrieved_contexts":  [context_texts],
    })

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge_llm,
        embeddings=mistral_embeddings,
    )

    scores = {
        "faithfulness":      round(float(result["faithfulness"]),      3),
        "answer_relevancy":  round(float(result["answer_relevancy"]),  3),
        "context_precision": round(float(result["context_precision"]), 3),
    }
    return scores



def _call_judge(prompt_template: ChatPromptTemplate, variables: dict) -> dict:
    chain  = prompt_template | judge_llm
    result = chain.invoke(variables)
    raw    = result.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return {"score": 0.0, "reason": f"Parse error: {raw[:200]}"}


JUDGE_FAITHFULNESS_PROMPT = ChatPromptTemplate.from_template("""
You are an evaluation assistant checking if an answer is grounded in the provided context.

Retrieved Context:
{context}

Generated Answer:
{answer}

Scoring:
- 1.0 → every claim supported by context
- 0.5 → mostly supported, minor additions
- 0.0 → significant hallucination

Respond ONLY with valid JSON:
{{"score": <0.0-1.0>, "reason": "<one sentence>"}}
""")

JUDGE_ANSWER_RELEVANCY_PROMPT = ChatPromptTemplate.from_template("""
You are an evaluation assistant checking if the answer addresses the query.

User Query:
{query}

Generated Answer:
{answer}

Scoring:
- 1.0 → directly and completely answers the query
- 0.5 → partially answers, misses key aspects
- 0.0 → does not address the query

Respond ONLY with valid JSON:
{{"score": <0.0-1.0>, "reason": "<one sentence>"}}
""")

JUDGE_CONTEXT_RELEVANCE_PROMPT = ChatPromptTemplate.from_template("""
You are an evaluation assistant checking if the retrieved context is relevant to the query.

User Query:
{query}

Retrieved Context:
{context}

Scoring:
- 1.0 → context directly and fully addresses the query
- 0.5 → partially relevant
- 0.0 → unrelated

Respond ONLY with valid JSON:
{{"score": <0.0-1.0>, "reason": "<one sentence>"}}
""")


def run_llm_judge( query: str,answer: str, context_docs: List[Document], metrics_to_judge: List[str],) -> Dict[str, Any]:
    """
    Runs LLM-as-judge only for the metrics flagged as mid-range by RAGAS.
    Returns dict of metric → {score, reason}
    """
    context_text = "\n\n".join( f"[{i+1}] Source: {doc.metadata.get('source','Unknown')}\n{doc.page_content}"
        for i, doc in enumerate(context_docs) )

    judge_results = {}

    if "faithfulness" in metrics_to_judge:
        r = _call_judge(JUDGE_FAITHFULNESS_PROMPT,
                        {"context": context_text, "answer": answer})
        judge_results["faithfulness"] = {"score": float(r["score"]), "reason": r["reason"]}

    if "answer_relevancy" in metrics_to_judge:
        r = _call_judge(JUDGE_ANSWER_RELEVANCY_PROMPT,
                        {"query": query, "answer": answer})
        judge_results["answer_relevancy"] = {"score": float(r["score"]), "reason": r["reason"]}

    if "context_precision" in metrics_to_judge:
        r = _call_judge(JUDGE_CONTEXT_RELEVANCE_PROMPT,
                        {"query": query, "context": context_text})
        judge_results["context_precision"] = {"score": float(r["score"]), "reason": r["reason"]}

    return judge_results


def resolve_scores(ragas_scores: Dict[str, float], query:str,answer:str,context_docs: List[Document],) -> Dict[str, Any]:
    """
    Logic:
        score >= 0.75  → trust RAGAS directly
        0.4 <= score < 0.75  → uncertain band → LLM judge
        score < 0.4    → trust RAGAS (clearly bad, no need for judge)
    """
    metrics_needing_judge = []
    final_scores          = {}

    for metric, score in ragas_scores.items():
        if RAGAS_LOW_THRESHOLD <= score < RAGAS_MID_THRESHOLD:
            # uncertain band — escalate
            metrics_needing_judge.append(metric)
            print(f"  [{metric}] RAGAS={score:.3f} → in uncertain band → escalating to LLM judge")
        else:
            # clear high or clear low — trust RAGAS
            final_scores[metric] = {
                "score":  score,
                "source": "ragas",
                "reason": "RAGAS score outside uncertain band — trusted directly",
            }
            band = "HIGH" if score >= RAGAS_MID_THRESHOLD else "LOW"
            print(f"  [{metric}] RAGAS={score:.3f} → {band} → trusted directly")

    # Run LLM judge only for uncertain metrics
    if metrics_needing_judge:
        print(f"\n  Running LLM judge for: {metrics_needing_judge}")
        judge_results = run_llm_judge(query, answer, context_docs, metrics_needing_judge)

        for metric, result in judge_results.items():
            final_scores[metric] = {
                "score":  result["score"],
                "source": "llm_judge",
                "reason": result["reason"],
            }
            print(f"  [{metric}] LLM judge={result['score']:.3f} — {result['reason']}")

    return final_scores



SELF_CORRECTION_PROMPT = ChatPromptTemplate.from_template("""
You are a careful AI assistant. Your previous answer may contain information not found in the provided context.

User Query:
{query}

Retrieved Context (use ONLY this):
{context}

Your Previous Answer:
{answer}

Why it was flagged:
{reason}

Instructions:
- Rewrite using ONLY information from the retrieved context.
- Do not add external knowledge or assumptions.
- If context is insufficient, say: "Based on the available context, I cannot fully answer this."
- Be concise and cite the source essay.
""")

def self_correct(  query: str, answer: str, context_docs: List[Document], reason: str,) -> str:
    context_text = "\n\n".join(
        f"[{i+1}] Source: {doc.metadata.get('source','Unknown')}\n{doc.page_content}"
        for i, doc in enumerate(context_docs)
    )
    chain  = SELF_CORRECTION_PROMPT | judge_llm
    result = chain.invoke({
        "query":   query,
        "context": context_text,
        "answer":  answer,
        "reason":  reason,
    })
    return result.content.strip()



def evaluate_and_correct( query:  str, answer:str,context_docs: List[Document])-> Dict[str, Any]:
    """
    Full pipeline:
        1. RAGAS scores all three metrics
        2. Mid-range scores → LLM judge re-evaluates
        3. If faithfulness still low → self-correction loop
        4. Returns final answer + all scores + metadata
    """

    ragas_scores = run_ragas(query, answer, context_docs)
    print(f"  RAGAS raw → {ragas_scores}")

    final_scores = resolve_scores(ragas_scores, query, answer, context_docs)

    # Self-correction if faithfulness is low
    final_answer         = answer
    correction_triggered = False
    correction_attempts  = 0

    faith_score  = final_scores["faithfulness"]["score"]
    faith_reason = final_scores["faithfulness"]["reason"]

    if faith_score < FAITHFULNESS_CORRECTION_THRESHOLD:
        print(f"\n  [Step 3] ⚠ Faithfulness={faith_score:.3f} below {FAITHFULNESS_CORRECTION_THRESHOLD} → self-correction...")
        correction_triggered = True

        for attempt in range(1, MAX_CORRECTION_ATTEMPTS + 1):
            correction_attempts += 1
            print(f"  Attempt {attempt}/{MAX_CORRECTION_ATTEMPTS}...")

            corrected = self_correct(query, final_answer, context_docs, faith_reason)

            # Re-score corrected answer with RAGAS only
            new_ragas   = run_ragas(query, corrected, context_docs)
            new_faith   = new_ragas["faithfulness"]
            print(f"  New faithfulness (RAGAS): {new_faith:.3f}")

            final_answer = corrected
            faith_score  = new_faith
            faith_reason = "Re-scored after self-correction"

            # Update faithfulness in final scores
            final_scores["faithfulness"] = {
                "score":  new_faith,
                "source": "ragas_post_correction",
                "reason": faith_reason,
            }

            if faith_score >= FAITHFULNESS_CORRECTION_THRESHOLD:
                print(f"  ✓ Faithfulness restored.")
                break
        else:
            print("  ✗ Max attempts reached — returning best corrected answer.")
    else:
        print(f"\n  Faithfulness={faith_score:.3f} OK — no correction needed.")


    return {
        "final_answer":         final_answer,
        "scores":               final_scores,
        "correction_triggered": correction_triggered,
        "correction_attempts":  correction_attempts,
    }


if __name__ == "__main__":


    query = "What does Paul Graham say about doing things that don't scale?"

    rag_result = run_llm(query)

    if rag_result["blocked"]:
        print("Blocked:", rag_result["answer"])
    else:
        eval_result = evaluate_and_correct(
            query        = query,
            answer       = rag_result["answer"],
            context_docs = rag_result["context"],  )

        print("Final Answer      :", eval_result["final_answer"])
        print("Correction Ran    :", eval_result["correction_triggered"])
        print("Correction Rounds :", eval_result["correction_attempts"])
        print("Scores            :", json.dumps(eval_result["scores"], indent=2))