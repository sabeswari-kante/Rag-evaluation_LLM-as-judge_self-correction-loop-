"""
evaluation.py — Self-correction RAG evaluation pipeline.

Metrics (no RAGAS):
    1. Context Relevance  → cosine similarity (fast) + LLM judge (borderline)
    2. Faithfulness       → LLM judge directly (no fast metric — reasoning required)
    3. Answer Relevance   → cosine similarity (fast) + LLM judge (borderline)

Correction methods:
    1. Context Relevance  → query rewriting + reranking (parallel)
    2. Faithfulness       → stronger system prompt + claim verification
    3. Answer Relevance   → aspect decomposition rewrite

Rules:
    - score >= 0.7  → pass, no correction
    - 0.4–0.7       → LLM judge (for cosine-scored metrics)
    - < 0.4         → correction immediately
    - max 2 correction rounds per metric
    - cascade: improved chunks from Step 1 flow into Step 2 and Step 3
    - always carry best score/answer/chunks seen across all rounds
"""

import os
import json
import asyncio
import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
load_dotenv()

import numpy as np
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

from logger import build_log, save_query_log

# thresholds 
HIGH_THRESHOLD   = 0.7   # pass immediately
MID_THRESHOLD  = 0.7
LOW_THRESHOLD    = 0.4   # skip judge, correct immediately
MAX_ROUNDS       = 2

# models 
# Judge LLM: Groq Llama (free tier, fast inference)
judge_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1,
)
# Embeddings: Mistral (unchanged)
embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=os.getenv("MISTRAL_API_KEY"),
)

CHROMA_DIR      = "chroma_store"
COLLECTION_NAME = "New_collection"

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
    collection_metadata={"hnsw:space": "cosine"},
)


def _embed(texts: List[str]) -> np.ndarray:
    """Embed a list of strings, returns (N, D) float array."""
    vecs = embeddings.embed_documents(texts)
    return np.array(vecs, dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _context_text(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[{i+1}] Source: {doc.metadata.get('source','Unknown')}\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )


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


def score_context_relevance_cosine(query: str, docs: List[Document]) -> float:
    """
    Average cosine similarity between the query embedding and each chunk embedding.
    Handles long-doc / short-query asymmetry well because embeddings compress
    full semantic meaning regardless of length.
    """
    if not docs:
        return 0.0
    texts     = [doc.page_content for doc in docs]
    all_texts = [query] + texts
    vecs      = _embed(all_texts)
    q_vec     = vecs[0]
    scores    = [_cosine(q_vec, vecs[i + 1]) for i in range(len(texts))]
    return round(float(np.mean(scores)), 3)


def score_answer_relevance_cosine(query: str, answer: str) -> float:
    """
    Cosine similarity between the query and the answer embeddings.
    High → answer is topically aligned with what was asked.
    """
    vecs = _embed([query, answer])
    return round(_cosine(vecs[0], vecs[1]), 3)



JUDGE_CONTEXT_RELEVANCE = ChatPromptTemplate.from_template("""
You are an evaluation assistant. Judge whether the retrieved context is relevant to the query.

User Query:
{query}

Retrieved Context:
{context}

Scoring:
- 1.0 → context directly and fully addresses the query
- 0.5 → partially relevant, some useful chunks mixed with noise
- 0.0 → unrelated to the query

Respond ONLY with valid JSON:
{{"score": <0.0-1.0>, "reason": "<one sentence>"}}
""")

JUDGE_FAITHFULNESS = ChatPromptTemplate.from_template("""
You are an evaluation assistant. Check if the answer is strictly grounded in the retrieved context.

Retrieved Context:
{context}

Generated Answer:
{answer}

Scoring:
- 1.0 → every claim in the answer is supported by the context
- 0.5 → mostly supported, minor additions from outside knowledge
- 0.0 → significant hallucination or contradiction with context

Also classify the failure mode if score < 0.7:
- "hallucination_by_addition"  → answer adds facts not present in context
- "hallucination_by_contradiction" → answer contradicts what context says
- "none" → no failure

Respond ONLY with valid JSON:
{{"score": <0.0-1.0>, "reason": "<one sentence>", "failure_mode": "<type>"}}
""")

JUDGE_ANSWER_RELEVANCE = ChatPromptTemplate.from_template("""
You are an evaluation assistant. Check if the answer fully addresses the user's query.

User Query:
{query}

Generated Answer:
{answer}

Scoring:
- 1.0 → directly and completely answers all parts of the query
- 0.5 → partially answers, misses key aspects
- 0.0 → does not address the query

Respond ONLY with valid JSON:
{{"score": <0.0-1.0>, "reason": "<one sentence>"}}
""")




QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_template("""
You are a search query expert for Paul Graham essays.

Original query: {query}

Rewrite this into 3 alternative search queries that:
- Use precise Paul Graham terminology
- Cover different angles of the same question
- Are concise (under 15 words each)

Respond ONLY with valid JSON:
{{"queries": ["<query1>", "<query2>", "<query3>"]}}
""")

FAITHFULNESS_CORRECTION_PROMPT = ChatPromptTemplate.from_template("""
You are a careful AI assistant answering questions about Paul Graham essays.

STRICT RULES:
- Answer ONLY using information explicitly present in the retrieved context below.
- Do NOT include any information not stated in the context.
- Do NOT use your pre-trained knowledge about Paul Graham.
- If the context does not contain the answer, say: "The provided documentation does not contain this information."
- Every claim you make must be traceable to a specific chunk in the context.
- Do NOT say anything that is not in the context.

User Query:
{query}

Retrieved Context (YOUR ONLY SOURCE):
{context}

Previous Answer (was flagged for: {failure_mode}):
{answer}

Reason it was flagged:
{reason}

Write a corrected answer below, citing the source chunk for each claim:
""")

CLAIM_VERIFICATION_PROMPT = ChatPromptTemplate.from_template("""
You are a fact-checker. For each claim in the answer, verify if it is supported by the context.

Retrieved Context:
{context}

Answer to verify:
{answer}

For each claim:
1. State the claim
2. Is it supported? (yes / partial / no)
3. Which chunk supports it (or none)

Then provide a revised answer that removes or softens any unsupported claims.
Add a citation tag like [Chunk N] after each supported claim.

Respond ONLY with valid JSON:
{{
  "claim_analysis": [
    {{"claim": "...", "supported": "yes|partial|no", "chunk": "N or none"}}
  ],
  "revised_answer": "..."
}}
""")

ASPECT_DECOMPOSE_PROMPT = ChatPromptTemplate.from_template("""
You are a question analyst. Break the following question into individual sub-questions.

Question: {query}

Rules:
- Each sub-question should be answerable independently
- Cover all aspects of the original question
- Keep sub-questions concise

Respond ONLY with valid JSON:
{{"sub_questions": ["<q1>", "<q2>", "<q3>"]}}
""")

ASPECT_REWRITE_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful AI assistant answering questions about Paul Graham Essays.

The current answer is missing responses to some sub-questions.

Original Question: {query}

Retrieved Context (use ONLY this):
{context}

Current Answer:
{answer}

Missing Sub-questions not addressed:
{missing}

Rewrite the answer to fully address ALL sub-questions using ONLY the retrieved context.
If context doesn't cover a sub-question, say so explicitly for that part.
""")



# CONTEXT RELEVANCE


def _rewrite_queries(query: str) -> List[str]:
    result = _call_judge(QUERY_REWRITE_PROMPT, {"query": query})
    return result.get("queries", [query])


def _rerank_docs(query: str, docs: List[Document]) -> List[Document]:
    """
    Re-rank docs by cosine similarity to query, descending.
    Acts as a fast cross-encoder substitute using the same embedding model.
    """
    if not docs:
        return docs
    texts  = [doc.page_content for doc in docs]
    vecs   = _embed([query] + texts)
    q_vec  = vecs[0]
    scored = [(docs[i], _cosine(q_vec, vecs[i + 1])) for i in range(len(docs))]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in scored]


def _retrieve_for_query(query: str, k: int = 3) -> List[Document]:
    return vectorstore.as_retriever(search_kwargs={"k": k}).invoke(query)


def _parallel_rewrite_and_rerank(
    original_query: str,
    original_docs:  List[Document],
) -> Tuple[List[Document], List[Document]]:
    """
    Runs query rewriting and reranking in parallel threads.
    Returns (rewritten_docs, reranked_docs).
    """
    def do_rewrite():
        alt_queries = _rewrite_queries(original_query)
        seen, merged = set(), []
        for q in alt_queries:
            for doc in _retrieve_for_query(q):
                key = doc.page_content[:100]
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)
        return merged

    def do_rerank():
        return _rerank_docs(original_query, original_docs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f_rewrite = ex.submit(do_rewrite)
        f_rerank  = ex.submit(do_rerank)
        rewritten_docs = f_rewrite.result()
        reranked_docs  = f_rerank.result()

    return rewritten_docs, reranked_docs


def _merge_and_dedupe(
    query: str,
    rewritten_docs: List[Document],
    reranked_docs:  List[Document],
    k: int = 5,
) -> List[Document]:
    """Merge both doc lists, deduplicate, re-rank combined set, return top-k."""
    seen, merged = set(), []
    for doc in rewritten_docs + reranked_docs:
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            merged.append(doc)
    return _rerank_docs(query, merged)[:k]


def evaluate_context_relevance(
    query: str,
    docs:  List[Document],
) -> Tuple[float, List[Document], Dict]:
    """
    Returns (final_score, best_docs, metric_log_block).

    Cascade output: best_docs are what Faithfulness + Answer Relevance should use.
    """
    metric_log = {
        "initial_metric_score": None,
        "llm_judge_score":      None,
        "status":               "passed",
        "failure_mode":         None,
        "corrections":          [],
        "final_score":          None,
    }

    best_score = score_context_relevance_cosine(query, docs)
    best_docs  = docs
    metric_log["initial_metric_score"] = best_score

    print(f"\n[Context Relevance] Cosine={best_score:.3f}")

    # ── decide path 
    if best_score >= HIGH_THRESHOLD:
        print("  → PASS")
        metric_log["final_score"] = best_score
        return best_score, best_docs, metric_log

    needs_correction = False

    if LOW_THRESHOLD <= best_score < HIGH_THRESHOLD:
        # borderline → LLM judge
        r = _call_judge(JUDGE_CONTEXT_RELEVANCE,
                        {"query": query, "context": _context_text(docs)})
        judge_score = float(r.get("score", 0.0))
        metric_log["llm_judge_score"] = judge_score
        print(f"  → Borderline, LLM judge={judge_score:.3f} — {r.get('reason','')}")

        if judge_score >= HIGH_THRESHOLD:
            best_score = judge_score
            metric_log["status"]      = "passed"
            metric_log["final_score"] = best_score
            return best_score, best_docs, metric_log
        else:
            needs_correction = True
    else:
        # clearly low
        needs_correction = True

    #  correction loop (max 2 rounds) 
    if needs_correction:
        metric_log["status"] = "triggered_correction"

    for rnd in range(1, MAX_ROUNDS + 1):
        print(f"  → Correction round {rnd}...")

        rewritten_docs, reranked_docs = _parallel_rewrite_and_rerank(query, best_docs)
        combined_docs                 = _merge_and_dedupe(query, rewritten_docs, reranked_docs)

        # score all three candidates
        score_rewritten = score_context_relevance_cosine(query, rewritten_docs)
        score_reranked  = score_context_relevance_cosine(query, reranked_docs)
        score_combined  = score_context_relevance_cosine(query, combined_docs)

        candidates = [
            (score_combined,  combined_docs,  ["query_rewriting", "reranking"]),
            (score_rewritten, rewritten_docs, ["query_rewriting"]),
            (score_reranked,  reranked_docs,  ["reranking"]),
        ]
        best_candidate = max(candidates, key=lambda x: x[0])
        round_score, round_docs, methods = best_candidate

        improved = round_score > best_score
        if improved:
            best_score = round_score
            best_docs  = round_docs

        metric_log["corrections"].append({
            "round":        rnd,
            "methods_used": methods,
            "score_after":  round_score,
            "improved":     improved,
        })
        print(f"     Combined={score_combined:.3f}  Rewritten={score_rewritten:.3f}  Reranked={score_reranked:.3f}")
        print(f"     Best this round={round_score:.3f} (improved={improved})")

        if best_score >= HIGH_THRESHOLD:
            metric_log["status"] = "triggered_correction"
            break

    # if still not passing, status stays triggered_correction or becomes failed
    if best_score < HIGH_THRESHOLD:
        metric_log["status"] = "triggered_correction_failed"

    metric_log["final_score"] = best_score
    return best_score, best_docs, metric_log



# STEP 2 — FAITHFULNESS
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_faithfulness(
    query:   str,
    answer:  str,
    docs:    List[Document],   # <-- always the best_docs from Step 1
) -> Tuple[float, str, Dict]:
    """
    Returns (final_score, best_answer, metric_log_block).
    """
    metric_log = {
        "initial_metric_score": None,
        "llm_judge_score":      None,
        "status":               "passed",
        "failure_mode":         None,
        "corrections":          [],
        "final_score":          None,
    }

    ctx = _context_text(docs)

    # faithfulness uses LLM judge directly as first scorer
    r           = _call_judge(JUDGE_FAITHFULNESS, {"context": ctx, "answer": answer})
    init_score  = float(r.get("score", 0.0))
    failure_mode = r.get("failure_mode", "none")
    reason      = r.get("reason", "")

    metric_log["initial_metric_score"] = init_score
    metric_log["llm_judge_score"]      = init_score   # same call — LLM is the first scorer
    metric_log["failure_mode"]         = failure_mode

    print(f"\n[Faithfulness] LLM judge={init_score:.3f} | mode={failure_mode} | {reason}")

    best_score  = init_score
    best_answer = answer

    if best_score >= HIGH_THRESHOLD:
        print("  → PASS")
        metric_log["final_score"] = best_score
        return best_score, best_answer, metric_log

    metric_log["status"] = "triggered_correction"

    for rnd in range(1, MAX_ROUNDS + 1):
        print(f"  → Correction round {rnd}...")

        # Layer 1: stronger system prompt regeneration
        corrected_raw = _call_judge(
            FAITHFULNESS_CORRECTION_PROMPT,
            {
                "query":        query,
                "context":      ctx,
                "answer":       best_answer,
                "failure_mode": failure_mode,
                "reason":       reason,
            },
        )
        # The correction prompt returns free text via judge_llm directly
        chain    = FAITHFULNESS_CORRECTION_PROMPT | judge_llm
        corrected_answer = chain.invoke({
            "query":        query,
            "context":      ctx,
            "answer":       best_answer,
            "failure_mode": failure_mode,
            "reason":       reason,
        }).content.strip()

        # Layer 2: claim verification + citation
        claim_result = _call_judge(
            CLAIM_VERIFICATION_PROMPT,
            {"context": ctx, "answer": corrected_answer},
        )
        verified_answer = claim_result.get("revised_answer", corrected_answer)

        # re-score verified answer
        r2         = _call_judge(JUDGE_FAITHFULNESS, {"context": ctx, "answer": verified_answer})
        new_score  = float(r2.get("score", 0.0))
        failure_mode = r2.get("failure_mode", "none")
        reason     = r2.get("reason", "")

        improved = new_score > best_score
        if improved:
            best_score  = new_score
            best_answer = verified_answer

        metric_log["corrections"].append({
            "round":        rnd,
            "methods_used": ["stronger_system_prompt", "claim_verification"],
            "score_after":  new_score,
            "improved":     improved,
        })
        print(f"     New score={new_score:.3f} (improved={improved})")

        if best_score >= HIGH_THRESHOLD:
            break

    if best_score < HIGH_THRESHOLD:
        metric_log["status"] = "triggered_correction_failed"

    metric_log["final_score"]  = best_score
    metric_log["failure_mode"] = failure_mode   # final failure mode
    return best_score, best_answer, metric_log


#ANSWER RELEVANCE


def _decompose_question(query: str) -> List[str]:
    result = _call_judge(ASPECT_DECOMPOSE_PROMPT, {"query": query})
    return result.get("sub_questions", [query])


def _find_missing_aspects(
    query:         str,
    answer:        str,
    sub_questions: List[str],
) -> List[str]:
    """
    For each sub-question, check cosine similarity with the answer.
    Those below 0.5 are considered 'missing'.
    """
    if not sub_questions:
        return []
    texts = [answer] + sub_questions
    vecs  = _embed(texts)
    a_vec = vecs[0]
    missing = []
    for i, sq in enumerate(sub_questions):
        sim = _cosine(a_vec, vecs[i + 1])
        if sim < 0.5:
            missing.append(sq)
    return missing


def evaluate_answer_relevance(
    query:  str,
    answer: str,
    docs:   List[Document],   # best_docs from Step 1 (for rewrite context)
) -> Tuple[float, str, Dict]:
    """
    Returns (final_score, best_answer, metric_log_block).
    """
    metric_log = {
        "initial_metric_score": None,
        "llm_judge_score":      None,
        "status":               "passed",
        "failure_mode":         None,
        "corrections":          [],
        "final_score":          None,
    }

    best_score  = score_answer_relevance_cosine(query, answer)
    best_answer = answer
    metric_log["initial_metric_score"] = best_score

    print(f"\n[Answer Relevance] Cosine={best_score:.3f}")

    if best_score >= HIGH_THRESHOLD:
        print("  → PASS")
        metric_log["final_score"] = best_score
        return best_score, best_answer, metric_log

    needs_correction = False

    if LOW_THRESHOLD <= best_score < HIGH_THRESHOLD:
        r = _call_judge(JUDGE_ANSWER_RELEVANCE, {"query": query, "answer": answer})
        judge_score = float(r.get("score", 0.0))
        metric_log["llm_judge_score"] = judge_score
        print(f"  → Borderline, LLM judge={judge_score:.3f} — {r.get('reason','')}")

        if judge_score >= HIGH_THRESHOLD:
            best_score = judge_score
            metric_log["final_score"] = best_score
            return best_score, best_answer, metric_log
        else:
            needs_correction = True
    else:
        needs_correction = True

    if needs_correction:
        metric_log["status"] = "triggered_correction"

    ctx = _context_text(docs)

    for rnd in range(1, MAX_ROUNDS + 1):
        print(f"  → Correction round {rnd}...")

        sub_questions   = _decompose_question(query)
        missing_aspects = _find_missing_aspects(query, best_answer, sub_questions)

        print(f"     Sub-questions: {sub_questions}")
        print(f"     Missing:       {missing_aspects}")

        if not missing_aspects:
            # nothing identifiably missing — try a full relevance rewrite anyway
            missing_aspects = sub_questions

        chain = ASPECT_REWRITE_PROMPT | judge_llm
        new_answer = chain.invoke({
            "query":   query,
            "context": ctx,
            "answer":  best_answer,
            "missing": "\n".join(f"- {q}" for q in missing_aspects),
        }).content.strip()

        new_score = score_answer_relevance_cosine(query, new_answer)
        improved  = new_score > best_score
        if improved:
            best_score  = new_score
            best_answer = new_answer

        metric_log["corrections"].append({
            "round":        rnd,
            "methods_used": ["aspect_decomposition"],
            "score_after":  new_score,
            "improved":     improved,
        })
        print(f"     New score={new_score:.3f} (improved={improved})")

        if best_score >= HIGH_THRESHOLD:
            break

    if best_score < HIGH_THRESHOLD:
        metric_log["status"] = "triggered_correction_failed"

    metric_log["final_score"] = best_score
    return best_score, best_answer, metric_log



#  REPORT


def build_confidence_report(
    cr_score: float,
    fa_score: float,
    ar_score: float,
    cr_log:   Dict,
    fa_log:   Dict,
    ar_log:   Dict,
) -> Dict[str, Any]:
    scores = {
        "context_relevance": cr_score,
        "faithfulness":      fa_score,
        "answer_relevance":  ar_score,
    }
    min_score = min(cr_score, fa_score, ar_score)
    flags     = []

    if cr_score < HIGH_THRESHOLD:
        flags.append(f"Context Relevance={cr_score:.2f} — retrieved chunks may not be fully relevant")
    if fa_score < HIGH_THRESHOLD:
        mode = fa_log.get("failure_mode", "unknown")
        flags.append(f"Faithfulness={fa_score:.2f} — answer may contain unverified claims ({mode})")
    if ar_score < HIGH_THRESHOLD:
        flags.append(f"Answer Relevance={ar_score:.2f} — answer may not fully address your question")

    if min_score >= HIGH_THRESHOLD:
        level   = "high"
        message = "Answer is well-supported by the documentation."
    elif min_score >= 0.5:
        level   = "medium"
        message = "Answer is mostly supported — verify critical details."
    else:
        level   = "low"
        message = "Answer may be incomplete or contain unverified claims — please cross-check."

    return {
        "level":         level,
        "message":       message,
        "flags":         flags,
        "scores_summary": scores,
    }


# main function starts 

def evaluate_and_correct(
    query:        str,
    answer:       str,
    context_docs: List[Document],
) -> Dict[str, Any]:
    """
    Full 3-step self-correction pipeline.

    Returns:
        final_answer      : str
        confidence        : dict  (level, message, flags, scores_summary)
        metric_logs       : dict  (per-metric structured logs)
        log_file          : str   (path to saved flow log)
    """
    print(f"\n{'='*60}")
    print(f"EVALUATION PIPELINE")
    print(f"Query: {query[:80]}...")
    print(f"{'='*60}")

    # Context Relevance
    cr_score, best_docs, cr_log = evaluate_context_relevance(query, context_docs)

    # Faithfulness  
    fa_score, best_answer, fa_log = evaluate_faithfulness(query, answer, best_docs)

    # Answer Relevance 
    ar_score, final_answer, ar_log = evaluate_answer_relevance(query, best_answer, best_docs)

    # Confidence report 
    confidence = build_confidence_report(
        cr_score, fa_score, ar_score,
        cr_log, fa_log, ar_log,
    )

    print(f"\n{'='*60}")
    print(f"RESULTS  CR={cr_score:.2f}  FA={fa_score:.2f}  AR={ar_score:.2f}  "
          f"Confidence={confidence['level'].upper()}")
    print(f"{'='*60}\n")

    # log writings
    log_data = build_log(
        query              = query,
        context_relevance  = cr_log,
        faithfulness       = fa_log,
        answer_relevance   = ar_log,
        final_answer       = final_answer,
        confidence         = confidence,
    )
    log_file = save_query_log(log_data)

    #  Collect correction metadata for frontend 
    any_correction = any(
        l["status"] != "passed"
        for l in [cr_log, fa_log, ar_log]
    )
    total_correction_rounds = sum(
        len(l.get("corrections", []))
        for l in [cr_log, fa_log, ar_log]
    )

    return {
        "final_answer":              final_answer,
        "confidence":                confidence,
        "scores": {
            "context_relevance": {"score": cr_score, "log": cr_log},
            "faithfulness":      {"score": fa_score, "log": fa_log},
            "answer_relevance":  {"score": ar_score, "log": ar_log},
        },
        "correction_triggered":      any_correction,
        "total_correction_rounds":   total_correction_rounds,
        "log_file":                  log_file,
        # keep old keys so frontend doesn't break
        "correction_attempts":       total_correction_rounds,
    }


if __name__ == "__main__":


    query      = "Paul Graham about startups"
    rag_result = run_llm(query)

    if rag_result["blocked"]:
        print("Blocked:", rag_result["answer"])
    else:
        result = evaluate_and_correct(
            query        = query,
            answer       = rag_result["answer"],
            context_docs = rag_result["context"],
        )
        print("Final Answer :", result["final_answer"])
        print("Confidence   :", result["confidence"]["level"], "—", result["confidence"]["message"])
        print("Flags        :", result["confidence"]["flags"])
        print("Log file     :", result["log_file"])
        print("Scores       :", json.dumps(
            {k: v["score"] for k, v in result["scores"].items()}, indent=2))