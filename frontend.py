from typing import Any, Dict, List
import os
import uuid
import streamlit as st

from core import run_llm
from evaluation import evaluate_and_correct

# ── session state init ────────────────────────────────────────────────────────
if "sessions" not in st.session_state:
    st.session_state.sessions = {}

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role":    "assistant",
            "content": (
                "Ask me anything about Paul Graham Essays. "
                "I'll retrieve relevant context, cite sources, and show evaluation scores."
            ),
            "sources": [],
        }
    ]

if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "total_checks":              0,
        "prompt_injections_blocked": 0,
        "harmful_intent_blocked":    0,
        "corrections_triggered":     0,
    }

if "eval_logs" not in st.session_state:
    st.session_state.eval_logs = []

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Paul Graham Essays RAG",
    page_icon="🤷‍♂️",
    layout="centered",
)
st.title(" Paul Graham Essays  — RAG Q&A")
st.caption("Answers grounded in Paul Graham Essays with self-correcting evaluation.")


#  helpers 
def _format_sources(context_docs: List[Any]) -> List[str]:
    seen, sources = set(), []
    for doc in (context_docs or []):
        meta = getattr(doc, "metadata", {}) or {}
        src  = str(meta.get("source") or meta.get("filename") or "Unknown")
        if src not in seen:
            seen.add(src)
            sources.append(src)
    return sources


def _agent_history() -> List[Dict]:
    return [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m["role"] in ("user", "assistant") and m.get("content")
    ]


def _score_color(score: float) -> str:
    if score >= 0.7:
        return "🟢"
    elif score >= 0.4:
        return "🟡"
    else:
        return "🔴"


def _confidence_color(level: str) -> str:
    return {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(level, "⚪")


def _render_correction_detail(log: Dict, metric_name: str):
    """Renders per-round correction detail inside an expander."""
    corrections = log.get("corrections", [])
    if not corrections:
        return
    for c in corrections:
        improved_icon = "✅" if c.get("improved") else "➡️"
        st.caption(
            f"Round {c['round']} — {', '.join(c['methods_used'])} "
            f"→ score {c['score_after']:.2f} {improved_icon}"
        )


def _render_eval_scores(eval_result: Dict):
    """Renders the full evaluation panel — confidence + per-metric scores."""
    scores      = eval_result.get("scores", {})
    confidence  = eval_result.get("confidence", {})
    any_correct = eval_result.get("correction_triggered", False)
    total_rnds  = eval_result.get("total_correction_rounds", 0)
    log_file    = eval_result.get("log_file", "")

    with st.expander("📊 RAG Evaluation — Scores & Self-Correction", expanded=False):

        # confidence banner
        conf_level   = confidence.get("level", "unknown")
        conf_message = confidence.get("message", "")
        conf_flags   = confidence.get("flags", [])
        conf_icon    = _confidence_color(conf_level)

        if conf_level == "high":
            st.success(f"{conf_icon} **Confidence: HIGH** — {conf_message}")
        elif conf_level == "medium":
            st.warning(f"{conf_icon} **Confidence: MEDIUM** — {conf_message}")
        else:
            st.error(f"{conf_icon} **Confidence: LOW** — {conf_message}")

        if conf_flags:
            for flag in conf_flags:
                st.caption(f"⚠️ {flag}")

        st.divider()

        # per-metric columns 
        col1, col2, col3 = st.columns(3)

        cr = scores.get("context_relevance", {})
        fa = scores.get("faithfulness", {})
        ar = scores.get("answer_relevance", {})

        cr_score = cr.get("score", 0.0)
        fa_score = fa.get("score", 0.0)
        ar_score = ar.get("score", 0.0)

        cr_log = cr.get("log", {})
        fa_log = fa.get("log", {})
        ar_log = ar.get("log", {})

        with col1:
            st.metric(
                "Context Relevance",
                f"{cr_score:.2f}",
                help="Are the retrieved chunks relevant to the query? (Cosine similarity + LLM judge)",
            )
            st.caption(f"{_score_color(cr_score)} Status: {cr_log.get('status','—')}")
            init = cr_log.get("initial_metric_score")
            if init is not None:
                st.caption(f"Initial cosine: {init:.2f}")
            judge = cr_log.get("llm_judge_score")
            if judge is not None:
                st.caption(f"LLM judge:      {judge:.2f}")
            _render_correction_detail(cr_log, "Context Relevance")

        with col2:
            st.metric(
                "Faithfulness",
                f"{fa_score:.2f}",
                help="Is the answer grounded in retrieved context? (LLM judge with claim verification)",
            )
            st.caption(f"{_score_color(fa_score)} Status: {fa_log.get('status','—')}")
            fm = fa_log.get("failure_mode")
            if fm and fm != "none":
                st.caption(f"Failure mode: {fm}")
            init = fa_log.get("initial_metric_score")
            if init is not None:
                st.caption(f"Initial judge: {init:.2f}")
            _render_correction_detail(fa_log, "Faithfulness")

        with col3:
            st.metric(
                "Answer Relevance",
                f"{ar_score:.2f}",
                help="Does the answer fully address the query? (Cosine similarity + aspect decomposition)",
            )
            st.caption(f"{_score_color(ar_score)} Status: {ar_log.get('status','—')}")
            init = ar_log.get("initial_metric_score")
            if init is not None:
                st.caption(f"Initial cosine: {init:.2f}")
            judge = ar_log.get("llm_judge_score")
            if judge is not None:
                st.caption(f"LLM judge:      {judge:.2f}")
            _render_correction_detail(ar_log, "Answer Relevance")

        st.divider()

        #  correction summary 
        if any_correct:
            st.warning(
                f"Self-correction ran — {total_rnds} correction round(s) across all metrics."
            )
        else:
            st.success("No self-correction needed.")

        #  log file path 
        if log_file:
            st.caption(f"📁 Log saved → `{log_file}`")


# side bars 

with st.sidebar:

    st.subheader("Session")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = [
                {
                    "role":    "assistant",
                    "content": "Chat cleared. Ask me anything about Paul Graham Essays.",
                    "sources": [],
                }
            ]
            st.rerun()

    with col2:
        if st.button("New chat", use_container_width=True):
            sid = st.session_state.current_session_id
            st.session_state.sessions[sid] = st.session_state.messages.copy()
            st.session_state.current_session_id = str(uuid.uuid4())
            st.session_state.messages = [
                {
                    "role":    "assistant",
                    "content": "New conversation started. Ask me anything about Paul Graham Essays!",
                    "sources": [],
                }
            ]
            st.rerun()

    if st.session_state.sessions:
        st.divider()
        st.subheader("Past chats")
        for sid, msgs in reversed(list(st.session_state.sessions.items())):
            user_msgs = [m for m in msgs if m["role"] == "user"]
            label = (user_msgs[0]["content"][:38] + "…") if user_msgs else "Empty chat"
            if st.button(label, key=f"session_{sid}", use_container_width=True):
                cur = st.session_state.current_session_id
                st.session_state.sessions[cur] = st.session_state.messages.copy()
                st.session_state.current_session_id = sid
                st.session_state.messages = msgs
                st.rerun()

    st.divider()
    st.subheader("Security Metrics")
    m = st.session_state.metrics
    st.caption(f"Total checks             : {m['total_checks']}")
    st.caption(f"Injections blocked       : {m['prompt_injections_blocked']}")
    st.caption(f"Harmful intent blocked   : {m['harmful_intent_blocked']}")
    st.caption(f"Self-corrections ran     : {m['corrections_triggered']}")

    st.divider()
    st.subheader("📋 Eval Log")

    if st.button("Clear eval log", use_container_width=True):
        st.session_state.eval_logs = []
        st.rerun()

    if st.session_state.eval_logs:
        st.caption(f"{len(st.session_state.eval_logs)} queries evaluated")
        for i, log_entry in enumerate(reversed(st.session_state.eval_logs)):
            scores     = log_entry.get("scores", {})
            cr = scores.get("context_relevance", {}).get("score", 0.0)
            fa = scores.get("faithfulness",      {}).get("score", 0.0)
            ar = scores.get("answer_relevance",  {}).get("score", 0.0)
            conf     = log_entry.get("confidence", {}).get("level", "?")
            corrected = "⚠️" if log_entry.get("correction_triggered") else "✅"
            q_label  = log_entry["query"][:40]
            idx      = len(st.session_state.eval_logs) - i

            with st.expander(f"{corrected} Q{idx}: {q_label}…"):
                st.caption(f"Confidence        : {_confidence_color(conf)} {conf.upper()}")
                st.caption(f"Context Relevance : {cr:.2f} {_score_color(cr)}")
                st.caption(f"Faithfulness      : {fa:.2f} {_score_color(fa)}")
                st.caption(f"Answer Relevance  : {ar:.2f} {_score_color(ar)}")
                if log_entry.get("correction_triggered"):
                    st.caption(f"Correction rounds : {log_entry.get('total_correction_rounds', 0)}")
                if log_entry.get("log_file"):
                    st.caption(f"Log → `{log_entry['log_file']}`")
    else:
        st.caption("No evaluations yet.")


# chat history 

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- `{s}`")
        if msg.get("eval_result"):
            _render_eval_scores(msg["eval_result"])


# query inputs

prompt = st.chat_input("Ask about Paul Graham Essays")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Step 1 — RAG retrieval + generation
            with st.spinner("Retrieving context…"):
                result: Dict[str, Any] = run_llm(prompt, chat_history=_agent_history())

            st.session_state.metrics["total_checks"] += 1

            # Blocked response
            if result.get("blocked"):
                answer_text = result.get("answer", "Request blocked.")
                if "injection" in answer_text.lower():
                    st.session_state.metrics["prompt_injections_blocked"] += 1
                else:
                    st.session_state.metrics["harmful_intent_blocked"] += 1

                st.warning(answer_text)
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": answer_text,
                    "sources": [],
                    "blocked": True,
                })

            # Safe response — evaluate + self-correct
            
            else:
                # If context is empty, generic reply — skip evaluation entirely
                if not result.get("context"):
                    generic_answer = result["answer"]
                    st.markdown(generic_answer)
                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": generic_answer,
                        "sources": [],
                        "blocked": False,
                    })
                else:
                    with st.spinner("Evaluating and self-correcting if needed…"):
                        eval_result = evaluate_and_correct(
                            query        = prompt,
                            answer       = result["answer"],
                            context_docs = result["context"],
                        )

                    if eval_result.get("correction_triggered"):
                        st.session_state.metrics["corrections_triggered"] += 1

                    final_answer = eval_result["final_answer"]
                    sources      = _format_sources(result.get("context", []))
                    confidence   = eval_result.get("confidence", {})

                    conf_level = confidence.get("level", "unknown")
                    conf_icon  = _confidence_color(conf_level)
                    st.caption(
                        f"{conf_icon} **Confidence: {conf_level.upper()}** — "
                        f"{confidence.get('message', '')}"
                    )

                    st.markdown(final_answer)

                    if sources:
                        with st.expander("📄 Sources"):
                            for s in sources:
                                st.markdown(f"- `{s}`")

                    _render_eval_scores(eval_result)

                    st.session_state.eval_logs.append({
                        "query":                   prompt,
                        "scores":                  eval_result["scores"],
                        "confidence":              eval_result["confidence"],
                        "correction_triggered":    eval_result["correction_triggered"],
                        "total_correction_rounds": eval_result.get("total_correction_rounds", 0),
                        "log_file":                eval_result.get("log_file", ""),
                    })

                    st.session_state.messages.append({
                        "role":        "assistant",
                        "content":     final_answer,
                        "sources":     sources,
                        "blocked":     False,
                        "eval_result": eval_result,
                    })



        except Exception as e:
            st.error("Failed to generate a response.")
            st.exception(e)