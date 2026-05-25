from typing import Any, Dict, List
import os, uuid
import streamlit as st
from core import run_llm
from evaluation import evaluate_and_correct

if "sessions" not in st.session_state:
    st.session_state.sessions = {}

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me anything about Paul Graham's essays. I'll retrieve relevant context, cite sources, and show evaluation scores.",
            "sources": [],
        }
    ]

if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "total_checks": 0,
        "prompt_injections_blocked": 0,
        "harmful_intent_blocked": 0,
        "corrections_triggered": 0,
    }

if "eval_logs" not in st.session_state:
    st.session_state.eval_logs = []   # stores per-query eval results

st.set_page_config(
    page_title="Paul Graham Essays RAG",
    page_icon="📝",
    layout="centered"
)
st.title("📝 Paul Graham Essays — RAG Q&A")
st.caption("Answers grounded in: *Do Things That Don't Scale · How to Get Startup Ideas · Keep Your Identity Small*")

def _format_sources(context_docs: List[Any]) -> List[str]:
    seen = set()
    sources = []
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
    if score >= 0.75:
        return "🟢"
    elif score >= 0.5:
        return "🟡"
    else:
        return "🔴"

def _render_eval_scores(eval_result: Dict):
    """Renders RAG Triad scores as a compact expander."""
    scores               = eval_result.get("scores", {})
    correction_triggered = eval_result.get("correction_triggered", False)
    correction_attempts  = eval_result.get("correction_attempts", 0)

    with st.expander("📊 RAG Evaluation Scores", expanded=False):
        col1, col2, col3 = st.columns(3)

        faith  = scores.get("faithfulness",      {})
        ans    = scores.get("answer_relevancy",  {})
        ctx    = scores.get("context_precision", {})

        with col1:
            f_score = faith.get("score", 0.0)
            st.metric("Faithfulness", f"{f_score:.2f}", help="Is the answer grounded in the retrieved context?")
            st.caption(f"{_score_color(f_score)} {faith.get('source','').upper()}")
            if faith.get("reason"):
                st.caption(faith["reason"])

        with col2:
            a_score = ans.get("score", 0.0)
            st.metric("Answer Relevancy", f"{a_score:.2f}", help="Does the answer address the query?")
            st.caption(f"{_score_color(a_score)} {ans.get('source','').upper()}")
            if ans.get("reason"):
                st.caption(ans["reason"])

        with col3:
            c_score = ctx.get("score", 0.0)
            st.metric("Context Precision", f"{c_score:.2f}", help="Did the retriever find the right chunks?")
            st.caption(f"{_score_color(c_score)} {ctx.get('source','').upper()}")
            if ctx.get("reason"):
                st.caption(ctx["reason"])

        if correction_triggered:
            st.warning(
                f"Self-correction triggered — "
                f"{correction_attempts} round(s) ran to fix low faithfulness."
            )
        else:
            st.success("No self-correction needed.")

with st.sidebar:

    # Session controls
    st.subheader("Session")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Chat cleared. Ask me anything about Paul Graham's essays.",
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
                    "role": "assistant",
                    "content": "New conversation started. Ask me anything about Paul Graham's essays!",
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

    # Security metrics
    st.divider()
    st.subheader("Security Metrics")
    m = st.session_state.metrics
    st.caption(f"Total checks         : {m['total_checks']}")
    st.caption(f"Injections blocked   : {m['prompt_injections_blocked']}")
    st.caption(f"Harmful intent blocked: {m['harmful_intent_blocked']}")
    st.caption(f"Self-corrections ran : {m['corrections_triggered']}")

    # Eval log
    st.divider()
    st.subheader("📋 Eval Log")
    if st.button("Clear eval log", use_container_width=True):
        st.session_state.eval_logs = []
        st.rerun()

    if st.session_state.eval_logs:
        st.caption(f"{len(st.session_state.eval_logs)} queries evaluated")
        for i, log in enumerate(reversed(st.session_state.eval_logs)):
            scores = log.get("scores", {})
            f = scores.get("faithfulness",     {}).get("score", 0.0)
            a = scores.get("answer_relevancy", {}).get("score", 0.0)
            c = scores.get("context_precision",{}).get("score", 0.0)
            corrected = "⚠️" if log.get("correction_triggered") else "✅"
            with st.expander(f"{corrected} Q{len(st.session_state.eval_logs)-i}: {log['query'][:40]}…"):
                st.caption(f"Faithfulness     : {f:.2f} {_score_color(f)}")
                st.caption(f"Answer Relevancy : {a:.2f} {_score_color(a)}")
                st.caption(f"Context Precision: {c:.2f} {_score_color(c)}")
                if log.get("correction_triggered"):
                    st.caption(f"Correction rounds: {log.get('correction_attempts')}")
    else:
        st.caption("No evaluations yet.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- `{s}`")
        if msg.get("eval_result"):
            _render_eval_scores(msg["eval_result"])

prompt = st.chat_input("Ask about Paul Graham's essays…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Step 1: RAG
            with st.spinner("Retrieving context…"):
                result: Dict[str, Any] = run_llm(prompt, chat_history=_agent_history())

            st.session_state.metrics["total_checks"] += 1

            # Step 2: Blocked response
            if result.get("blocked"):
                answer_text = result.get("answer", "Request blocked by security layer.")
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

            # Step 3: Safe response → evaluate
            else:
                with st.spinner("Evaluating answer quality…"):
                    eval_result = evaluate_and_correct(
                        query        = prompt,
                        answer       = result["answer"],
                        context_docs = result["context"],
                    )

                if eval_result.get("correction_triggered"):
                    st.session_state.metrics["corrections_triggered"] += 1

                final_answer = eval_result["final_answer"]
                sources      = _format_sources(result.get("context", []))

                st.markdown(final_answer)

                if sources:
                    with st.expander("📄 Sources"):
                        for s in sources:
                            st.markdown(f"- `{s}`")

                _render_eval_scores(eval_result)

                # Log to sidebar eval log
                st.session_state.eval_logs.append({
                    "query":                prompt,
                    "scores":               eval_result["scores"],
                    "correction_triggered": eval_result["correction_triggered"],
                    "correction_attempts":  eval_result["correction_attempts"],
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