
from typing import Any,Dict,List
import os,uuid 
import streamlit as st

from core import run_llm

if "sessions" not in st.session_state:
    st.session_state.sessions = {}          # {session_id: [messages]}

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me anything about LangChain docs. I’ll retrieve relevant context and cite sources.",
            "sources": [],
        }
    ]
    
if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "total_checks": 0,
        "prompt_injections_blocked": 0,
        "harmful_intent_blocked": 0,
    }

if "langsmith_logs" not in st.session_state:
    st.session_state.langsmith_logs = []

st.set_page_config(page_title="LangChain Documentation Helper", layout="centered")
st.title("LangChain Documentation Helper")

def _format_sources(context_docs: List[Any]) -> List[str]:
    return [
        str((meta.get("source") or "Unknown"))
        for doc in (context_docs or [])
        if (meta := (getattr(doc, "metadata", None) or {})) is not None
    ]

def _agent_history() -> List[Dict]:
    return [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m["role"] in ("user", "assistant") and m.get("content")
    ]

with st.sidebar:

    # session controls
    st.subheader("Session")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Chat cleared. Ask me anything about LangChain docs.",
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
                    "content": "New conversation started. Ask me anything about LangChain docs!",
                    "sources": [],
                }
            ]
            st.rerun()

    # past sessions
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

    # metrics
    st.divider()
    st.subheader("Security metrics")
    m = st.session_state.metrics
    st.caption(f"Total checks: {m['total_checks']}")
    st.caption(f"Injections blocked: {m['prompt_injections_blocked']}")
    st.caption(f"Harmful intent blocked: {m['harmful_intent_blocked']}")

    # langsmith logs
    st.divider()
    st.subheader("LangSmith logs")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Fetch runs", use_container_width=True):
            try:
                from langsmith import Client
                ls_client = Client()
                project_name = os.getenv("LANGCHAIN_PROJECT")

                # FIX: get project ID first, then pass as session_id
                project = ls_client.read_project(project_name=project_name)

                runs = list(ls_client.list_runs(
                    project_id=project.id,   # use id not name
                    limit=20,
                    order="desc",
                ))

                if not runs:
                    st.warning("No runs found.")
                else:
                    logs = []
                    for run in runs:
                        latency = None
                        if run.end_time and run.start_time:
                            latency = round(
                                (run.end_time - run.start_time).total_seconds(), 2
                            )
                        logs.append({
                            "name": run.name or "unnamed",
                            "latency_sec": latency,
                            "total_tokens": getattr(run, "total_tokens", None),
                            "status": run.status,
                            "start_time": str(run.start_time),
                            "input": str(run.inputs)[:200],
                            "output": str(run.outputs)[:200],
                        })

                    st.session_state.langsmith_logs = logs
                    st.success(f"Fetched {len(logs)} runs")

            except Exception as e:
                st.error(f"Failed to fetch: {e}")

    with col_b:
        if st.button("Clear logs", use_container_width=True):
            st.session_state.langsmith_logs = []
            st.rerun()

    if st.session_state.langsmith_logs:
        st.caption(f"{len(st.session_state.langsmith_logs)} Previous Logs stored")
        for log in st.session_state.langsmith_logs:
            status_icon = "🟢" if log["status"] == "success" else "🔴"
            with st.expander(f"{status_icon} {log['name']} — {log['latency_sec']}s"):
                st.json(log)


#chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

prompt = st.chat_input("Ask a question about LangChain…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # generating response
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking…"): # ("Retrieving docs and generating answer…"):
                result: Dict[str, Any] = run_llm(prompt)
            # for prompts
            st.session_state.metrics["total_checks"] += 1
            if result.get("blocked"):
                if "injection" in result.get("answer", "").lower():
                    st.session_state.metrics["prompt_injections_blocked"] += 1
                else:
                    st.session_state.metrics["harmful_intent_blocked"] += 1

            #  blocked responses
            if result.get("blocked"):
                blocked_msg = result.get("answer", "Request blocked by security layer.")
                st.warning(blocked_msg)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": blocked_msg,
                        "sources": [],
                        "blocked": True,
                    }
                )

            # safe responses now 
            else:
                answer = str(result.get("answer", "")).strip() or "(No answer returned.)"
                sources = _format_sources(result.get("context", []))

                st.markdown(answer)
                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            st.markdown(f"- {s}")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "blocked": False,
                    }
                )

        except Exception as e:
            st.error(" Failed to generate a response.")
            st.exception(e)
