
import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional


LOGS_ROOT        = "logs"
FLOW_LOGS_DIR    = os.path.join(LOGS_ROOT, "flow_logs")
CORRECTED_DIR    = os.path.join(LOGS_ROOT, "correction_logs", "corrected")
FAILED_DIR       = os.path.join(LOGS_ROOT, "correction_logs", "failed")

for d in [FLOW_LOGS_DIR, CORRECTED_DIR, FAILED_DIR]:
    os.makedirs(d, exist_ok=True)

# ── stdlib logger (console)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rag_pipeline")


# ── helpers 
def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")


def _write(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── main entry 
def save_query_log(log_data: Dict[str, Any]) -> str:
    """
    Saves a structured log for one query.

    Always writes to flow_logs/.
    If any correction was triggered:
        - improved score  → also writes to correction_logs/corrected/
        - did not improve → also writes to correction_logs/failed/

    Returns the path of the flow-log file.
    """
    ts       = _timestamp()
    filename = f"query_{ts}.json"

    # ── flow log (always)
    flow_path = os.path.join(FLOW_LOGS_DIR, filename)
    _write(flow_path, log_data)
    log.info(f"Flow log saved → {flow_path}")

    # ── correction log (only when correction ran) 
    meta = log_data.get("pipeline_meta", {})
    if meta.get("any_correction_triggered"):
        if meta.get("any_correction_improved"):
            dest = os.path.join(CORRECTED_DIR, filename)
            _write(dest, log_data)
            log.info(f"Correction log (improved) → {dest}")
        else:
            dest = os.path.join(FAILED_DIR, filename)
            _write(dest, log_data)
            log.info(f"Correction log (failed)   → {dest}")

    return flow_path


# ── log-record builder 
def build_log(
    query:           str,
    context_relevance: Dict,
    faithfulness:      Dict,
    answer_relevance:  Dict,
    final_answer:      str,
    confidence:        Dict,
) -> Dict[str, Any]:
    """
    Assembles the full structured log record.

    Each metric block has the shape:
    {
        "initial_metric_score":  float,
        "llm_judge_score":       float | None,
        "status":                "passed" | "triggered_correction" | "triggered_correction_failed",
        "failure_mode":          str | None,   # faithfulness only
        "corrections": [
            {
                "round":         int,
                "methods_used":  list[str],
                "score_after":   float,
                "improved":      bool,
            }
        ],
        "final_score":           float,
    }
    """
    any_triggered = any(
        m.get("status", "passed") != "passed"
        for m in [context_relevance, faithfulness, answer_relevance]
    )
    any_improved = any(
        m.get("status") == "triggered_correction"
        for m in [context_relevance, faithfulness, answer_relevance]
    )

    return {
        "query":     query,
        "timestamp": datetime.utcnow().isoformat(),

        "context_relevance": context_relevance,
        "faithfulness":      faithfulness,
        "answer_relevance":  answer_relevance,

        "confidence_report": confidence,
        "final_answer":      final_answer,

        "pipeline_meta": {
            "any_correction_triggered": any_triggered,
            "any_correction_improved":  any_improved,
        },
    }