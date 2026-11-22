"""Structured request usage shared by generation, retrieval and evaluation."""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import json
import logging
from pathlib import Path


_CURRENT = ContextVar("eeg_medrag_request_usage", default=None)
logger = logging.getLogger(__name__)


def request_namespace() -> str | None:
    context = _CURRENT.get()
    return context["labels"].get("namespace") if context else None


@contextmanager
def request_context(**labels):
    parent = _CURRENT.get()
    current = {"labels": {**(parent["labels"] if parent else {}), **labels},
               "events": parent["events"] if parent else []}
    token = _CURRENT.set(current)
    try:
        yield current["events"]
    finally:
        _CURRENT.reset(token)


def record_request(**event) -> None:
    current = _CURRENT.get()
    if current is None:
        return
    labels = dict(current["labels"])
    ledger = labels.pop("ledger", None)
    row = {**labels, "timestamp": datetime.now(timezone.utc).isoformat(), **event}
    current["events"].append(row)
    if ledger is not None:
        path = Path(ledger)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        except OSError as error:
            row["journal_error"] = type(error).__name__
            logger.warning("Request usage journal unavailable path=%s error=%s", path, type(error).__name__)


def summarize_requests(events: list[dict]) -> dict:
    calls = [r for r in events if r["event"] == "sdk_call"]
    observed = {r.get("request_key") for r in calls}
    unknown_calls = any(r["event"] == "request_started" and r.get("request_key") not in observed
                        for r in events)
    result = {"sdk_calls": len(calls),
              "journal_errors": sum("journal_error" in r for r in events),
              "cache_hits": sum(r.get("count", 1) for r in events if r["event"] == "cache_hit"),
              "rejected_responses": sum(r.get("status") == "invalid" for r in calls),
              "failed_calls": sum(r.get("status") == "error" for r in calls),
              "sdk_elapsed_seconds": sum(r["elapsed_seconds"] for r in calls),
              "request_keys": list(dict.fromkeys(r["request_key"] for r in events if "request_key" in r))}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [0 if name == "completion_tokens" and r["operation"] == "embedding" else
                  (r.get("usage") or {}).get(name) for r in calls]
        known = [v for v in values if isinstance(v, int)]
        result[name] = sum(known) if len(known) == len(calls) and not unknown_calls else None
        result["observed_" + name] = sum(known)
    return result
