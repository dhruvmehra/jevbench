"""Render results/<run>/*.json into a markdown summary."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

COLUMNS = [
    ("Classifier", lambda r: r["classifier"]),
    ("Model", lambda r: r["model_id"]),
    ("Acc", lambda r: _pct(r["metrics"]["accuracy"])),
    ("Macro-F1", lambda r: _pct(r["metrics"]["macro_f1"])),
    ("ECE", lambda r: _f(r["metrics"]["ece"], 3)),
    ("Err%", lambda r: _pct(r["metrics"]["error_rate"])),
    ("p50 ms", lambda r: _f(r["metrics"]["latency_p50_ms"], 0)),
    ("p95 ms", lambda r: _f(r["metrics"]["latency_p95_ms"], 0)),
    ("ex/s", lambda r: _f(r["metrics"].get("throughput_ex_s"), 1)),
    ("$/1k", lambda r: _money(r["metrics"]["cost_per_1k_usd"])),
]

DASH = "–"


def _f(v, nd):
    return DASH if v is None else f"{v:.{nd}f}"


def _pct(v):
    return DASH if v is None else f"{100 * v:.1f}"


def _money(v):
    return DASH if v is None else f"${v:.4f}"


def render(results: list[dict]) -> str:
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_ds[r["dataset"]].append(r)
    lines = ["# JEV classifier benchmark", ""]
    if results:
        r0 = results[0]
        lines += [f"Hardware for local models: {r0['hardware']}", ""]
    for ds in sorted(by_ds):
        rows = sorted(by_ds[ds], key=lambda r: -(r["metrics"]["accuracy"] or 0))
        lines += [f"## {ds}  (n={rows[0]['n']}, {rows[0]['n_labels']} labels)", ""]
        lines.append("| " + " | ".join(c for c, _ in COLUMNS) + " |")
        lines.append("|" + "---|" * len(COLUMNS))
        for r in rows:
            lines.append("| " + " | ".join(str(fn(r)) for _, fn in COLUMNS) + " |")
        notes = []
        for r in rows:
            if r["metrics"].get("latency_from_cache"):
                notes.append(f"{r['classifier']}: latency omitted (loaded from cache)")
            if r.get("train_seconds"):
                notes.append(f"{r['classifier']}: fine-tuning took {r['train_seconds']:.0f}s")
            tp = r["metrics"].get("throughput_parallelism")
            if tp:
                notes.append(f"{r['classifier']}: throughput at parallelism {tp}")
        if notes:
            lines += [""] + [f"- {n}" for n in notes]
        lines.append("")
    lines += [
        "Notes: Acc/F1 exclude errored examples; Err% counts them. ECE needs probabilities,",
        "so it is blank for LLMs. Latency is per example at parallelism 1. Local models cost $0 in API fees.",
    ]
    return "\n".join(lines) + "\n"


def write_summary(run_dir: Path) -> Path:
    results = [json.loads(p.read_text()) for p in sorted(Path(run_dir).glob("*.json"))]
    out = Path(run_dir) / "summary.md"
    out.write_text(render(results))
    return out
