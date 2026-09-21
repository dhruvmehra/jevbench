from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
from pathlib import Path

from . import datasets as D
from .cache import Cache
from .config import api_key, load_config
from .report import write_summary
from .runner import build_classifiers, evaluate, group_classifiers

ALL_CLASSIFIERS = ["jev", "laya", "llm-cheap", "llm-frontier", "bert-ft", "bert-zs"]


def parse(argv=None):
    p = argparse.ArgumentParser(prog="jevbench")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the benchmark")
    r.add_argument("--datasets", nargs="+", default=list(D.DATASETS), choices=list(D.DATASETS))
    r.add_argument("--classifiers", nargs="+", default=ALL_CLASSIFIERS, choices=ALL_CLASSIFIERS)
    r.add_argument("--n", type=int)
    r.add_argument("--seed", type=int)
    r.add_argument("--skip-throughput", action="store_true")
    r.add_argument("--no-cache", action="store_true", help="always hit the network for the latency pass")
    r.add_argument("--smoke", action="store_true", help="n=5, skip throughput")
    r.add_argument("--out", default="results")
    r.add_argument("--config", default="config.toml")
    rep = sub.add_parser("report", help="re-render summary.md for a run dir")
    rep.add_argument("run_dir")
    return p.parse_args(argv)


async def run(args) -> Path:
    cfg = load_config(args.config)
    n = 5 if args.smoke else (args.n or cfg["run"]["n"])
    seed = args.seed if args.seed is not None else cfg["run"]["seed"]
    skip_tp = args.skip_throughput or args.smoke
    needs_key = any(c.startswith(("jev", "llm")) for c in args.classifiers)
    key = api_key() if needs_key else None

    run_dir = Path(args.out) / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    cache = Cache("cache.sqlite")
    clfs = build_classifiers(args.classifiers, cfg, key)

    for ds_name in args.datasets:
        print(f"\n=== {ds_name}: loading (n={n}, seed={seed}) ===", flush=True)
        ds = D.load(ds_name, n, seed, cfg["bert"]["train_cap"])
        async def run_one(clf):
            print(f"--- {clf.name} ({clf.model_id}) on {ds_name}: started", flush=True)
            try:
                res = await evaluate(
                    clf, ds, cache,
                    parallelism_api=cfg["run"]["api_concurrency"],
                    local_batch=cfg["run"]["local_batch_size"],
                    skip_throughput=skip_tp,
                    use_cache=not args.no_cache,
                )
            except Exception as e:  # noqa: BLE001 - one classifier failing must not kill the run
                print(f"--- {clf.name} on {ds_name}: FAILED: {e!r}", flush=True)
                return
            m = res["metrics"]
            print(
                f"--- {clf.name} on {ds_name}: done  acc={m['accuracy']:.3f} f1={m['macro_f1']:.3f} "
                f"err={m['error_rate']:.2%} p50={m['latency_p50_ms'] or 0:.0f}ms "
                f"tp={m['throughput_ex_s'] or 0:.1f}/s "
                f"$/1k={m['cost_per_1k_usd'] if m['cost_per_1k_usd'] is not None else 0:.4f}",
                flush=True,
            )
            (run_dir / f"{clf.name}__{ds_name}.json").write_text(json.dumps(res, indent=1, default=str))

        api_clfs, local_clfs = group_classifiers(clfs)
        # API classifiers run concurrently with each other (each still sequential internally);
        # local models run one at a time so they do not contend for the GPU.
        if api_clfs:
            await asyncio.gather(*(run_one(c) for c in api_clfs))
        for c in local_clfs:
            await run_one(c)
    out = write_summary(run_dir)
    print(f"\nWrote {out}\n")
    print(out.read_text())
    return run_dir


def main(argv=None):
    args = parse(argv)
    if args.cmd == "report":
        out = write_summary(Path(args.run_dir))
        print(out.read_text())
        return
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
