#!/usr/bin/env python3
"""Replay the fixed LaMP profile-adaptation cohort into a new output prefix.

GPU runs must use the existing qrun dispatcher on Thor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guard_new_output(out):
    out = Path(out)
    existing = list(out.parent.glob(out.stem + "*")) if out.parent.exists() else []
    if existing:
        raise FileExistsError("new output prefix required; existing companions: " + ", ".join(str(p) for p in existing))


def atomic_write(path, payload):
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", choices=["dense_qwen15b", "hybrid_qwen27b"], required=True)
    ap.add_argument("--lamp-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--authors", type=int, default=6)
    ap.add_argument("--max-wall-s", type=int, required=True)
    ap.add_argument("--preflight", action="store_true", help="CPU-only source/data/config inspection; writes no receipts")
    args = ap.parse_args()
    budget_s, memory_gb = (5400, 80.) if args.backbone == "hybrid_qwen27b" else (3600, 12.)
    if args.authors != 6 or not 0 < args.max_wall_s <= budget_s:
        raise ValueError("fixed six-author cohort and positive wall time within prospective budget required")
    out, data = Path(args.out), Path(args.lamp_dir)
    guard_new_output(out)
    manifest_path = data / "_manifest.json"
    manifest = json.loads(manifest_path.read_text())[:args.authors]
    paths = [data / (row["author_id"] + ".json") for row in manifest]
    authors = [json.loads(path.read_text()) for path in paths]
    if len(authors) != args.authors:
        raise ValueError("six authors unavailable")
    expected_queries = 6 if args.backbone == "hybrid_qwen27b" else 10
    for author in authors:
        if len(author["heldout_queries"]) < expected_queries or len(author["history_pairs"]) < 24:
            raise ValueError("insufficient profile/query data: " + author["author_id"])
    from planning import V2_MODEL_PATHS
    model_path = Path(V2_MODEL_PATHS[args.backbone])
    config_paths = [model_path / "config.json", model_path / "tokenizer_config.json"]
    if not config_paths[0].is_file():
        raise FileNotFoundError(config_paths[0])
    config_paths += sorted(model_path.glob("*.safetensors.index.json"))
    source = Path(__file__).parent
    payload = dict(schema="tierserve_author_lamp_v1", ok=False, args=vars(args),
        started_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for folder in ("src", "experiments") for p in sorted((ROOT/folder).glob("*.py"))},
        data_sha256={p.name:sha256(p) for p in [manifest_path, *paths]},
        model_path=str(model_path), model_config_sha256={p.name:sha256(p) for p in config_paths if p.is_file()},
        seed=20261107, model_recipe=dict(precision="BF16", lora_r=8, lora_alpha=16, quant_4bit=False),
        protocol="one_profile_adaptation_transfer_not_online_RentFlush",
        negative_labels="dataset_deterministic_lead_sentence_or_generic_headline_for_all_methods",
        energy_boundary="estimated_prefill_plus_attributed_refresh_not_complete_lifecycle",
        reference_policy="cold_reference_for_both_executors_no_precomputed_cache",
        foreground_adapter="base_fallback_during_refresh",
        memory_budget_gib=memory_gb, expected_authors=6, expected_queries=6*expected_queries,
        events=[])
    if args.preflight:
        print(json.dumps(payload, indent=2))
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    provenance = out.with_suffix(".provenance.json")
    # Exclusive prefix reservation: a second invocation cannot resume this run.
    with provenance.open("x") as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
    started = time.perf_counter()
    def timeout_handler(*_):
        raise TimeoutError("prospective LaMP wall-time budget exhausted")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.max_wall_s)
    try:
        import torch
        import transformers
        import peft
        from headlines import run_lamp4_backbone
        from core import set_seed
        set_seed(payload["seed"])
        if not torch.cuda.is_available():
            raise RuntimeError("GPU replay must run on Thor through canonical qrun")
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(.9, (memory_gb-.25)*1024**3/total_bytes))
        payload["versions"] = dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__)
        payload["result"] = run_lamp4_backbone(args.backbone, authors,
            out.with_suffix(".partial.json"), event_callback=payload["events"].append,
            memory_limit_gb=memory_gb)
        if payload["result"]["n_authors"] != 6 or payload["result"]["n_queries"] != 6*expected_queries:
            raise RuntimeError("incomplete author/query contract")
        payload.update(ok=True, wall_time_s=time.perf_counter()-started)
        atomic_write(out, payload)
    except BaseException as exc:
        payload.update(error=repr(exc), ok=False, wall_time_s=time.perf_counter()-started)
        atomic_write(out.with_suffix(".failed.json"), payload)
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
