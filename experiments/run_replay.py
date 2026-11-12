#!/usr/bin/env python3
"""Run stream or executor replays and record the source and inputs.

GPU runs must use the existing qrun dispatcher on Thor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_receipt(path, payload):
    tmp = path.with_suffix(".writing.json")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def require_work_contract(meta, steps, pairs, label):
    if meta.get("oom"):
        raise RuntimeError(label + ": " + str(meta.get("error", "out of memory")))
    if meta.get("optimizer_updates") != steps or meta.get("training_pairs") != pairs:
        raise RuntimeError(label + ": execution work contract violated")


def executor_suite(args, users, out, payload):
    import torch
    from core import (TegraWatch, _encode_prompt_comp, format_bare_prompt,
                               mean_logp_no_grad, mem_snapshot, reset_peft_lora,
                               sum_and_mean_logp_no_grad, word_edit_ratio)
    from snapshots import TrainableSnapshot
    from planning import load_v2_backbone, rouge_l_f1
    from execution import (generate_coupled_coserve_arrivals,
        measure_unloaded_single_query_baseline, run_adjoint_microtxn_engine,
        run_step_boundary_preempt_engine)
    from serving import summarize_slo_metrics, serve_single_request_kv

    model, tok, load_info = load_v2_backbone(args.backbone)
    # This experiment isolates scheduling, not dropout realization differences.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout): module.p = 0.0
    pool = users[0]["train_k160"][:12]
    held = users[0]["heldout"][:8]
    if len(pool) < 8 or len(held) < 4:
        raise ValueError("executor study requires at least 8 train and 4 held-out pairs")
    max_p, max_c = 384, 96
    reset_peft_lora(model, seed=args.seed)
    refs = []
    with TegraWatch("author_reference_cache") as reference_power:
        with model.disable_adapter():
            for pair in pool:
                pr = format_bare_prompt(pair["request"])
                sp, _ = sum_and_mean_logp_no_grad(model, tok, pr, pair["pos"], max_p, max_c)
                sn, _ = sum_and_mean_logp_no_grad(model, tok, pr, pair["neg"], max_p, max_c)
                ep = _encode_prompt_comp(tok, pr, pair["pos"], max_p, max_c)
                en = _encode_prompt_comp(tok, pr, pair["neg"], max_p, max_c)
                if None in (sp, sn) or ep is None or en is None:
                    raise ValueError("unscorable training pair")
                refs.append((sp, sn, len(ep[1]), len(en[1])))
    # Warm the full 12-item pool through both pair and branch paths before timing.
    # The subsequent previous-adapter build must not uniquely warm branch shapes.
    preparation = {}
    for name, engine in [("pair_warmup", run_step_boundary_preempt_engine),
                         ("branch_warmup", run_adjoint_microtxn_engine)]:
        _, preparation[name] = engine(model, tok, pool, [], [], 3, 4, 1e-4,
            max_p, max_c, 16, drain_future_arrivals=False, reset_lora=True,
            precomputed_refs=refs)
        require_work_contract(preparation[name], 3, 12, name)
    # Reinitialize after both warmups, then construct the common previous adapter.
    reset_peft_lora(model, seed=args.seed)
    _, preparation["previous_adapter_training"] = run_adjoint_microtxn_engine(
        model, tok, pool, [], [], 3, 4, 1e-4, max_p, max_c, 16,
        drain_future_arrivals=False, reset_lora=False, precomputed_refs=refs)
    require_work_contract(preparation["previous_adapter_training"], 3, 12, "previous adapter")
    committed = TrainableSnapshot(model, "committed-v1")
    prompts = [format_bare_prompt(held[i % len(held)]["request"]) for i in range(args.requests)]
    # Exercise snapshot copies and all eight held-out prompt shapes untimed.
    preparation["committed_foreground_warmup"] = []
    for prompt in prompts[:len(held)]:
        with committed.activate(model):
            request = serve_single_request_kv(model, tok, prompt, time.perf_counter(),
                max_new_tokens=16, use_adapter=True)
        preparation["committed_foreground_warmup"].append(request)
    preparation["base_unloaded"] = measure_unloaded_single_query_baseline(
        model, tok, prompts, gen_toks=16, use_adapter=False)
    unloaded = measure_unloaded_single_query_baseline(model, tok, prompts,
        gen_toks=16, use_adapter=True)
    ttft = max(350.0, 2.5 * unloaded["unloaded_ttft_p95_ms"])
    tpot = max(20.0, 1.5 * unloaded["unloaded_tpot_p95_ms"])
    payload.update(load_info=load_info, effective_quality_items=len(held),
        committed_adapter_bytes=committed.bytes, cache_build_power=reference_power.result, preparation=preparation,
        unloaded=unloaded, unloaded_reader="committed-v1", ttft_budget_ms=ttft, tpot_budget_ms=tpot,
        note="Warm reference cache; previous adapter retained except explicit BaseFallback. Full replay board energy includes foreground and idle time. No claim of user-level inference from repeated queries.", runs=[])
    methods = ["StepPreempt", "PairPreempt", "AdjointPass-Committed", "AdjointPass-BaseFallback"]
    for rep in range(args.repetitions):
        arrivals = generate_coupled_coserve_arrivals(args.requests,
            qps=0.34 * unloaded["service_capacity_qps"],
            mean_service_ms=unloaded["mean_service_ms"], seed=args.seed + rep,
            is_high_load=False)
        # Balanced cyclic order rather than choosing the best observed run.
        order = methods[rep % len(methods):] + methods[:rep % len(methods)]
        for method in order:
            committed.restore(model)
            snapshot = None if method.endswith("BaseFallback") else committed
            common = dict(n_steps=6, grad_accum=4, lr=1e-4, max_p_train=max_p,
                max_c_train=max_c, gen_toks=16, drain_future_arrivals=True,
                reset_lora=False, seed=args.seed, precomputed_refs=refs,
                foreground_snapshot=snapshot)
            engine = run_adjoint_microtxn_engine if method.startswith("Adjoint") else run_step_boundary_preempt_engine
            if not method.startswith("Adjoint"):
                common["yield_granularity"] = "pair" if method == "PairPreempt" else "step"
            reqs, meta = engine(model, tok, pool, prompts, arrivals, **common)
            require_work_contract(meta, 6, 24, method)
            if len(reqs) != args.requests:
                raise RuntimeError("execution request contract violated")
            for i, request in enumerate(reqs):
                gold = held[i % len(held)]["pos"]
                quality_ids = request["generated_ids"]
                eos = tok.eos_token_id
                if eos is not None and eos in quality_ids:
                    quality_ids = quality_ids[:quality_ids.index(eos)]
                text = tok.decode(quality_ids, skip_special_tokens=True)
                request.update(request_index=i, held_index=i % len(held), arrival_s=arrivals[i],
                    text=text, quality_generated_ids=quality_ids, during_refresh=arrivals[i] < meta["bg_s"],
                    rouge_l=rouge_l_f1(text, gold),
                    edit_similarity=1.0 - word_edit_ratio(text, gold))
            metrics = summarize_slo_metrics(reqs, ttft, tpot, meta["bg_s"], meta["peak_gb"], meta["tegra"])
            payload["runs"].append(dict(method=method, repetition=rep,
                requests=reqs, execution=meta, metrics=metrics,
                mean_rouge_l=sum(r["rouge_l"] for r in reqs)/len(reqs),
                mean_edit_similarity=sum(r["edit_similarity"] for r in reqs)/len(reqs)))
            payload["mem_final"] = mem_snapshot()
            write_receipt(out.with_suffix(".partial.json"), payload)
            print(f"AUTHOR_EXECUTOR rep={rep} method={method} SLO={metrics['slo_joint_attainment_pct']} power={meta['tegra']}", flush=True)
    payload["ok"] = True


def stream_suite(args, users, out, payload):
    from stream import run_backbone_streaming_suite
    # No prev_result, no resume, no synthetic per-turn reconstruction, no old output reads.
    result = run_backbone_streaming_suite(args.backbone, users, args.turns,
        run_coreset_gen=False, partial_path=out.with_suffix(".raw-partial.json"),
        prev_result=None, only_open_gen=False, refresh_policies=None)
    for user in result["per_user"]:
        for policy, record in user["policies"].items():
            if len(record.get("per_turn_dm", [])) != record["n_turns"]:
                raise RuntimeError("missing actual per-turn observations: " + policy)
    payload.update(result=result,
        reference_policy="cold reference scores recomputed with exact training truncation for every executor",
        energy_boundary="estimated prompt prefill plus attributed refresh board energy; excludes generation and auxiliary scoring; not complete lifecycle",
        ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["executor", "stream"], required=True)
    ap.add_argument("--backbone", choices=["dense_qwen15b", "mamba_falcon15b", "hybrid_qwen27b", "gdn_qwen2b"], required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--users", type=int, default=6)
    ap.add_argument("--turns", type=int, default=24)
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument("--requests", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--max-wall-s", type=int, default=7200)
    args = ap.parse_args()
    out = Path(args.out)
    companions = [out, *(out.with_suffix(s) for s in [".partial.json", ".raw-partial.json", ".failed.json", ".provenance.json"])]
    if any(p.exists() for p in companions):
        raise FileExistsError("use a new output path; no result overwrite or automatic resume")
    out.parent.mkdir(parents=True, exist_ok=True)
    root = Path(args.data_dir)
    manifest = json.loads((root/"_manifest.json").read_text())[:args.users]
    paths = [root/(row["prefix"]+".json") for row in manifest]
    users = [json.loads(p.read_text()) for p in paths]
    if len(users) != args.users:
        raise ValueError("requested users unavailable")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("real-model validation requires a serialized Thor GPU job")
    torch.cuda.set_per_process_memory_fraction(0.90)
    import transformers, peft
    started = time.perf_counter()
    payload = dict(schema="tierserve_author_validation_w5_raw_power_preserved", ok=False,
        started_utc=datetime.now(timezone.utc).isoformat(), args=vars(args),
        source_sha256={str(p.relative_to(ROOT)):digest(p) for folder in ("src", "experiments") for p in sorted((ROOT/folder).glob("*.py"))},
        data_sha256={p.name:digest(p) for p in paths},
        versions=dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__))
    write_receipt(out.with_suffix(".provenance.json"), payload)
    def timeout_handler(*_):
        raise TimeoutError("prospective wall-time budget exhausted")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.max_wall_s)
    try:
        (executor_suite if args.suite == "executor" else stream_suite)(args, users, out, payload)
        payload["wall_time_s"] = time.perf_counter() - started
        write_receipt(out, payload)
    except BaseException as exc:
        payload.update(ok=False, error=repr(exc), wall_time_s=time.perf_counter()-started)
        write_receipt(out.with_suffix(".failed.json"), payload)
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
