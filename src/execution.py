#!/usr/bin/env python3
"""Run context calibration, gradient comparisons and co-serving measurements.

The suite compares a local optimizer-step control with branchwise execution,
plus trainer and serving baselines. Arrival rates use the unloaded request
latency as a reference. Historical method labels are retained in result keys.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.90, 0)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


from core import (  # noqa: E402
    TegraWatch,
    _encode_prompt_comp,
    _nullcontext,
    drop_kv,
    embed_texts,
    format_bare_prompt,
    format_vanilla_rag_prompt,
    log,
    mem_snapshot,
    mean_logp_no_grad,
    reset_peft_lora,
    select_topk_cosine,
    set_seed,
    sum_and_mean_logp_no_grad,
    sum_logp_with_grad,
)
from planning import (  # noqa: E402
    _capture_rng,
    _restore_rng,
    center_embeddings_zero_param,
    compute_scalar_functional_adjoints,
    load_v2_backbone,
    run_specprobe_calibration,
    verify_adjoint_microtxn_exactness,
)
from serving import (  # noqa: E402
    generate_poisson_arrivals,
    percentile,
    serve_single_request_kv,
    summarize_slo_metrics,
)


def load_wc20_users(wc_dir: str | Path, max_users: int) -> list[dict]:
    wc_dir = Path(wc_dir)
    manifest = json.loads((wc_dir / "_manifest.json").read_text())
    out = []
    for row in manifest[:max_users]:
        out.append(json.loads((wc_dir / f"{row['prefix']}.json").read_text()))
    return out


# Retrieval depth and context retention

@torch.no_grad()
def run_empirical_kshot_phase_curve(
    model,
    tok,
    users: Sequence[dict],
    k_grid: Sequence[int] = (0, 1, 2, 4, 8),
    max_held_per_user: int = 16,
) -> dict:
    """Measure real heldout personalization margin dm(k), acc(k), and prompt tokens across
    k in {0, 1, 2, 4, 8} retrieved personal history shots under frozen resident weights.
    """
    was_training = model.training
    model.eval()
    ad_ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else _nullcontext()

    per_k_margins: dict[int, list[float]] = {k: [] for k in k_grid}
    per_k_accs: dict[int, list[float]] = {k: [] for k in k_grid}
    per_k_toks: dict[int, list[int]] = {k: [] for k in k_grid}

    with ad_ctx:
        for u in users:
            hist = u["stream_episodes"][:24]
            held = u["heldout"][:max_held_per_user]
            if len(hist) < 4 or not held:
                continue
            h_emb_raw = embed_texts(model, tok, [x["request"] for x in hist], max_len=192)
            q_emb_raw = embed_texts(model, tok, [x["request"] for x in held], max_len=192)
            h_emb, q_emb = center_embeddings_zero_param(h_emb_raw, h_emb_raw, q_emb_raw)

            for qi, item in enumerate(held):
                for k in k_grid:
                    if k == 0:
                        pr = format_bare_prompt(item["request"])
                    else:
                        idxs, _ = select_topk_cosine(q_emb[qi], h_emb, k)
                        exs = [hist[j] for j in reversed(idxs)]
                        pr = format_vanilla_rag_prompt(exs, item["request"], max_ex_chars=1400)
                    lp_pos = mean_logp_no_grad(model, tok, pr, item["pos"], max_prompt=2560, max_comp=224)
                    lp_neg = mean_logp_no_grad(model, tok, pr, item["neg"], max_prompt=2560, max_comp=224)
                    if lp_pos is None or lp_neg is None:
                        continue
                    ptoks = min(2560, len(tok(pr, add_special_tokens=True).input_ids))
                    per_k_margins[k].append(lp_pos - lp_neg)
                    per_k_accs[k].append(1.0 if lp_pos > lp_neg else 0.0)
                    per_k_toks[k].append(ptoks)

    if was_training:
        model.train()
    drop_kv()

    base_margins = per_k_margins[0]
    m0 = sum(base_margins) / max(len(base_margins), 1)
    curve = {}
    for k in k_grid:
        ms = per_k_margins[k]
        acs = per_k_accs[k]
        tks = per_k_toks[k]
        n = max(len(ms), 1)
        curve[str(k)] = {
            "k": k,
            "n_pairs": len(ms),
            "acc": round(sum(acs) / n, 4),
            "mean_margin": round(sum(ms) / n, 4),
            "dm_vs_k0": round((sum(ms) / n) - m0, 4),
            "mean_prompt_toks": round(sum(tks) / n, 1),
        }
    return curve


# Co-serving controls

def measure_unloaded_single_query_baseline(
    model, tok, prompts: Sequence[str], gen_toks: int = 20, n_probe: int = 8, use_adapter: bool = False
) -> dict:
    """Measure true UNLOADED single-request TTFT and TPOT (one request at a time with zero
    queueing backlog) to establish rigorous MLPerf/Llumnix interactive SLO budgets!
    """
    drop_kv()
    reqs = []
    for i in range(min(n_probe, len(prompts))):
        now = time.perf_counter()
        reqs.append(serve_single_request_kv(model, tok, prompts[i], now, max_new_tokens=gen_toks, use_adapter=use_adapter))
    # Discard index 0 cold warmup if >= 3 probes
    steady = reqs[1:] if len(reqs) >= 3 else reqs
    ttfts = [r["ttft_ms"] for r in steady]
    tpots = [r["tpot_ms"] for r in steady]
    e2es = [r["e2e_ms"] for r in steady]
    mean_service_s = (sum(e2es) / max(len(e2es), 1)) / 1000.0
    max_mu_qps = 1.0 / max(mean_service_s, 1e-3)
    return {
        "unloaded_ttft_p50_ms": percentile(ttfts, 50),
        "unloaded_ttft_p95_ms": percentile(ttfts, 95),
        "unloaded_tpot_p50_ms": percentile(tpots, 50),
        "unloaded_tpot_p95_ms": percentile(tpots, 95),
        "mean_service_ms": round(mean_service_s * 1000.0, 2),
        "service_capacity_qps": round(max_mu_qps, 3),
    }


def generate_coupled_coserve_arrivals(
    n_req: int,
    qps: float,
    mean_service_ms: float,
    seed: int = 20261107,
    is_high_load: bool = False,
) -> list[float]:
    """Generate seeded renewal offsets with bounded gaps and optional bursts.
    
    qps sets the nominal time scale. Gap clipping and the high-load burst pattern
    also affect the realized arrival rate; this is not a Poisson arrival process.
    """
    rng = random.Random(seed)
    srv_s = max(0.05, mean_service_ms / 1000.0)
    target_mean_s = 1.0 / max(qps, 1e-3)
    unit_draws = [rng.expovariate(1.0) for _ in range(n_req)]
    mean_draw = sum(unit_draws) / max(len(unit_draws), 1)
    norm_draws = [x / max(mean_draw, 1e-6) for x in unit_draws]

    t = 0.18
    arrivals: list[float] = []
    for i, u in enumerate(norm_draws):
        arrivals.append(round(t, 4))
        if not is_high_load:
            min_gap = srv_s * 1.25
            slack = max(0.05, target_mean_s - min_gap)
            gap = min_gap + slack * min(2.2, max(0.25, u))
        else:
            # High-QPS (rho=0.52): realistic interactive micro-bursts (pairs/triplets) at higher arrival rate
            if i % 3 in (1, 2):
                gap = srv_s * 0.46
            else:
                min_gap = srv_s * 1.10
                slack = max(0.04, target_mean_s * 1.35 - min_gap)
                gap = min_gap + slack * min(1.8, max(0.25, u))
        t += gap
    return arrivals


def concatenated_completion_scores(model, p_ids, c_pos, c_neg, pad_id, grad):
    """Batch both branches, casting only completion logits to FP32, like AdjointPass."""
    device = next(model.parameters()).device
    seq1, seq2 = p_ids + c_pos, p_ids + c_neg
    length = max(len(seq1), len(seq2))
    batch = torch.tensor([seq1 + [pad_id] * (length-len(seq1)),
                          seq2 + [pad_id] * (length-len(seq2))], dtype=torch.long, device=device)
    attn = torch.tensor([[1]*len(seq1)+[0]*(length-len(seq1)),
                         [1]*len(seq2)+[0]*(length-len(seq2))], dtype=torch.long, device=device)
    with (_nullcontext() if grad else torch.no_grad()):
        output = model(input_ids=batch, attention_mask=attn, use_cache=False)
        start = len(p_ids)-1
        lp_pos = -F.cross_entropy(output.logits[0,start:len(seq1)-1,:].float(),
                                 batch[0,len(p_ids):len(seq1)], reduction="sum")
        lp_neg = -F.cross_entropy(output.logits[1,start:len(seq2)-1,:].float(),
                                 batch[1,len(p_ids):len(seq2)], reduction="sum")
        return lp_pos, -lp_pos/len(c_pos), lp_neg


def pair_training_prompt(pair):
    """Use the exact generated context when supplied; legacy callers keep their frame."""
    if "training_prompt" in pair:
        prompt = pair["training_prompt"]
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("invalid explicit training_prompt")
        return prompt
    return format_bare_prompt(pair["request"])


def validate_reference_cache(tok, pairs, cache, max_prompt, max_comp):
    """Cache entries are aligned with exact encoded branch lengths, including None misses."""
    if cache is None:
        return
    if len(cache) != len(pairs):
        raise ValueError("reference cache length must equal train_pairs length")
    for pair, item in zip(pairs, cache):
        if item is None:
            continue
        pos = _encode_prompt_comp(tok, pair_training_prompt(pair), pair["pos"], max_prompt, max_comp)
        neg = _encode_prompt_comp(tok, pair_training_prompt(pair), pair["neg"], max_prompt, max_comp)
        if pos is None or neg is None or (int(item[2]), int(item[3])) != (len(pos[1]), len(neg[1])):
            raise ValueError("cached completion length does not match encoded pair")


def serve_foreground_contract(model, tok, prompt, arrival, gen_toks, use_adapter, snapshot):
    if snapshot is not None and not use_adapter:
        with snapshot.activate(model):
            result = serve_single_request_kv(model, tok, prompt, arrival,
                                            max_new_tokens=gen_toks, use_adapter=True)
        result["adapter_version"] = snapshot.version
    else:
        result = serve_single_request_kv(model, tok, prompt, arrival,
                                        max_new_tokens=gen_toks, use_adapter=use_adapter)
        result["adapter_version"] = "published" if use_adapter else "base"
    return result


def run_step_boundary_preempt_engine(
    model,
    tok,
    train_pairs: list[dict],
    prompts: list[str],
    arrivals: list[float],
    n_steps: int,
    grad_accum: int,
    lr: float,
    max_p_train: int,
    max_c_train: int,
    gen_toks: int,
    tag: str = "llmstation_step_preempt",
    drain_future_arrivals: bool = True,
    sft_alpha: float = 0.1,
    precomputed_refs: Sequence[tuple[float, float, int, int] | None] | None = None,
    foreground_snapshot=None,
    yield_granularity: str = "step",
    reset_lora: bool = True,
    seed: int = 20261107,
    foreground_callback=None,
) -> tuple[list[dict], dict]:
    """Local optimizer-step preemption control; not LLMStation's suspendable backward."""
    validate_reference_cache(tok, train_pairs, precomputed_refs, max_p_train, max_c_train)
    drop_kv()
    if yield_granularity not in ("step", "pair"):
        raise ValueError("yield_granularity must be step or pair")
    if reset_lora:
        reset_peft_lora(model, seed=seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    opt_p = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(opt_p, lr=lr, weight_decay=0.0)
    device = next(model.parameters()).device

    reqs: list[dict] = []
    next_q = 0
    step_durations_ms: list[float] = []
    training_pairs_count = 0
    optimizer_updates = 0
    scheduling_windows_ms = []
    reference_cache_hits = 0
    reference_pairs_computed = 0

    def _drain_arrived(t0: float, use_ad: bool):
        nonlocal next_q
        if foreground_callback is not None:
            foreground_callback(use_ad, {"optimizer_updates": optimizer_updates, "training_pairs": training_pairs_count})
        while next_q < len(arrivals) and time.perf_counter() >= (t0 + arrivals[next_q]):
            reqs.append(
                serve_foreground_contract(
                    model, tok, prompts[next_q], t0 + arrivals[next_q],
                    gen_toks, use_ad, foreground_snapshot,
                )
            )
            next_q += 1

    def _concat_seq_logps(p_ids, c_pos, c_neg, grad):
        return concatenated_completion_scores(model, p_ids, c_pos, c_neg, tok.pad_token_id or 0, grad)

    with TegraWatch(tag) as tw:
        t0 = time.perf_counter()
        bg_s = 0.0
        ref_dur_s = 0.0
        try:
            # Coarse reference batch precomputation (yields once after ref batch)
            t_ref0 = time.perf_counter()
            refs = []
            model.eval()
            with model.disable_adapter():
                for idx_p, p in enumerate(train_pairs):
                    enc_p = _encode_prompt_comp(tok, pair_training_prompt(p), p["pos"], max_p_train, max_c_train)
                    enc_n = _encode_prompt_comp(tok, pair_training_prompt(p), p["neg"], max_p_train, max_c_train)
                    if enc_p is None or enc_n is None:
                        continue
                    pr_item = (
                        precomputed_refs[idx_p]
                        if (precomputed_refs is not None and idx_p < len(precomputed_refs))
                        else None
                    )
                    if pr_item is not None:
                        refs.append((float(pr_item[0]), float(pr_item[1]), enc_p[0], enc_p[1], enc_n[1]))
                        reference_cache_hits += 1
                    else:
                        rp, _, rn = _concat_seq_logps(enc_p[0], enc_p[1], enc_n[1], grad=False)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        reference_pairs_computed += 1
                        refs.append((float(rp.item()), float(rn.item()), enc_p[0], enc_p[1], enc_n[1]))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            wall_ref_s = time.perf_counter() - t_ref0
            ref_dur_s = wall_ref_s
            _drain_arrived(t0, use_ad=False)

            pair_cursor = 0
            step_idx = 0
            while step_idx < n_steps:
                t_s0 = time.perf_counter()
                model.train()
                if hasattr(model, "config"):
                    model.config.use_cache = False
                opt.zero_grad(set_to_none=True)
                for _ in range(grad_accum):
                    pair_t0 = time.perf_counter()
                    rp, rn, pids, cpos, cneg = refs[pair_cursor % len(refs)]
                    pair_cursor += 1
                    plp, pce, nlp = _concat_seq_logps(pids, cpos, cneg, grad=True)
                    loss = (-F.logsigmoid(0.1 * ((plp - rp) - (nlp - rn))) + sft_alpha * pce) / grad_accum
                    loss.backward()
                    del plp, pce, nlp, loss
                    training_pairs_count += 1
                    if yield_granularity == "pair":
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        scheduling_windows_ms.append((time.perf_counter() - pair_t0) * 1000.0)
                        _drain_arrived(t0, use_ad=False)
                if step_idx < n_steps:
                    torch.nn.utils.clip_grad_norm_(opt_p, 1.0)
                    opt.step()
                    optimizer_updates += 1
                opt.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                step_durations_ms.append((time.perf_counter() - t_s0) * 1e3)
                step_idx += 1
                if step_idx == n_steps:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    bg_s = time.perf_counter() - t0
                if yield_granularity == "step":
                    scheduling_windows_ms.append(step_durations_ms[-1])
                _drain_arrived(t0, use_ad=(step_idx >= n_steps))
                if not drain_future_arrivals and step_idx >= n_steps:
                    break

            if bg_s <= 0.0:
                bg_s = time.perf_counter() - t0
            if drain_future_arrivals:
                while next_q < len(arrivals):
                    time.sleep(max(0.0, t0 + arrivals[next_q] - time.perf_counter()))
                    _drain_arrived(t0, use_ad=True)
            del opt
            drop_kv()
        except RuntimeError as e:
            drop_kv()
            if "out of memory" in str(e).lower():
                return reqs, {"oom": True, "error": str(e)[:180], "peak_gb": mem_snapshot().get("max_allocated_gb"), "tegra": tw.result}
            raise

    pure_bg_s = ref_dur_s + (sum(step_durations_ms[:n_steps]) / 1000.0)
    return reqs, {
        "oom": False,
        "bg_s": bg_s,
        "pure_bg_s": round(pure_bg_s, 3) if yield_granularity == "step" else None,
        "optimizer_step_spans_s": sum(step_durations_ms[:n_steps]) / 1000.0,
        "training_pairs": training_pairs_count,
        "optimizer_updates": optimizer_updates,
        "reference_cache_hits": reference_cache_hits,
        "reference_pairs_computed": reference_pairs_computed,
        "reference_time_s": ref_dur_s,
        "non_preempt_window_ms_mean": round(sum(scheduling_windows_ms) / max(len(scheduling_windows_ms), 1), 1),
        "scheduling_windows_ms": scheduling_windows_ms,
        "peak_gb": mem_snapshot().get("max_allocated_gb"),
        "tegra": tw.result,
    }


def run_adjoint_microtxn_engine(
    model,
    tok,
    train_pairs: list[dict],
    prompts: list[str],
    arrivals: list[float],
    n_steps: int,
    grad_accum: int,
    lr: float,
    max_p_train: int,
    max_c_train: int,
    gen_toks: int,
    objective: str = "dpo",
    drain_future_arrivals: bool = True,
    sft_alpha: float = 0.1,
    reset_lora: bool = True,
    seed: int = 20261107,
    precomputed_refs: Sequence[tuple[float, float, int, int] | None] | None = None,
    foreground_snapshot=None,
    foreground_callback=None,
) -> tuple[list[dict], dict]:
    """Branchwise preference execution with graph-free foreground boundaries."""
    validate_reference_cache(tok, train_pairs, precomputed_refs, max_p_train, max_c_train)
    drop_kv()
    if reset_lora:
        reset_peft_lora(model, seed=seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    opt_p = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(opt_p, lr=lr, weight_decay=0.0)

    reqs: list[dict] = []
    next_q = 0
    preempt_count = 0
    micro_durations_ms: list[float] = []
    ref_micro_ms: list[float] = []
    training_pairs_count = 0
    optimizer_updates = 0
    scheduling_windows_ms = []
    reference_cache_hits = 0
    reference_pairs_computed = 0

    def _poll_fg(t0: float, use_ad: bool):
        nonlocal next_q, preempt_count
        if foreground_callback is not None:
            foreground_callback(use_ad, {"optimizer_updates": optimizer_updates, "training_pairs": training_pairs_count})
        served = False
        while next_q < len(arrivals) and time.perf_counter() >= (t0 + arrivals[next_q]):
            reqs.append(
                serve_foreground_contract(
                    model, tok, prompts[next_q], t0 + arrivals[next_q],
                    gen_toks, use_ad, foreground_snapshot,
                )
            )
            next_q += 1
            preempt_count += 1
            served = True
        if served and hasattr(model, "config"):
            model.config.use_cache = False

    with TegraWatch(f"adjoint_microtxn_{objective}") as tw:
        t0 = time.perf_counter()
        bg_s = 0.0
        # Reuse compatible reference scores or compute them with per-item yields.
        # SimPO does not need reference scores.
        refs: list[tuple[float, float, int, int] | None] = []
        model.eval()
        for pair_idx, p in enumerate(train_pairs):
                item = precomputed_refs[pair_idx] if precomputed_refs is not None else None
                if item is not None:
                    refs.append(item)
                    reference_cache_hits += 1
                    _poll_fg(t0, use_ad=False)
                    continue
                _poll_fg(t0, use_ad=False)
                pr = pair_training_prompt(p)
                enc_p = _encode_prompt_comp(tok, pr, p["pos"], max_p_train, max_c_train)
                enc_n = _encode_prompt_comp(tok, pr, p["neg"], max_p_train, max_c_train)
                if enc_p is None or enc_n is None:
                    refs.append(None)
                    continue
                if objective == "simpo":
                    refs.append((0.0, 0.0, len(enc_p[1]), len(enc_n[1])))
                else:
                    reference_pairs_computed += 1
                    tr0 = time.perf_counter()
                    with model.disable_adapter():
                        sp, _ = sum_and_mean_logp_no_grad(model, tok, pr, p["pos"], max_prompt=max_p_train, max_comp=max_c_train)
                    ref_micro_ms.append((time.perf_counter() - tr0) * 1e3)
                    _poll_fg(t0, use_ad=False)
                    tr1 = time.perf_counter()
                    with model.disable_adapter():
                        sn, _ = sum_and_mean_logp_no_grad(model, tok, pr, p["neg"], max_prompt=max_p_train, max_comp=max_c_train)
                    ref_micro_ms.append((time.perf_counter() - tr1) * 1e3)
                    if sp is None or sn is None:
                        refs.append(None)
                    else:
                        refs.append((sp, sn, len(enc_p[1]), len(enc_n[1])))

        valid_ids = [i for i, r in enumerate(refs) if r is not None]

        # Stage 2: M_act = 0 Adjoint Micro-Transactions
        beta_v = 2.0 if objective == "simpo" else 0.1
        pair_cursor = 0
        step_idx = 0
        while step_idx < n_steps:
            use_ad_now = (step_idx >= n_steps)
            opt.zero_grad(set_to_none=True)
            used = 0
            for _ in range(grad_accum):
                _poll_fg(t0, use_ad=use_ad_now)
                if drain_future_arrivals and step_idx >= n_steps and next_q >= len(arrivals):
                    break
                idx = valid_ids[pair_cursor % len(valid_ids)]
                pair_cursor += 1
                p = train_pairs[idx]
                rp, rn, lp_len, ln_len = refs[idx]  # type: ignore[misc]
                pr = pair_training_prompt(p)

                # Micro-txn (a): neg scalar probe (M_act = 0)
                tm0 = time.perf_counter()
                model.train()
                if hasattr(model, "config"):
                    model.config.use_cache = False
                rng_neg = _capture_rng()
                with torch.no_grad():
                    sn_t, ce_n = sum_logp_with_grad(model, tok, pr, p["neg"], max_prompt=max_p_train, max_comp=max_c_train)
                    sn_val = float(sn_t.item()) if sn_t is not None else None
                    del sn_t, ce_n
                if sn_val is None:
                    continue
                rng_pos = _capture_rng()
                if step_idx < n_steps:
                    micro_durations_ms.append((time.perf_counter() - tm0) * 1e3)
                _poll_fg(t0, use_ad=use_ad_now)

                # Micro-txn (b): pos forward + scalar adjoint + immediate backward (M_act -> 0)
                tm1 = time.perf_counter()
                model.train()
                if hasattr(model, "config"):
                    model.config.use_cache = False
                _restore_rng(rng_pos)
                pos_lp, ce_p = sum_logp_with_grad(model, tok, pr, p["pos"], max_prompt=max_p_train, max_comp=max_c_train)
                rng_after = _capture_rng()
                if pos_lp is None:
                    continue
                gp, gn, _ = compute_scalar_functional_adjoints(
                    objective, float(pos_lp.detach().item()), sn_val, lp_len, ln_len, rp, rn,
                    beta=beta_v, sft_alpha=sft_alpha,
                )
                ((gp / grad_accum) * pos_lp).backward()
                del pos_lp, ce_p
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                if step_idx < n_steps:
                    micro_durations_ms.append((time.perf_counter() - tm1) * 1e3)
                _poll_fg(t0, use_ad=use_ad_now)

                # Micro-txn (c): neg forward + immediate backward with exact rng_neg (M_act -> 0)
                tm2 = time.perf_counter()
                model.train()
                if hasattr(model, "config"):
                    model.config.use_cache = False
                _restore_rng(rng_neg)
                neg_lp, ce_n2 = sum_logp_with_grad(model, tok, pr, p["neg"], max_prompt=max_p_train, max_comp=max_c_train)
                if neg_lp is not None:
                    ((gn / grad_accum) * neg_lp).backward()
                    del neg_lp, ce_n2
                    used += 1
                    training_pairs_count += 1
                _restore_rng(rng_after)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                if step_idx < n_steps:
                    micro_durations_ms.append((time.perf_counter() - tm2) * 1e3)
                _poll_fg(t0, use_ad=use_ad_now)

            if used > 0 and step_idx < n_steps:
                torch.nn.utils.clip_grad_norm_(opt_p, 1.0)
                opt.step()
                optimizer_updates += 1
            opt.zero_grad(set_to_none=True)
            step_idx += 1
            if step_idx == n_steps:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                bg_s = time.perf_counter() - t0
            _poll_fg(t0, use_ad=(step_idx >= n_steps))
            if not drain_future_arrivals and step_idx >= n_steps:
                break

        if bg_s <= 0.0:
            bg_s = time.perf_counter() - t0
        if drain_future_arrivals:
            while next_q < len(arrivals):
                time.sleep(max(0.0, t0 + arrivals[next_q] - time.perf_counter()))
                _poll_fg(t0, use_ad=True)
        del opt
        drop_kv()

    pure_bg_s = (sum(ref_micro_ms) + sum(micro_durations_ms)) / 1000.0
    return reqs, {
        "oom": False,
        "bg_s": bg_s,
        "pure_bg_s": round(pure_bg_s, 3),
        "training_pairs": training_pairs_count,
        "optimizer_updates": optimizer_updates,
        "reference_cache_hits": reference_cache_hits,
        "reference_pairs_computed": reference_pairs_computed,
        "reference_time_s": sum(ref_micro_ms) / 1000.0,
        "preempt_events": preempt_count,
        "non_preempt_window_ms_mean": round(sum(micro_durations_ms) / max(len(micro_durations_ms), 1), 1),
        "scheduling_windows_ms": micro_durations_ms,
        "peak_gb": mem_snapshot().get("max_allocated_gb"),
        "tegra": tw.result,
    }

