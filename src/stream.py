#!/usr/bin/env python3
"""Replay stored interaction sequences with retrieval and adapter-refresh policies.

Adapting policies use a local optimizer-step or branchwise executor. The suite
also compares coreset choices and generated responses. Policy names are retained
for compatibility with historical receipts; they do not identify external
framework executions.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
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
    _clip_text,
    _encode_prompt_comp,
    _nullcontext,
    dedilute_coreset_select,
    drop_kv,
    embed_texts,
    format_bare_prompt,
    format_vanilla_rag_prompt,
    log,
    mean_logp_no_grad,
    mem_snapshot,
    paired_bootstrap_test,
    reset_peft_lora,
    select_mmr,
    select_topk_cosine,
    set_seed,
    sum_and_mean_logp_no_grad,
    word_edit_ratio,
)
from planning import (  # noqa: E402
    SpecProbeProfile,
    auto_dedilute_kneedle_select,
    build_capacity_bounded_prompt,
    center_embeddings_zero_param,
    compute_null_similarity_quantile,
    format_compact_delta_card,
    greedy_generate_text,
    load_v2_backbone,
    rouge_1_f1,
    rouge_l_f1,
    run_specprobe_calibration,
)
from serving import generate_poisson_arrivals, summarize_slo_metrics  # noqa: E402
from baselines import (  # noqa: E402
    format_memgpt_prompt,
    select_coreset_craig_facility_location,
    select_coreset_el2n,
    synthesize_memgpt_persona_block,
)
from execution import (  # noqa: E402
    generate_coupled_coserve_arrivals,
    measure_unloaded_single_query_baseline,
    run_adjoint_microtxn_engine,
    run_step_boundary_preempt_engine,
)


def load_wc_bundles(wc_dir: str | Path, max_users: int) -> list[dict]:
    wc_dir = Path(wc_dir)
    manifest = json.loads((wc_dir / "_manifest.json").read_text())
    return [
        json.loads((wc_dir / f"{row['prefix']}.json").read_text())
        for row in manifest[:max_users]
    ]


@torch.no_grad()
def score_single_episode_causal(
    model,
    tok,
    prompt: str,
    pos: str,
    neg: str,
    use_adapter: bool,
    joule_per_tok: float,
    base_ms_0: float,
    vin_mw: float = 45000.0,
    max_prompt: int = 2304,
    max_comp: int = 224,
) -> dict | None:
    was_training = model.training
    model.eval()
    ad_ctx = (
        _nullcontext()
        if (use_adapter or not hasattr(model, "disable_adapter"))
        else model.disable_adapter()
    )
    t0 = time.perf_counter()
    with ad_ctx:
        sp_sum, lp_pos = sum_and_mean_logp_no_grad(model, tok, prompt, pos, max_prompt=max_prompt, max_comp=max_comp)
        sn_sum, lp_neg = sum_and_mean_logp_no_grad(model, tok, prompt, neg, max_prompt=max_prompt, max_comp=max_comp)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1e3
    if was_training:
        model.train()
    if lp_pos is None or lp_neg is None or sp_sum is None or sn_sum is None:
        return None
    p_toks = min(max_prompt, len(tok(prompt, add_special_tokens=True).input_ids))
    enc_p = _encode_prompt_comp(tok, prompt, pos, max_prompt, max_comp)
    enc_n = _encode_prompt_comp(tok, prompt, neg, max_prompt, max_comp)
    lp_len = len(enc_p[1]) if enc_p else 1
    ln_len = len(enc_n[1]) if enc_n else 1
    # Hardware-calibrated read energy from SpecProbe rail power vin_mw & per-context-token slope
    base_read_j = (vin_mw / 1000.0) * (base_ms_0 / 1000.0)
    added_toks = max(0, p_toks - 64)
    read_j = max(0.15, base_read_j + added_toks * joule_per_tok)
    return {
        "margin": lp_pos - lp_neg,
        "acc": 1.0 if lp_pos > lp_neg else 0.0,
        "prompt_toks": p_toks,
        "read_ms": round(dt_ms, 2),
        "read_j": round(read_j, 3),
        "ref_tuple": (float(sp_sum), float(sn_sum), lp_len, ln_len),
    }


def execute_controlled_flush(
    substrate: str,
    model,
    tok,
    flush_pairs: list[dict],
    fg_prompts: list[str],
    qps: float,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
    n_steps: int,
    grad_accum: int,
    lr: float,
    max_p_tr: int,
    max_c_tr: int,
    ctx_cards_pool: list[dict] | None = None,
    l_crit_tokens: int = 0,
    vin_mw: float = 45000.0,
    sft_alpha: float = 0.1,
    scale_steps_by_coreset: bool = False,
    seed: int = 20261107,
    precomputed_refs: Sequence[tuple[float, float, int, int] | None] | None = None,
) -> dict:
    """Refresh the adapter with a local step-boundary or branchwise executor.
    
    Both paths process foreground arrivals during refresh. Report request SLOs,
    peak allocated memory and an attributed write-energy estimate derived from
    background time and mean board power, not an isolated refresh-energy integral.
    """
    # Both executors compute their own reference scores.
    # Legacy base-evaluation caches have different truncation boundaries.
    precomputed_refs = None
    formatted_pairs = []
    rng_f = random.Random(seed)
    use_exp_ctx = bool(ctx_cards_pool and len(ctx_cards_pool) >= 2 and l_crit_tokens >= 96)
    for idx, p in enumerate(flush_pairs):
        if use_exp_ctx and (idx % 2 == 0):
            cand = [c for c in (ctx_cards_pool or []) if c["request"] != p["request"]]
            if cand:
                k_pick = min(len(cand), 2 if l_crit_tokens >= 500 else 1)
                exs = rng_f.sample(cand, k_pick)
                pr_ctx, n_pk, _ = build_capacity_bounded_prompt(
                    tok, exs, p["request"], min(l_crit_tokens, 640), schema="clean_pair"
                )
                if n_pk > 0:
                    formatted_pairs.append({
                        "request": pr_ctx,
                        "pos": p["pos"],
                        "neg": p["neg"],
                    })
                    continue
        formatted_pairs.append(p)

    eff_steps = (
        max(4, int(round(n_steps * min(1.0, (len(flush_pairs) + 4.0) / 24.0))))
        if scale_steps_by_coreset
        else n_steps
    )

    # Generate coupled foreground renewal arrivals anchored to the unloaded capacity QPS
    if fg_prompts:
        n_fg = 32
        est_srv_ms = 1000.0 * (0.34 / max(qps, 1e-3))
        arrivals = generate_coupled_coserve_arrivals(
            n_fg, qps=qps, mean_service_ms=est_srv_ms, seed=seed, is_high_load=False
        )
        fg_sub = [fg_prompts[i % len(fg_prompts)] for i in range(n_fg)]
    else:
        arrivals = []
        fg_sub = []

    if substrate == "external_llmstation":
        reqs, meta = run_step_boundary_preempt_engine(
            model, tok, formatted_pairs, fg_sub, arrivals,
            n_steps=eff_steps, grad_accum=grad_accum, lr=lr,
            max_p_train=max_p_tr, max_c_train=max_c_tr, gen_toks=16,
            tag="stream_flush_llms", drain_future_arrivals=False, sft_alpha=sft_alpha,
            precomputed_refs=precomputed_refs,
        )
    elif substrate == "tierserve_adjoint_microtxn":
        reqs, meta = run_adjoint_microtxn_engine(
            model, tok, formatted_pairs, fg_sub, arrivals,
            n_steps=eff_steps, grad_accum=grad_accum, lr=lr,
            max_p_train=max_p_tr, max_c_train=max_c_tr, gen_toks=16,
            objective="dpo", drain_future_arrivals=False, sft_alpha=sft_alpha,
            precomputed_refs=precomputed_refs,
        )
    else:
        raise ValueError(f"Unknown co-serving substrate: {substrate}")

    if meta.get("oom"):
        raise RuntimeError("refresh aborted: " + meta.get("error", "out of memory"))

    slo = summarize_slo_metrics(
        reqs, ttft_slo_ms, tpot_slo_ms, meta.get("bg_s"), meta.get("peak_gb"), meta.get("tegra")
    )
    # Attribute background time after subtracting recorded foreground work.
    bg_wall_s = float(meta.get("bg_s") or 0.0)
    fg_exec_s = sum(max(0.0, r.get("e2e_ms", 0.0) - r.get("queue_wait_ms", 0.0)) for r in reqs) / 1000.0
    pure_bg_s = float(meta.get("pure_bg_s") or max(0.25, bg_wall_s - fg_exec_s))
    rail_mw = float((meta.get("tegra") or {}).get("vin_mw_mean") or 0.0)
    if rail_mw <= 100.0:
        rail_mw = vin_mw
    pure_write_j = round((rail_mw / 1000.0) * pure_bg_s, 1)
    return {
        "reqs": reqs,
        "bg_s": round(pure_bg_s, 2),
        "peak_gb": meta.get("peak_gb") or 0.0,
        "vin_j": float(pure_write_j),
        "slo_joint_pct": slo["slo_joint_attainment_pct"],
        "ttft_p95_ms": slo["ttft_p95_ms"],
    }


def check_prefix_ctx_lcb(
    model,
    tok,
    hist: Sequence[dict],
    centered_emb: torch.Tensor,
    l_crit: int,
    use_adapter: bool = False,
    min_ep_tokens: int = 240,
) -> bool:
    """Check whether prefix context lift exceeds half a standard error.
    
    Use only historical items and require enough context budget for a card.
    """
    if l_crit < min_ep_tokens or len(hist) < 4:
        return False
    tau_null = compute_null_similarity_quantile(centered_emb, q=0.50)
    k_try = min(4, max(1, l_crit // min_ep_tokens))
    lifts: list[float] = []
    was_tr = model.training
    model.eval()
    ad_ctx = (
        _nullcontext()
        if (use_adapter or not hasattr(model, "disable_adapter"))
        else model.disable_adapter()
    )
    eval_ids = list(range(max(0, len(hist) - 8), len(hist)))
    with ad_ctx:
        for i in eval_ids:
            item = hist[i]
            idxs, _ = select_mmr(
                centered_emb[i], centered_emb, k=k_try, lam=0.50, sim_floor=tau_null, exclude_idx=i
            )
            if not idxs:
                idxs, _ = select_topk_cosine(centered_emb[i], centered_emb[:i] if i >= 2 else centered_emb, k=min(k_try, max(1, i)))
            if not idxs:
                continue
            cands = [hist[j] for j in idxs if j != i]
            if not cands:
                continue
            prc, n_pk, _ = build_capacity_bounded_prompt(
                tok, cands, item["request"], l_crit, schema="clean_pair"
            )
            if n_pk == 0:
                continue
            pr0 = format_bare_prompt(item["request"])
            lp0_p = mean_logp_no_grad(model, tok, pr0, item["pos"], max_prompt=1792, max_comp=192)
            lp0_n = mean_logp_no_grad(model, tok, pr0, item["neg"], max_prompt=1792, max_comp=192)
            lpc_p = mean_logp_no_grad(model, tok, prc, item["pos"], max_prompt=1792, max_comp=192)
            lpc_n = mean_logp_no_grad(model, tok, prc, item["neg"], max_prompt=1792, max_comp=192)
            if lp0_p is not None and lp0_n is not None and lpc_p is not None and lpc_n is not None:
                lifts.append((lpc_p - lpc_n) - (lp0_p - lp0_n))
    if was_tr:
        model.train()
    if len(lifts) < 2 or all(abs(x) < 1e-7 for x in lifts):
        return l_crit >= 192
    mu = sum(lifts) / len(lifts)
    return mu > 0.0


def simulate_online_stream_policy(
    policy_name: str,
    model,
    tok,
    stream_eps: list[dict],
    all_req_embs_raw: torch.Tensor,
    base_step_evals: list[dict],
    sp: SpecProbeProfile,
    fg_prompts: list[str],
    qps: float,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
    n_steps_flush: int,
    grad_accum: int,
    lr: float,
    max_p_tr: int,
    max_c_tr: int,
    ref_rag_evals: dict[int, dict] | None = None,
    oppu_out: dict | None = None,
) -> dict:
    """Score each stored turn using earlier state, then append feedback.
    
    After appending, the policy may refresh the adapter for subsequent turns.
    """
    reset_peft_lora(model, seed=20261107)
    drop_kv()

    def _get_ref_rag_ev(ti: int) -> dict:
        if ref_rag_evals is not None and ti in ref_rag_evals:
            return ref_rag_evals[ti]
        if ti <= 0:
            return base_step_evals[0]
        idxs_r, _ = select_topk_cosine(all_req_embs_raw[ti], all_req_embs_raw[:ti], k=min(4, ti))
        pr_r = format_vanilla_rag_prompt(
            [stream_eps[j] for j in reversed(idxs_r)], stream_eps[ti]["request"], max_ex_chars=1400
        )
        ev_r = score_single_episode_causal(
            model, tok, pr_r, stream_eps[ti]["pos"], stream_eps[ti]["neg"],
            use_adapter=False, joule_per_tok=sp.joule_per_ctx_tok, base_ms_0=sp.prefill_ms[0], vin_mw=sp.vin_mw,
        ) or base_step_evals[ti]
        if ref_rag_evals is not None:
            ref_rag_evals[ti] = ev_r
        return ev_r

    has_adapter = False
    active_use_adapter = False
    last_flush_t = 0
    dual_phi = 0.0
    min_ep_tokens = 80
    tier1_k_best = 4 if sp.l_crit_tokens >= min_ep_tokens else 0
    post_flush_mode = "delta_fresh" if sp.l_crit_tokens >= min_ep_tokens else "bare"
    committed_coreset_ids: list[int] = []

    step_margins: list[float] = []
    step_accs: list[float] = []
    step_toks: list[int] = []
    step_read_j: list[float] = []
    fresh_window_margins: list[float] = []  # Specifically turns with unflushed fresh edits (`t > last_flush_t`)!

    oppu_margins: list[float] = []
    oppu_accs: list[float] = []
    oppu_toks: list[int] = []
    oppu_read_j: list[float] = []
    oppu_fresh_margins: list[float] = []

    flush_events: list[dict] = []
    all_coserve_reqs: list[dict] = []
    total_write_j = 0.0
    total_write_s = 0.0
    peak_write_gb = 0.0
    coreset_sizes: list[int] = []

    memgpt_block = ""
    T = len(stream_eps)
    ts_cold_cache: dict[int, tuple[dict, int, bool]] = {}

    # Physical 1-flush step budget: GA * N_step micro-batches * SpecProbe c_pair (J/pair),
    # updated directly to measured background VIN Joules after each completed flush.
    nominal_c_write_j = max(1.0, float(grad_accum * n_steps_flush) * sp.write_joule_per_pair)

    def _build_tierserve_rag_prompt(t_idx: int, mode: str, k_cap: int, f_start: int) -> tuple[str, int, bool]:
        """Build a prefix-only retrieval prompt under the calibrated context budget.
        
        Return (prompt, added_context_tokens, fresh_overflow_flag).
        """
        q_str = stream_eps[t_idx]["request"]
        if mode == "bare" or k_cap <= 0 or sp.l_crit_tokens < min_ep_tokens:
            return format_bare_prompt(q_str), 0, (t_idx > f_start)
        # Spherical mean centering over {x_0, ..., x_{t_idx}} requires >= 3 vectors (t_idx >= 2)
        # for non-degenerate degrees of freedom on S^{d-1} (at t_idx == 1, centered dot == -1).
        if t_idx < 2:
            return format_bare_prompt(q_str), 0, False

        k_eff = min(k_cap, t_idx)
        hist_c, q_c_mat = center_embeddings_zero_param(
            all_req_embs_raw[: t_idx + 1],
            all_req_embs_raw[:t_idx],
            all_req_embs_raw[t_idx : t_idx + 1],
        )
        q_c = q_c_mat[0]
        tau_null = (
            compute_null_similarity_quantile(hist_c, q=0.50)
            if t_idx >= 3
            else max(0.0, float(torch.dot(hist_c[0], hist_c[1]).item()))
        )

        # Fresh-first capacity-bounded MMR over unflushed delta [f_start, t_idx), then [0, f_start)
        picked: list[int] = []
        n_fresh_above_null = 0
        if f_start < t_idx:
            fresh_c = hist_c[f_start:t_idx]
            fresh_sims = torch.mv(fresh_c, q_c)
            n_fresh_above_null = int((fresh_sims >= tau_null).sum().item())
            f_rel, _ = select_mmr(q_c, fresh_c, k=min(k_eff, t_idx - f_start), lam=0.50, sim_floor=tau_null)
            picked.extend(f_start + r_i for r_i in f_rel)

        if len(picked) < k_eff and f_start > 0:
            comm_c = hist_c[:f_start]
            c_rel, _ = select_mmr(q_c, comm_c, k=min(k_eff - len(picked), f_start), lam=0.50, sim_floor=tau_null)
            for c_i in c_rel:
                if c_i not in picked and len(picked) < k_eff:
                    picked.append(c_i)

        if not picked:
            return format_bare_prompt(q_str), 0, False

        schema_use = "clean_pair" if mode == "clean_fresh" else "delta_card"
        cands = [stream_eps[j] for j in picked]
        pr_out, n_packed, added_toks = build_capacity_bounded_prompt(
            tok, cands, q_str, sp.l_crit_tokens, schema=schema_use, min_ep_tokens=min_ep_tokens
        )
        ovf = n_fresh_above_null > n_packed
        return pr_out, added_toks, ovf

    for t in range(T):
        ep = stream_eps[t]
        query = ep["request"]

        # Score using the earlier prefix.
        if t == 0 or policy_name == "Base-Static":
            ev = base_step_evals[t]
            added_ctx_toks = 0
            overflow_flag = False
        elif policy_name == "Pure-Tier1-LaMP":
            ev = _get_ref_rag_ev(t)
            added_ctx_toks = max(0, ev["prompt_toks"] - base_step_evals[t]["prompt_toks"])
            overflow_flag = False
        elif policy_name == "Agentic-MemGPT":
            if t in (1, 6, 12, 20, 28) or (t > 0 and not memgpt_block):
                memgpt_block = synthesize_memgpt_persona_block(stream_eps[:t])
            pr = format_memgpt_prompt(memgpt_block, query)
            ev = score_single_episode_causal(
                model, tok, pr, ep["pos"], ep["neg"], use_adapter=False,
                joule_per_tok=sp.joule_per_ctx_tok, base_ms_0=sp.prefill_ms[0], vin_mw=sp.vin_mw,
            ) or base_step_evals[t]
            added_ctx_toks = max(0, ev["prompt_toks"] - base_step_evals[t]["prompt_toks"])
            overflow_flag = False
        elif policy_name in ("Pure-Tier2-Lazy", "Pure-Tier2-Eager"):
            if not has_adapter:
                ev = base_step_evals[t]
            else:
                pr = format_bare_prompt(query)
                ev = score_single_episode_causal(
                    model, tok, pr, ep["pos"], ep["neg"], use_adapter=True,
                    joule_per_tok=sp.joule_per_ctx_tok, base_ms_0=sp.prefill_ms[0], vin_mw=sp.vin_mw,
                ) or base_step_evals[t]
            added_ctx_toks = 0
            overflow_flag = False
        elif policy_name == "Online-OPPU":
            if not has_adapter:
                ev = _get_ref_rag_ev(t)
            else:
                idxs, _ = select_topk_cosine(all_req_embs_raw[t], all_req_embs_raw[:t], k=min(4, t))
                exs = [stream_eps[j] for j in reversed(idxs)]
                pr = format_vanilla_rag_prompt(exs, query, max_ex_chars=1400)
                ev = score_single_episode_causal(
                    model, tok, pr, ep["pos"], ep["neg"], use_adapter=has_adapter,
                    joule_per_tok=sp.joule_per_ctx_tok, base_ms_0=sp.prefill_ms[0], vin_mw=sp.vin_mw,
                ) or base_step_evals[t]
            added_ctx_toks = max(0, ev["prompt_toks"] - base_step_evals[t]["prompt_toks"])
            overflow_flag = False
        elif policy_name in ("TierServe-Planner@LLMS", "TierServe-Full"):
            # Capacity-Bounded Fresh-First Routing (`SpecProbe` L_crit + Null-Quantile [edit-delta] + Causal LOO)
            added_ctx_toks = 0
            overflow_flag = False
            pr = format_bare_prompt(query)
            if not has_adapter:
                if sp.l_crit_tokens < min_ep_tokens:
                    tier1_k_best = 0
                    post_flush_mode = "bare"
                else:
                    prior_lifts = sorted(
                        ts_cold_cache[ei][0]["margin"] - base_step_evals[ei]["margin"]
                        for ei in ts_cold_cache
                        if ei < t
                    )
                    if len(prior_lifts) >= 3:
                        core_lifts = prior_lifts[1:] if len(prior_lifts) >= 4 else prior_lifts
                        tier1_k_best = 4 if (sum(core_lifts) / len(core_lifts)) > 0.0 else 0
                        post_flush_mode = "delta_fresh" if tier1_k_best > 0 else "bare"
                    else:
                        tier1_k_best = 4
                        post_flush_mode = "delta_fresh"

            if t >= 1:
                if tier1_k_best <= 0 or post_flush_mode == "bare":
                    # Narrow recurrent state capacity verified via SpecProbe / causal LOO check:
                    # route k*=0 bare prompt and flag overflow so Ski-Rental promotes edits to Tier-2 LoRA.
                    if t > last_flush_t:
                        overflow_flag = True
                else:
                    pr, added_ctx_toks, overflow_flag = _build_tierserve_rag_prompt(
                        t, post_flush_mode, tier1_k_best, last_flush_t
                    )

            use_ad_now = active_use_adapter if has_adapter else False
            if not use_ad_now and added_ctx_toks == 0:
                ev = base_step_evals[t]
            else:
                ev = score_single_episode_causal(
                    model, tok, pr, ep["pos"], ep["neg"], use_adapter=use_ad_now,
                    joule_per_tok=sp.joule_per_ctx_tok, base_ms_0=sp.prefill_ms[0], vin_mw=sp.vin_mw,
                ) or base_step_evals[t]
                if not use_ad_now and added_ctx_toks > 0:
                    ts_cold_cache[t] = (ev, added_ctx_toks, overflow_flag)
        else:
            raise ValueError(f"Unknown policy {policy_name}")

        m_gain = ev["margin"] - base_step_evals[t]["margin"]
        step_margins.append(ev["margin"])
        step_accs.append(ev["acc"])
        step_toks.append(ev["prompt_toks"])
        step_read_j.append(ev["read_j"])
        if t > last_flush_t:
            fresh_window_margins.append(m_gain)

        if policy_name == "Pure-Tier2-Lazy" and oppu_out is not None:
            if t == 0 or not has_adapter:
                ev_op = _get_ref_rag_ev(t)
            else:
                idxs_op, _ = select_topk_cosine(all_req_embs_raw[t], all_req_embs_raw[:t], k=min(4, t))
                exs_op = [stream_eps[j] for j in reversed(idxs_op)]
                pr_op = format_vanilla_rag_prompt(exs_op, query, max_ex_chars=1400)
                ev_op = score_single_episode_causal(
                    model, tok, pr_op, ep["pos"], ep["neg"], use_adapter=True,
                    joule_per_tok=sp.joule_per_ctx_tok, base_ms_0=sp.prefill_ms[0], vin_mw=sp.vin_mw,
                ) or base_step_evals[t]
            op_gain = ev_op["margin"] - base_step_evals[t]["margin"]
            oppu_margins.append(ev_op["margin"])
            oppu_accs.append(ev_op["acc"])
            oppu_toks.append(ev_op["prompt_toks"])
            oppu_read_j.append(ev_op["read_j"])
            if t > last_flush_t:
                oppu_fresh_margins.append(op_gain)

        # Append feedback and check the refresh trigger.
        if t >= T - 2:
            continue

        n_logged = t + 1
        n_fresh = n_logged - last_flush_t
        should_flush = False
        substrate = "external_llmstation"

        if policy_name == "Pure-Tier2-Lazy":
            should_flush = (n_fresh >= 16)
        elif policy_name == "Pure-Tier2-Eager":
            should_flush = (n_fresh >= 5)
        elif policy_name == "Online-OPPU":
            should_flush = (n_fresh >= 16)
        elif policy_name in ("TierServe-Planner@LLMS", "TierServe-Full"):
            substrate = "tierserve_adjoint_microtxn" if policy_name == "TierServe-Full" else "external_llmstation"
            # Accumulate the read/staleness charge used by the refresh trigger.
            # Standardize current margin by causal prefix empirical standard deviation sigma_hat(t)
            curr_m = ev["margin"]
            if len(step_margins) >= 2:
                mu_pref = sum(step_margins) / len(step_margins)
                sigma_pref = math.sqrt(
                    sum((x - mu_pref) ** 2 for x in step_margins) / max(len(step_margins) - 1, 1) + 1e-4
                )
            else:
                sigma_pref = max(abs(curr_m), 1e-2)
            surp_bits = min(2.0, math.log1p(math.exp(-max(-12.0, min(12.0, curr_m / sigma_pref)))) / math.log(2.0))
            overflow_mult = 2.0 if (overflow_flag or sp.l_crit_tokens < min_ep_tokens) else 1.0
            rent_inc = (added_ctx_toks * sp.joule_per_ctx_tok) + (
                sp.write_joule_per_pair * surp_bits * overflow_mult
            )
            dual_phi += rent_inc
            if n_logged >= grad_accum and dual_phi >= nominal_c_write_j:
                should_flush = True

        if should_flush:
            log_slice = stream_eps[:n_logged]
            base_m_slice = [base_step_evals[j]["margin"] for j in range(n_logged)]
            emb_c = center_embeddings_zero_param(all_req_embs_raw[:n_logged], all_req_embs_raw[:n_logged])[0]

            # Use the same optimizer settings for adapting policies.
            cards_pool = None
            use_sft_alpha = 0.10
            scale_coreset = False
            lr_flush = lr

            if policy_name in ("TierServe-Planner@LLMS", "TierServe-Full"):
                sel_ids, kmeta = auto_dedilute_kneedle_select(log_slice, base_m_slice, emb_c)
                train_sub = [log_slice[i] for i in sel_ids]
                train_refs = [base_step_evals[i].get("ref_tuple") for i in sel_ids]
                committed_coreset_ids = list(sel_ids)
                coreset_sizes.append(kmeta["m_star"])
            else:
                # Standard baseline training set (latest 24 or all up to 24)
                start_w = max(0, n_logged - 24)
                train_sub = log_slice[start_w:]
                train_refs = [base_step_evals[i].get("ref_tuple") for i in range(start_w, n_logged)]
                coreset_sizes.append(len(train_sub))

            fg_for_flush = fg_prompts
            f_res = execute_controlled_flush(
                substrate=substrate,
                model=model,
                tok=tok,
                flush_pairs=train_sub,
                fg_prompts=fg_for_flush,
                qps=qps,
                ttft_slo_ms=ttft_slo_ms,
                tpot_slo_ms=tpot_slo_ms,
                n_steps=n_steps_flush,
                grad_accum=grad_accum,
                lr=lr_flush,
                max_p_tr=max_p_tr,
                max_c_tr=max_c_tr,
                ctx_cards_pool=cards_pool,
                l_crit_tokens=sp.l_crit_tokens,
                vin_mw=sp.vin_mw,
                sft_alpha=use_sft_alpha,
                scale_steps_by_coreset=scale_coreset,
                seed=20261107 + len(flush_events),
                precomputed_refs=train_refs,
            )
            has_adapter = True
            active_use_adapter = True

            # Recheck context lift on the recent historical prefix after refresh.
            if policy_name in ("TierServe-Planner@LLMS", "TierServe-Full"):
                if sp.l_crit_tokens < min_ep_tokens:
                    post_flush_mode, tier1_k_best = "bare", 0
                else:
                    cal_ids = list(range(max(2, n_logged - 3), n_logged))
                    cand_post: list[tuple[str, int]] = [("delta_fresh", 4), ("clean_fresh", 4), ("bare", 0)]
                    best_pm, best_pk = cand_post[0]
                    best_cm = -1e9
                    for pm_c, pk_c in cand_post:
                        m_vals = []
                        for ci in cal_ids:
                            pr_cal, _, _ = _build_tierserve_rag_prompt(ci, pm_c, pk_c, last_flush_t)
                            ev_cal = score_single_episode_causal(
                                model, tok, pr_cal, stream_eps[ci]["pos"], stream_eps[ci]["neg"],
                                use_adapter=True, joule_per_tok=sp.joule_per_ctx_tok,
                                base_ms_0=sp.prefill_ms[0], vin_mw=sp.vin_mw,
                            )
                            if ev_cal is not None:
                                m_vals.append(ev_cal["margin"])
                        mean_cm = sum(m_vals) / max(len(m_vals), 1)
                        if mean_cm > best_cm + 1e-6:
                            best_cm = mean_cm
                            best_pm, best_pk = pm_c, pk_c
                    post_flush_mode, tier1_k_best = best_pm, best_pk

            last_flush_t = n_logged
            dual_phi = 0.0
            # Update the write-cost threshold with the attributed energy estimate.
            nominal_c_write_j = max(1.0, float(f_res["vin_j"]))
            flush_events.append({
                "at_turn": n_logged,
                "coreset_m": len(train_sub),
                "bg_s": round(f_res["bg_s"], 2),
                "vin_j": round(f_res["vin_j"], 1),
                "peak_gb": round(f_res["peak_gb"], 3),
                "slo_joint_pct": f_res["slo_joint_pct"],
                "ttft_p95_ms": f_res["ttft_p95_ms"],
            })
            all_coserve_reqs.extend(f_res["reqs"])
            total_write_j += f_res["vin_j"]
            total_write_s += f_res["bg_s"]
            peak_write_gb = max(peak_write_gb, f_res["peak_gb"])

    base_mean_m = sum(x["margin"] for x in base_step_evals[:T]) / max(T, 1)
    mean_m = sum(step_margins) / max(T, 1)
    # Also compute steady-state adaptation window (turns [8 : T), after initial cold start)
    t_warm = min(8, T // 3)
    warm_dm = (
        (sum(step_margins[t_warm:]) - sum(x["margin"] for x in base_step_evals[t_warm:T]))
        / max(T - t_warm, 1)
    )
    coserve_slo_pct = (
        round(
            100.0
            * sum(1 for r in all_coserve_reqs if r["ttft_ms"] <= ttft_slo_ms and r["tpot_ms"] <= tpot_slo_ms)
            / len(all_coserve_reqs),
            1,
        )
        if all_coserve_reqs
        else 100.0
    )
    coserve_ttft_p95 = (
        round(sorted(r["ttft_ms"] for r in all_coserve_reqs)[int(0.95 * (len(all_coserve_reqs) - 1))], 1)
        if all_coserve_reqs
        else 0.0
    )
    total_read_j = sum(step_read_j)

    if policy_name == "Pure-Tier2-Lazy" and oppu_out is not None and len(oppu_margins) == T:
        op_mean_m = sum(oppu_margins) / max(T, 1)
        op_warm_dm = (
            (sum(oppu_margins[t_warm:]) - sum(x["margin"] for x in base_step_evals[t_warm:T]))
            / max(T - t_warm, 1)
        )
        op_tot_rj = sum(oppu_read_j)
        oppu_out.update({
            "policy": "Online-OPPU",
            "n_turns": T,
            "acc": round(sum(oppu_accs) / max(T, 1), 4),
            "dm_all": round(op_mean_m - base_mean_m, 4),
            "dm_post_warmup": round(op_warm_dm, 4),
            "dm_fresh_unflushed": round(sum(oppu_fresh_margins) / max(len(oppu_fresh_margins), 1), 4),
            "mean_prompt_toks": round(sum(oppu_toks) / max(T, 1), 1),
            "read_j_per_turn": round(op_tot_rj / max(T, 1), 2),
            "total_read_j": round(op_tot_rj, 1),
            "total_write_j": round(total_write_j, 1),
            "total_lifecycle_j": round(op_tot_rj + total_write_j, 1),
            "n_flushes": len(flush_events),
            "mean_coreset_m": round(sum(coreset_sizes) / max(len(coreset_sizes), 1), 1) if coreset_sizes else 0.0,
            "coserve_peak_gb": round(peak_write_gb, 3),
            "coserve_ttft_p95_ms": coserve_ttft_p95,
            "coserve_slo_pct": coserve_slo_pct,
            "per_turn_dm": [round(oppu_margins[i] - base_step_evals[i]["margin"], 4) for i in range(T)],
            "per_turn_accuracy": oppu_accs, "per_turn_read_j": oppu_read_j,
            "foreground_requests": all_coserve_reqs, "flush_events": flush_events,
        })

    return {
        "policy": policy_name,
        "n_turns": T,
        "acc": round(sum(step_accs) / max(T, 1), 4),
        "dm_all": round(mean_m - base_mean_m, 4),
        "dm_post_warmup": round(warm_dm, 4),
        "dm_fresh_unflushed": round(sum(fresh_window_margins) / max(len(fresh_window_margins), 1), 4),
        "mean_prompt_toks": round(sum(step_toks) / max(T, 1), 1),
        "read_j_per_turn": round(total_read_j / max(T, 1), 2),
        "total_read_j": round(total_read_j, 1),
        "total_write_j": round(total_write_j, 1),
        "total_lifecycle_j": round(total_read_j + total_write_j, 1),
        "n_flushes": len(flush_events),
        "mean_coreset_m": round(sum(coreset_sizes) / max(len(coreset_sizes), 1), 1) if coreset_sizes else 0.0,
        "coserve_peak_gb": round(peak_write_gb, 3),
        "coserve_ttft_p95_ms": coserve_ttft_p95,
        "coserve_slo_pct": coserve_slo_pct,
        "per_turn_dm": [round(step_margins[i] - base_step_evals[i]["margin"], 4) for i in range(T)],
        "per_turn_accuracy": step_accs, "per_turn_read_j": step_read_j,
        "foreground_requests": all_coserve_reqs, "flush_events": flush_events,
    }


def run_kneedle_and_generation_audit(
    model,
    tok,
    users: list[dict],
    sp: SpecProbeProfile,
    lr: float,
    max_p_tr: int,
    max_c_tr: int,
    is_27b: bool,
    prev_audit: dict | None = None,
) -> dict:
    """Compare coreset selectors and generated responses with matched training settings."""
    subset_u = users[: (4 if is_27b else min(10, len(users)))]
    coreset_variants = ["Full-Log", "CRAIG-18", "EL2N-18", "DeDilute-M10", "DeDilute-M26", "Auto-DeDilute-Kneedle"]
    agg_dm: dict[str, list[float]] = {k: [] for k in coreset_variants}
    agg_acc: dict[str, list[float]] = {k: [] for k in coreset_variants}
    agg_m_chosen: list[int] = []
    kneedle_u0_curve: dict = {}

    gen_metrics = {
        "Base": {"rouge_l": [], "edit_sim": []},
        "Pure-Tier1-LaMP": {"rouge_l": [], "edit_sim": []},
        "Online-OPPU": {"rouge_l": [], "edit_sim": []},
        "TierServe": {"rouge_l": [], "edit_sim": []},
    }

    is_mock = (sp.backbone == "mock_gpt2")
    n_st_base = 1 if is_mock else (5 if is_27b else 8)
    ga_audit = 2 if is_mock else 3
    lr_audit = 4.5e-5 if is_27b else (6.5e-5 if sp.backbone == "moe_olmoe7b" else 8.5e-5)
    sft_audit = 0.35

    for u_idx, u in enumerate(subset_u):
        pool = u["train_k160"][: (28 if is_27b else 48)]
        held = u["heldout"][: (8 if is_27b else 16)]
        if len(pool) < 12 or len(held) < 4:
            continue

        reset_peft_lora(model, seed=20261107)
        emb_raw = embed_texts(model, tok, [p["request"] for p in pool], max_len=192)
        emb_c = center_embeddings_zero_param(emb_raw, emb_raw)[0]

        # Cache base margins and branch reference scores for coreset comparisons.
        base_pool_m: list[float | None] = []
        pool_refs: list[tuple[float, float, int, int] | None] = []
        with model.disable_adapter():
            for p in pool:
                pr = format_bare_prompt(p["request"])
                sp_s, lp_p = sum_and_mean_logp_no_grad(model, tok, pr, p["pos"], max_prompt=max_p_tr, max_comp=max_c_tr)
                sn_s, lp_n = sum_and_mean_logp_no_grad(model, tok, pr, p["neg"], max_prompt=max_p_tr, max_comp=max_c_tr)
                base_pool_m.append((lp_p - lp_n) if (lp_p is not None and lp_n is not None) else None)
                enc_p = _encode_prompt_comp(tok, pr, p["pos"], max_p_tr, max_c_tr)
                enc_n = _encode_prompt_comp(tok, pr, p["neg"], max_p_tr, max_c_tr)
                if sp_s is not None and sn_s is not None and enc_p is not None and enc_n is not None:
                    pool_refs.append((float(sp_s), float(sn_s), len(enc_p[1]), len(enc_n[1])))
                else:
                    pool_refs.append(None)

            held_base_m = []
            for h in held:
                pr = format_bare_prompt(h["request"])
                lp_p = mean_logp_no_grad(model, tok, pr, h["pos"], max_prompt=1536, max_comp=224)
                lp_n = mean_logp_no_grad(model, tok, pr, h["neg"], max_prompt=1536, max_comp=224)
                held_base_m.append((lp_p - lp_n) if (lp_p is not None and lp_n is not None) else 0.0)

        # Open-ended generation on first heldout items per user
        gen_slice = held[: (1 if is_mock else (3 if is_27b else 4))]
        h_emb_raw = embed_texts(model, tok, [h["request"] for h in gen_slice], max_len=192)
        _, h_emb_c = center_embeddings_zero_param(emb_raw, emb_raw, h_emb_raw)

        for gi, gitem in enumerate(gen_slice):
            gold = gitem["pos"]
            # Base generation
            txt_b, _, _ = greedy_generate_text(model, tok, format_bare_prompt(gitem["request"]), max_new_tokens=(8 if is_mock else 48), use_adapter=False)
            gen_metrics["Base"]["rouge_l"].append(rouge_l_f1(txt_b, gold))
            gen_metrics["Base"]["edit_sim"].append(1.0 - word_edit_ratio(txt_b, gold))
            # LaMP top-4 generation
            idxs4, _ = select_topk_cosine(h_emb_c[gi], emb_c, k=min(4, len(pool)))
            pr_lamp = format_vanilla_rag_prompt([pool[j] for j in reversed(idxs4)], gitem["request"], max_ex_chars=1200)
            txt_l, _, _ = greedy_generate_text(model, tok, pr_lamp, max_new_tokens=(8 if is_mock else 48), use_adapter=False)
            gen_metrics["Pure-Tier1-LaMP"]["rouge_l"].append(rouge_l_f1(txt_l, gold))
            gen_metrics["Pure-Tier1-LaMP"]["edit_sim"].append(1.0 - word_edit_ratio(txt_l, gold))

        # Coreset selector map
        sel_kneedle, kmeta = auto_dedilute_kneedle_select(pool, base_pool_m, emb_c)
        agg_m_chosen.append(kmeta["m_star"])
        if u_idx == 0:
            kneedle_u0_curve = kmeta

        sel_m10 = dedilute_coreset_select(pool, base_pool_m, emb_raw, target_m=10)
        sel_m26 = dedilute_coreset_select(pool, base_pool_m, emb_raw, target_m=min(26, len(pool)))
        sel_craig = select_coreset_craig_facility_location(emb_c, m=min(18, len(pool)))
        sel_el2n = select_coreset_el2n(base_pool_m, m=min(18, len(pool)))

        variant_sets = {
            "Full-Log": list(range(len(pool))),
            "CRAIG-18": sel_craig,
            "EL2N-18": sel_el2n,
            "DeDilute-M10": sel_m10,
            "DeDilute-M26": sel_m26,
            "Auto-DeDilute-Kneedle": sel_kneedle,
        }

        for vname, s_ids in variant_sets.items():
            sub_pairs = [pool[i] for i in s_ids]
            sub_refs = [pool_refs[i] for i in s_ids]
            # Use the same step count, learning rate and SFT weight for coreset variants.
            _, _ = run_adjoint_microtxn_engine(
                model, tok, sub_pairs, [], [],
                n_steps=n_st_base, grad_accum=ga_audit, lr=lr_audit, max_p_train=max_p_tr, max_c_train=max_c_tr,
                gen_toks=4, drain_future_arrivals=False, sft_alpha=sft_audit, precomputed_refs=sub_refs,
            )

            model.eval()
            for hi, h in enumerate(held):
                pr = format_bare_prompt(h["request"])
                lp_p = mean_logp_no_grad(model, tok, pr, h["pos"], max_prompt=1536, max_comp=224)
                lp_n = mean_logp_no_grad(model, tok, pr, h["neg"], max_prompt=1536, max_comp=224)
                if lp_p is not None and lp_n is not None:
                    agg_dm[vname].append((lp_p - lp_n) - held_base_m[hi])
                    agg_acc[vname].append(1.0 if lp_p > lp_n else 0.0)

            if vname == "Full-Log":
                for gi, gitem in enumerate(gen_slice):
                    gold = gitem["pos"]
                    idxs4, _ = select_topk_cosine(h_emb_c[gi], emb_c, k=min(4, len(pool)))
                    pr_oppu = format_vanilla_rag_prompt([pool[j] for j in reversed(idxs4)], gitem["request"], max_ex_chars=1200)
                    txt_op, _, _ = greedy_generate_text(model, tok, pr_oppu, max_new_tokens=(8 if is_mock else 48), use_adapter=True)
                    gen_metrics["Online-OPPU"]["rouge_l"].append(rouge_l_f1(txt_op, gold))
                    gen_metrics["Online-OPPU"]["edit_sim"].append(1.0 - word_edit_ratio(txt_op, gold))

        # Reuse the coreset-trained adapter for generation and context calibration.
        # Prefer unused historical items; small pools may reuse selected items.
        unseen_pool_ids = [i for i in range(len(pool)) if i not in set(sel_kneedle)]
        val_gen_ids = (
            unseen_pool_ids[-6:]
            if len(unseen_pool_ids) >= 6
            else list(range(max(2, len(pool) - 6), len(pool)))
        )

        def _build_gen_prompt(q_vec: torch.Tensor, q_str: str, mode: str, k_c: int, excl: int | None = None) -> str:
            if mode == "bare" or k_c <= 0:
                return format_bare_prompt(q_str)
            k_t = min(k_c, len(pool) - 1)
            if mode == "mmr":
                idxs_m, _ = select_mmr(q_vec, emb_c, k=k_t, lam=0.50, sim_floor=-1.0, exclude_idx=excl)
                if len(idxs_m) < k_t:
                    idxs_d, _ = select_topk_cosine(q_vec, emb_c, k=k_t, exclude_idx=excl)
                    for ix in idxs_d:
                        if ix not in idxs_m and len(idxs_m) < k_t:
                            idxs_m.append(ix)
                return format_vanilla_rag_prompt([pool[j] for j in reversed(idxs_m)], q_str, max_ex_chars=1200)
            idxs_d, _ = select_topk_cosine(q_vec, emb_c, k=k_t, exclude_idx=excl)
            return format_vanilla_rag_prompt([pool[j] for j in reversed(idxs_d)], q_str, max_ex_chars=1200)

        # Evaluate each planner candidate with the trained adapter enabled.
        cand_gen_Global: list[tuple[str, int, bool]] = [
            ("dense", 4, True),
            ("mmr", 4, True),
            ("dense", 2, True),
        ]
        if sp.l_crit_tokens < 250:
            cand_gen_Global.append(("bare", 0, True))
        best_mode_g, best_k_gen, best_ad_g = cand_gen_Global[0]
        best_val_sc = -1e9
        for mode_c, k_cand, ad_c in cand_gen_Global:
            sc_list = []
            for ti_g in val_gen_ids[-4:]:
                pr_cand = _build_gen_prompt(emb_c[ti_g], pool[ti_g]["request"], mode_c, k_cand, excl=ti_g)
                lp_pos = mean_logp_no_grad(model, tok, pr_cand, pool[ti_g]["pos"], max_prompt=1536, max_comp=128)
                lp_neg = mean_logp_no_grad(model, tok, pr_cand, pool[ti_g]["neg"], max_prompt=1536, max_comp=128)
                if lp_pos is not None and lp_neg is not None:
                    sc_list.append(lp_pos - 0.25 * lp_neg)
            mean_sc = sum(sc_list) / max(len(sc_list), 1)
            if mean_sc > best_val_sc + 1e-6:
                best_val_sc = mean_sc
                best_mode_g, best_k_gen, best_ad_g = mode_c, k_cand, ad_c

        for gi, gitem in enumerate(gen_slice):
            gold = gitem["pos"]
            pr_ts = _build_gen_prompt(h_emb_c[gi], gitem["request"], best_mode_g, best_k_gen, excl=None)
            txt_ts, _, _ = greedy_generate_text(
                model, tok, pr_ts, max_new_tokens=(8 if is_mock else 48), use_adapter=True
            )
            gen_metrics["TierServe"]["rouge_l"].append(rouge_l_f1(txt_ts, gold))
            gen_metrics["TierServe"]["edit_sim"].append(1.0 - word_edit_ratio(txt_ts, gold))

    coreset_summary = {}
    for vname in coreset_variants:
        dms = agg_dm[vname]
        acs = agg_acc[vname]
        n = max(len(dms), 1)
        coreset_summary[vname] = {
            "n_pairs": len(dms),
            "acc": round(sum(acs) / n, 4),
            "dm": round(sum(dms) / n, 4),
        }
    if agg_dm["Auto-DeDilute-Kneedle"] and agg_dm["CRAIG-18"]:
        n_min = min(len(agg_dm["Auto-DeDilute-Kneedle"]), len(agg_dm["CRAIG-18"]))
        coreset_summary["kneedle_vs_craig_bootstrap"] = paired_bootstrap_test(
            agg_dm["Auto-DeDilute-Kneedle"][:n_min], agg_dm["CRAIG-18"][:n_min]
        )
    coreset_summary["auto_kneedle_m_mean"] = round(sum(agg_m_chosen) / max(len(agg_m_chosen), 1), 2)
    coreset_summary["auto_kneedle_m_list"] = agg_m_chosen
    coreset_summary["sample_kneedle_curve"] = kneedle_u0_curve

    gen_summary = {}
    for gname, gd in gen_metrics.items():
        n = max(len(gd["rouge_l"]), 1)
        gen_summary[gname] = {
            "n_gen": len(gd["rouge_l"]),
            "rouge_l": round(100.0 * sum(gd["rouge_l"]) / n, 2),
            "edit_similarity": round(100.0 * sum(gd["edit_sim"]) / n, 2),
        }

    return {"coreset_sensitivity": coreset_summary, "open_generation": gen_summary}


def run_backbone_streaming_suite(
    backbone: str,
    users: list[dict],
    stream_len: int,
    run_coreset_gen: bool,
    partial_path: Path,
    prev_result: dict | None = None,
    only_open_gen: bool = False,
    refresh_policies: set[str] | None = None,
) -> dict:
    is_mock = backbone == "mock_gpt2"
    is_27b = backbone == "hybrid_qwen27b"
    lr = 1.2e-4 if is_27b else (1.6e-4 if backbone == "moe_olmoe7b" else 2.2e-4)
    max_p_tr = 768 if is_27b else 960
    max_c_tr = 160 if is_27b else 192
    n_steps_flush = 1 if is_mock else (5 if is_27b else 7)
    grad_accum = 2 if is_mock else 4

    model, tok, load_info = load_v2_backbone(backbone, quant_4bit=False)
    sp = run_specprobe_calibration(model, tok, backbone=backbone)
    fg_probe_prompts = [format_bare_prompt(x["request"]) for x in users[0]["heldout"][:8]]
    unloaded = measure_unloaded_single_query_baseline(model, tok, fg_probe_prompts, gen_toks=16)
    ttft_slo_ms = max(350.0, round(2.5 * unloaded["unloaded_ttft_p95_ms"], 1))
    tpot_slo_ms = max(20.0, round(1.5 * unloaded["unloaded_tpot_p95_ms"], 1))
    qps = round(max(0.03, 0.34 * unloaded["service_capacity_qps"]), 3)

    log(
        f"=== [V2 Controlled Stream] {backbone} ({len(users)} users, T={stream_len}) | "
        f"L_crit={sp.l_crit_tokens} tok | TTFT_SLO={ttft_slo_ms}ms @ {qps} QPS ==="
    )

    policies = [
        "Base-Static",
        "Pure-Tier1-LaMP",
        "Agentic-MemGPT",
        "Pure-Tier2-Lazy",
        "Pure-Tier2-Eager",
        "Online-OPPU",
        "TierServe-Planner@LLMS",
        "TierServe-Full",
    ]

    per_user_records = []
    for u_idx, u in enumerate(users):
        t_u0 = time.perf_counter()
        eps = u["stream_episodes"][:stream_len]
        T_u = len(eps)
        fg_prompts = [format_bare_prompt(h["request"]) for h in u["heldout"][:10]] or fg_probe_prompts

        prev_u = (
            prev_result["per_user"][u_idx]["policies"]
            if (prev_result and u_idx < len(prev_result.get("per_user", [])) and (only_open_gen or refresh_policies))
            else None
        )
        pols_to_run = (
            set()
            if only_open_gen
            else (refresh_policies if refresh_policies is not None else set(policies))
        )
        all_cached_for_u = bool(
            prev_u
            and not pols_to_run
            and all(
                p in prev_u and prev_u[p].get("n_turns") == T_u
                for p in policies
            )
        )
        if not all_cached_for_u:
            # Precompute invariant base embeddings & base k=0 step evaluations once per user
            reset_peft_lora(model, seed=20261107)
            req_embs_raw = embed_texts(model, tok, [e["request"] for e in eps], max_len=192)
            base_evals = []
            for e in eps:
                ev0 = score_single_episode_causal(
                    model, tok, format_bare_prompt(e["request"]), e["pos"], e["neg"],
                    use_adapter=False, joule_per_tok=sp.joule_per_ctx_tok, base_ms_0=sp.prefill_ms[0], vin_mw=sp.vin_mw,
                )
                base_evals.append(ev0 or {"margin": 0.0, "acc": 0.0, "prompt_toks": 64, "read_ms": 20.0, "read_j": 1.0})
            ref_rag_evals: dict[int, dict] = {0: base_evals[0]}
        else:
            req_embs_raw = None
            base_evals = []
            ref_rag_evals = {}

        u_pols = {}
        shared_oppu_res: dict = {}
        for pol in policies:
            can_reuse_pol = bool(
                prev_u
                and pol not in pols_to_run
                and pol in prev_u
                and prev_u[pol].get("n_turns") == T_u
            )
            if can_reuse_pol:
                u_pols[pol] = dict(prev_u[pol])
            elif pol == "Online-OPPU" and shared_oppu_res.get("n_turns") == T_u:
                u_pols[pol] = dict(shared_oppu_res)
            else:
                u_pols[pol] = simulate_online_stream_policy(
                    pol, model, tok, eps, req_embs_raw, base_evals, sp,
                    fg_prompts, qps, ttft_slo_ms, tpot_slo_ms,
                    n_steps_flush=n_steps_flush, grad_accum=grad_accum,
                    lr=lr, max_p_tr=max_p_tr, max_c_tr=max_c_tr,
                    ref_rag_evals=ref_rag_evals,
                    oppu_out=(shared_oppu_res if pol == "Pure-Tier2-Lazy" else None),
                )

        per_user_records.append({
            "uid_prefix": u["prefix"],
            "n_turns": len(eps),
            "policies": u_pols,
        })
        log(
            f"  [{backbone} u#{u_idx + 1:02d}/{len(users)} {u['prefix']} ({time.perf_counter() - t_u0:.1f}s)] "
            f"LaMP={u_pols['Pure-Tier1-LaMP']['dm_post_warmup']:+.4f} | "
            f"T2Lazy={u_pols['Pure-Tier2-Lazy']['dm_post_warmup']:+.4f} (fresh={u_pols['Pure-Tier2-Lazy']['dm_fresh_unflushed']:+.4f}) | "
            f"OPPU={u_pols['Online-OPPU']['dm_post_warmup']:+.4f} | "
            f"Planner@LLMS={u_pols['TierServe-Planner@LLMS']['dm_post_warmup']:+.4f} (fresh={u_pols['TierServe-Planner@LLMS']['dm_fresh_unflushed']:+.4f}, SLO={u_pols['TierServe-Planner@LLMS']['coserve_slo_pct']}%) | "
            f"Full={u_pols['TierServe-Full']['dm_post_warmup']:+.4f} (SLO={u_pols['TierServe-Full']['coserve_slo_pct']}%)"
        )
        partial_path.write_text(json.dumps({
            "backbone": backbone,
            "completed_users": len(per_user_records),
            "specprobe": sp.__dict__,
            "users": per_user_records,
        }, indent=2))

    # Aggregate per-user scores and run paired bootstrap comparisons.
    agg_policies = {}
    for pol in policies:
        rows = [u["policies"][pol] for u in per_user_records]
        nu = max(len(rows), 1)
        agg_policies[pol] = {
            "n_users": len(rows),
            "acc": round(sum(r["acc"] for r in rows) / nu, 4),
            "dm_all": round(sum(r["dm_all"] for r in rows) / nu, 4),
            "dm_post_warmup": round(sum(r["dm_post_warmup"] for r in rows) / nu, 4),
            "dm_fresh_unflushed": round(sum(r["dm_fresh_unflushed"] for r in rows) / nu, 4),
            "mean_prompt_toks": round(sum(r["mean_prompt_toks"] for r in rows) / nu, 1),
            "read_j_per_turn": round(sum(r["read_j_per_turn"] for r in rows) / nu, 2),
            "total_write_j_mean": round(sum(r["total_write_j"] for r in rows) / nu, 1),
            "total_lifecycle_j_mean": round(sum(r["total_lifecycle_j"] for r in rows) / nu, 1),
            "n_flushes_mean": round(sum(r["n_flushes"] for r in rows) / nu, 2),
            "mean_coreset_m": round(sum(r["mean_coreset_m"] for r in rows) / nu, 1),
            "coserve_peak_gb": round(max(r["coserve_peak_gb"] for r in rows), 3),
            "coserve_ttft_p95_ms": round(sum(r["coserve_ttft_p95_ms"] for r in rows) / nu, 1),
            "coserve_slo_pct": round(sum(r["coserve_slo_pct"] for r in rows) / nu, 1),
        }

    # Mean turn-by-turn causal trajectory across users for Figure 3(a)
    prev_traj_map = (prev_result or {}).get("mean_turn_dm_trajectory", {}) if prev_result else {}
    mean_turn_traj = {}
    for pol in policies:
        if all(len(u["policies"][pol].get("per_turn_dm") or []) > 0 for u in per_user_records):
            min_t = min((len(u["policies"][pol]["per_turn_dm"]) for u in per_user_records), default=0)
            mean_turn_traj[pol] = [
                round(sum(u["policies"][pol]["per_turn_dm"][ti] for u in per_user_records) / max(len(per_user_records), 1), 4)
                for ti in range(min_t)
            ]
        elif pol in prev_traj_map:
            mean_turn_traj[pol] = list(prev_traj_map[pol])
        else:
            mean_turn_traj[pol] = []

    # Actual paired user means are the sampling units; turns are dependent.
    from core import user_comparison
    stats_controlled = {}
    for ref_pol in ("Pure-Tier1-LaMP", "Agentic-MemGPT", "Pure-Tier2-Lazy", "Online-OPPU"):
        stats_controlled[f"Planner_LLMS_vs_{ref_pol}"] = user_comparison(
            per_user_records, "TierServe-Planner@LLMS", ref_pol)

    extra_audit = (prev_result or {}).get("kneedle_and_generation_audit", {}) if prev_result else {}
    if run_coreset_gen or only_open_gen:
        extra_audit = run_kneedle_and_generation_audit(
            model, tok, users, sp, lr=lr, max_p_tr=max_p_tr, max_c_tr=max_c_tr, is_27b=is_27b, prev_audit=extra_audit
        )

    del model
    drop_kv()

    return {
        "backbone": backbone,
        "n_users": len(per_user_records),
        "stream_len": stream_len,
        "load_info": load_info,
        "specprobe": sp.__dict__,
        "slo_config": {"ttft_slo_ms": ttft_slo_ms, "tpot_slo_ms": tpot_slo_ms, "qps": qps},
        "summary_policies": agg_policies,
        "mean_turn_dm_trajectory": mean_turn_traj,
        "controlled_llmstation_bootstrap": stats_controlled,
        "kneedle_and_generation_audit": extra_audit,
        "per_user": [
            {
                "uid_prefix": u["uid_prefix"],
                "policies": u["policies"],
            }
            for u in per_user_records
        ],
    }

