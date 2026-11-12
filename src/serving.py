"""Foreground generation and serving latency summaries."""
from __future__ import annotations
import random
import time
from typing import Sequence
import torch
from core import _nullcontext


def percentile(vals: Sequence[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(round((q / 100.0) * (len(s) - 1)))))
    return round(float(s[idx]), 2)



def generate_poisson_arrivals(n_req: int, qps: float, seed: int = 20261106) -> list[float]:
    """Deterministic seeded Poisson arrival offsets (seconds from t=0)."""
    rng = random.Random(seed)
    t = 0.15
    arrivals = []
    for _ in range(n_req):
        arrivals.append(round(t, 4))
        t += rng.expovariate(qps)
    return arrivals



@torch.no_grad()
def serve_single_request_kv(
    model,
    tok,
    prompt: str,
    arrival_wall_t: float,
    max_new_tokens: int = 24,
    max_prompt_tok: int = 512,
    use_adapter: bool = False,
    cuda_stream=None,
) -> dict:
    """Run prefill and cached greedy decode; include queue wait in TTFT."""
    device = next(model.parameters()).device
    ids_list = tok(prompt, truncation=True, max_length=max_prompt_tok).input_ids
    input_ids = torch.tensor([ids_list], dtype=torch.long, device=device)

    t_dispatch = time.perf_counter()
    queue_wait_ms = max(0.0, (t_dispatch - arrival_wall_t) * 1e3)

    was_training = model.training
    model.eval()
    ad_ctx = (
        _nullcontext()
        if (use_adapter or not hasattr(model, "disable_adapter"))
        else model.disable_adapter()
    )
    stream_ctx = torch.cuda.stream(cuda_stream) if (cuda_stream is not None and torch.cuda.is_available()) else _nullcontext()

    t_sw0 = time.perf_counter()
    with ad_ctx, stream_ctx:
        switch_ms = (time.perf_counter() - t_sw0) * 1e3
        # Prefill pass -> first output token
        out = model(input_ids=input_ids, use_cache=True)
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        past_kv = out.past_key_values
        generated_ids = [int(next_tok.item())]
        if torch.cuda.is_available():
            if cuda_stream is not None:
                cuda_stream.synchronize()
            else:
                torch.cuda.synchronize()
        t_first = time.perf_counter()

        # Autoregressive KV-cached decode steps
        n_dec = max(1, max_new_tokens - 1)
        for _ in range(n_dec):
            out = model(input_ids=next_tok, past_key_values=past_kv, use_cache=True)
            next_tok = out.logits[:, -1:, :].argmax(dim=-1)
            past_kv = out.past_key_values
            generated_ids.append(int(next_tok.item()))
        if torch.cuda.is_available():
            if cuda_stream is not None:
                cuda_stream.synchronize()
            else:
                torch.cuda.synchronize()
        t_end = time.perf_counter()
        del out, past_kv, next_tok

    if was_training:
        model.train()

    ttft_ms = (t_first - arrival_wall_t) * 1e3
    prefill_exec_ms = (t_first - t_dispatch) * 1e3
    tpot_ms = ((t_end - t_first) * 1e3) / max(1, max_new_tokens - 1)
    e2e_ms = (t_end - arrival_wall_t) * 1e3
    return {
        "generated_ids": generated_ids,
        "queue_wait_ms": round(queue_wait_ms, 2),
        "switch_ms": round(switch_ms, 3),
        "prefill_exec_ms": round(prefill_exec_ms, 2),
        "ttft_ms": round(ttft_ms, 2),
        "tpot_ms": round(tpot_ms, 2),
        "e2e_ms": round(e2e_ms, 2),
        "prompt_tok": len(ids_list),
        "gen_tok": max_new_tokens,
    }



def summarize_slo_metrics(
    req_logs: list[dict],
    ttft_slo_ms: float,
    tpot_slo_ms: float,
    bg_s: float | None,
    peak_gb: float | None,
    tegra: dict | None,
    extra: dict | None = None,
) -> dict:
    ttfts = [r["ttft_ms"] for r in req_logs]
    tpots = [r["tpot_ms"] for r in req_logs]
    waits = [r["queue_wait_ms"] for r in req_logs]
    n = max(len(req_logs), 1)
    met_ttft = sum(1 for x in ttfts if x <= ttft_slo_ms)
    met_both = sum(1 for r in req_logs if r["ttft_ms"] <= ttft_slo_ms and r["tpot_ms"] <= tpot_slo_ms)
    res = {
        "n_requests": len(req_logs),
        "ttft_p50_ms": percentile(ttfts, 50),
        "ttft_p95_ms": percentile(ttfts, 95),
        "ttft_p99_ms": percentile(ttfts, 99),
        "ttft_max_ms": round(max(ttfts), 2) if ttfts else 0.0,
        "queue_wait_p95_ms": percentile(waits, 95),
        "tpot_mean_ms": round(sum(tpots) / n, 2),
        "tpot_p95_ms": percentile(tpots, 95),
        "tpot_p99_ms": percentile(tpots, 99),
        "slo_ttft_budget_ms": round(ttft_slo_ms, 1),
        "slo_tpot_budget_ms": round(tpot_slo_ms, 1),
        "slo_ttft_attainment_pct": round(100.0 * met_ttft / n, 1),
        "slo_joint_attainment_pct": round(100.0 * met_both / n, 1),
        "bg_adaptation_s": round(bg_s, 2) if bg_s is not None else None,
        "peak_cuda_gb": round(peak_gb, 3) if peak_gb is not None else None,
        "vin_j_total": (tegra or {}).get("vin_j"),
    }
    if extra:
        res.update(extra)
    return res

