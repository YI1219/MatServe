#!/usr/bin/env python3
"""Shared data, scoring, retrieval and LoRA training utilities for MatServe.

Includes completion-only log probabilities, cached reference scores, retrieval
prompts, fixed-size coreset selection and paired bootstrap comparisons.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

GB = 1024 ** 3

# Stored V13 group order.
V13_USER_PREFIXES: tuple[str, ...] = (
    "4d89480b",
    "87b7f490",
    "955078d4",
    "c2373ab4",
    "f248475d",
    "314cdcc8",
    "bf057e7f",
    "ab057c61",
    "f7c8d6f",
    "0180d13e",
)

HELDOUT_LO = 200
HELDOUT_HI = 400


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def set_seed(seed: int = 20261105) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def drop_kv() -> None:
    """Collect Python garbage, release unused CUDA allocations and synchronize."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def mem_snapshot() -> dict:
    sysm: dict[str, float] = {}
    if Path("/proc/meminfo").exists():
        with open("/proc/meminfo") as f:
            for line in f:
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                if k in ("MemTotal", "MemFree", "MemAvailable"):
                    sysm[k] = round(int(v.strip().split()[0]) / (1024 ** 2), 2)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return {
            "allocated_gb": round(torch.cuda.memory_allocated() / GB, 3),
            "reserved_gb": round(torch.cuda.memory_reserved() / GB, 3),
            "max_allocated_gb": round(torch.cuda.max_memory_allocated() / GB, 3),
            **sysm,
        }
    return sysm


# Hardware Rail Power Telemetry (Jetson AGX Thor tegrastats)

def parse_tegrastats(path: Path, dt_s: float) -> dict:
    gpu_mw, vin_mw = [], []
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            m_gpu = re.search(r"VDD_GPU\s+(\d+)mW", line)
            if m_gpu:
                gpu_mw.append(int(m_gpu.group(1)))
            m_vin = re.search(r"(?:^|\s)VIN\s+(\d+)mW", line)
            if m_vin:
                vin_mw.append(int(m_vin.group(1)))
    mg = sum(gpu_mw) / max(len(gpu_mw), 1)
    mv = sum(vin_mw) / max(len(vin_mw), 1)
    return {
        "dt_s": round(dt_s, 2),
        "n_samples": len(gpu_mw),
        "gpu_mw_mean": round(mg, 1),
        "vin_mw_mean": round(mv, 1),
        "gpu_j": round((mg / 1000.0) * dt_s, 2),
        "vin_j": round((mv / 1000.0) * dt_s, 2),
    }


class TegraWatch:
    def __init__(self, tag: str, log_dir: str | Path = "/tmp/deltaview_tegra"):
        self.tag = tag
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{tag}_{os.getpid()}_{time.time_ns()}.log"
        self.proc: subprocess.Popen | None = None
        self.t0 = 0.0
        self.result: dict = {}

    def __enter__(self):
        if not torch.cuda.is_available():
            self.proc = None
            self.t0 = time.perf_counter()
            return self
        subprocess.run(
            ["tegrastats", "--stop"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if self.path.exists():
            self.path.unlink()
        try:
            self.proc = subprocess.Popen(
                ["tegrastats", "--interval", "200", "--logfile", str(self.path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            time.sleep(0.22)
        except FileNotFoundError:
            self.proc = None
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if not torch.cuda.is_available():
            self.result = {
                "dt_s": round(time.perf_counter() - self.t0, 2),
                "n_samples": 0,
                "gpu_mw_mean": 0.0,
                "vin_mw_mean": 0.0,
                "gpu_j": 0.0,
                "vin_j": 0.0,
            }
            return False
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        subprocess.run(
            ["tegrastats", "--stop"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.result = parse_tegrastats(self.path, time.perf_counter() - self.t0)
        self.result["raw_power_log"] = str(self.path)
        self.result["raw_power_sha256"] = hashlib.sha256(self.path.read_bytes()).hexdigest() if self.path.exists() else None
        return False


# Data Normalization & V13 TrackA Episode Loading

def word_edit_ratio(a: str, b: str) -> float:
    wa, wb = a.split(), b.split()
    if not wa and not wb:
        return 0.0
    if len(wa) > 400:
        wa = wa[:400]
    if len(wb) > 400:
        wb = wb[:400]
    m, n = len(wa), len(wb)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            prev, dp[j] = dp[j], (
                prev if wa[i - 1] == wb[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            )
    return dp[n] / max(m, n, 1)


def normalize_pair(item: dict) -> dict | None:
    """Extract request, pos, neg and ep_idx from an episode or pair dictionary."""
    req = (
        item.get("request")
        or item.get("prompt")
        or item.get("query")
        or item.get("user_request")
        or item.get("input")
        or ""
    ).strip()
    pos = (
        item.get("pos")
        or item.get("revised")
        or item.get("chosen")
        or item.get("user_revised")
        or item.get("preferred")
        or ""
    ).strip()
    neg = (
        item.get("neg")
        or item.get("draft")
        or item.get("rejected")
        or item.get("assistant_draft")
        or item.get("initial_draft")
        or ""
    ).strip()
    if not req or not pos or not neg or pos == neg:
        return None
    return {
        "request": req,
        "pos": pos,
        "neg": neg,
        "ep_idx": item.get("ep_idx", item.get("idx", item.get("turn_idx", -1))),
    }


def load_v13_user_bundle(data_dir: str | Path, uid_prefix: str) -> dict:
    """Load stored episodes or a prepared training/held-out bundle.
    
    Prepared files contain uid, train_k40, train_k80, train_k160 and heldout.
    Pair labels and episode indices follow the stored archive conventions.
    """
    data_dir = Path(data_dir)
    matches = sorted(data_dir.glob(f"{uid_prefix}*.json"))
    if not matches:
        raise FileNotFoundError(f"No user bundle matching {uid_prefix} in {data_dir}")
    bundle = json.loads(matches[0].read_text())
    return bundle


# Completion log probabilities

def _encode_prompt_comp(
    tok, prompt: str, completion: str, max_prompt: int, max_comp: int
) -> tuple[list[int], list[int]] | None:
    p_full = tok(prompt, truncation=False, add_special_tokens=True).input_ids
    if len(p_full) > max_prompt:
        # Left-truncate old exemplar prefix while preserving BOS + trailing query & cue
        p_ids = p_full[:1] + p_full[-(max_prompt - 1) :]
    else:
        p_ids = p_full
    c_ids = tok(
        completion, truncation=True, max_length=max_comp, add_special_tokens=False
    ).input_ids
    if not p_ids or not c_ids:
        return None
    return p_ids, c_ids


@torch.no_grad()
def mean_logp_no_grad(
    model,
    tok,
    prompt: str,
    completion: str,
    max_prompt: int = 3680,
    max_comp: int = 384,
) -> float | None:
    enc = _encode_prompt_comp(tok, prompt, completion, max_prompt, max_comp)
    if enc is None:
        return None
    p_ids, c_ids = enc
    device = next(model.parameters()).device
    full = torch.tensor([p_ids + c_ids], dtype=torch.long, device=device)
    p_len = len(p_ids)
    out = model(input_ids=full, use_cache=False)
    comp_logits = out.logits[0, p_len - 1 : full.shape[1] - 1, :].float()
    target = full[0, p_len:]
    ce = F.cross_entropy(comp_logits, target, reduction="mean")
    return float(-ce.item())


@torch.no_grad()
def sum_and_mean_logp_no_grad(
    model,
    tok,
    prompt: str,
    completion: str,
    max_prompt: int = 1792,
    max_comp: int = 256,
) -> tuple[float | None, float | None]:
    """Return summed and mean completion log probabilities from one forward pass."""
    enc = _encode_prompt_comp(tok, prompt, completion, max_prompt, max_comp)
    if enc is None:
        return None, None
    p_ids, c_ids = enc
    device = next(model.parameters()).device
    full = torch.tensor([p_ids + c_ids], dtype=torch.long, device=device)
    p_len = len(p_ids)
    out = model(input_ids=full, use_cache=False)
    comp_logits = out.logits[0, p_len - 1 : full.shape[1] - 1, :].float()
    target = full[0, p_len:]
    ce_sum = F.cross_entropy(comp_logits, target, reduction="sum")
    n_tok = max(int(target.numel()), 1)
    return float(-ce_sum.item()), float(-(ce_sum / n_tok).item())


def sum_logp_with_grad(
    model,
    tok,
    prompt: str,
    completion: str,
    max_prompt: int = 1792,
    max_comp: int = 256,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    """Return completion log probability and mean cross-entropy with gradients."""
    enc = _encode_prompt_comp(tok, prompt, completion, max_prompt, max_comp)
    if enc is None:
        return None, None
    p_ids, c_ids = enc
    device = next(model.parameters()).device
    full = torch.tensor([p_ids + c_ids], dtype=torch.long, device=device)
    p_len = len(p_ids)
    out = model(input_ids=full, use_cache=False)
    comp_logits = out.logits[0, p_len - 1 : full.shape[1] - 1, :].float()
    target = full[0, p_len:]
    ce_sum = F.cross_entropy(comp_logits, target, reduction="sum")
    ce_mean = ce_sum / max(int(target.numel()), 1)
    return -ce_sum, ce_mean


# Embeddings and retrieval

@torch.no_grad()
def embed_texts(model, tok, texts: Sequence[str], max_len: int = 256) -> torch.Tensor:
    """Mean-pooled last hidden state embeddings (L2-normalized, CPU float32)."""
    device = next(model.parameters()).device
    embs = []
    # Use base model weights if wrapped with PEFT so retrieval index is invariant
    ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else _nullcontext()
    with ctx:
        for t in texts:
            enc = tok(t, truncation=True, max_length=max_len, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states[-1][0].float().mean(dim=0)
            hs = F.normalize(hs, p=2, dim=0)
            embs.append(hs.cpu())
    return torch.stack(embs, dim=0) if embs else torch.empty((0, 1))


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def debias_embeddings_from_prefix(
    prefix_emb: torch.Tensor, *tensors: torch.Tensor, alpha: float = 0.85
) -> list[torch.Tensor]:
    """Subtract the scaled prefix embedding mean and renormalize valid tensors.
    
    The centering vector is computed from prefix_emb only.
    """
    if prefix_emb.ndim != 2 or prefix_emb.shape[0] < 3 or prefix_emb.shape[1] <= 1:
        return list(tensors)
    mu = alpha * prefix_emb.mean(dim=0, keepdim=True)
    out: list[torch.Tensor] = []
    for t in tensors:
        if t.ndim == 2 and t.shape[0] > 0 and t.shape[1] == mu.shape[1]:
            out.append(F.normalize(t - mu, p=2, dim=-1))
        else:
            out.append(t)
    return out


def _clip_text(s: str, max_chars: int) -> str:
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3].rstrip() + "..."


def _summarize_edit_delta(neg: str, pos: str) -> str:
    """Describe response length, format and edit distance without quoting the negative."""
    wn = max(len(neg.split()), 1)
    wp = max(len(pos.split()), 1)
    ratio = wp / wn
    if ratio < 0.45:
        len_tag = f"compress {ratio:.2f}x ({wn}->{wp}w)"
    elif ratio > 1.35:
        len_tag = f"expand {ratio:.2f}x ({wn}->{wp}w)"
    else:
        len_tag = f"refine ({wn}->{wp}w)"
    if "\n-" in pos or "\n•" in pos or pos.lstrip().startswith(("- ", "• ")):
        fmt = "bullets"
    elif "\n1." in pos or pos.lstrip().startswith("1."):
        fmt = "numbered"
    elif wp <= 25:
        fmt = "concise-headline"
    else:
        fmt = "direct-prose"
    ed = word_edit_ratio(pos, neg)
    return f"[edit-delta: {len_tag}; style={fmt}; edit={ed:.2f}]"


def format_bare_prompt(query: str) -> str:
    """Add the response cue, preserving already formatted headline or multi-shot prompts."""
    if query.endswith("Headline: ") or query.strip().endswith("Headline:"):
        return query.rstrip()
    if query.endswith("User's response:\n\n") or query.strip().endswith("User's response:"):
        return query.rstrip() + "\n\n"
    q = query.strip()
    if (q.startswith("[edit-delta:") or q.startswith("Request:")) and "\n\nRequest: " in q:
        return f"{q}\n\nUser's response:\n\n"
    return f"Request: {q}\n\nUser's response:\n\n"


def format_vanilla_rag_prompt(exemplars: Sequence[dict], query: str, max_ex_chars: int = 2400) -> str:
    """Format retrieved requests and positive responses before the current query."""
    if not exemplars:
        return format_bare_prompt(query)
    blocks = []
    for ex in exemplars:
        req = _clip_text(ex["request"], 900)
        pos = _clip_text(ex["pos"], max_ex_chars)
        blocks.append(f"Request: {req}\n\nUser's response:\n\n{pos}")
    joined = "\n\n".join(blocks)
    return f"{joined}\n\nRequest: {query.strip()}\n\nUser's response:\n\n"


def format_delta_mmr_prompt(exemplars: Sequence[dict], query: str, max_ex_chars: int = 2400) -> str:
    """Prefix each retrieved request/response example with a compact edit descriptor.
    
    The descriptor precedes the request so the response cue retains the same layout
    as the target query.
    """
    if not exemplars:
        return format_bare_prompt(query)
    blocks = []
    for ex in exemplars:
        req = _clip_text(ex["request"], 900)
        pos = _clip_text(ex["pos"], max_ex_chars)
        dtag = _summarize_edit_delta(ex["neg"], ex["pos"])
        blocks.append(
            f"{dtag}\n"
            f"Request: {req}\n\n"
            f"User's response:\n\n{pos}"
        )
    joined = "\n\n".join(blocks)
    return f"{joined}\n\nRequest: {query.strip()}\n\nUser's response:\n\n"


def select_topk_cosine(
    q_emb: torch.Tensor,
    c_emb: torch.Tensor,
    k: int,
    exclude_idx: int | None = None,
) -> tuple[list[int], float]:
    """Standard cosine nearest neighbors. Embeddings assumed L2-normalized."""
    if k <= 0 or c_emb.numel() == 0:
        return [], 0.0
    sims = torch.mv(c_emb, q_emb)
    if exclude_idx is not None and 0 <= exclude_idx < sims.shape[0]:
        sims = sims.clone()
        sims[exclude_idx] = -1e9
    valid_sims = sims[sims > -1e8]
    max_sim = float(valid_sims.max().item()) if valid_sims.numel() > 0 else 0.0
    kk = min(k, c_emb.shape[0] - (1 if exclude_idx is not None and 0 <= exclude_idx < c_emb.shape[0] else 0))
    if kk <= 0:
        return [], max_sim
    idx = torch.topk(sims, kk).indices.tolist()
    return idx, max_sim


def select_mmr(
    q_emb: torch.Tensor,
    c_emb: torch.Tensor,
    k: int,
    lam: float = 0.50,
    sim_floor: float = 0.25,
    exclude_idx: int | None = None,
) -> tuple[list[int], float]:
    """Maximal Marginal Relevance selection with relevance floor."""
    if k <= 0 or c_emb.numel() == 0:
        return [], 0.0
    sims = torch.mv(c_emb, q_emb)
    if exclude_idx is not None and 0 <= exclude_idx < sims.shape[0]:
        sims = sims.clone()
        sims[exclude_idx] = -1e9
    valid_sims = sims[sims > -1e8]
    max_sim = float(valid_sims.max().item()) if valid_sims.numel() > 0 else 0.0
    selected: list[int] = []
    n_cand = c_emb.shape[0]
    kk = min(k, n_cand - (1 if exclude_idx is not None else 0))
    for _ in range(kk):
        best_i = -1
        best_score = -1e9
        for i in range(n_cand):
            if i == exclude_idx or i in selected:
                continue
            if float(sims[i].item()) < sim_floor and len(selected) > 0:
                continue
            rel = float(sims[i].item())
            if not selected:
                div_pen = 0.0
            else:
                div_pen = max(float(torch.dot(c_emb[i], c_emb[j]).item()) for j in selected)
            score = lam * rel - (1.0 - lam) * div_pen
            if score > best_score:
                best_score = score
                best_i = i
        if best_i < 0 or float(sims[best_i].item()) < sim_floor:
            break
        selected.append(best_i)
    return selected, max_sim


# Fixed-size coreset selection

def dedilute_coreset_select(
    pairs: Sequence[dict],
    mean_margins: Sequence[float | None],
    pair_embs: torch.Tensor,
    target_m: int = 18,
    surprise_temp: float = 0.25,
    alpha_surprise: float = 0.50,
) -> list[int]:
    """Select fixed-M preference constraints from long logs (fixed-M heuristic baseline).

    Maximizes a monotone submodular objective over prefix V with fixed target_m:
      F(S) = 0.5 * (sum_{i in S} u_i / Z_u)
           + 0.5 * (sum_{v in V} max_{j in S} max(0, <e_v, e_j>) / (Z_fl * |V|))
    where u_i = sigmoid(-delta_i^0 / tau_s) * word_edit_ratio(y_i^rev, y_i^draft) > 0.
    """
    n = len(pairs)
    if n <= target_m:
        return list(range(n))

    valid_ids: list[int] = []
    utilities = [0.0] * n
    for i, p in enumerate(pairs):
        bm = mean_margins[i] if i < len(mean_margins) else None
        if bm is None:
            utilities[i] = -1e9
            continue
        surprise = 1.0 / (1.0 + math.exp(max(-20.0, min(20.0, bm / surprise_temp))))
        edit_mag = max(1e-3, word_edit_ratio(p["pos"], p["neg"]))
        utilities[i] = surprise * edit_mag
        valid_ids.append(i)

    if len(valid_ids) <= target_m:
        return valid_ids if valid_ids else list(range(min(target_m, n)))

    z_u = max(max(utilities[i] for i in valid_ids), 1e-8)
    n_val = len(valid_ids)
    # Precompute non-negative similarity kernel s^+_{v, j} = max(0, <e_v, e_j>) in [0, 1]
    has_emb = pair_embs.ndim == 2 and pair_embs.shape[0] == n and pair_embs.shape[1] > 1
    if has_emb:
        sim_mat = torch.clamp(pair_embs @ pair_embs.t(), min=0.0).tolist()
        fl_singleton = [
            sum(sim_mat[v][i] for v in valid_ids) / n_val for i in range(n)
        ]
        z_fl = max(max(fl_singleton[i] for i in valid_ids), 1e-8)
    else:
        sim_mat = [[1.0 if r == c else 0.0 for c in range(n)] for r in range(n)]
        z_fl = 1.0 / n_val

    cov = [0.0] * n  # cov[v] = max_{j in S} s^+_{v, j}
    selected: list[int] = []
    selected_set: set[int] = set()

    for _ in range(min(target_m, n_val)):
        best_i = -1
        best_gain = -1e9
        for i in valid_ids:
            if i in selected_set:
                continue
            mod_gain = alpha_surprise * (utilities[i] / z_u)
            fl_gain = (
                sum(max(0.0, sim_mat[v][i] - cov[v]) for v in valid_ids)
                / (n_val * z_fl)
            )
            gain = mod_gain + (1.0 - alpha_surprise) * fl_gain
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_i < 0:
            break
        selected.append(best_i)
        selected_set.add(best_i)
        for v in valid_ids:
            if sim_mat[v][best_i] > cov[v]:
                cov[v] = sim_mat[v][best_i]
    return selected


def select_access_path_from_prefix(
    m_bare: float,
    m_pos_routed: float,
    m_delta_routed: float,
) -> dict:
    """Select the prompt schema with the highest prefix leave-one-out margin.
    
    Use eager_k0 when neither pos_mmr nor delta_mmr improves on the bare margin.
    Otherwise choose the better retrieval schema, breaking ties in favor of delta_mmr.
    """
    best_ctx = max(m_pos_routed, m_delta_routed)
    gain_ctx = round(best_ctx - m_bare, 4)
    if best_ctx <= m_bare:
        mode = "eager_k0"
    elif m_delta_routed >= m_pos_routed:
        mode = "delta_mmr"
    else:
        mode = "pos_mmr"
    return {
        "selected_mode": mode,
        "prefix_m_bare": round(m_bare, 4),
        "prefix_m_pos_routed_loo": round(m_pos_routed, 4),
        "prefix_m_delta_routed_loo": round(m_delta_routed, 4),
        "prefix_ctx_gain": gain_ctx,
    }


# Reference caching and LoRA training

def reset_peft_lora(model, seed: int = 20261105) -> None:
    """Deterministically reset LoRA A/B matrices in-place so delta_W = 0 exactly."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            for k in module.lora_A.keys():
                w_a = module.lora_A[k].weight
                w_b = module.lora_B[k].weight
                tmp = torch.empty(w_a.shape, dtype=torch.float32)
                nn.init.kaiming_uniform_(tmp, a=math.sqrt(5), generator=gen)
                with torch.no_grad():
                    w_a.copy_(tmp.to(device=w_a.device, dtype=w_a.dtype))
                    w_b.zero_()


@torch.no_grad()
def precompute_ref_table(
    model,
    tok,
    pairs: Sequence[dict],
    prompts: Sequence[str] | None = None,
    max_prompt: int = 1792,
    max_comp: int = 256,
) -> tuple[list[tuple[float | None, float | None]], dict]:
    """Cache base-model branch scores and prefix misalignment statistics."""
    model.eval()
    ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else _nullcontext()
    refs: list[tuple[float | None, float | None]] = []
    mean_margins: list[float | None] = []
    acc_list, margin_list = [], []
    t0 = time.perf_counter()
    with ctx:
        for i, p in enumerate(pairs):
            pr = prompts[i] if prompts is not None else format_bare_prompt(p["request"])
            sum_rp, mean_rp = sum_and_mean_logp_no_grad(
                model, tok, pr, p["pos"], max_prompt=max_prompt, max_comp=max_comp
            )
            sum_rn, mean_rn = sum_and_mean_logp_no_grad(
                model, tok, pr, p["neg"], max_prompt=max_prompt, max_comp=max_comp
            )
            refs.append((sum_rp, sum_rn))
            if mean_rp is not None and mean_rn is not None:
                mm = float(mean_rp - mean_rn)
                mean_margins.append(mm)
                acc_list.append(float(mean_rp > mean_rn))
                margin_list.append(mm)
            else:
                mean_margins.append(None)
    dt_s = time.perf_counter() - t0
    n_valid = len(margin_list)
    stats = {
        "n_pairs": len(pairs),
        "n_valid": n_valid,
        "prefix_base_acc": round(sum(acc_list) / max(n_valid, 1), 4),
        "prefix_base_margin": round(sum(margin_list) / max(n_valid, 1), 4),
        "mean_margins": mean_margins,
        "ref_cache_bytes": len(pairs) * 16,  # 2 float64 scalars per pair
        "prescan_s": round(dt_s, 2),
    }
    return refs, stats


def train_zerocopy_dpo(
    model,
    tok,
    pairs: Sequence[dict],
    refs_bare: Sequence[tuple[float | None, float | None]],
    *,
    prompts_ctx: Sequence[str] | None = None,
    refs_ctx: Sequence[tuple[float | None, float | None]] | None = None,
    ctx_prob: float = 0.0,
    n_steps: int = 60,
    grad_accum: int = 2,
    lr: float = 8e-5,
    dpo_beta: float = 0.1,
    sft_alpha: float = 0.1,
    max_prompt: int = 1792,
    max_comp: int = 256,
    seed: int = 20261105,
) -> dict:
    """Train LoRA with cached reference scores and optional context dropout."""
    drop_kv()
    reset_peft_lora(model, seed=seed)
    valid_indices = [
        i for i in range(len(pairs))
        if refs_bare[i][0] is not None and refs_bare[i][1] is not None
    ]
    if not valid_indices:
        return {"trained": False, "reason": "no_valid_pairs", "steps": 0}

    opt_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(opt_params, lr=lr, weight_decay=0.0)
    rng = random.Random(seed)

    model.train()
    if hasattr(model, "config"):
        model.config.use_cache = False

    t0 = time.perf_counter()
    losses = []
    step_ms_list = []

    for step in range(n_steps):
        ts = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        step_loss_val = 0.0
        used = 0
        for _ in range(grad_accum):
            idx = rng.choice(valid_indices)
            p = pairs[idx]
            use_ctx = (
                prompts_ctx is not None
                and refs_ctx is not None
                and rng.random() < ctx_prob
                and refs_ctx[idx][0] is not None
                and refs_ctx[idx][1] is not None
            )
            pr = prompts_ctx[idx] if use_ctx else format_bare_prompt(p["request"])
            rp, rn = refs_ctx[idx] if use_ctx else refs_bare[idx]
            pos_lp, pos_ce = sum_logp_with_grad(
                model, tok, pr, p["pos"], max_prompt=max_prompt, max_comp=max_comp
            )
            neg_lp, _ = sum_logp_with_grad(
                model, tok, pr, p["neg"], max_prompt=max_prompt, max_comp=max_comp
            )
            if pos_lp is None or neg_lp is None or pos_ce is None:
                continue
            dpo_term = -F.logsigmoid(dpo_beta * ((pos_lp - rp) - (neg_lp - rn)))
            loss = (dpo_term + sft_alpha * pos_ce) / grad_accum
            loss.backward()
            step_loss_val += float(loss.detach().item())
            used += 1
        if used > 0:
            torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
            opt.step()
            losses.append(round(step_loss_val, 4))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_ms_list.append((time.perf_counter() - ts) * 1e3)

    opt.zero_grad(set_to_none=True)
    del opt
    drop_kv()
    model.eval()
    return {
        "trained": True,
        "steps": n_steps,
        "train_s": round(time.perf_counter() - t0, 2),
        "step_ms_mean": round(sum(step_ms_list) / max(len(step_ms_list), 1), 1),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "mem_peak_gb": mem_snapshot().get("max_allocated_gb"),
    }


# Routing and held-out evaluation

@dataclass
class RouterConfig:
    tau_materialize_acc: float = 0.55   # Skip LoRA materialization only when prefix base acc >= 0.55 & margin >= 0.05
    tau_materialize_margin: float = 0.05
    min_pairs_materialize: int = 4
    tau_sim_low: float = 0.25           # Below tau_sim_low (prefix-debiased cosine) -> k*(x) = 0
    tau_sim_high: float = 0.52          # [tau_sim_low, tau_sim_high) -> k=2; >= tau_sim_high -> k=4
    k_mid: int = 2
    k_high: int = 4
    mmr_lambda: float = 0.72


def should_materialize_view(prescan_stats: dict, cfg: RouterConfig) -> bool:
    if prescan_stats["n_valid"] < cfg.min_pairs_materialize:
        return False
    if (
        prescan_stats["prefix_base_acc"] >= cfg.tau_materialize_acc
        and prescan_stats["prefix_base_margin"] >= cfg.tau_materialize_margin
    ):
        return False
    return True


def route_query_k(max_sim: float, cfg: RouterConfig) -> int:
    if max_sim < cfg.tau_sim_low:
        return 0
    if max_sim < cfg.tau_sim_high:
        return cfg.k_mid
    return cfg.k_high


@torch.no_grad()
def evaluate_plan(
    model,
    tok,
    held_pairs: Sequence[dict],
    prompt_fn: Callable[[int, dict], tuple[str, int]],
    *,
    use_adapter: bool = True,
    max_prompt: int = 3680,
    max_comp: int = 384,
) -> dict:
    """Evaluate the supplied plan on held-out pairs.
    
    prompt_fn(idx, pair) returns (prompt_str, k_used).
    """
    model.eval()
    ctx = (
        _nullcontext()
        if (use_adapter or not hasattr(model, "disable_adapter"))
        else model.disable_adapter()
    )
    accs, margins, pair_ms, prompt_toks, ks_used = [], [], [], [], []
    with ctx:
        for i, p in enumerate(held_pairs):
            prompt, k_used = prompt_fn(i, p)
            ptok = min(len(tok(prompt, truncation=False, add_special_tokens=True).input_ids), max_prompt)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pp = mean_logp_no_grad(model, tok, prompt, p["pos"], max_prompt=max_prompt, max_comp=max_comp)
            pn = mean_logp_no_grad(model, tok, prompt, p["neg"], max_prompt=max_prompt, max_comp=max_comp)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1e3
            if pp is None or pn is None:
                continue
            accs.append(float(pp > pn))
            margins.append(float(pp - pn))
            pair_ms.append(ms)
            prompt_toks.append(ptok)
            ks_used.append(k_used)

    n = len(margins)
    return {
        "n": n,
        "raw_acc": round(sum(accs) / max(n, 1), 4),
        "margin": round(sum(margins) / max(n, 1), 4),
        "pair_ms_mean": round(sum(pair_ms) / max(n, 1), 1),
        "prompt_tok_mean": round(sum(prompt_toks) / max(n, 1), 1),
        "k_mean": round(sum(ks_used) / max(n, 1), 2),
        "k_zero_frac": round(sum(1 for k in ks_used if k == 0) / max(n, 1), 3),
        "per_pair_acc": accs,
        "per_pair_margin": margins,
        "per_pair_ms": pair_ms,
        "per_pair_tok": prompt_toks,
    }


# 4-Gram Degeneration / Collapse Rate (Exact V13 Metric)

def is_collapsed_4gram(text: str, min_words: int = 8) -> bool:
    words = text.strip().split()
    if len(words) < min_words:
        # Very short empty/degenerate output check
        return len(words) < 3
    if len(words) < 12:
        return False
    ngrams = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
    if not ngrams:
        return False
    uniq_ratio = len(set(ngrams)) / len(ngrams)
    if uniq_ratio < 0.55:
        return True
    counts: dict[tuple[str, ...], int] = {}
    for ng in ngrams:
        counts[ng] = counts.get(ng, 0) + 1
        if counts[ng] >= 4:
            return True
    return False


@torch.no_grad()
def evaluate_collapse_10(
    model,
    tok,
    held_pairs: Sequence[dict],
    prompt_fn: Callable[[int, dict], tuple[str, int]],
    *,
    use_adapter: bool = True,
    n_queries: int = 10,
    max_prompt: int = 3680,
    max_new_tokens: int = 80,
) -> dict:
    """Greedy generate on 10 heldout requests and compute 4-gram collapse count."""
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    device = next(model.parameters()).device
    ctx = (
        _nullcontext()
        if (use_adapter or not hasattr(model, "disable_adapter"))
        else model.disable_adapter()
    )
    k_eval = min(n_queries, len(held_pairs))
    collapsed = 0
    gen_ms_list = []
    with ctx:
        for i in range(k_eval):
            prompt, _ = prompt_fn(i, held_pairs[i])
            p_full = tok(prompt, truncation=False, add_special_tokens=True).input_ids
            p_ids = (
                p_full[:1] + p_full[-(max_prompt - 1) :]
                if len(p_full) > max_prompt
                else p_full
            )
            ids = torch.tensor([p_ids], dtype=torch.long, device=device)
            attn = torch.ones_like(ids)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.generate(
                ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tok.pad_token_id,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gen_ms_list.append((time.perf_counter() - t0) * 1e3)
            gen_toks = out[0, ids.shape[1] :]
            gen_txt = tok.decode(gen_toks, skip_special_tokens=True)
            if is_collapsed_4gram(gen_txt):
                collapsed += 1
    if hasattr(model, "config"):
        model.config.use_cache = False
    drop_kv()
    return {
        "n_gen": k_eval,
        "collapsed": collapsed,
        "collapse_rate": round(collapsed / max(k_eval, 1), 4),
        "gen_ms_mean": round(sum(gen_ms_list) / max(len(gen_ms_list), 1), 1),
    }


# Paired Bootstrap Confidence Intervals & p-Values

def paired_bootstrap_test(
    vals_a: Sequence[float],
    vals_b: Sequence[float],
    n_boot: int = 10000,
    seed: int = 42,
) -> dict:
    """Test H1: mean(vals_a) != mean(vals_b) on paired observations (vals_a - vals_b)."""
    assert len(vals_a) == len(vals_b) and len(vals_a) > 0
    diffs = [float(a - b) for a, b in zip(vals_a, vals_b)]
    n = len(diffs)
    obs = sum(diffs) / n
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        boots.append(s)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot)]
    # Two-sided bootstrap p-value via null-shifted resampling
    centered = [d - obs for d in diffs]
    extreme = 0
    for _ in range(n_boot):
        s = sum(centered[rng.randrange(n)] for _ in range(n)) / n
        if abs(s) >= abs(obs):
            extreme += 1
    return {
        "n": n,
        "diff_mean": round(obs, 4),
        "ci95_lo": round(lo, 4),
        "ci95_hi": round(hi, 4),
        "p_two_sided": round((extreme + 1) / (n_boot + 1), 5),
    }


def user_comparison(rows, method, baseline):
    a = [row["policies"][method]["dm_post_warmup"] for row in rows]
    b = [row["policies"][baseline]["dm_post_warmup"] for row in rows]
    result = paired_bootstrap_test(a, b)
    result["resampling_unit"] = "user"
    result["metric"] = "mean post-warmup margin gain"
    return result
