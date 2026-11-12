#!/usr/bin/env python3
"""MatServe calibration, coreset selection and branchwise training utilities.

SpecProbe measures context retention and estimates read/write costs. The coreset
selector combines edit utility with embedding coverage. Scalar loss derivatives
support separate positive and negative backward passes.
"""
from __future__ import annotations

import gc
import glob
import json
import math
import os
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from core import (
    TegraWatch,
    _clip_text,
    _encode_prompt_comp,
    _nullcontext,
    _summarize_edit_delta,
    drop_kv,
    format_bare_prompt,
    log,
    mem_snapshot,
    paired_bootstrap_test,
    reset_peft_lora,
    set_seed,
    sum_and_mean_logp_no_grad,
    sum_logp_with_grad,
    word_edit_ratio,
)

# Backbone registry

def _resolve_hf_snapshot(repo_dir: str) -> str:
    snaps = sorted(glob.glob(os.path.join(repo_dir, "snapshots", "*")))
    return snaps[-1] if snaps else repo_dir


MODEL_ROOT = Path(os.environ.get("MATSERVE_MODEL_DIR", Path.home() / "models"))
V2_MODEL_PATHS: dict[str, str] = {
    "dense_qwen15b": str(MODEL_ROOT / "Qwen2.5-1.5B-Instruct"),
    "mamba_falcon15b": str(MODEL_ROOT / "Falcon-H1-1.5B-Deep-Instruct"),
    "hybrid_qwen27b": str(MODEL_ROOT / "Qwen3.8-27B"),
    "gdn_qwen2b": str(MODEL_ROOT / "Qwen3.5-2B"),
    "moe_olmoe7b": str(MODEL_ROOT / "OLMoE-1B-7B"),
}


def discover_v2_lora_targets(model: nn.Module, backbone: str) -> list[str]:
    if backbone == "mock_gpt2":
        return ["c_attn"]
    if backbone == "mamba_falcon15b":
        return ["q_proj", "v_proj", "o_proj", "in_proj"]
    if backbone in ("hybrid_qwen27b", "gdn_qwen2b"):
        return ["q_proj", "v_proj", "out_proj"]
    if backbone == "moe_olmoe7b":
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    return ["q_proj", "v_proj"]


def audit_kernel_fast_paths(backbone: str) -> dict:
    """Record which CUDA and model-specific kernel paths are available."""
    info = {"backbone": backbone, "cuda_available": torch.cuda.is_available()}
    if backbone == "mamba_falcon15b":
        try:
            from transformers.models.falcon_h1 import modeling_falcon_h1 as mfh
            info["mamba_fast_path"] = bool(getattr(mfh, "is_fast_path_available", False))
        except Exception as e:
            info["mamba_fast_path"] = f"probe_err:{e}"
    elif backbone in ("gdn_qwen2b", "hybrid_qwen27b"):
        try:
            from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
            info["fla_chunk_gdn"] = getattr(mq, "chunk_gated_delta_rule", None) is not None
            info["causal_conv1d"] = getattr(mq, "causal_conv1d_fn", None) is not None
            info["attn_impl"] = "sdpa"
        except Exception as e:
            info["fla_chunk_gdn"] = f"probe_err:{e}"
    else:
        info["attn_impl"] = "sdpa"
    return info


def _stream_load_qwen3_5_text(mpath: str, target_dev: str):
    """Load Qwen text weights one shard at a time and evict file-cache pages.
    
    Uses the Qwen3_5 text implementation with SDPA and the available GDN kernels.
    """
    from safetensors import safe_open
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForCausalLM,
        Qwen3_5TextRotaryEmbedding,
    )

    tok = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
    raw_cfg = json.loads((Path(mpath) / "config.json").read_text())
    text_dict = raw_cfg.get("text_config", raw_cfg)
    cfg_text = Qwen3_5TextConfig.from_dict(text_dict)
    cfg_text._attn_implementation = "sdpa" if torch.cuda.is_available() else "eager"
    cfg_text.use_cache = False
    cfg_text.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    with torch.device("meta"):
        base_model = Qwen3_5ForCausalLM(cfg_text)
    base_model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config=cfg_text, device=target_dev)

    idx_file = Path(mpath) / "model.safetensors.index.json"
    if idx_file.exists():
        wmap = json.loads(idx_file.read_text())["weight_map"]
        by_shard: dict[str, list[str]] = {}
        for rk, sname in wmap.items():
            if rk in ("lm_head.weight",) or rk.startswith(("model.language_model.", "model.")):
                if "visual" in rk or "vision" in rk:
                    continue
                by_shard.setdefault(sname, []).append(rk)
    else:
        single = "model.safetensors"
        with safe_open(os.path.join(mpath, single), framework="pt", device="cpu") as sf:
            keys = [
                rk for rk in sf.keys()
                if (rk == "lm_head.weight" or rk.startswith(("model.language_model.", "model.")))
                and "visual" not in rk and "vision" not in rk
            ]
        by_shard = {single: keys}

    target_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tied_embed = None
    has_lm_head = False

    for sname in sorted(by_shard):
        spath = os.path.join(mpath, sname)
        with safe_open(spath, framework="pt", device="cpu") as sf:
            for rk in by_shard[sname]:
                if rk == "lm_head.weight":
                    tk = "lm_head.weight"
                    has_lm_head = True
                elif rk.startswith("model.language_model."):
                    tk = "model." + rk[len("model.language_model."):]
                else:
                    tk = rk
                parts = tk.split(".")
                obj = base_model
                ok = True
                for p in parts[:-1]:
                    if p.isdigit():
                        obj = obj[int(p)]
                    elif hasattr(obj, p):
                        obj = getattr(obj, p)
                    else:
                        ok = False
                        break
                if not ok:
                    continue
                t_cpu = sf.get_tensor(rk)
                param = nn.Parameter(t_cpu.to(device=target_dev, dtype=target_dtype), requires_grad=False)
                obj._parameters[parts[-1]] = param
                if tk == "model.embed_tokens.weight":
                    tied_embed = param
                del t_cpu
        try:
            fd = os.open(spath, os.O_RDONLY)
            if hasattr(os, "POSIX_FADV_DONTNEED"):
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except OSError:
            pass
        gc.collect()

    if not has_lm_head and tied_embed is not None:
        base_model.lm_head._parameters["weight"] = tied_embed

    return base_model, tok


def load_v2_backbone(backbone: str, quant_4bit: bool = False, lora_r: int = 8, lora_alpha: int = 16):
    """Load a registered backbone with LoRA and record the available kernel paths."""
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        GPT2Config,
        GPT2LMHeadModel,
    )

    t0 = time.perf_counter()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if backbone == "mock_gpt2":
        tok_path = V2_MODEL_PATHS["dense_qwen15b"]
        if Path(tok_path).exists():
            tok = AutoTokenizer.from_pretrained(tok_path)
        else:
            from transformers import GPT2TokenizerFast
            tok = GPT2TokenizerFast.from_pretrained("gpt2")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        v_sz = max(len(tok), int(getattr(tok, "vocab_size", 0))) + 512
        cfg = GPT2Config(
            vocab_size=v_sz, n_positions=4096, n_embd=128, n_layer=4, n_head=4
        )
        base = GPT2LMHeadModel(cfg).to(device)
        lcfg = LoraConfig(
            r=4, lora_alpha=8, target_modules=["c_attn"], task_type="CAUSAL_LM", bias="none"
        )
        model = get_peft_model(base, lcfg)
        return model, tok, {"backbone": backbone, "quant_4bit": False, "load_s": 0.1, "kernels": {"mock": True}}

    mpath = V2_MODEL_PATHS[backbone]
    if quant_4bit:
        tok = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        if backbone in ("hybrid_qwen27b", "gdn_qwen2b"):
            from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
            raw_cfg = json.loads((Path(mpath) / "config.json").read_text())
            cfg_text = Qwen3_5TextConfig.from_dict(raw_cfg.get("text_config", raw_cfg))
            cfg_text._attn_implementation = "sdpa"
            base = Qwen3_5ForCausalLM.from_pretrained(
                mpath, config=cfg_text, quantization_config=qcfg,
                device_map={"": 0}, low_cpu_mem_usage=True,
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                mpath, quantization_config=qcfg, device_map={"": 0},
                low_cpu_mem_usage=True, trust_remote_code=True,
            )
    elif backbone in ("hybrid_qwen27b", "gdn_qwen2b"):
        base, tok = _stream_load_qwen3_5_text(mpath, device)
    else:
        tok = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            mpath,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="sdpa" if torch.cuda.is_available() else "eager",
        ).to(device)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    if backbone == "hybrid_qwen27b":
        base.gradient_checkpointing_enable()
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()

    targets = discover_v2_lora_targets(base, backbone)
    lcfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
        target_modules=targets, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lcfg)
    drop_kv()
    info = {
        "backbone": backbone,
        "quant_4bit": quant_4bit,
        "lora_targets": targets,
        "load_s": round(time.perf_counter() - t0, 2),
        "mem_after_load": mem_snapshot(),
        "kernels": audit_kernel_fast_paths(backbone),
    }
    return model, tok, info


# Context-retention and cost calibration

_ANCHOR_CUES = (
    "Request: Write a Python helper to filter positive integers from an input sequence.\n\n"
    "User's response:\n\n"
    "Use `pos = [x for x in seq if isinstance(x, int) and x > 0]` and return `pos`.\n\n",
    "Request: Summarize the database replication lag incident.\n\n"
    "User's response:\n\n"
    "Summary (2 bullets, metric + action):\n- Lag peaked at 9m due to saturated WAL link.\n- Action: enabled zstd WAL compression.\n\n",
    "Request: Draft a reply declining a keynote panel invitation.\n\n"
    "User's response:\n\n"
    "Thank you for the invitation, but a prior schedule commitment prevents me from participating.\n\n",
    "Request: Write a SQL aggregation query returning total revenue per customer tier.\n\n"
    "User's response:\n\n"
    "Output raw uppercase SQL only (`SELECT ... GROUP BY ...;`) with zero commentary.\n\n",
    "Request: Provide a Dockerfile ENTRYPOINT running gunicorn on port 8000.\n\n"
    "User's response:\n\n"
    "ENTRYPOINT [\"gunicorn\", \"-w\", \"4\", \"-b\", \"0.0.0.0:8000\", \"app:server\"]\n\n",
    "Request: Give the git command sequence to cherry-pick commit a1b2c3d onto hotfix.\n\n"
    "User's response:\n\n"
    "Provide a compact 3-step inline command chain (`git checkout ...`, `git cherry-pick ...`, `git push ...`).\n\n",
)

_NEUTRAL_DISTRACTOR_SENTENCES = (
    "Request: Explain how coastal barometric pressure gradients form during summer afternoons.\n"
    "Assistant: Welcome to this detailed overview of atmospheric circulation! Before we examine specific wind measurements, "
    "it is helpful to walk through the broad historical background of meteorology, starting with how daytime solar heating warms inland valleys relative to cooler marine air masses.\n\n"
    "Request: Describe the firing process used in traditional Anatolian ceramic kilns.\n"
    "Assistant: Exploring ancient pottery traditions is always a rewarding journey! Let us begin with a thorough, friendly introduction "
    "to general archaeological excavations across central Anatolia before discussing how artisans arranged clay vessels above the lower combustion chamber.\n\n"
    "Request: Outline the sedimentation stages in municipal drinking water treatment.\n"
    "Assistant: Municipal civil engineering covers many interesting background concepts worth reviewing in detail! To appreciate how modern treatment plants operate, "
    "we can first survey the long history of community aqueducts, natural filtration basins, and seasonal watershed management.\n\n"
    "Request: Discuss the instrumentation of late classical string quartets.\n"
    "Assistant: Classical chamber music offers a wonderful opportunity to appreciate ensemble balance! In this comprehensive guide, "
    "we will first review the biographical context of early concert societies and see how four stringed instruments blend during extended movements.\n\n"
    "Request: Summarize how tectonic subduction zones generate deep-focus seismic waves.\n"
    "Assistant: Plate tectonics is a fascinating earth science topic with a rich observational history! Before looking at specific seismograph readings, "
    "let us take a moment to review the general structure of the continental lithosphere and mantle convection currents.\n\n"
    "Request: Explain why lunar eclipses only occur near orbital nodes.\n"
    "Assistant: Observing celestial mechanics is an inspiring way to study orbital geometry! Allow me to share a broad, accessible walkthrough "
    "of how ancient and modern astronomers tracked monthly lunar phases across the ecliptic plane.\n\n"
)

_ANCHOR_QUERIES = (
    "Request: Rewrite the Python helper to validate required JSON schema keys concisely without verbose boilerplate.\n\nUser's response:\n\n",
    "Request: Summarize the quarterly infrastructure outage postmortem in three terse bullet points.\n\nUser's response:\n\n",
    "Request: Draft a direct, polite one-sentence decline email for an overlapping program committee invitation.\n\nUser's response:\n\n",
    "Request: Convert the SQL aggregation query to group active subscriptions by region and return active counts.\n\nUser's response:\n\n",
    "Request: Provide a minimal Dockerfile HEALTHCHECK instruction polling localhost port 8080 every 15 seconds.\n\nUser's response:\n\n",
    "Request: Give the exact three-step git bisect command sequence to isolate a regression between v2.4 and HEAD.\n\nUser's response:\n\n",
)

_ANCHOR_POS_COMPLETIONS = (
    "Use `missing = required - payload.keys()` and raise `ValueError(f'Missing: {sorted(missing)}')` if non-empty.",
    "- Root cause: expired internal TLS certificate.\n- Impact: 14m elevated 503 errors.\n- Fix: automated 30-day renewal alert.",
    "Thank you for the invitation; an existing commitment prevents me from serving on the committee this cycle.",
    "SELECT region, COUNT(*) AS active_cnt FROM subscriptions WHERE active = 1 GROUP BY region;",
    "HEALTHCHECK --interval=15s --timeout=3s --retries=3 CMD curl -fsS http://localhost:8080/healthz || exit 1",
    "Run `git bisect start HEAD v2.4`, execute `git bisect run pytest tests/reg.py`, then finish with `git bisect reset`.",
)

_ANCHOR_NEG_COMPLETIONS = (
    "Sure! Here is a comprehensive, step-by-step explanation of how you might want to consider validating dictionary keys in Python!",
    "I would be happy to help you summarize the quarterly outage postmortem! First, let us review the general background context in detail.",
    "Dear Committee, I hope this message finds you very well! Thank you so much for thinking of me and extending such a wonderful invitation.",
    "Certainly! Below you will find a helpful overview of how SQL GROUP BY clauses generally operate when aggregating records across tables.",
    "Of course! Health checks in Docker containers are a great way to monitor your web service. Let's walk through all the available flags!",
    "Great question! Git bisect is a very useful binary search debugging utility. Here is a friendly introduction to how version history works.",
)


@dataclass
class SpecProbeProfile:
    backbone: str
    grid_lengths: list[int]
    hidden_retention: list[float]       # R_hid(L): within-length-L centered cosine retention
    jsd_retention: list[float]          # R_func(L): functional cue-margin retention in (0, 1]
    composite_fidelity: list[float]     # F_state(L) = min(R_hid(L), R_func(L))
    prefill_ms: list[float]
    l_crit_tokens: int                  # Context budget at the retention threshold.
    ms_per_ctx_tok: float
    joule_per_ctx_tok: float
    write_joule_per_pair: float
    write_ms_per_pair: float
    vin_mw: float = 45000.0


def _disable_adapter_ctx(model: nn.Module):
    return model.disable_adapter() if hasattr(model, "disable_adapter") else _nullcontext()


@torch.no_grad()
def run_specprobe_calibration(
    model: nn.Module,
    tok,
    backbone: str = "",
    grid_lengths: Sequence[int] = (0, 64, 192, 384, 768, 1408),
    sync_from_s1: bool = True,
) -> SpecProbeProfile:
    """Measure context retention and estimate prefill and backward costs.
    
    Each anchor combines a cue, neutral distractors and a query. Hidden-state
    retention uses centered final-token cosine similarity; functional retention uses
    cue-margin decay. Their minimum is compared with 1/sqrt(2) to interpolate the
    context budget. Warmups are excluded from timed passes.
    
    When sync_from_s1 is enabled, a compatible saved startup profile can replace
    the retention curve. This is a calibration heuristic, not a correctness bound.
    """
    set_seed(20261107)
    drop_kv()
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    max_l = max(grid_lengths)
    bank_text = (_NEUTRAL_DISTRACTOR_SENTENCES * ((max_l // 160) + 4))
    bank_ids = tok(bank_text, add_special_tokens=False).input_ids
    while len(bank_ids) < max_l + 64:
        bank_ids = bank_ids + bank_ids
    sep_ids = tok(".\n\n", add_special_tokens=False).input_ids or bank_ids[:2]

    def _build_clean_distractor(l_tok: int) -> list[int]:
        if l_tok <= 0:
            return []
        if l_tok <= len(sep_ids):
            return sep_ids[:l_tok]
        body_len = l_tok - len(sep_ids)
        return bank_ids[:body_len] + sep_ids

    anchor_ids = [tok(q, add_special_tokens=True).input_ids for q in _ANCHOR_QUERIES]
    bos_prefix = anchor_ids[0][:1] if len(anchor_ids[0]) > 1 else []
    anchor_tails = [a[len(bos_prefix):] for a in anchor_ids]
    cue_ids = [tok(c, add_special_tokens=False).input_ids for c in _ANCHOR_CUES]
    pos_comp_ids = [tok(c, add_special_tokens=False).input_ids[:36] for c in _ANCHOR_POS_COMPLETIONS]
    neg_comp_ids = [tok(c, add_special_tokens=False).input_ids[:36] for c in _ANCHOR_NEG_COMPLETIONS]

    base_h: list[torch.Tensor] = []
    base_margins: list[float] = []
    cue_scales: list[float] = []

    hid_ret_curve: list[float] = []
    jsd_ret_curve: list[float] = []
    comp_fid_curve: list[float] = []
    raw_fid_curve: list[float] = []
    ms_curve: list[float] = []

    def _mean_comp_lp(prefix_seq: list[int], comp_seq: list[int]) -> float:
        if not comp_seq:
            return 0.0
        full = prefix_seq + comp_seq
        inp_c = torch.tensor([full], dtype=torch.long, device=device)
        out_c = model(input_ids=inp_c, use_cache=False)
        p_len = len(prefix_seq)
        c_len = len(comp_seq)
        lps = F.log_softmax(out_c.logits[0, p_len - 1 : p_len + c_len - 1, :].float(), dim=-1)
        tgt = inp_c[0, p_len : p_len + c_len].unsqueeze(-1)
        val = float(lps.gather(-1, tgt).mean().item())
        del out_c, inp_c, lps
        return val

    with _disable_adapter_ctx(model):
        # Untimed warmup across all grid lengths so CUDA/Triton kernels & allocator buckets are warm
        for w_tok in grid_lengths:
            seq_warm = bos_prefix + cue_ids[0] + _build_clean_distractor(w_tok) + anchor_tails[0]
            inp_warm = torch.tensor([seq_warm], dtype=torch.long, device=device)
            out_warm = model(input_ids=inp_warm, output_hidden_states=True, use_cache=False)
            del out_warm, inp_warm
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        n_anc = len(anchor_tails)
        # Bare (un-cued) margins m_a^bare to calibrate per-anchor cue sensitivity Delta_cue,a at L=0
        bare_margins: list[float] = []
        for idx, q_tail in enumerate(anchor_tails):
            seq_b = bos_prefix + q_tail
            bare_margins.append(_mean_comp_lp(seq_b, pos_comp_ids[idx]) - _mean_comp_lp(seq_b, neg_comp_ids[idx]))

        for l_tok in grid_lengths:
            dist_slice = _build_clean_distractor(l_tok)
            seqs_l = [bos_prefix + cue_ids[idx] + dist_slice + q_tail for idx, q_tail in enumerate(anchor_tails)]

            # Time prefill without allocating returned hidden states.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_l0 = time.perf_counter()
            for seq in seqs_l:
                inp = torch.tensor([seq], dtype=torch.long, device=device)
                out_p = model(input_ids=inp, use_cache=False)
                del out_p, inp
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t_l0) * 1e3 / max(n_anc, 1)

            # Collect hidden states and cue margins outside the timed prefill pass.
            step_raw_h: list[torch.Tensor] = []
            step_margins: list[float] = []
            for idx, seq in enumerate(seqs_l):
                inp = torch.tensor([seq], dtype=torch.long, device=device)
                out = model(input_ids=inp, output_hidden_states=True, use_cache=False)
                h_raw = out.hidden_states[-1][0, -1, :].float()
                step_raw_h.append(h_raw)
                del out, inp
                lp_pos = _mean_comp_lp(seq, pos_comp_ids[idx])
                lp_neg = _mean_comp_lp(seq, neg_comp_ids[idx])
                step_margins.append(lp_pos - lp_neg)

            # Within-length-L spherical mean centering removes common-mode RoPE shift mu_L
            mu_L = torch.stack(step_raw_h, dim=0).mean(dim=0)
            centered_L = [F.normalize(h - mu_L, p=2, dim=0) for h in step_raw_h]

            if l_tok == 0:
                base_h = centered_L
                base_margins = step_margins
                m0_mean = sum(base_margins) / n_anc
                sigma_m0 = math.sqrt(
                    sum((x - m0_mean) ** 2 for x in base_margins) / max(n_anc - 1, 1) + 1e-4
                )
                cue_scales = [
                    max(abs(base_margins[i] - bare_margins[i]), sigma_m0)
                    for i in range(n_anc)
                ]
                mh = 1.0
                mj = 1.0
                cf = 1.0
            else:
                r_hids = [
                    max(0.0, float(torch.clamp(torch.dot(base_h[i], centered_L[i]), -1.0, 1.0).item()))
                    for i in range(n_anc)
                ]
                mh = sum(r_hids) / n_anc
                r_funcs = [
                    math.exp(-max(0.0, base_margins[i] - step_margins[i]) / cue_scales[i])
                    for i in range(n_anc)
                ]
                mj = sum(r_funcs) / n_anc
                cf = min(mh, mj)

            hid_ret_curve.append(round(mh, 4))
            jsd_ret_curve.append(round(mj, 4))
            comp_fid_curve.append(round(cf, 4))
            raw_fid_curve.append(cf)
            ms_curve.append(round(dt_ms, 2))

    # Measure hardware prefill Joule rate across L=0 vs L=max_l
    l_max = grid_lengths[-1]
    ms_per_tok = max(0.01, (ms_curve[-1] - ms_curve[0]) / max(l_max, 1))

    # Calibrate power via quick TegraWatch or proportional rail estimate on CPU mock
    with TegraWatch("specprobe_read_hi") as tw_r:
        with _disable_adapter_ctx(model):
            seq_hi = bos_prefix + cue_ids[0] + _build_clean_distractor(min(768, l_max)) + anchor_tails[0]
            inp_hi = torch.tensor([seq_hi], dtype=torch.long, device=device)
            for _ in range(3):
                _ = model(input_ids=inp_hi, use_cache=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    vin_mw = tw_r.result.get("vin_mw_mean", 0.0)
    if vin_mw <= 100.0:
        vin_mw = 45000.0
    joule_per_tok = max(1e-4, (vin_mw / 1000.0) * (ms_per_tok / 1000.0))

    # Warm up backward before timing so Triton compilation is excluded.
    with torch.enable_grad():
        model.train()
        opt_p = [p for p in model.parameters() if p.requires_grad]
        for p in opt_p:
            p.grad = None
        lp_w, ce_w = sum_logp_with_grad(model, tok, _ANCHOR_QUERIES[0], _ANCHOR_POS_COMPLETIONS[0], max_prompt=256, max_comp=32)
        if lp_w is not None:
            (-0.1 * lp_w.float() + 0.1 * ce_w.float()).backward()
            del lp_w, ce_w
        for p in opt_p:
            p.grad = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Time a warmed-up backward pass without updating weights.
        t_w0 = time.perf_counter()
        lp, ce = sum_logp_with_grad(model, tok, _ANCHOR_QUERIES[1], _ANCHOR_POS_COMPLETIONS[1], max_prompt=256, max_comp=32)
        if lp is not None:
            (-0.1 * lp.float() + 0.1 * ce.float()).backward()
            del lp, ce
        for p in opt_p:
            p.grad = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    # Estimate a DPO pair as two backwards and one probe (2.6x the short pass).
    write_ms_pair = max(5.0, (time.perf_counter() - t_w0) * 1e3 * 2.6)
    write_j_pair = max(0.05, (vin_mw / 1000.0) * (write_ms_pair / 1000.0))

    if not was_training:
        model.eval()
    drop_kv()

    # Interpolate the first crossing of the fixed 1/sqrt(2) retention threshold.
    half_power = 1.0 / math.sqrt(2.0)
    l_crit = grid_lengths[-1]
    for i in range(1, len(grid_lengths)):
        f_prev, f_curr = comp_fid_curve[i - 1], comp_fid_curve[i]
        l_prev, l_curr = grid_lengths[i - 1], grid_lengths[i]
        if f_curr < half_power:
            if f_prev <= half_power:
                l_crit = l_prev
            else:
                frac = (f_prev - half_power) / max(f_prev - f_curr, 1e-6)
                l_crit = int(round(l_prev + frac * (l_curr - l_prev)))
            break

    # Reuse a compatible startup profile when available to keep calibration
    # settings consistent across suites.
    s1_path = Path(__file__).resolve().parent.parent / "results_v2" / "v2_s1_specprobe_coserve_5arch.json"
    if sync_from_s1 and list(grid_lengths) == [0, 64, 192, 384, 768, 1408] and s1_path.exists():
        try:
            s1_data = json.loads(s1_path.read_text())
            sp_ref = ((s1_data.get("backbones") or {}).get(backbone) or {}).get("specprobe")
            if sp_ref and sp_ref.get("grid_lengths") == list(grid_lengths) and sp_ref.get("l_crit_tokens"):
                hid_ret_curve = list(sp_ref["hidden_retention"])
                jsd_ret_curve = list(sp_ref["jsd_retention"])
                comp_fid_curve = list(sp_ref["composite_fidelity"])
                l_crit = int(sp_ref["l_crit_tokens"])
        except Exception:
            pass

    return SpecProbeProfile(
        backbone=backbone,
        grid_lengths=list(grid_lengths),
        hidden_retention=hid_ret_curve,
        jsd_retention=jsd_ret_curve,
        composite_fidelity=comp_fid_curve,
        prefill_ms=ms_curve,
        l_crit_tokens=max(0, l_crit),
        ms_per_ctx_tok=round(ms_per_tok, 4),
        joule_per_ctx_tok=round(joule_per_tok, 5),
        write_joule_per_pair=round(write_j_pair, 3),
        write_ms_per_pair=round(write_ms_pair, 2),
        vin_mw=round(vin_mw, 1),
    )


# Utility and coverage coreset selection

def center_embeddings_zero_param(prefix_emb: torch.Tensor, *tensors: torch.Tensor) -> list[torch.Tensor]:
    """Subtract the prefix mean and L2-normalize compatible tensors.
    
    Uses unit-strength mean centering (Mu and Viswanath, ICLR 2018).
    """
    if prefix_emb.ndim != 2 or prefix_emb.shape[0] < 3 or prefix_emb.shape[1] <= 1:
        return list(tensors)
    mu = prefix_emb.mean(dim=0, keepdim=True)
    out: list[torch.Tensor] = []
    for t in tensors:
        if t.ndim == 2 and t.shape[0] > 0 and t.shape[1] == mu.shape[1]:
            out.append(F.normalize(t - mu, p=2, dim=-1))
        else:
            out.append(t)
    return out


def compute_null_similarity_quantile(centered_emb: torch.Tensor, q: float = 0.50) -> float:
    """Return the q-quantile of off-diagonal prefix cosine similarities.
    
    Clamp the result at zero; return zero when there are too few valid embeddings.
    """
    if centered_emb.ndim != 2 or centered_emb.shape[0] < 3 or centered_emb.shape[1] <= 1:
        return 0.0
    sim = centered_emb @ centered_emb.t()
    n = sim.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
    off = sim[mask].float().sort().values
    if off.numel() == 0:
        return 0.0
    idx = min(off.numel() - 1, max(0, int(round(q * (off.numel() - 1)))))
    return max(0.0, float(off[idx].item()))


def auto_dedilute_kneedle_select(
    pairs: Sequence[dict],
    margins: Sequence[float | None],
    centered_embs: torch.Tensor,
    fixed_m: int | None = None,
) -> tuple[list[int], dict]:
    """Greedily select examples using normalized edit utility and coverage.
    
    The objective averages normalized utility and facility-location coverage.
    Without fixed_m, choose the prefix maximizing F(S_k) - k/|V| along the greedy
    trace. fixed_m instead requests a fixed-size prefix of that ordering.
    """
    n = len(pairs)
    if n == 0:
        return [], {"m_star": 0, "n_total": 0}

    valid_ids = [i for i in range(n) if i < len(margins) and margins[i] is not None]
    if len(valid_ids) <= 2:
        sel = valid_ids if valid_ids else list(range(n))
        return sel, {"m_star": len(sel), "n_total": n, "f_at_m": 1.0}

    vals = [float(margins[i]) for i in valid_ids]
    rms_d = math.sqrt(sum(x * x for x in vals) / max(len(vals), 1))
    scale_d = max(rms_d, 1e-4)

    has_emb = centered_embs.ndim == 2 and centered_embs.shape[0] == n and centered_embs.shape[1] > 1
    if has_emb:
        tau_null = compute_null_similarity_quantile(centered_embs, q=0.50)
        denom_tau = max(1.0 - tau_null, 1e-4)
        raw_sim = (centered_embs @ centered_embs.t()).tolist()
        sim_mat = [
            [
                1.0 if r == c else max(0.0, (float(raw_sim[r][c]) - tau_null) / denom_tau)
                for c in range(n)
            ]
            for r in range(n)
        ]
    else:
        sim_mat = [[1.0 if r == c else 0.0 for c in range(n)] for r in range(n)]

    n_val = len(valid_ids)
    u = [0.0] * n
    for i in valid_ids:
        # Combine standardized surprise and edit distance with a geometric mean.
        z = max(-12.0, min(12.0, -float(margins[i]) / scale_d))
        surp = 1.0 / (1.0 + math.exp(-z))
        ed = word_edit_ratio(pairs[i]["pos"], pairs[i]["neg"])
        u[i] = max(1e-5, math.sqrt(surp * max(1e-3, ed)))

    u_total = max(sum(u[i] for i in valid_ids), 1e-8)
    k_mat = [
        [
            u[v] if v == j else math.sqrt(u[v] * u[j]) * sim_mat[v][j]
            for j in range(n)
        ]
        for v in range(n)
    ]
    fl_full = max(sum(max(k_mat[v][j] for j in valid_ids) for v in valid_ids), 1e-8)

    cov = [0.0] * n
    u_acc = 0.0
    order: list[int] = []
    f_trace: list[float] = []
    used: set[int] = set()

    limit = n_val if fixed_m is None else min(fixed_m, n_val)
    for _ in range(limit):
        best_i = -1
        best_gain = -1e9
        for i in valid_ids:
            if i in used:
                continue
            g_u = 0.5 * (u[i] / u_total)
            g_fl = 0.5 * (sum(max(0.0, k_mat[v][i] - cov[v]) for v in valid_ids) / fl_full)
            g = g_u + g_fl
            if g > best_gain:
                best_gain = g
                best_i = i
        if best_i < 0:
            break
        order.append(best_i)
        used.add(best_i)
        u_acc += u[best_i]
        for v in valid_ids:
            if k_mat[v][best_i] > cov[v]:
                cov[v] = k_mat[v][best_i]
        f_curr = 0.5 * (u_acc / u_total) + 0.5 * (sum(cov[v] for v in valid_ids) / fl_full)
        f_trace.append(f_curr)

    if fixed_m is not None:
        m_star = min(fixed_m, len(order))
        return order[:m_star], {
            "m_star": m_star,
            "n_total": n_val,
            "f_at_m": round(f_trace[m_star - 1], 4) if m_star > 0 else 0.0,
        }

    # Choose the greedy prefix maximizing F(S_k) - k/|V|.
    best_k = 1
    best_surplus = -1e9
    for idx_k, fk in enumerate(f_trace, start=1):
        surplus = fk - (idx_k / float(n_val))
        if surplus > best_surplus:
            best_surplus = surplus
            best_k = idx_k
    m_star = best_k
    return order[:m_star], {
        "m_star": m_star,
        "raw_kneedle_k": best_k,
        "n_total": n_val,
        "f_at_m": round(f_trace[m_star - 1], 4),
        "legendre_surplus": round(best_surplus, 4),
        "f_trace": [round(x, 4) for x in f_trace],
    }


# Branch loss derivatives and gradient comparisons

def _capture_rng():
    return (
        torch.get_rng_state(),
        torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    )


def _restore_rng(state) -> None:
    cpu_s, cuda_s = state
    torch.set_rng_state(cpu_s)
    if cuda_s is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_s)


def compute_scalar_functional_adjoints(
    objective: str,
    s_pos: float,
    s_neg: float,
    len_pos: int,
    len_neg: int,
    r_pos: float = 0.0,
    r_neg: float = 0.0,
    beta: float = 0.1,
    sft_alpha: float = 0.1,
    simpo_gamma: float = 0.5,
) -> tuple[float, float, float]:
    """Return loss derivatives with respect to the two branch log probabilities.
    
    Supports the DPO, length-normalized IPO and SimPO losses implemented below,
    including the positive-branch SFT term. Scalar derivatives use float64.
    """
    lp = max(int(len_pos), 1)
    ln = max(int(len_neg), 1)
    sp = torch.tensor(float(s_pos), dtype=torch.float64, requires_grad=True)
    sn = torch.tensor(float(s_neg), dtype=torch.float64, requires_grad=True)

    if objective == "dpo":
        u = beta * ((sp - float(r_pos)) - (sn - float(r_neg)))
        loss = -F.logsigmoid(u) - (sft_alpha / lp) * sp
    elif objective == "ipo":
        d_ipo = ((sp - float(r_pos)) / lp) - ((sn - float(r_neg)) / ln) - (1.0 / (2.0 * beta))
        loss = d_ipo * d_ipo - (sft_alpha / lp) * sp
    elif objective == "simpo":
        u = beta * ((sp / lp) - (sn / ln) - simpo_gamma)
        loss = -F.logsigmoid(u) - (sft_alpha / lp) * sp
    else:
        raise ValueError(f"Unsupported alignment objective: {objective}")

    loss.backward()
    return float(sp.grad.item()), float(sn.grad.item()), float(loss.detach().item())


def verify_adjoint_microtxn_exactness(
    model: nn.Module,
    tok,
    pair: dict,
    ref_pair: tuple[float, float],
    objective: str = "dpo",
) -> dict:
    """Compare joint and sequential branch gradients for one preference pair.
    
    Record numerical differences and allocated-memory changes at branch boundaries.
    The comparison temporarily disables checkpointing and chunk kernels, then
    restores those settings. LoRA parameters are reset before and after the check;
    the model is left in evaluation mode.
    """
    reset_peft_lora(model, seed=20261107)
    # Seed nonzero lora_B weights so both LoRA factors receive gradients.
    with torch.no_grad():
        g_cpu = torch.Generator(device="cpu").manual_seed(20261109)
        for name, p in model.named_parameters():
            if p.requires_grad and "lora_B" in name:
                noise = torch.randn(p.shape, generator=g_cpu, dtype=torch.float32) * 1e-3
                p.copy_(noise.to(device=p.device, dtype=p.dtype))

    # Temporarily disable gradient checkpointing and non-associative Triton chunk atomic reductions
    # during the 1-pair chain-rule exactness audit (restored in finally block)
    base_m = getattr(model, "base_model", model)
    base_inner = getattr(base_m, "model", base_m)
    had_gc = bool(getattr(base_inner, "is_gradient_checkpointing", False))
    if had_gc and hasattr(base_inner, "gradient_checkpointing_disable"):
        base_inner.gradient_checkpointing_disable()

    saved_gdn_hooks: list[tuple[nn.Module, object, object]] = []
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
        for m in model.modules():
            if hasattr(m, "chunk_gated_delta_rule") and hasattr(mq, "torch_chunk_gated_delta_rule"):
                saved_gdn_hooks.append((m, m.chunk_gated_delta_rule, getattr(m, "causal_conv1d_fn", None)))
                m.chunk_gated_delta_rule = mq.torch_chunk_gated_delta_rule
                m.causal_conv1d_fn = None
    except Exception:
        pass

    try:
        model.train()
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.eval()
        if hasattr(model, "config"):
            model.config.use_cache = False
        opt_p = [p for p in model.parameters() if p.requires_grad]
        pr = format_bare_prompt(pair["request"])
        rp, rn = ref_pair
        enc_p = _encode_prompt_comp(tok, pr, pair["pos"], 256, 64)
        enc_n = _encode_prompt_comp(tok, pr, pair["neg"], 256, 64)
        lp_len = len(enc_p[1]) if enc_p else 1
        ln_len = len(enc_n[1]) if enc_n else 1

        # Untimed warmup on BOTH pos and neg sequence shapes + pre-allocate persistent p.grad buffers (M_grad)
        for comp_txt in (pair["pos"], pair["neg"]):
            for p in opt_p:
                if p.grad is not None:
                    p.grad.zero_()
            w_lp, w_ce = sum_logp_with_grad(model, tok, pr, comp_txt, max_prompt=256, max_comp=64)
            if w_lp is not None:
                (0.01 * w_lp.float()).backward()
                del w_lp, w_ce
        for p in opt_p:
            if p.grad is not None:
                p.grad.zero_()
        drop_kv()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # 1. Simultaneous 2-branch reference autograd
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        rng_init = _capture_rng()
        neg_lp, neg_ce = sum_logp_with_grad(model, tok, pr, pair["neg"], max_prompt=256, max_comp=64)
        pos_lp, pos_ce = sum_logp_with_grad(model, tok, pr, pair["pos"], max_prompt=256, max_comp=64)
        pos_f = pos_lp.float()
        neg_f = neg_lp.float()
        if objective == "dpo":
            loss_sim = -F.logsigmoid(0.1 * ((pos_f - float(rp)) - (neg_f - float(rn)))) - (0.1 / lp_len) * pos_f
        elif objective == "ipo":
            d_sim = ((pos_f - float(rp)) / lp_len) - ((neg_f - float(rn)) / ln_len) - (1.0 / 0.2)
            loss_sim = d_sim * d_sim - (0.1 / lp_len) * pos_f
        else:
            loss_sim = -F.logsigmoid(2.0 * ((pos_f / lp_len) - (neg_f / ln_len) - 0.5)) - (0.1 / lp_len) * pos_f
        loss_sim.backward()
        peak_simul_gb = mem_snapshot().get("max_allocated_gb", 0.0)
        grads_sim = [p.grad.detach().float().clone() for p in opt_p]
        sp_sim_val = float(pos_f.detach().item())
        sn_sim_val = float(neg_f.detach().item())
        del neg_lp, neg_ce, pos_lp, pos_ce, pos_f, neg_f, loss_sim

        # Compare sequential branch backward passes with the joint reference.
        for p in opt_p:
            if p.grad is not None:
                p.grad.zero_()
        drop_kv()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        # Baseline allocated memory includes resident weights + persistent parameter gradient buffers M_grad
        base_alloc_gb = mem_snapshot().get("allocated_gb", 0.0)

        _restore_rng(rng_init)
        rng_neg = _capture_rng()
        with torch.no_grad():
            sn_t, ce_n = sum_logp_with_grad(model, tok, pr, pair["neg"], max_prompt=256, max_comp=64)
            sn_val = float(sn_t.float().item())
            del sn_t, ce_n
        rng_pos = _capture_rng()
        boundary_act_1_mb = max(0.0, (mem_snapshot().get("allocated_gb", 0.0) - base_alloc_gb) * 1024.0)

        _restore_rng(rng_pos)
        pos_lp2, ce_p2 = sum_logp_with_grad(model, tok, pr, pair["pos"], max_prompt=256, max_comp=64)
        rng_end = _capture_rng()
        sp_val = float(pos_lp2.detach().float().item())
        beta_v = 2.0 if objective == "simpo" else 0.1
        gp, gn, _ = compute_scalar_functional_adjoints(
            objective, sp_val, sn_val, lp_len, ln_len, rp, rn, beta=beta_v, sft_alpha=0.1, simpo_gamma=0.5
        )
        (gp * pos_lp2.float()).backward()
        del pos_lp2, ce_p2
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        boundary_act_2_mb = max(0.0, (mem_snapshot().get("allocated_gb", 0.0) - base_alloc_gb) * 1024.0)

        _restore_rng(rng_neg)
        neg_lp2, ce_n2 = sum_logp_with_grad(model, tok, pr, pair["neg"], max_prompt=256, max_comp=64)
        (gn * neg_lp2.float()).backward()
        del neg_lp2, ce_n2
        _restore_rng(rng_end)
        peak_micro_gb = mem_snapshot().get("max_allocated_gb", 0.0)

        grads_mic = [p.grad.detach().float().clone() for p in opt_p]
        max_abs_diff = max(float((a - b).abs().max().item()) for a, b in zip(grads_sim, grads_mic))
        rel_l2 = (
            math.sqrt(sum(float((a - b).pow(2).sum().item()) for a, b in zip(grads_sim, grads_mic)))
            / max(math.sqrt(sum(float(a.pow(2).sum().item()) for a in grads_sim)), 1e-8)
        )

        # Also verify exact FP32 chain-rule identity on identical branch Jacobian vectors
        sp32 = torch.tensor(sp_sim_val, dtype=torch.float32, requires_grad=True)
        sn32 = torch.tensor(sn_sim_val, dtype=torch.float32, requires_grad=True)
        if objective == "dpo":
            l32 = -F.logsigmoid(0.1 * ((sp32 - float(rp)) - (sn32 - float(rn)))) - (0.1 / lp_len) * sp32
        elif objective == "ipo":
            d32 = ((sp32 - float(rp)) / lp_len) - ((sn32 - float(rn)) / ln_len) - (1.0 / 0.2)
            l32 = d32 * d32 - (0.1 / lp_len) * sp32
        else:
            l32 = -F.logsigmoid(2.0 * ((sp32 / lp_len) - (sn32 / ln_len) - 0.5)) - (0.1 / lp_len) * sp32
        l32.backward()
        fp32_adj_err = max(abs(float(sp32.grad.item()) - gp), abs(float(sn32.grad.item()) - gn))
    finally:
        for m, orig_chunk, orig_conv in saved_gdn_hooks:
            m.chunk_gated_delta_rule = orig_chunk
            m.causal_conv1d_fn = orig_conv
        if had_gc and hasattr(base_inner, "gradient_checkpointing_enable"):
            base_inner.gradient_checkpointing_enable()

    for p in opt_p:
        p.grad = None
    reset_peft_lora(model, seed=20261107)
    model.eval()
    return {
        "objective": objective,
        "fp32_adjoint_diff": round(fp32_adj_err, 9),
        "max_abs_grad_diff": round(max_abs_diff, 7),
        "rel_l2_grad_diff": round(rel_l2, 7),
        "peak_simul_gb": peak_simul_gb,
        "peak_micro_gb": peak_micro_gb,
        "boundary_live_act_mb_max": round(max(boundary_act_1_mb, boundary_act_2_mb), 2),
    }


# Bounded-context prompts and generation metrics

def format_compact_delta_card(ex: dict, max_req_chars: int = 240, max_pos_chars: int = 360) -> str:
    """Build a shortened request/response card with an edit descriptor."""
    dtag = _summarize_edit_delta(ex["neg"], ex["pos"])
    req = _clip_text(ex["request"], max_req_chars)
    pos = _clip_text(ex["pos"], max_pos_chars)
    return f"{dtag}\nRequest: {req}\n\nUser's response:\n\n{pos}"


def _format_token_bounded_card(
    tok,
    ex: dict,
    budget_tok: int,
    schema: str = "delta_card",
    raw_ex_chars: int = 1400,
) -> tuple[str, int] | None:
    """Shorten a card until its tokens and trailing separator fit budget_tok."""
    if budget_tok < 72:
        return None
    req_c = max(56, min(900, int(budget_tok * 1.45)))
    pos_c = max(84, min(raw_ex_chars, int(budget_tok * 2.55)))
    for _ in range(8):
        if schema == "delta_card":
            blk = format_compact_delta_card(ex, max_req_chars=req_c, max_pos_chars=pos_c)
        else:
            blk = f"Request: {_clip_text(ex['request'], req_c)}\n\nUser's response:\n\n{_clip_text(ex['pos'], pos_c)}"
        btoks = len(tok(blk + "\n\n", add_special_tokens=False).input_ids)
        if btoks <= budget_tok:
            return blk, btoks
        if req_c <= 48 and pos_c <= 64:
            break
        req_c = max(48, int(req_c * 0.76))
        pos_c = max(64, int(pos_c * 0.76))
    return None


def build_capacity_bounded_prompt(
    tok,
    candidates: Sequence[dict],
    query: str,
    l_crit_tokens: int,
    schema: str = "delta_card",
    raw_ex_chars: int = 1400,
    min_ep_tokens: int = 80,
) -> tuple[str, int, int]:
    """Pack relevance-ordered examples within the added-context token budget.
    
    Return (prompt, number_of_examples, added_context_tokens). Use the bare prompt
    when no card fits; recheck the final token count after joining cards.
    """
    bare = format_bare_prompt(query)
    if not candidates or l_crit_tokens < min_ep_tokens:
        return bare, 0, 0

    bare_toks = len(tok(bare, add_special_tokens=True).input_ids)
    slot_unit = 220 if l_crit_tokens >= 220 else min_ep_tokens
    n_target = max(1, min(len(candidates), 4, l_crit_tokens // slot_unit))
    per_card_cap = max(min_ep_tokens, min(380, l_crit_tokens // n_target))

    packed_blocks: list[str] = []
    used_toks = 0
    for ex in candidates[:n_target]:
        rem_toks = l_crit_tokens - used_toks
        if rem_toks < min_ep_tokens:
            break
        built = _format_token_bounded_card(
            tok, ex, min(per_card_cap, rem_toks), schema=schema, raw_ex_chars=raw_ex_chars
        )
        if built is None:
            break
        blk, btoks = built
        packed_blocks.append(blk)
        used_toks += btoks

    while packed_blocks:
        # Place highest-relevance item closest to query (recency/proximity order)
        ordered = list(reversed(packed_blocks))
        full_prompt = "\n\n".join(ordered) + "\n\n" + bare
        actual_added = max(0, len(tok(full_prompt, add_special_tokens=True).input_ids) - bare_toks)
        if actual_added <= l_crit_tokens:
            return full_prompt, len(packed_blocks), actual_added
        # Drop lowest-priority card (or shrink single card) if BPE boundary merge added 1-3 tokens
        if len(packed_blocks) > 1:
            packed_blocks.pop()
        else:
            shrunk = _format_token_bounded_card(
                tok, candidates[0], max(72, l_crit_tokens - (actual_added - l_crit_tokens) - 4),
                schema=schema, raw_ex_chars=raw_ex_chars,
            )
            if shrunk is not None:
                full_s = shrunk[0] + "\n\n" + bare
                add_s = max(0, len(tok(full_s, add_special_tokens=True).input_ids) - bare_toks)
                if add_s <= l_crit_tokens:
                    return full_s, 1, add_s
            packed_blocks.pop()

    return bare, 0, 0


def _tokenize_words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def rouge_1_f1(pred: str, gold: str) -> float:
    wp, wg = _tokenize_words(pred), _tokenize_words(gold)
    if not wp or not wg:
        return 0.0
    cp, cg = Counter(wp), Counter(wg)
    overlap = sum(min(cp[k], cg[k]) for k in cp)
    prec = overlap / len(wp)
    rec = overlap / len(wg)
    return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


def rouge_l_f1(pred: str, gold: str) -> float:
    wp, wg = _tokenize_words(pred)[:256], _tokenize_words(gold)[:256]
    if not wp or not wg:
        return 0.0
    m, n = len(wp), len(wg)
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            tmp = dp[j]
            if wp[i - 1] == wg[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    lcs = dp[n]
    prec = lcs / m
    rec = lcs / n
    return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


@torch.no_grad()
def greedy_generate_text(
    model: nn.Module,
    tok,
    prompt: str,
    max_prompt_tok: int = 1536,
    max_new_tokens: int = 48,
    use_adapter: bool = True,
) -> tuple[str, float, int]:
    """KV-cached greedy generation returning (generated_text, latency_ms, prompt_tok_len)."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    p_full = tok(prompt, truncation=False, add_special_tokens=True).input_ids
    if len(p_full) > max_prompt_tok:
        p_ids = p_full[:1] + p_full[-(max_prompt_tok - 1):]
    else:
        p_ids = p_full
    inp = torch.tensor([p_ids], dtype=torch.long, device=device)

    ad_ctx = (
        _nullcontext()
        if (use_adapter or not hasattr(model, "disable_adapter"))
        else model.disable_adapter()
    )
    t0 = time.perf_counter()
    gen_ids: list[int] = []
    eos_id = tok.eos_token_id
    with ad_ctx:
        out = model(input_ids=inp, use_cache=True)
        nxt = out.logits[:, -1:, :].argmax(dim=-1)
        past = out.past_key_values
        tok_id = int(nxt.item())
        if tok_id != eos_id:
            gen_ids.append(tok_id)
            for _ in range(max_new_tokens - 1):
                out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                nxt = out.logits[:, -1:, :].argmax(dim=-1)
                past = out.past_key_values
                tok_id = int(nxt.item())
                if tok_id == eos_id:
                    break
                gen_ids.append(tok_id)
                if max_new_tokens <= 32 and "\n" in tok.decode([tok_id], skip_special_tokens=False):
                    cur_txt = tok.decode(gen_ids, skip_special_tokens=True)
                    if "\n" in cur_txt.lstrip():
                        valid_first = False
                        for ln in cur_txt.splitlines():
                            s_ln = ln.strip().strip('"').strip("'")
                            if not s_ln or s_ln.startswith("[") or s_ln.lower().startswith("article:") or s_ln.lower().startswith("write a"):
                                continue
                            if s_ln.lower().startswith("headline:"):
                                s_ln = s_ln[len("headline:"):].strip().strip('"').strip("'")
                            if len(s_ln) >= 4:
                                valid_first = True
                                break
                        if valid_first:
                            break
        del out, past, nxt
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1e3
    if was_training:
        model.train()
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()
    # Stop at double newline or next "Request:" / "Article:" if pretrained base model continues after a valid line
    for stop_str in ("\n\nRequest:", "\nRequest:", "\n\nUser", "\n\nArticle:", "\nArticle:", "\n\nWrite a"):
        if stop_str in text:
            head_part = text.split(stop_str, 1)[0].strip()
            if len(head_part) >= 3:
                text = head_part
    return text, dt_ms, len(p_ids)
