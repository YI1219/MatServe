#!/usr/bin/env python3
"""Evaluate local retrieval and profile-adaptation configurations on LaMP-4.

Reports generated-headline ROUGE and separate gold-versus-synthetic-negative
margins. Adapting methods use local optimizer-step or branchwise execution.
This is a prepared benchmark replay, not the official leaderboard protocol.
"""
from __future__ import annotations

import argparse
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

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.90, 0)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from core import (  # noqa: E402
    _clip_text,
    _encode_prompt_comp,
    _nullcontext,
    _summarize_edit_delta,
    drop_kv,
    embed_texts,
    format_bare_prompt,
    log,
    mean_logp_no_grad,
    mem_snapshot,
    paired_bootstrap_test,
    reset_peft_lora,
    select_mmr,
    select_topk_cosine,
    set_seed,
    sum_and_mean_logp_no_grad,
)
from planning import (  # noqa: E402
    SpecProbeProfile,
    auto_dedilute_kneedle_select,
    center_embeddings_zero_param,
    compute_null_similarity_quantile,
    greedy_generate_text,
    load_v2_backbone,
    rouge_1_f1,
    rouge_l_f1,
    run_specprobe_calibration,
)
from baselines import select_lamp_bm25  # noqa: E402
from stream import execute_controlled_flush  # noqa: E402
from execution import (  # noqa: E402
    measure_unloaded_single_query_baseline,
    run_adjoint_microtxn_engine,
)


def load_lamp4_authors(lamp_dir: str | Path, max_authors: int) -> list[dict]:
    lamp_dir = Path(lamp_dir)
    manifest = json.loads((lamp_dir / "_manifest.json").read_text())
    return [
        json.loads((lamp_dir / f"{row['author_id']}.json").read_text())
        for row in manifest[:max_authors]
    ]


def fmt_lamp_bare(article: str) -> str:
    return f"Write a concise news headline matching the author's style.\nArticle: {_clip_text(article, 750)}\nHeadline:"


def fmt_lamp_shots(exs: Sequence[dict], article: str, with_delta: bool = False, max_art_chars: int = 420) -> str:
    if not exs:
        return fmt_lamp_bare(article)
    blocks = ["Write a concise news headline matching the author's style."]
    for ex in exs:
        art = _clip_text(ex.get("article", ex["request"]), max_art_chars)
        hdr = f"{_summarize_edit_delta(ex['neg'], ex['pos'])}\n" if with_delta else ""
        blocks.append(f"{hdr}Article: {art}\nHeadline: {ex['pos'].strip()}")
    blocks.append(f"Article: {_clip_text(article, 750)}\nHeadline:")
    return "\n\n".join(blocks)


def fmt_lamp_persona(history: Sequence[dict], article: str) -> str:
    lens = [len(h["pos"].split()) for h in history]
    avg_w = round(sum(lens) / max(len(lens), 1), 1)
    sample_titles = " | ".join(h["pos"].strip()[:65] for h in history[-4:])
    persona = (
        f"[Author Headline Profile: typical length={avg_w} words; "
        f"recent headlines: {sample_titles}]\n\n"
    )
    return persona + fmt_lamp_bare(article)


def clean_headline_output(raw: str) -> str:
    text = raw.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    # First priority: explicit "Headline:" or "Title:" line if model emitted preamble
    for line in text.splitlines():
        s = line.strip().strip('"').strip("'")
        if s.lower().startswith("headline:"):
            cand = s[len("headline:"):].strip().strip('"').strip("'")
            if len(cand) >= 3:
                return cand[:160]
        if s.lower().startswith("title:"):
            cand = s[len("title:"):].strip().strip('"').strip("'")
            if len(cand) >= 3:
                return cand[:160]
    for line in text.splitlines():
        s = line.strip().strip('"').strip("'")
        if (
            not s
            or s.startswith("[")
            or s.startswith("<think")
            or s.lower().startswith("article:")
            or s.lower().startswith("write a")
        ):
            continue
        if len(s) >= 3:
            return s[:160]
    return text[:160]


def _generic_article_lead_headline(article: str) -> str:
    """Construct a synthetic negative headline from the article lead clause."""
    clean = " ".join(article.strip().split())
    if not clean:
        return "News update and analysis."
    first_clause = clean.split(".")[0].strip()
    words = first_clause.split()
    if len(words) >= 6:
        return " ".join(words[:12]).rstrip(",;:") + "."
    return " ".join(clean.split()[:12]).rstrip(",;:") + "."


def to_lamp_train_pair(
    item: dict,
    ctx_exs: Sequence[dict] | None = None,
    prefix_ctx: str = "",
    max_art_chars: int = 420,
    use_lead_neg: bool = False,
) -> dict:
    """Use the Article-to-Headline prompt layout for a training pair."""
    art = item.get("article", item["request"])
    if ctx_exs:
        req_body = fmt_lamp_shots(ctx_exs, art, with_delta=False, max_art_chars=max_art_chars)
    elif prefix_ctx:
        req_body = f"{prefix_ctx}\n\n{fmt_lamp_bare(art)}"
    else:
        req_body = fmt_lamp_bare(art)
    neg_txt = _generic_article_lead_headline(art) if use_lead_neg else item["neg"].strip()
    if neg_txt.lower() == item["pos"].strip().lower():
        neg_txt = item["neg"].strip()
    return {"request": req_body, "pos": item["pos"].strip(), "neg": neg_txt}


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def item_source_id(item: dict, fallback: str) -> str:
    return str(item.get("source_id", item.get("id", fallback)))


def build_budget_prompt(tok, history, article, indices, added_token_budget, max_art_chars=420):
    """Greedily admit whole cards; cap the exact added tokens relative to bare.

    Query tokens are outside this budget. Character clipping formats individual
    articles, but never substitutes for the added-token admission check.
    """
    if added_token_budget < 0:
        raise ValueError("added-token budget must be nonnegative")
    bare = fmt_lamp_bare(article)
    base_tokens = len(tok(bare, add_special_tokens=True).input_ids)
    chosen, prompt = [], bare
    for idx in indices:
        if idx in chosen:
            continue
        trial = fmt_lamp_shots([history[j] for j in reversed(chosen + [idx])], article,
                               max_art_chars=max_art_chars)
        added = max(0, len(tok(trial, add_special_tokens=True).input_ids) - base_tokens)
        if added <= added_token_budget:
            chosen.append(idx)
            prompt = trial
    rendered_tokens = len(tok(prompt, add_special_tokens=True).input_ids)
    return dict(prompt=prompt, retrieved_indices=chosen,
                retrieved_source_ids=[item_source_id(history[j], f"history:{j}") for j in chosen],
                actual_k=len(chosen), requested_k=len(indices), base_prompt_tokens=base_tokens,
                rendered_prompt_tokens=rendered_tokens,
                added_prompt_tokens=max(0, rendered_tokens - base_tokens),
                added_token_budget=added_token_budget, budget_semantics="exact_added_tokens_vs_bare")


def author_comparison(rows, method, baseline, metric="rouge_l"):
    result = paired_bootstrap_test([r["methods"][method][metric] for r in rows],
                                   [r["methods"][baseline][metric] for r in rows])
    result.update(resampling_unit="author", metric=metric,
                  author_ids=[r["author_id"] for r in rows])
    return result


@torch.no_grad()
def evaluate_lamp_queries(model, tok, queries, prompt_fn, use_adapter, sp, event_callback=None):
    """Measure headline ROUGE and pairwise margins with the same prompt and adapter.
    
    The synthetic-negative margin and generated-headline scores are separate metrics.
    """
    items = []
    base_read_j = (sp.vin_mw / 1000.0) * (sp.prefill_ms[0] / 1000.0)
    for qi, q in enumerate(queries):
        built = prompt_fn(qi, q)
        meta = dict(built) if isinstance(built, dict) else {"prompt": built}
        pr = meta.pop("prompt")
        gen_raw, dt_ms, p_tok = greedy_generate_text(
            model, tok, pr, max_prompt_tok=2200, max_new_tokens=32, use_adapter=use_adapter)
        pred, gold, negative = clean_headline_output(gen_raw), q["pos"].strip(), q["neg"].strip()
        with (_nullcontext() if (use_adapter or not hasattr(model, "disable_adapter")) else model.disable_adapter()):
            lp_p = mean_logp_no_grad(model, tok, pr, gold, max_prompt=2200, max_comp=64)
            lp_n = mean_logp_no_grad(model, tok, pr, negative, max_prompt=2200, max_comp=64)
        if lp_p is None or lp_n is None:
            raise ValueError(f"unscorable held-out item {qi}")
        item = dict(meta, source_id=item_source_id(q, f"heldout:{qi}"), query_index=qi,
                    prompt=pr, rendered_prompt_tokens=len(tok(pr, add_special_tokens=True).input_ids),
                    prompt_tokens=p_tok, generation_ms=dt_ms, raw_generation=gen_raw,
                    prediction=pred, gold=gold, negative=negative,
                    gold_source_id=q.get("gold_source_id", item_source_id(q, f"heldout:{qi}") + ":gold"),
                    negative_source_id=q.get("negative_source_id", item_source_id(q, f"heldout:{qi}") + ":synthetic"),
                    negative_construction="dataset_deterministic_lead_sentence_or_generic_headline",
                    use_adapter=use_adapter, rouge_1=rouge_1_f1(pred, gold), rouge_l=rouge_l_f1(pred, gold),
                    margin=lp_p-lp_n, acc=float(lp_p > lp_n),
                    estimated_prefill_j=max(.2, base_read_j + max(0, p_tok-64)*sp.joule_per_ctx_tok))
        if item.get("added_prompt_tokens", 0) > item.get("added_token_budget", float("inf")):
            raise RuntimeError("strict added-token contract violated")
        items.append(item)
        if event_callback:
            event_callback(item)
    if not items:
        raise ValueError("empty held-out query set")
    def mean(key):
        return sum(x[key] for x in items) / len(items)
    return dict(rouge_1=100*mean("rouge_1"), rouge_l=100*mean("rouge_l"),
                margin=mean("margin"), acc=mean("acc"), prompt_toks=mean("prompt_tokens"),
                read_j=mean("estimated_prefill_j"), n_queries=len(items),
                per_item_rl=[x["rouge_l"] for x in items], per_item_m=[x["margin"] for x in items], per_item=items)


def run_lamp4_backbone(backbone, authors, partial_path, event_callback=None, memory_limit_gb=None):
    """Fresh one-profile adaptation transfer; no RentFlush online decision loop."""
    is_27b = backbone == "hybrid_qwen27b"
    lr, max_p_tr = (4.5e-5, 768) if is_27b else (7.5e-5, 960)
    max_c_tr, n_steps, grad_accum, sft_lamp = 64, (5 if is_27b else 7), 4, .45
    model, tok, load_info = load_v2_backbone(backbone, quant_4bit=False)
    sp = run_specprobe_calibration(model, tok, backbone=backbone)
    unloaded = measure_unloaded_single_query_baseline(model, tok,
        [fmt_lamp_bare(q["article"]) for q in authors[0]["heldout_queries"][:8]], gen_toks=16)
    ttft_slo_ms = max(350., round(2.5*unloaded["unloaded_ttft_p95_ms"], 1))
    tpot_slo_ms = max(20., round(1.5*unloaded["unloaded_tpot_p95_ms"], 1))
    qps = round(max(.03, .34*unloaded["service_capacity_qps"]), 3)
    methods = ["Base", "LaMP-BM25@4", "LaMP-Dense@4", "LaMP-Dense@8", "MemGPT-Persona",
               "OPPU-StepPreempt", "RAFT-StepPreempt", "TierServe-Planner-StepPreempt", "TierServe-Full"]
    rows, events = [], []
    recipe = dict(lr=lr, max_training_prompt_tokens=max_p_tr, max_completion_tokens=max_c_tr,
                  optimizer_steps=n_steps, grad_accum=grad_accum, sft_alpha=sft_lamp,
                  heldout_prompt_tokens=2200, generation_tokens=32, foreground_generation_tokens=16,
                  reference_policy="cold_for_both_executors", foreground_adapter="base_fallback",
                  history_items=60, heldout_items=(6 if is_27b else 10),
                  protocol="one_profile_adaptation_transfer_not_online_RentFlush",
                  planner_budget="exact_added_tokens_relative_to_bare_query")
    def emit(kind, author_id, method=None, **record):
        if memory_limit_gb and torch.cuda.is_available() and torch.cuda.max_memory_allocated()/1024**3 >= memory_limit_gb:
            raise RuntimeError("prospective memory budget exceeded")
        event = dict(kind=kind, author_id=author_id, method=method, monotonic_s=time.perf_counter(), **record)
        events.append(event)
        atomic_json(partial_path, dict(backbone=backbone, completed_authors=len(rows), per_author=rows,
                    events=events, recipe=recipe, specprobe=sp.__dict__, load_info=load_info))
        if event_callback:
            event_callback(event)
    for ai, au in enumerate(authors):
        aid, hist = au["author_id"], [dict(h) for h in au["history_pairs"][-60:]]
        held = [dict(q) for q in au["heldout_queries"][:(6 if is_27b else 10)]]
        for hi, item in enumerate(hist):
            item["source_id"] = item_source_id(item, f"{aid}:history:{item.get('ep_idx', hi)}")
        for qi, item in enumerate(held):
            item["source_id"] = item_source_id(item, f"{aid}:heldout:{qi}")
        if not hist or not held:
            raise ValueError("empty author profile or heldout set")
        emit("author_start", aid, n_history=len(hist), n_queries=len(held),
             history_source_ids=[h["source_id"] for h in hist], heldout_source_ids=[h["source_id"] for h in held])
        reset_peft_lora(model, seed=20261107)
        he = embed_texts(model, tok, [h["article"] for h in hist], max_len=192)
        qe = embed_texts(model, tok, [q["article"] for q in held], max_len=192)
        he, qe = center_embeddings_zero_param(he, he, qe)
        bm25 = [{"request": h["article"]} for h in hist]
        row = {}
        def shots(q, ids, mode, k, chars=420):
            return dict(prompt=fmt_lamp_shots([hist[j] for j in reversed(ids)], q["article"], max_art_chars=chars),
                        retrieval_mode=mode, requested_k=k, actual_k=len(ids),
                        retrieved_source_ids=[hist[j]["source_id"] for j in ids], article_char_limit=chars)
        def evaluate(name, builder, adapter):
            result = evaluate_lamp_queries(model, tok, held, builder, adapter, sp,
                event_callback=lambda item: emit("heldout_query", aid, name, item=item))
            row[name] = result
            emit("method_evaluation_complete", aid, name, summary={k:v for k,v in result.items() if k != "per_item"})
            return result
        evaluate("Base", lambda i,q: dict(prompt=fmt_lamp_bare(q["article"]), retrieval_mode="bare", requested_k=0, actual_k=0, retrieved_source_ids=[]), False)
        evaluate("LaMP-BM25@4", lambda i,q: shots(q, select_lamp_bm25(q["article"], bm25, 4), "bm25", 4), False)
        evaluate("LaMP-Dense@4", lambda i,q: shots(q, select_topk_cosine(qe[i], he, 4)[0], "dense", 4), False)
        evaluate("LaMP-Dense@8", lambda i,q: shots(q, select_topk_cosine(qe[i], he, 8)[0], "dense", 8, 300), False)
        evaluate("MemGPT-Persona", lambda i,q: dict(prompt=fmt_lamp_persona(hist, q["article"]), retrieval_mode="persona", requested_k=0, actual_k=0, retrieved_source_ids=[h["source_id"] for h in hist[-4:]]), False)
        fg = [fmt_lamp_bare(q["article"]) for q in held]
        def train(name, pairs, source_ids, substrate):
            token_counts = []
            for pair in pairs:
                prompt = format_bare_prompt(pair["request"])
                positive = _encode_prompt_comp(tok, prompt, pair["pos"], max_p_tr, max_c_tr)
                negative = _encode_prompt_comp(tok, prompt, pair["neg"], max_p_tr, max_c_tr)
                if positive is None or negative is None:
                    raise ValueError("unscorable training pair")
                token_counts.append(dict(prompt_tokens=len(positive[0]), positive_tokens=len(positive[1]),
                                         negative_tokens=len(negative[1])))
            emit("refresh_start", aid, name, training_pairs=pairs, training_source_ids=source_ids,
                 training_token_counts=token_counts, recipe=recipe, seed=20261107+ai)
            result = execute_controlled_flush(substrate, model, tok, pairs, fg,
                qps, ttft_slo_ms, tpot_slo_ms, n_steps, grad_accum, lr, max_p_tr, max_c_tr,
                vin_mw=sp.vin_mw, sft_alpha=sft_lamp, scale_steps_by_coreset=False, seed=20261107+ai)
            emit("refresh_complete", aid, name, execution=result)
            return result
        def add_execution(result, execution):
            result.update(write_j=execution["vin_j"], coserve_slo_pct=execution["slo_joint_pct"],
                          coserve_peak_gb=execution["peak_gb"], execution=execution)
        f = train("OPPU-StepPreempt", [to_lamp_train_pair(h) for h in hist[-24:]],
                  [h["source_id"] for h in hist[-24:]], "external_llmstation")
        add_execution(evaluate("OPPU-StepPreempt", lambda i,q: shots(q, select_topk_cosine(qe[i], he, 4)[0], "dense", 4), True), f)
        raft, rng = [], random.Random(20261107+ai)
        for hi in range(max(0, len(hist)-24), len(hist)):
            ids, _ = select_topk_cosine(he[hi], he[:hi] if hi >= 4 else he, k=3)
            ctx = [hist[j] for j in ids if j != hi][:2] + ([hist[rng.randrange(len(hist))]] if len(hist)>4 else [])
            prefix = "\n".join(f"Article: {_clip_text(h['article'], 220)}\nHeadline: {h['pos']}" for h in ctx)
            pair = to_lamp_train_pair(hist[hi], prefix_ctx=prefix)
            pair["context_source_ids"] = [h["source_id"] for h in ctx]
            raft.append(pair)
        f = train("RAFT-StepPreempt", raft, [h["source_id"] for h in hist[-24:]], "external_llmstation")
        add_execution(evaluate("RAFT-StepPreempt", lambda i,q: shots(q, select_topk_cosine(qe[i], he, 4)[0], "dense", 4), True), f)
        reset_peft_lora(model, seed=20261107)
        margins = []
        with model.disable_adapter():
            for h in hist:
                pr = fmt_lamp_bare(h["article"])
                pos = mean_logp_no_grad(model, tok, pr, h["pos"], max_prompt=960, max_comp=64)
                neg = mean_logp_no_grad(model, tok, pr, h["neg"], max_prompt=960, max_comp=64)
                if pos is None or neg is None:
                    raise ValueError("unscorable profile item")
                margins.append(pos-neg)
                emit("profile_reference", aid, source_id=h["source_id"], margin=pos-neg)
        selected, kmeta = auto_dedilute_kneedle_select(hist, margins, he)
        tau = compute_null_similarity_quantile(he, q=.50)
        def route(article, emb, mode, k, exclude=None, chars=420):
            if mode == "bare" or k == 0:
                ids = []
            elif mode == "dense":
                ids = select_topk_cosine(emb, he, k, exclude_idx=exclude)[0]
            else:
                ids = select_mmr(emb, he, k=k, lam=.50, sim_floor=tau, exclude_idx=exclude)[0]
                for j in select_topk_cosine(emb, he, k, exclude_idx=exclude)[0]:
                    if j not in ids and len(ids)<k:
                        ids.append(j)
            built = build_budget_prompt(tok, hist, article, ids, sp.l_crit_tokens, chars)
            built.update(retrieval_mode=mode, requested_k=k, article_char_limit=chars)
            return built
        ts_pairs = []
        for hi in selected:
            k = 0 if sp.l_crit_tokens < 96 else (2 if sp.l_crit_tokens < 250 else 4)
            built = route(hist[hi]["article"], he[hi], "dense" if k else "bare", k, hi, 240)
            ts_pairs.append(dict(request=built["prompt"], pos=hist[hi]["pos"].strip(), neg=hist[hi]["neg"].strip(), prompt_metadata={k:v for k,v in built.items() if k!="prompt"}))
        f = train("TierServe-Planner-StepPreempt", ts_pairs, [hist[j]["source_id"] for j in selected], "external_llmstation")
        unseen = [j for j in range(len(hist)) if j not in set(selected)]
        valid = unseen[-5:] if len(unseen)>=5 else list(range(max(0,len(hist)-5),len(hist)))
        candidates = []
        if sp.l_crit_tokens >= 600:
            candidates.extend([("dense", 8), ("mmr", 8)])
        if sp.l_crit_tokens >= 96:
            candidates.extend([("dense", 4), ("mmr", 4), ("dense", 2)])
        candidates.append(("bare", 0))
        best, best_score = candidates[0], -math.inf
        for mode,k in candidates:
            scores=[]
            for vi,hi in enumerate(valid):
                q = hist[hi]
                built = route(q["article"], he[hi], mode, k, hi, 300 if k>=8 else 420)
                pr = built["prompt"]
                pos = mean_logp_no_grad(model, tok, pr, q["pos"], max_prompt=2200, max_comp=64)
                neg = mean_logp_no_grad(model, tok, pr, q["neg"], max_prompt=2200, max_comp=64)
                if pos is None or neg is None:
                    raise ValueError("unscorable validation item")
                raw, pred, rl = None, None, 0.
                if vi < 3:
                    raw, _, _ = greedy_generate_text(model, tok, pr, max_prompt_tok=2200, max_new_tokens=32, use_adapter=True)
                    pred=clean_headline_output(raw); rl=rouge_l_f1(pred,q["pos"].strip())
                score=.35*(pos-neg)+rl; scores.append(score)
                emit("routing_validation", aid, "TierServe-Planner-StepPreempt", source_id=q["source_id"],
                     prompt_record=built, gold=q["pos"], negative=q["neg"], margin=pos-neg,
                     gold_source_id=q["source_id"]+":gold", negative_source_id=q["source_id"]+":synthetic",
                     raw_generation=raw, prediction=pred, rouge_l=rl, score=score)
            score=sum(scores)/max(len(scores),1)
            if score > best_score+1e-6:
                best,best_score=(mode,k),score
        emit("routing_selected", aid, mode=best[0], requested_k=best[1], score=best_score,
             validation_source_ids=[hist[j]["source_id"] for j in valid], coreset_source_ids=[hist[j]["source_id"] for j in selected], coreset=kmeta)
        builder=lambda i,q: route(q["article"], qe[i], *best, chars=300 if best[1]>=8 else 420)
        add_execution(evaluate("TierServe-Planner-StepPreempt", builder, True), f)
        f = train("TierServe-Full", ts_pairs, [hist[j]["source_id"] for j in selected], "tierserve_adjoint_microtxn")
        add_execution(evaluate("TierServe-Full", builder, True), f)
        for result in row.values():
            result["dm"] = result["margin"]-row["Base"]["margin"]
            result.setdefault("write_j",0.); result.setdefault("coserve_slo_pct",None); result.setdefault("coserve_peak_gb",0.)
        rows.append(dict(author_id=aid, n_queries=len(held), history_source_ids=[h["source_id"] for h in hist], methods=row,
                         selected_mode=best[0], selected_k=best[1], coreset_source_ids=[hist[j]["source_id"] for j in selected]))
        emit("author_complete", aid, completed_authors=len(rows))
        log(f"AUTHOR_LAMP backbone={backbone} author={aid} completed={len(rows)}/{len(authors)}")
    aggregate={}
    for name in methods:
        values=[r["methods"][name] for r in rows]
        aggregate[name]={key:sum(v[key] for v in values)/len(values) for key in
            ("rouge_1","rouge_l","acc","margin","dm","prompt_toks","read_j","write_j")}
        aggregate[name].update(n_authors=len(values), n_queries=sum(v["n_queries"] for v in values))
    comparisons={}
    for baseline in ["Base","LaMP-BM25@4","LaMP-Dense@4","OPPU-StepPreempt","RAFT-StepPreempt"]:
        for metric in ["rouge_l","dm","acc"]:
            comparisons[f"Planner_vs_{baseline}_{metric}"]=author_comparison(rows,"TierServe-Planner-StepPreempt",baseline,metric)
    del model
    drop_kv()
    return dict(backbone=backbone,n_authors=len(rows),n_queries=sum(r["n_queries"] for r in rows),
                load_info=load_info,specprobe=sp.__dict__,recipe=recipe,per_author=rows,events=events,
                summary_methods=aggregate,author_paired_comparisons=comparisons)

