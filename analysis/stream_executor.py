#!/usr/bin/env python3
"""Summarize Dense/Mamba stream replays and matched executor runs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results"
LABELS = {"dense_qwen15b": "Dense 1.5B", "mamba_falcon15b": "Mamba 1.5B", "hybrid_qwen27b": "GDN 27B"}
METHODS = [("Base-Static", "Bare base"), ("Pure-Tier1-LaMP", "RAG@4"), ("Pure-Tier2-Lazy", "Lazy"),
    ("Pure-Tier2-Eager", "Eager"), ("Online-OPPU", "OPPU"),
    ("TierServe-Planner@LLMS", r"\sysplan"), ("TierServe-Full", r"\sysfull")]
COLORS = ["#333333", "#8C8C8C", "#CC79A7", "#D55E00", "#E69F00", "#56B4E9", "#0072B2"]


def ci(values, seed=20261112):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
    return float(values.mean()), *np.quantile(samples, [.025, .975]).tolist()


def latex_mean_ci(values, places=3):
    m, lo, hi = ci(values)
    return f"{m:+.{places}f} [{lo:+.{places}f}, {hi:+.{places}f}]"


def saving_ci(candidate, baseline, seed=20261112):
    a, b = np.asarray(candidate, dtype=float), np.asarray(baseline, dtype=float)
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, len(a), (10000, len(a)))
    savings = 1-a[ids].mean(axis=1)/b[ids].mean(axis=1)
    return float(1-a.mean()/b.mean()), *np.quantile(savings,[.025,.975]).tolist()


def stream_outputs(receipts):
    out = []
    lines = [r"\begin{table*}[t]", r"\centering", r"\caption{Fresh policy replays: history means, post-eight-turn-warmup margin and all-turn accuracy. Cost estimates prefill plus attributed refresh (kJ/history), excluding decoding and auxiliary work. OPPU shares Lazy's freshly trained adapter sequence and refresh accounting. Intervals resample histories (10,000 paired draws); they do not establish equivalence.}",
        r"\label{tab:author_stream}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
        r"Backbone / histories / turns & Policy & Margin gain & Accuracy & Cost (kJ) & Refreshes \\", r"\midrule"]
    for payload in receipts:
        d = payload["result"]; b = d["backbone"]; users = d["per_user"]
        for i, (method, label) in enumerate(METHODS):
            row = d["summary_policies"][method]
            head = f"{LABELS[b]} / {len(users)} / {d['stream_len']}" if i == 0 else ""
            lines.append(f"{head} & {label} & {row['dm_post_warmup']:+.3f} & {row['acc']:.3f} & {row['total_lifecycle_j_mean']/1000:.2f} & {row['n_flushes_mean']:.1f} \\\\")
        lines.append(r"\midrule")
        values = lambda m,k: [u["policies"][m][k] for u in users]
        a = np.array(values("TierServe-Planner@LLMS", "dm_post_warmup"))
        o = np.array(values("Online-OPPU", "dm_post_warmup"))
        e = np.array(values("Pure-Tier2-Eager", "dm_post_warmup"))
        summary = dict(backbone=b, n_users=len(users), turns=d["stream_len"],
            planner_margin_ci=ci(a), planner_minus_oppu_ci=ci(a-o), planner_minus_eager_ci=ci(a-e),
            full_margin_ci=ci(values("TierServe-Full", "dm_post_warmup")),
            full_minus_oppu_ci=ci(np.array(values("TierServe-Full", "dm_post_warmup"))-o),
            full_minus_planner_ci=ci(np.array(values("TierServe-Full", "dm_post_warmup"))-a),
            planner_user_wins_vs_oppu=int(np.sum(a>o)),
            full_user_wins_vs_oppu=int(np.sum(np.array(values("TierServe-Full", "dm_post_warmup"))>o)),
            planner_estimated_saving_vs_eager_ci=saving_ci(values("TierServe-Planner@LLMS", "total_lifecycle_j"),values("Pure-Tier2-Eager", "total_lifecycle_j")),
            planner_estimated_saving_vs_oppu_ci=saving_ci(values("TierServe-Planner@LLMS", "total_lifecycle_j"),values("Online-OPPU", "total_lifecycle_j")),
            full_estimated_saving_vs_eager_ci=saving_ci(values("TierServe-Full", "total_lifecycle_j"),values("Pure-Tier2-Eager", "total_lifecycle_j")))
        conditional = {}
        cfg = d["slo_config"]
        for method, _ in METHODS:
            reqs = [r for u in users for r in u["policies"][method]["foreground_requests"]]
            conditional[method] = dict(n_requests=len(reqs),
                n_users_with_requests=sum(bool(u["policies"][method]["foreground_requests"]) for u in users),
                pooled_joint_slo_pct=(100*sum(r["ttft_ms"]<=cfg["ttft_slo_ms"] and r["tpot_ms"]<=cfg["tpot_slo_ms"] for r in reqs)/len(reqs) if reqs else None))
        summary["conditional_refresh_slo"] = conditional
        summary["conditional_slo_note"] = "Different refresh durations admit different subsets of independent probe arrivals; not common-trace scheduling evidence. Zero-request cases are undefined, not 100%."
        out.append(summary)
        lines.append(r"\multicolumn{6}{@{}l}{" + f"{LABELS[b]}: Planner margin {latex_mean_ci(a)}; Planner $-$ OPPU {latex_mean_ci(a-o)}" + r"} \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}"]
    (ROOT/"outputs/tables/tab_author_stream.tex").write_text("\n".join(lines)+"\n")
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55))
    x = np.arange(len(receipts)); w = .105
    for j, (method, label) in enumerate(METHODS):
        margin = []; errors = []; cost = []
        for payload in receipts:
            users = payload["result"]["per_user"]
            m, lo, hi = ci([u["policies"][method]["dm_post_warmup"] for u in users])
            margin.append(m); errors.append((m-lo, hi-m))
            row = payload["result"]["summary_policies"]
            cost.append(row[method]["total_lifecycle_j_mean"]/row["Pure-Tier2-Eager"]["total_lifecycle_j_mean"])
        xpos = x + (j-3)*w
        axes[0].bar(xpos, margin, w, color=COLORS[j], label=label.replace(r"\sysplan", "Planner").replace(r"\sysfull", "Full"))
        axes[0].errorbar(xpos, margin, yerr=np.array(errors).T, fmt="none", ecolor="#333333", capsize=1.5, lw=.6)
        axes[1].bar(xpos, cost, w, color=COLORS[j])
    names = [LABELS[p["result"]["backbone"]] for p in receipts]
    for ax in axes:
        ax.set_xticks(x, names); ax.axhline(0, color="#555555", lw=.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Margin gain (history 95% interval)")
    axes[1].set_ylabel("Estimated cost / Eager")
    axes[0].set_title("(a) Preference quality")
    axes[1].set_title("(b) Prefill + refresh cost")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=7, frameon=False)
    fig.tight_layout(rect=(0,0,1,.88))
    save(fig, "author_stream")
    return out


def executor_outputs(receipts):
    out = []
    lines = [r"\begin{table*}[t]", r"\centering", r"\caption{Matched executors: common warm cache and previous adapter, six updates/24 pair exposures, four balanced orders. SLO and empirical P95 are run means; CUDA peak includes snapshots, excludes reservations/OS pages. VIN energy spans the entire replay, excluding common preparation. Quality trims at EOS; each model has eight distinct items from one history. Repeated requests do not enlarge this quality sample.}",
        r"\label{tab:author_executor}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
        r"Backbone & Executor / reader & Runs $\times$ req. & P95 TTFT (ms) & SLO (\%) & CUDA GiB & Energy kJ & R-L (\%) \\", r"\midrule"]
    method_names = ["StepPreempt", "PairPreempt", "AdjointPass-Committed", "AdjointPass-BaseFallback"]
    for payload in receipts:
        backbone = payload["args"]["backbone"]
        for i, method in enumerate(method_names):
            rows = [r for r in payload["runs"] if r["method"] == method]
            mean = lambda fn: float(np.mean([fn(r) for r in rows]))
            requests = [r for run in rows for r in run["requests"]]
            # Use W5 EOS-trimmed quality records, excluding earlier pilot outputs.
            summary = dict(backbone=backbone, method=method, n_runs=len(rows),
                n_requests_per_run=len(rows[0]["requests"]),
                p95_ttft_ms=mean(lambda r:r["metrics"]["ttft_p95_ms"]),
                slo_pct=mean(lambda r:r["metrics"]["slo_joint_attainment_pct"]),
                peak_gib=max(r["execution"]["peak_gb"] for r in rows),
                replay_kj=mean(lambda r:r["execution"]["tegra"]["vin_j"])/1000,
                rouge_l_pct=100*mean(lambda r:r["mean_rouge_l"]),
                during_refresh_requests=sum(bool(r["during_refresh"]) for r in requests),
                publication_s=mean(lambda r:r["execution"]["bg_s"]),
                observed_reader_versions={v:sum(r.get("adapter_version")==v for r in requests) for v in sorted({r.get("adapter_version", "unrecorded") for r in requests})},
                run_metrics=[dict(repetition=r["repetition"],ttft_p95_ms=r["metrics"]["ttft_p95_ms"],slo_pct=r["metrics"]["slo_joint_attainment_pct"],publication_s=r["execution"]["bg_s"],replay_j=r["execution"]["tegra"]["vin_j"]) for r in rows])
            warm = payload["preparation"]["committed_foreground_warmup"]
            old = [r for r in requests if r.get("adapter_version")=="committed-v1"]
            identical = sum(r["generated_ids"]==warm[r["held_index"]]["generated_ids"] for r in old)
            summary["committed_output_audit"] = dict(n_committed_requests=len(old),
                n_exact_raw_greedy_ids_equal_unloaded_committed=identical,
                note="Checks the retained-state service contract over identical fixed prompts; not useful-learning or task-quality confirmation.")
            summary["quality_by_reader"] = {v:dict(n=sum(r.get("adapter_version")==v for r in requests),
                rouge_l_mean=float(np.mean([r["rouge_l"] for r in requests if r.get("adapter_version")==v])),
                distinct_held_items=sorted({r["held_index"] for r in requests if r.get("adapter_version")==v}))
                for v in summary["observed_reader_versions"]}
            out.append(summary)
            label = {"StepPreempt":"Step / committed", "PairPreempt":"Pair / committed", "AdjointPass-Committed":"Branch / committed", "AdjointPass-BaseFallback":"Branch / base fallback"}[method]
            head = LABELS[backbone] if i == 0 else ""
            lines.append(f"{head} & {label} & {len(rows)} $\\times$ {len(rows[0]['requests'])} & {summary['p95_ttft_ms']:.0f} & {summary['slo_pct']:.1f} & {summary['peak_gib']:.2f} & {summary['replay_kj']:.2f} & {summary['rouge_l_pct']:.1f} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}",r"\end{table*}"]
    (ROOT/"outputs/tables/tab_author_executor.tex").write_text("\n".join(lines)+"\n")
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.35))
    palette = ["#E69F00", "#009E73", "#0072B2", "#56B4E9"]
    for model_idx, payload in enumerate(receipts):
        ax=axes[model_idx]
        rows=[r for r in out if r["backbone"]==payload["args"]["backbone"]]
        means=[r["p95_ttft_ms"] for r in rows]
        ax.bar(range(4),means,color=palette)
        lows=[m-min(v["ttft_p95_ms"] for v in row["run_metrics"]) for m,row in zip(means,rows)]
        highs=[max(v["ttft_p95_ms"] for v in row["run_metrics"])-m for m,row in zip(means,rows)]
        ax.errorbar(range(4),means,yerr=[lows,highs],fmt="none",ecolor="#333333",capsize=2,lw=.6)
        ax.set_xticks(range(4),["Step", "Pair", "Branch\ncommitted", "Branch\nbase"])
        ax.set_ylabel("Mean empirical P95 TTFT (ms)")
        ax.set_title(LABELS[payload["args"]["backbone"]])
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig,"author_executor")
    return out


def save(fig, name):
    fig.savefig(ROOT/"outputs/figures"/(name+".pdf"))
    fig.savefig(ROOT/"outputs/figures"/(name+".png"),dpi=200)
    plt.close(fig)


def main():
    (ROOT / "outputs/tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs/figures").mkdir(parents=True, exist_ok=True)
    ap=argparse.ArgumentParser(); ap.add_argument("--allow-partial-cohorts",action="store_true"); args=ap.parse_args()
    plt.rcParams.update({"font.family":"serif", "font.serif":["Times New Roman","STIXGeneral"],
        "font.size":8, "pdf.fonttype":42, "ps.fonttype":42})
    streams=[]; executors=[]; missing=[]
    for name in ["dense_stream_20u_r1.json","mamba_stream_6u_r1.json"]:
        p=INPUT/name
        if not p.exists(): missing.append(name); continue
        d=json.loads(p.read_text())
        if not d.get("ok"): raise ValueError("Incomplete result: "+name)
        expected = d["args"]
        cohort = d["result"]["per_user"]
        if len(cohort) != expected["users"]: raise ValueError("Incomplete user cohort: "+name)
        for user in cohort:
            if len(user["policies"]) != 8: raise ValueError("Incomplete policy set: "+name)
            for record in user["policies"].values():
                if record["n_turns"] != expected["turns"] or any(len(record[k]) != expected["turns"] for k in ["per_turn_dm","per_turn_accuracy","per_turn_read_j"]):
                    raise ValueError("Incomplete actual turn records: "+name)
        streams.append(d)
    for name in ["dense_executor_w5.json","hybrid_executor_w5_16q.json"]:
        p=INPUT/name
        if not p.exists(): missing.append(name); continue
        d=json.loads(p.read_text())
        if not d.get("ok"): raise ValueError("Incomplete result: "+name)
        if len(d["runs"]) != 16: raise ValueError("Incomplete balanced executor runs: "+name)
        for run in d["runs"]:
            meta=run["execution"]
            if len(run["requests"])!=d["args"]["requests"] or meta["optimizer_updates"]!=6 or meta["training_pairs"]!=24 or meta["reference_cache_hits"]!=12 or meta["reference_pairs_computed"]!=0:
                raise ValueError("Unmatched executor work/request/cache contract: "+name)
            if meta["tegra"]["vin_mw_mean"]<=0 or meta["tegra"]["n_samples"]<=0:
                raise ValueError("Missing measured executor power: "+name)
        executors.append(d)
    if missing and not args.allow_partial_cohorts: raise FileNotFoundError("Pending receipts: "+", ".join(missing))
    summary={"pending":missing,
        "analysis_unit":"pseudonymous history for stream; fixed-trace repetition for executor; no equivalence claim"}
    if streams: summary["streams"]=stream_outputs(streams)
    if executors: summary["executors"]=executor_outputs(executors)
    (ROOT/"outputs/stream_executor.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__": main()
