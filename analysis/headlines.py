#!/usr/bin/env python3
"""Build headline-generation tables and item audits from LaMP receipts.

Analyze each model recipe separately. Intervals resample profile groups;
generated-headline ROUGE and synthetic-negative accuracy are separate metrics.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys


PAPER = Path(__file__).resolve().parents[1]
RECIPES = {
    "dense_qwen15b": dict(items_per_group=10, total_items=60, label="Dense 1.5B", short="D", tag="dense"),
    "hybrid_qwen27b": dict(items_per_group=6, total_items=36, label="GDN 27B", short="H", tag="hybrid27b"),
}
METHODS = (
    "Base", "LaMP-BM25@4", "LaMP-Dense@4", "LaMP-Dense@8", "MemGPT-Persona",
    "OPPU-StepPreempt", "RAFT-StepPreempt", "TierServe-Planner-StepPreempt", "TierServe-Full",
)
LABELS = {
    "Base": "Base", "LaMP-BM25@4": "BM25@4", "LaMP-Dense@4": "Dense@4",
    "LaMP-Dense@8": "Dense@8", "MemGPT-Persona": "Persona",
    "OPPU-StepPreempt": "OPPU + Step", "RAFT-StepPreempt": "RAFT + Step",
    "TierServe-Planner-StepPreempt": r"\textsc{TierServe}$_{\rm plan}$",
    "TierServe-Full": r"\textsc{TierServe}$_{\rm full}$",
}
STRICT_METHODS = {"TierServe-Planner-StepPreempt", "TierServe-Full"}
BOOTSTRAPS = 10_000
SEED = 20261112


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, context):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), f"missing/nonfinite numeric value: {context}")
    return float(value)


def count(value, context):
    result = number(value, context)
    require(result >= 0 and result.is_integer(), f"invalid count: {context}")
    return int(result)


def load_cohort(path):
    payload = json.loads(path.read_text())
    require(payload.get("schema") in ("tierserve_author_lamp_v1", "tierserve_author_lamp_l2_v1"),
            "unsupported receipt schema")
    native_l2 = payload.get("schema") == "tierserve_author_lamp_l2_v1"
    require(payload.get("ok") is True, "only a complete ok=true receipt is eligible")
    require(payload.get("protocol") == "one_profile_adaptation_transfer_not_online_RentFlush",
            "unexpected task/service protocol")
    backbone = payload.get("args", {}).get("backbone")
    require(backbone in RECIPES, "only the frozen Dense and Hybrid recipes are supported")
    recipe = RECIPES[backbone]
    per_group, total = recipe["items_per_group"], recipe["total_items"]
    require(payload["args"].get("authors") == 6 and payload.get("expected_authors") == 6,
            "the frozen recipe requires six profile groups")
    require(payload.get("expected_queries") == total, "declared query count differs from backbone recipe")
    for name in ("source_sha256", "data_sha256", "model_config_sha256"):
        hashes = payload.get(name, {})
        require(bool(hashes) and all(isinstance(v, str) and len(v) == 64
                and all(c in "0123456789abcdef" for c in v) for v in hashes.values()),
                f"missing/invalid {name} manifest")
    result = payload.get("result", {})
    if native_l2:
        recipe_record = result.get("recipe", {})
        require(recipe_record.get("task_prompt_protocol") == "native_chat_enable_thinking_false"
                and recipe_record.get("calibration_and_embeddings") == "unchanged_raw_tokenizer",
                "missing native-task/raw-calibration protocol boundary")
    require(result.get("backbone") == backbone, "result/argument backbone mismatch")
    rows = result.get("per_author", [])
    require(len(rows) == 6 and result.get("n_authors") == 6, "exactly six completed profile groups are required")
    require(result.get("n_queries") == total, f"the {backbone} recipe requires exactly {total} items")
    ids = [row.get("author_id") for row in rows]
    require(all(isinstance(i, str) and i for i in ids) and len(set(ids)) == 6, "duplicate/missing profile-group ID")
    aligned = {}
    for row in rows:
        aid = row["author_id"]
        require(set(row.get("methods", {})) == set(METHODS), f"method set mismatch for {aid}")
        require(row.get("n_queries") == per_group, f"expected {per_group} queries for profile group {aid}")
        aligned[aid] = {}
        for method in METHODS:
            record = row["methods"][method]
            items = record.get("per_item", [])
            require(len(items) == per_group and record.get("n_queries") == per_group,
                    f"incomplete per-item records for {aid}/{method}")
            by_id = {}
            for item in items:
                sid = item.get("source_id")
                require(isinstance(sid, str) and sid and sid not in by_id,
                        f"duplicate/missing item ID for {aid}/{method}")
                for field in ("gold", "negative", "prediction", "raw_generation", "prompt",
                              "gold_source_id", "negative_source_id", "negative_construction"):
                    require(isinstance(item.get(field), str), f"missing {field}: {aid}/{method}/{sid}")
                for metric in ("rouge_1", "rouge_l", "acc"):
                    value = number(item.get(metric), f"{aid}/{method}/{sid}/{metric}")
                    require(0 <= value <= 1, f"metric outside [0,1]: {metric}")
                require(item["acc"] in (0, 1), f"nonbinary accuracy: {aid}/{method}/{sid}")
                number(item.get("margin"), f"{aid}/{method}/{sid}/margin")
                rendered = count(item.get("rendered_prompt_tokens"), "rendered prompt tokens")
                effective = count(item.get("prompt_tokens"), "effective prompt tokens")
                require(effective <= rendered, f"effective prompt exceeds rendered prompt: {aid}/{method}/{sid}")
                count(item.get("requested_k"), "requested k")
                count(item.get("actual_k"), "actual k")
                require(isinstance(item.get("retrieved_source_ids"), list), "missing retrieved source IDs")
                if native_l2:
                    native = item.get("native_input", {})
                    generation = item.get("generation_audit", {})
                    native_text = native.get("native_prompt")
                    ids = native.get("encoded_prompt_ids")
                    require(isinstance(native_text, str) and item["prompt"] in native_text
                            and "</think>" in native_text, "missing actual native input text")
                    require(hashlib.sha256(native_text.encode()).hexdigest()
                            == native.get("native_prompt_sha256"), "native text hash mismatch")
                    require(isinstance(ids, list) and len(ids) == effective
                            and all(isinstance(i, int) and i >= 0 for i in ids), "native input ID mismatch")
                    require(hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
                            == native.get("encoded_prompt_sha256"), "native encoded hash mismatch")
                    require(generation.get("encoded_prompt_ids") == ids,
                            "actual generation differs from native input audit")
                    emitted, generated = generation.get("emitted_ids"), generation.get("raw_generated_ids")
                    require(isinstance(emitted, list) and 1 <= len(emitted) <= 32
                            and isinstance(generated, list), "missing bounded actual emissions")
                    stop = generation.get("stop_reason")
                    require((stop == "max_new_tokens" and len(emitted) == 32 and emitted == generated)
                            or (stop == "eos" and emitted[:-1] == generated), "invalid EOS/cap stop record")
                    require(generation.get("raw_generation_with_special_tokens", "").strip()
                            == item["raw_generation"], "raw generation text mismatch")
                    require(item.get("empty_prediction") == (not bool(item["prediction"])),
                            "empty-answer flag mismatch")
                    require(item.get("format_failure") in (None, "unclosed_think", "empty_or_preamble_only"),
                            "unknown format-failure outcome")
                    require(item.get("format_failure") is None or not item["prediction"],
                            "format failure must retain an empty scored answer")
                by_id[sid] = item
            aligned[aid][method] = by_id
        base = aligned[aid]["Base"]
        for method in METHODS:
            require(set(aligned[aid][method]) == set(base), f"unpaired item IDs: {aid}/{method}")
            for sid, item in aligned[aid][method].items():
                for field in ("gold", "negative", "gold_source_id", "negative_source_id"):
                    require(item[field] == base[sid][field], f"changed label/source: {aid}/{method}/{sid}/{field}")
                added = max(0, item["rendered_prompt_tokens"] - base[sid]["rendered_prompt_tokens"])
                if method in STRICT_METHODS:
                    require(item.get("budget_semantics") == "exact_added_tokens_vs_bare",
                            f"missing strict added-token semantics: {aid}/{method}/{sid}")
                    require(count(item.get("base_prompt_tokens"), "base prompt tokens")
                            == base[sid]["rendered_prompt_tokens"], "strict builder used a different bare prompt")
                    require(count(item.get("added_prompt_tokens"), "added prompt tokens") == added,
                            f"stored/recomputed added-token mismatch: {aid}/{method}/{sid}")
                    budget = count(item.get("added_token_budget"), "added-token budget")
                    require(added <= budget, f"added-token cap violated: {aid}/{method}/{sid}: {added}>{budget}")
                    require(item["actual_k"] <= item["requested_k"]
                            and item["actual_k"] == len(item["retrieved_source_ids"]),
                            f"admitted card-count mismatch: {aid}/{method}/{sid}")
    return payload, aligned


def percentile(sorted_values, probability):
    position = (len(sorted_values) - 1) * probability
    lo = math.floor(position)
    hi = math.ceil(position)
    return sorted_values[lo] + (position - lo) * (sorted_values[hi] - sorted_values[lo])


def interval(values, samples):
    draws = sorted(statistics.fmean(values[i] for i in indices) for indices in samples)
    return dict(mean=statistics.fmean(values), ci95_lo=percentile(draws, .025),
                ci95_hi=percentile(draws, .975), sd=statistics.stdev(values),
                minimum=min(values), maximum=max(values), profile_group_values=list(values))


def analyze(payload, aligned, generation_reference):
    group_ids = list(aligned)
    backbone = payload["args"]["backbone"]
    recipe = RECIPES[backbone]
    rng = random.Random(SEED)
    samples = [tuple(rng.randrange(6) for _ in range(6)) for _ in range(BOOTSTRAPS)]
    vectors, audit = {}, []
    for method in METHODS:
        vectors[method] = {metric: [] for metric in ("rouge_1", "rouge_l", "margin", "acc",
                                                   "rendered_tokens", "effective_tokens", "added_tokens")}
        for aid in group_ids:
            items = aligned[aid][method]
            base = aligned[aid]["Base"]
            for metric in ("rouge_1", "rouge_l", "margin", "acc"):
                scale = 100 if metric in ("rouge_1", "rouge_l", "acc") else 1
                vectors[method][metric].append(scale * statistics.fmean(item[metric] for item in items.values()))
            added_values = []
            for sid, item in items.items():
                added = max(0, item["rendered_prompt_tokens"] - base[sid]["rendered_prompt_tokens"])
                added_values.append(added)
                audit.append(dict(profile_group_id=aid, method=method, source_id=sid,
                    prediction=item["prediction"], gold=item["gold"], negative=item["negative"],
                    retrieval_mode=item.get("retrieval_mode", "unspecified"), requested_k=item["requested_k"],
                    actual_k=item["actual_k"], base_rendered_tokens=base[sid]["rendered_prompt_tokens"],
                    rendered_tokens=item["rendered_prompt_tokens"], effective_tokens=item["prompt_tokens"],
                    added_rendered_tokens=added, added_token_budget=item.get("added_token_budget"),
                    truncated=item["prompt_tokens"] < item["rendered_prompt_tokens"],
                    rouge_1=item["rouge_1"], rouge_l=item["rouge_l"], margin=item["margin"], acc=item["acc"],
                    format_failure=item.get("format_failure"), empty_prediction=not bool(item["prediction"]),
                    literal_think=item["prediction"].strip() == "<think>",
                    native_prompt_sha256=item.get("native_input", {}).get("native_prompt_sha256"),
                    native_encoded_sha256=item.get("native_input", {}).get("encoded_prompt_sha256"),
                    stop_reason=item.get("generation_audit", {}).get("stop_reason")))
            vectors[method]["rendered_tokens"].append(statistics.fmean(i["rendered_prompt_tokens"] for i in items.values()))
            vectors[method]["effective_tokens"].append(statistics.fmean(i["prompt_tokens"] for i in items.values()))
            vectors[method]["added_tokens"].append(statistics.fmean(added_values))
    summaries = {}
    for method in METHODS:
        summary = {metric: interval(values, samples) for metric, values in vectors[method].items()}
        summary["paired_deltas"] = {}
        for metric, reference in (("rouge_l", generation_reference), ("margin", "Base"), ("acc", "Base")):
            differences = [a-b for a,b in zip(vectors[method][metric], vectors[reference][metric])]
            summary["paired_deltas"][metric] = dict(interval(differences, samples), reference=reference)
        entries = [entry for entry in audit if entry["method"] == method]
        summary["format_outcomes"] = dict(n_items=len(entries),
            empty_predictions=sum(e["empty_prediction"] for e in entries),
            literal_think_predictions=sum(e["literal_think"] for e in entries),
            failure_reasons=dict(Counter(e["format_failure"] for e in entries if e["format_failure"])),
            stop_reasons=dict(Counter(e["stop_reason"] for e in entries if e["stop_reason"])))
        summary["token_audit"] = dict(added_min=min(e["added_rendered_tokens"] for e in entries),
            added_max=max(e["added_rendered_tokens"] for e in entries), truncated_items=sum(e["truncated"] for e in entries),
            modes=dict(sorted(Counter(e["retrieval_mode"] for e in entries).items())),
            requested_k_counts=dict(sorted(Counter(e["requested_k"] for e in entries).items())),
            admitted_k_counts=dict(sorted(Counter(e["actual_k"] for e in entries).items())),
            budgets=sorted({e["added_token_budget"] for e in entries if e["added_token_budget"] is not None}),
            strict_cap_violations=0 if method in STRICT_METHODS else None)
        summaries[method] = summary
    return dict(schema="tierserve_profile_group_lamp_analysis_v2", backbone=backbone,
        n_profile_groups=6, n_items=recipe["total_items"], items_per_profile_group=recipe["items_per_group"],
        profile_group_ids=group_ids, methods=summaries, item_audit=audit,
        group_identity="prepared author_id profile packet; two-headline-fingerprint deduplication; not authenticated individual",
        sampling=dict(unit="profile_group", estimand="equal-profile-group mean and paired equal-profile-group difference",
                      resamples=BOOTSTRAPS, seed=SEED, interval="pointwise 95% percentile; linear quantiles",
                      p_values=False, simultaneous_intervals=False),
        generation_reference=generation_reference, preference_reference="Base",
        protocol=payload.get("protocol"), receipt_schema=payload.get("schema"),
        task_prompt_protocol=payload.get("result", {}).get("recipe", {}).get("task_prompt_protocol", "plain_article_headline_l1"),
        foreground_adapter=payload.get("foreground_adapter"),
        source_sha256=payload["source_sha256"], data_sha256=payload["data_sha256"],
        model_config_sha256=payload["model_config_sha256"])


def tex_escape(value):
    mapping = {"\\": r"\textbackslash{}", "_": r"\_", "&": r"\&", "%": r"\%",
               "#": r"\#", "{": r"\{", "}": r"\}"}
    return "".join(mapping.get(c, c) for c in str(value))


def formatted(stat, digits=1, signed=False):
    def render(value):
        if abs(value) < .5 * 10**(-digits):
            value = 0.
        return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return f"{render(stat['mean'])} [{render(stat['ci95_lo'])}, {render(stat['ci95_hi'])}]"


def table(caption, label, header, rows, columns):
    return "\n".join([r"\begin{table*}[t]", r"\centering", r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}", r"\caption{"+caption+"}", r"\label{"+label+"}",
        r"\begin{tabular}{@{}"+columns+r"@{}}", r"\toprule", header+r" \\", r"\midrule",
        *[row+r" \\" for row in rows], r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])


def render_tables(analysis):
    summaries = analysis["methods"]
    reference = LABELS[analysis["generation_reference"]]
    recipe = RECIPES[analysis["backbone"]]
    common = (recipe["label"]+f", six profile groups and {analysis['n_items']} held-out headlines; "
              +r"brackets are pointwise 95\% intervals from 10,000 paired profile-group resamples. ")
    generation = table(
        "Generated-headline quality after one profile adaptation. "+common
        +r"ROUGE values are percentages; $\Delta$ROUGE-L is in percentage points relative to "+reference
        +r". These are generated-output metrics, separate from the constructed preference proxy.",
        "tab:author_lamp_generation_"+recipe["tag"], r"Method & ROUGE-1 & ROUGE-L & $\Delta$ROUGE-L", [
            LABELS[m]+" & "+formatted(summaries[m]["rouge_1"])+" & "+formatted(summaries[m]["rouge_l"])
            +" & "+formatted(summaries[m]["paired_deltas"]["rouge_l"], signed=True) for m in METHODS], "lccc")
    preference = table(
        "Constructed preference proxy, reported separately from headline generation. "+common
        +r"Margin is the mean-token log-likelihood difference between gold and synthetic negative headlines. "
        +r"$\Delta$m and $\Delta$accuracy are paired differences from Base; accuracy and its difference are in percent and percentage points. "
        +r"Synthetic negatives are deterministic lead-sentence or generic headlines, not authentic user rejections.",
        "tab:author_lamp_preference_"+recipe["tag"], r"Method & Raw margin & $\Delta$m vs Base & Accuracy (\%) & $\Delta$accuracy", [
            LABELS[m]+" & "+formatted(summaries[m]["margin"], 3)+" & "
            +formatted(summaries[m]["paired_deltas"]["margin"], 3, True)+" & "+formatted(summaries[m]["acc"])
            +" & "+formatted(summaries[m]["paired_deltas"]["acc"], signed=True) for m in METHODS], "lcccc")
    budget_rows = []
    for method in METHODS:
        summary, audit = summaries[method], summaries[method]["token_audit"]
        modes = ",".join(audit["modes"])
        cap = ",".join(str(v) for v in audit["budgets"]) or "---"
        requested = ",".join(str(v) for v in audit["requested_k_counts"])
        admitted = ",".join(str(v) for v in audit["admitted_k_counts"])
        budget_rows.append(LABELS[method]+" & "+tex_escape(modes)+" & "+requested+"/"+admitted
            +f" & {summary['rendered_tokens']['mean']:.1f}/{summary['effective_tokens']['mean']:.1f}"
            +f" & {summary['added_tokens']['mean']:.1f} [{audit['added_min']}, {audit['added_max']}]"
            +" & "+cap+f" & {audit['truncated_items']}")
    budgets = table(
        f"{recipe['label']} prompt audit over the same {analysis['n_items']} items. "
        +r"Prompt values are mean rendered/effective generation tokens; "
        +r"added tokens are exact recorded rendered-token differences from the matching Base prompt, with item minimum/maximum. "
        +r"The strict cap applies only to TierServe added context, not the bare article or the fixed-retrieval controls. "
        +r"Requested/admitted $k$ lists observed values; persona is a profile summary rather than retrieved cards. "
        +r"Truncated is the number of generation prompts shorter than their rendered form. All strict-cap records passed admission checks.",
        "tab:author_lamp_budget_"+recipe["tag"], r"Method & Mode(s) & Req./adm. $k$ & Prompt R/E & Added [min,max] & Cap & Truncated",
        budget_rows, "llccccc")
    return {"tab_author_lamp_generation.tex": generation, "tab_author_lamp_preference.tex": preference,
            "tab_author_lamp_budget.tex": budgets}


def compact_table(analyses):
    """Render the compact quality table; detailed files retain the paired differences."""
    rows=[]
    for method in METHODS:
        cells=[LABELS[method]]
        for analysis in analyses:
            summary=analysis['methods'][method]
            cells.extend([formatted(summary['rouge_l']), f"{summary['acc']['mean']:.1f}"])
        rows.append(" & ".join(cells))
    cohorts="; ".join(RECIPES[a['backbone']]['short']+": "+RECIPES[a['backbone']]['label']+f", 6 profile groups/{a['n_items']} items" for a in analyses)
    header='Method'+''.join(' & '+RECIPES[a['backbone']]['short']+r' ROUGE-L (\%) [95\% CI] & Acc. (\%)' for a in analyses)
    protocol_note = ""
    if any(a["receipt_schema"] == "tierserve_author_lamp_l2_v1" for a in analyses):
        protocol_note = ("D uses original task prompts; H uses native non-thinking task prompts with raw synthetic calibration. "
                         "The failed H first-line decoder is preserved separately, without a causal version contrast. ")
    elif any(a["backbone"] == "hybrid_qwen27b" for a in analyses):
        protocol_note = (r"H retains a failed first-line decoder that stops on \texttt{<think>}; "
                         "its raw scores are not a clean maintenance-policy comparison. ")
    return table(
        "Generated headlines after single-profile adaptation ("+cohorts+r"). Backbones are analyzed separately. "
        +protocol_note
        +r"Pointwise intervals use 10,000 paired profile-group resamples. Accuracy compares gold with a constructed negative; "
        +r"this is not an online RentFlush or official leaderboard evaluation. Detailed paired differences are in the artifact.",
        "tab:author_lamp_compact",header,rows,'l'+'cc'*len(analyses))


def write_analysis(analysis, output_dir):
    artifacts = render_tables(analysis)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in artifacts.items():
        (output_dir/name).write_text(content)
    (output_dir/"author_lamp_analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False)+"\n")
    with (output_dir/"author_lamp_item_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(analysis["item_audit"][0]))
        writer.writeheader()
        writer.writerows(analysis["item_audit"])


def main():
    (PAPER / "outputs/tables").mkdir(parents=True, exist_ok=True)
    (PAPER / "outputs/figures").mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, action="append",
                        help="Completed Dense or Hybrid receipt; repeat once to include both")
    parser.add_argument("--output-dir", type=Path, default=PAPER/"outputs/tables/author_lamp_fresh_l1")
    parser.add_argument("--generation-reference", choices=["LaMP-Dense@4"], default="LaMP-Dense@4",
                        help="Fixed generation contrast; preference contrasts use Base")
    args = parser.parse_args()
    receipts = args.receipt or [PAPER/"results/dense_lamp_fresh_l1.json"]
    require(1 <= len(receipts) <= 2, "provide one recipe or the two distinct frozen recipes")
    analyses = []
    for receipt in receipts:
        payload, aligned = load_cohort(receipt)
        require(payload["args"]["backbone"] not in {a["backbone"] for a in analyses},
                "duplicate backbone receipts are not independent cohorts")
        analysis = analyze(payload, aligned, args.generation_reference)
        analysis.update(receipt_path=str(receipt.resolve()),
                        receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
                        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        python_version=sys.version)
        analyses.append(analysis)
    analyses.sort(key=lambda a: list(RECIPES).index(a["backbone"]))
    # Validate all receipts before writing outputs.
    for analysis in analyses:
        directory = args.output_dir if len(analyses) == 1 else args.output_dir/RECIPES[analysis["backbone"]]["tag"]
        write_analysis(analysis, directory)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"tab_author_lamp_compact.tex").write_text(compact_table(analyses))
    print(f"Rendered {len(analyses)} separate frozen recipe(s), detailed artifacts, and one 9-method compact table into {args.output_dir}")


if __name__ == "__main__":
    main()
