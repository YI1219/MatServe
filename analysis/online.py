#!/usr/bin/env python3
"""Summarize online replay quality, timing and board energy.

Checks record alignment, event ordering and raw power assets.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys

PAPER=Path(__file__).resolve().parents[1]
METHODS=("RAG@4","RAG@8","Fixed16-Pair","RentKnee-Pair","RentKnee-Branch")
CONTRASTS=("Fixed16-Pair","RAG@8","RentKnee-Pair","RAG@4")
SEED=20261112
NBOOT=10_000


def check(condition,message):
    if not condition: raise ValueError(message)


def finite(value):
    check(isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value),"nonfinite/nonnumeric measurement")
    return float(value)


def near(a,b,message,tol=1e-6):
    # Use absolute tolerances for monotonic timestamps.
    check(math.isclose(finite(a),finite(b),rel_tol=0.,abs_tol=tol),message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ids(values,cap=None):
    check(isinstance(values,list) and all(type(i) is int and i>=0 for i in values),"invalid token/source index array")
    if cap is not None: check(0<len(values)<=cap,"encoded token cap/empty input violation")
    return values


def rouge_l(pred,gold):
    a=re.findall(r"[a-z0-9]+",pred.lower())[:256]
    b=re.findall(r"[a-z0-9]+",gold.lower())[:256]
    if not a or not b: return 0.
    dp=[0]*(len(b)+1)
    for token in a:
        diagonal=0
        for j,other in enumerate(b,1):
            previous=dp[j]
            dp[j]=diagonal+1 if token==other else max(dp[j],dp[j-1])
            diagonal=previous
    return 2.*dp[-1]/(len(a)+len(b))


def integrate(samples,start,stop):
    if len(samples)<2 or samples[0][0]>start or samples[-1][0]<stop: return None
    total=0.
    for (a,p),(b,q) in zip(samples,samples[1:]):
        lo,hi=max(a,start),min(b,stop)
        if hi<=lo: continue
        left=p+(q-p)*(lo-a)/(b-a); right=p+(q-p)*(hi-a)/(b-a)
        total+=(hi-lo)*(left+right)/2.
    return total


class AssetAudit:
    def __init__(self,directory):
        self.directory=directory; self.warnings=[]; self.records=[]

    def energy(self,record):
        check(isinstance(record,dict) and record.get("unit")=="uW","missing/wrong INA238 record units")
        path=self.directory/Path(record["csv_path"]).name
        if not path.is_file():
            self.warnings.append(f"Missing CSV: {path.name}; energy is null, never zero")
            self.records.append(dict(asset=path.name,verified=False,energy_j=None))
            return None,[]
        check(sha(path)==record["csv_sha256"],f"CSV SHA mismatch: {path.name}")
        samples=[]; times=[]; errors=[]
        with path.open(newline="") as handle:
            reader=csv.DictReader(handle)
            check(reader.fieldnames==["monotonic_s","power_uW","power_W","error"],"unexpected VIN CSV columns")
            for row in reader:
                t=finite(float(row["monotonic_s"])); times.append(t)
                if row["error"]:
                    errors.append(row["error"]); continue
                raw=int(row["power_uW"]); watts=finite(float(row["power_W"]))
                check(0<watts<=1000 and raw>0,"invalid VIN power range")
                near(watts,raw/1e6,"uW/W conversion mismatch",1e-10)
                samples.append((t,watts))
        check(all(b>=a for a,b in zip(times,times[1:])),"nonmonotonic VIN sample timestamps")
        check(all(b[0]>a[0] for a,b in zip(samples,samples[1:])),"duplicate/reversed successful VIN timestamp")
        check(len(samples)==record["samples"],"successful VIN sample count mismatch")
        if record.get("errors") or errors:
            check(record.get("energy_j") is None,"failed sensor assigned nonnull energy")
            energy=None
            self.warnings.append(f"Sensor errors in {path.name}; complete-method energy remains null")
        else:
            check(samples,"empty successful VIN trace")
            near(record["start_monotonic_s"],samples[0][0],"VIN start mismatch")
            near(record["stop_monotonic_s"],samples[-1][0],"VIN stop mismatch")
            energy=integrate(samples,samples[0][0],samples[-1][0])
            check(energy is not None and energy>0,"uncovered/zero replay energy")
            near(record["energy_j"],energy,"VIN trapezoid/recorded-J mismatch")
        self.records.append(dict(asset=path.name,verified=True,sha256=sha(path),samples=len(samples),
                                 energy_j=energy,sensor_errors=record.get("errors",[])))
        return energy,samples

    def journal(self,payload):
        record=payload["event_journal"]
        path=self.directory/Path(record["path"]).name
        check(path.is_file(),"mandatory append-only journal asset missing")
        check(sha(path)==record["sha256"],"event journal SHA mismatch")
        events=[json.loads(line) for line in path.read_text().splitlines()]
        check(len(events)==record["records"] and events==payload["events"],"journal count/event equality mismatch")
        stamps=[finite(e["monotonic_s"]) for e in events]
        check(all(b>=a for a,b in zip(stamps,stamps[1:])),"nonmonotonic journal")
        self.records.append(dict(asset=path.name,verified=True,sha256=sha(path),records=len(events)))
        return events


def source_and_data(payload,smoke):
    check(payload.get("schema")=="tierserve_online_oracle_lamp_v1" and payload.get("ok") is True,
          "receipt is not complete ok=true online-oracle v1")
    groups=1 if smoke else 6
    args=payload["args"]
    check((args["groups"],args["queries"],args["seed"])==(groups,24,SEED),"undeclared/partial cohort recipe")
    check(list(payload["methods"])==list(METHODS),"five-method registry/order mismatch")
    directory=PAPER/"data/lamp4_time_20u"
    manifest=json.loads((directory/"_manifest.json").read_text())[6:6+groups]
    paths=[directory/(r["author_id"]+".json") for r in manifest]
    hashes={p.name:sha(p) for p in [directory/"_manifest.json",*paths]}
    check(hashes==payload["data_sha256"],"prepared-data hash mismatch")
    check(payload.get("analysis_unit")=="profile_group","wrong statistical unit")
    check(payload.get("oracle")=="gold_headline_oracle_correction_not_human_preference","wrong oracle construct")
    expected={}
    for entry,path in zip(manifest,paths):
        packet=json.loads(path.read_text())
        items=sorted(enumerate(packet["history_pairs"]),key=lambda x:(x[1]["date"],x[0]))[:24]
        expected[entry["author_id"]]=[dict(source_id=f"{entry['author_id']}:history:{item.get('ep_idx',i)}",
            article=item["article"],gold=item["pos"].strip(),date=item["date"]) for i,item in items]
    check(list(expected)==payload["profile_groups"],"profile group/order mismatch")
    check(payload.get("model_sidecar_sha256") and all(re.fullmatch(r"[0-9a-f]{64}",x)
          for x in payload["model_sidecar_sha256"].values()),"model sidecar hashes missing")
    recipe=payload["learning_recipe"]
    for key,value in dict(objective="DPO",beta=.1,sft_alpha=.45,lora_r=8,lora_alpha=16,updates=7,
        grad_accum=4,lr=7.5e-5,prompt_cap=512,completion_cap=64,generation_tokens=32,
        generation_prompt_cap=1792,dropout=0.,rematerialize_each_refresh=True).items():
        check(recipe.get(key)==value,f"learning recipe mismatch: {key}")
    return expected


def query_metrics(q,payload):
    content=ids(q["generated_ids"]); emitted=ids(q["all_emitted_ids"],32)
    check(emitted[:len(content)]==content and len(emitted)-len(content) in (0,1),"content/EOS IDs inconsistent")
    check(len(emitted)==32 or len(emitted)==len(content)+1,"early generation stopped without terminal EOS")
    stamps=[finite(t) for t in q["token_monotonic_s"]]
    check(len(stamps)==len(emitted)==q["emitted_token_count"],"emission timestamp/count mismatch")
    check(all(b>=a for a,b in zip(stamps,stamps[1:])),"nonmonotonic token timestamps")
    near(q["first_token_monotonic_s"],stamps[0],"first token mismatch")
    check(q["generation_start_monotonic_s"]<=stamps[0]<=stamps[-1]<=q["response_monotonic_s"]<=q["feedback_monotonic_s"],
          "generation/response/feedback order violation")
    check(q["routing_complete_monotonic_s"]<=q["generation_start_monotonic_s"],"routing timestamp after generation start")
    check(q["inter_token_intervals"]==len(emitted)-1,"EOS interval denominator mismatch")
    ttft=1000*(stamps[0]-finite(q["arrival_monotonic_s"]))
    tpot=1000*(stamps[-1]-stamps[0])/(len(stamps)-1) if len(stamps)>1 else 0.
    near(q["ttft_ms"],ttft,"TTFT recomputation mismatch")
    near(q["tpot_ms"],tpot,"TPOT recomputation mismatch")
    for field in ("ttft_budget_ms","tpot_budget_ms"): near(q[field],payload[field],"SLO budget mismatch")
    slo=ttft<=payload["ttft_budget_ms"] and tpot<=payload["tpot_budget_ms"]
    check(q["joint_slo"] is slo,"joint SLO mismatch")
    check(q["prediction"]==q["raw_prediction"].strip(),"visible prediction mismatch")
    tokens=ids(q["prompt_input_ids"],1792)
    check(len(tokens)==q["prompt_tokens"] and q["prompt_cap"]==1792,"actual prompt cap mismatch")
    check(q["added_tokens"]==max(0,len(tokens)-q["base_prompt_tokens"]),"exact added-token arithmetic mismatch")
    if q["added_token_budget"] is not None:
        check(q["added_tokens"]<=q["added_token_budget"]<=1408,"added-token cap violation")
    rl=rouge_l(q["prediction"],q["gold"])
    near(q["feedback"]["rouge_l"],rl,"independent regex-256 ROUGE-L mismatch")
    return dict(rouge_l=100*rl,slo=100*int(slo),ttft_ms=ttft,tpot_ms=tpot,joint_supplement=100*rl*int(slo))


def split_events(events,runs):
    chunks=[]; current=[]
    for event in events:
        current.append(event)
        if event["kind"]=="method_complete": chunks.append(current); current=[]
    check(not current and len(chunks)==len(runs),"incomplete method boundaries in journal")
    for chunk,run in zip(chunks,runs):
        tail=chunk[-1]
        check((tail["profile_group_id"],tail["method"])==(run["profile_group_id"],run["method"]),"journal/run boundary mismatch")
    return chunks


def validate_run(run,events,expected,payload,samples):
    method=run["method"]; result=run["result"]
    check(run.get("complete") is True and result["n_queries"]==24 and result["method"]==method,"incomplete run")
    queries=result["queries"]; check(len(queries)==24,"missing actual queries")
    offsets=result["arrival_offsets_s"]
    check(len(offsets)==24 and all(b>a for a,b in zip(offsets,offsets[1:])),"invalid arrival trace")
    anchor=queries[0]["arrival_monotonic_s"]-offsets[0]
    metrics=[]
    for index,(q,item) in enumerate(zip(queries,expected)):
        for field in ("source_id","article","gold","date"): check(q[field]==item[field],f"query/{field} differs from prepared episode")
        check(q["query_index"]==q["source_frontier_before_response"]==index,"query/source frontier mismatch")
        near(q["arrival_monotonic_s"],anchor+offsets[index],"global arrival timestamp/offset mismatch")
        metrics.append(query_metrics(q,payload))
    near(result["actual_arrival_rate_qps"],23/(offsets[-1]-offsets[0]),"realized arrival rate mismatch")
    near(result["realized_offered_load"],result["actual_arrival_rate_qps"]/payload["service_capacity_qps"],"realized load mismatch")
    peak=finite(result["peak_cuda_gib"]); check(peak<8,"declared peak budget violated")
    edits=[]; phi=0.; version=0; frontier=0; edit_frontier=0
    ticket=None; published=False; observed_queries=[]; publications=[]; refreshes=[]; pending_generation=None
    write_threshold=max(1e-6,28*payload["specprobe"]["write_joule_per_pair"])
    phases=Counter(); eos_ids=set(); read_versions=defaultdict(list)
    for event in events:
        kind=event["kind"]; phases[kind]+=1
        check(event.get("method")==method,"cross-method event contamination")
        if kind=="generation_complete":
            check(pending_generation is None,"new generation before previous feedback")
            index=len(observed_queries); check(index<24,"extra generated request")
            q=queries[index]; pending_generation=event
            check(event["source_id"]==q["source_id"] and event["source_frontier"]==index,"generation event/source mismatch")
            check(event["committed_frontier"]==frontier,"generation read stale/partial frontier")
            check("gold" not in event and "feedback" not in event,"oracle target leaked into generation record")
            for key in ("prompt","prediction","raw_prediction","generated_ids","all_emitted_ids","token_monotonic_s","adapter_read_version"):
                check(event[key]==q[key],"generation/feedback record mismatch: "+key)
            check(event["monotonic_s"]>=q["response_monotonic_s"],"generation log predates actual response")
            candidates=[e for e in edits if not method.startswith("Rent") or e["index"]>=frontier]
            available=[e["source_id"] for e in candidates]
            check(q["available_source_ids"]==available,"routing pool contains noncausal/wrong-frontier feedback")
            selected=q["selected_source_ids"]; requested=q["requested_source_ids"]
            check(len(set(selected))==len(selected) and set(selected)<=set(requested)<=set(available),"selected/requested IDs are not past feedback")
            check(len(set(requested))==len(requested) and len(requested)<=(8 if method=="RAG@8" else 4),"requested retrieval count/order invalid")
            declared_budget=min(1408,max(0,int(payload["specprobe"]["l_crit_tokens"]))) if method.startswith("Rent") else None
            check(q["added_token_budget"]==declared_budget,"routing budget differs from frozen variant")
            check(set(requested)-set(selected)==set(q["budget_excluded_source_ids"])|set(q["prompt_cap_excluded_source_ids"]),"capacity exclusion IDs mismatch")
            lookup={e["source_id"]:e for e in edits}
            bare="Write one concise news headline.\nArticle: "+q["article"][:750]+"\nHeadline:"
            cards=["Article: "+lookup[s]["article"][:240]+"\nCorrected headline: "+lookup[s]["pos"] for s in reversed(selected)]
            rendered="Previous oracle corrections:\n"+"\n\n".join(cards)+"\n\n"+bare if cards else bare
            check(rendered==q["prompt"],"prompt differs from selected causal correction cards")
            expected_version="base" if method.startswith("RAG") else f"committed-v{version}"
            check(q["adapter_read_version"]==expected_version,"request did not read the published/committed version")
            check(q["committed_frontier_at_read"]==frontier and q["committed_edit_frontier_at_read"]==edit_frontier,"read frontier metadata mismatch")
            if len(q["all_emitted_ids"])>len(q["generated_ids"]): eos_ids.add(q["all_emitted_ids"][-1])
            read_versions[expected_version].append(metrics[index]["rouge_l"])
        elif kind=="oracle_feedback":
            check(pending_generation is not None,"feedback without actual generation")
            q=event["record"]; check(q==queries[len(observed_queries)],"run/query-event mismatch")
            check(q["feedback_monotonic_s"]>=pending_generation["monotonic_s"] and event["monotonic_s"]>=q["feedback_monotonic_s"],"oracle feedback precedes response logging")
            fb=q["feedback"]; edited=q["prediction"]!=q["gold"]
            check(fb["edited"] is edited and fb["actual_prediction"]==q["prediction"],"on-policy feedback mismatch")
            negative=q["prediction"] or "[EMPTY_HEADLINE]"
            check(fb["negative_text"]==negative and fb["empty_prediction"] is (not bool(q["prediction"])),"empty/on-policy negative mismatch")
            check(fb["negative_is_synthetic_empty_marker"] is (not bool(q["prediction"])),"synthetic marker flag mismatch")
            suffix=":actual_generation" if q["prediction"] else ":synthetic_empty_marker"
            check(fb["negative_source_id"]==q["source_id"]+suffix,"negative provenance mismatch")
            check(fb["gold_source_id"]==q["source_id"]+":gold","oracle gold provenance mismatch")
            if edited: edits.append(dict(source_id=q["source_id"],index=len(observed_queries),article=q["article"],pos=q["gold"],neg=negative,query=q))
            charge=finite(fb["maintenance_charge_j"]); check(charge>=0,"negative maintenance potential increment")
            phi+=charge; near(q["phi_after_feedback"],phi,"feedback Phi accounting mismatch")
            observed_queries.append(q); pending_generation=None
        elif kind=="refresh_start":
            check(ticket is None and len(observed_queries)<24 and not method.startswith("RAG"),"late/overlapping/read-only refresh")
            ticket=event["ticket"]; published=False
            check(ticket["source_frontier"]==len(observed_queries) and ticket["edit_frontier"]==len(edits),"launch did not freeze current prefix")
            near(ticket["consumed_phi"],phi,"launch consumed a different Phi"); phi=0.
            selected=ticket["selected_source_ids"]; pairs=ticket["pairs"]
            check(selected and len(set(selected))==len(selected) and selected==[p["source_id"] for p in pairs],"invalid frozen selected pair IDs")
            lookup={e["source_id"]:e for e in edits}
            for pair in pairs:
                check(pair["source_id"] in lookup,"training selected future/unknown correction")
                prior=lookup[pair["source_id"]]
                for key in ("pos","neg","article"): check(pair[key]==prior[key],"frozen training label/article changed")
                check(pair["request"]==pair["training_prompt"]==prior["query"]["prompt"],"training context differs from actual generated prompt")
                check(pair["actual_prediction_ids"]==prior["query"]["generated_ids"] and pair["source_index"]==prior["index"],"frozen prediction/source index mismatch")
            if method=="Fixed16-Pair":
                check(len(edits)-edit_frontier>=16 and selected==[e["source_id"] for e in edits[-16:]],"fixed16 eligibility/selection mismatch")
            else:
                check(len(edits)>=4 and any(e["index"]>=frontier for e in edits) and ticket["consumed_phi"]>=write_threshold,"Rent eligibility mismatch")
            check(ticket["launch_monotonic_s"]<=event["monotonic_s"],"launch/event timestamp mismatch")
        elif kind=="publication":
            check(ticket is not None and not published,"publication without unique active refresh")
            pub=event["record"]
            check(pub["work"]=={"optimizer_updates":7,"training_pairs":28},"publication incomplete work")
            check(pub["source_frontier"]==ticket["source_frontier"] and pub["edit_frontier"]==ticket["edit_frontier"],"publication advanced beyond frozen prefix")
            check(pub["selected_source_ids"]==ticket["selected_source_ids"] and pub["version"]==version+1,"published selection/version mismatch")
            near(pub["refresh_s"],pub["publication_monotonic_s"]-ticket["launch_monotonic_s"],"publication delay mismatch")
            check(pub["refresh_s"]>=0 and pub["publication_monotonic_s"]<=event["monotonic_s"],"invalid publication timestamp")
            near(pub["pending_phi"],phi,"publication discarded or reset pending Phi")
            pending=[e["source_id"] for e in edits if e["index"]>=ticket["source_frontier"]]
            check(pub["pending_source_ids"]==pending,"publication discarded new correction IDs")
            phase=pub["refresh_window_vin_j"]
            if phase is not None:
                if samples:
                    actual=integrate(samples,ticket["launch_monotonic_s"],pub["publication_monotonic_s"])
                    check(actual is not None,"publication energy window not covered")
                    near(phase,actual,"publication VIN integral mismatch")
                write_threshold=max(finite(phase),1e-6)
            near(pub["updated_write_threshold"],write_threshold,"write-cost threshold update mismatch")
            version+=1; frontier=ticket["source_frontier"]; edit_frontier=ticket["edit_frontier"]
            publications.append(pub); published=True
        elif kind=="refresh_complete":
            check(ticket is not None and published,"refresh completion without publication")
            work=event["record"]
            check(not work.get("oom") and work["optimizer_updates"]==7 and work["training_pairs"]==28,"incomplete refresh work")
            check(work["reference_cache_hits"]==0 and work["reference_pairs_computed"]==len(ticket["pairs"]),"cold-reference work mismatch")
            encoded=work["actual_encoded_training_inputs"]
            check(len(encoded)==len(ticket["pairs"]),"missing exact training input records")
            lookup={e["source_id"]:e for e in edits}
            for record,pair in zip(encoded,ticket["pairs"]):
                check(record["source_id"]==pair["source_id"] and record["training_prompt"]==pair["training_prompt"],"encoded training source/prompt mismatch")
                original=lookup[pair["source_id"]]["query"]["prompt_input_ids"]
                truncated=original[:1]+original[-511:] if len(original)>512 else original
                check(ids(record["encoded_prompt_ids"],512)==truncated,"training input truncation differs from actual generated prompt")
                ids(record["positive_completion_ids"],64); ids(record["negative_completion_ids"],64)
            expected_exposure=[ticket["selected_source_ids"][i%len(ticket["pairs"])] for i in range(28)]
            check(work["training_exposure_source_ids"]==expected_exposure,"pair exposure order/count mismatch")
            refreshes.append(work); ticket=None; published=False
        elif kind=="method_complete": pass
        else: raise ValueError("unknown online event kind: "+kind)
    check(ticket is None and pending_generation is None and observed_queries==queries,"unfinished generation/refresh events")
    check(publications==result["publications"] and refreshes==result["refreshes"],"run/phase-event mismatch")
    check(result["edit_count"]==len(edits) and result["committed_frontier"]==frontier and result["committed_edit_frontier"]==edit_frontier,"final frontier/count mismatch")
    near(result["pending_phi"],phi,"final pending Phi mismatch")
    check(result["pending_source_ids"]==[e["source_id"] for e in edits if e["index"]>=frontier],"final pending edit IDs mismatch")
    check(len(eos_ids)<=1,"inconsistent observed terminal EOS IDs")
    rl=statistics.fmean(m["rouge_l"] for m in metrics); slo=statistics.fmean(m["slo"] for m in metrics)
    near(result["mean_rouge_l"],rl/100,"run ROUGE summary mismatch")
    near(result["joint_slo_fraction"],slo/100,"run SLO summary mismatch")
    near(result["mean_rouge_l_times_joint_slo"],statistics.fmean(m["joint_supplement"] for m in metrics)/100,"secondary joint metric mismatch")
    empty=sum(not q["prediction"] for q in queries)
    check(result["empty_prediction_count"]==result["synthetic_empty_negative_count"]==empty,"empty marker count mismatch")
    return dict(profile_group_id=run["profile_group_id"],method=method,rouge_l=rl,slo_pct=slo,
        p95_ttft_ms=sorted(m["ttft_ms"] for m in metrics)[math.ceil(.95*24)-1],
        publication_delay_s=statistics.fmean(p["refresh_s"] for p in publications) if publications else None,
        publication_delay_sum_s=sum(p["refresh_s"] for p in publications),
        refreshes=len(refreshes),optimizer_updates=7*len(refreshes),training_pairs=28*len(refreshes),
        peak_cuda_gib=peak,secondary_joint=statistics.fmean(m["joint_supplement"] for m in metrics),
        version_audit={k:dict(n=len(v),mean_rouge_l=statistics.fmean(v)) for k,v in read_versions.items()},
        phase_counts=dict(phases),empty_marker_count=empty,realized_offered_load=result["realized_offered_load"],
        source_ids=[q["source_id"] for q in queries],articles=[q["article"] for q in queries],
        golds=[q["gold"] for q in queries],arrival_offsets_s=offsets)


def percentile(values,p):
    ordered=sorted(values); x=(len(ordered)-1)*p; lo=math.floor(x); hi=math.ceil(x)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(x-lo)


def estimate(values,draws):
    if any(v is None for v in values): return dict(mean=None,ci95_lo=None,ci95_hi=None,group_values=values)
    means=[statistics.fmean(values[i] for i in sample) for sample in draws]
    return dict(mean=statistics.fmean(values),ci95_lo=percentile(means,.025),ci95_hi=percentile(means,.975),group_values=values)


def summarize(rows,group_ids):
    rng=random.Random(SEED); draws=[tuple(rng.randrange(6) for _ in range(6)) for _ in range(NBOOT)]
    by_method={m:[next(r for r in rows if r["method"]==m and r["profile_group_id"]==g) for g in group_ids] for m in METHODS}
    methods={}
    for method,records in by_method.items():
        stats={key:estimate([r[key] for r in records],draws) for key in
               ("rouge_l","slo_pct","p95_ttft_ms","energy_kj","refreshes","optimizer_updates","training_pairs","secondary_joint")}
        delays=[r["publication_delay_s"] for r in records]; present=[d for d in delays if d is not None]
        stats["publication_delay_s"]=estimate(delays,draws)
        stats["publication_delay_s"].update(mean=statistics.fmean(present) if present else None,
            groups_with_publication=len(present),conditional_on_at_least_one_publication=True)
        counts=[r["refreshes"] for r in records]; totals=[r["publication_delay_sum_s"] for r in records]
        pubmean=sum(totals)/sum(counts) if sum(counts) else None
        pubdraws=[sum(totals[i] for i in sample)/sum(counts[i] for i in sample) for sample in draws] if all(counts) else []
        stats["publication_delay_per_refresh_s"]=dict(mean=pubmean,
            ci95_lo=percentile(pubdraws,.025) if pubdraws else None,
            ci95_hi=percentile(pubdraws,.975) if pubdraws else None,
            completed_refreshes=sum(counts),groups_with_publication=len(present),
            note="mean over completed refreshes; paired profile resampling of totals/counts; CI only when all six groups publish")
        stats["maximum_peak_cuda_gib"]=max(r["peak_cuda_gib"] for r in records)
        methods[method]=stats
    contrasts={}
    for control in CONTRASTS:
        branch=by_method["RentKnee-Branch"]; baseline=by_method[control]
        contrast={key:estimate([a[key]-b[key] for a,b in zip(branch,baseline)],draws) for key in ("rouge_l","slo_pct")}
        a=[r["energy_kj"] for r in branch]; b=[r["energy_kj"] for r in baseline]
        if any(v is None for v in a+b): ratio=dict(mean=None,ci95_lo=None,ci95_hi=None)
        else:
            ratios=[statistics.fmean(a[i] for i in sample)/statistics.fmean(b[i] for i in sample) for sample in draws]
            ratio=dict(mean=statistics.fmean(a)/statistics.fmean(b),ci95_lo=percentile(ratios,.025),ci95_hi=percentile(ratios,.975))
        contrast["energy_ratio_branch_over_control"]=ratio; contrasts["RentKnee-Branch minus "+control]=contrast
    return methods,contrasts


def table(analysis):
    labels={"RAG@4":"RAG@4","RAG@8":"RAG@8","Fixed16-Pair":"Fixed16 / Pair", "RentKnee-Pair":"Rent+Knee / Pair","RentKnee-Branch":"Rent+Knee / Branch"}
    def fmt(value,digits=1): return "---" if value is None else f"{value:.{digits}f}"
    lines=[]
    for method in METHODS:
        s=analysis["methods"][method]; q=s["rouge_l"]
        quality=f"{q['mean']:.1f} [{q['ci95_lo']:.1f}, {q['ci95_hi']:.1f}]"
        lines.append(" & ".join((labels[method],quality,fmt(s["slo_pct"]["mean"]),fmt(s["p95_ttft_ms"]["mean"]),
            fmt(s["energy_kj"]["mean"],2),fmt(s["refreshes"]["mean"]),fmt(s["publication_delay_per_refresh_s"]["mean"],2)))+r" \\")
    return "\n".join([r"\begin{table*}[t]",r"\centering",r"\footnotesize",r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Event-driven gold-headline-oracle correction replay: six prepared profile packets, 144 source episodes and 720 method responses, with one within-packet repeated article retained. Packets are not authenticated people. ROUGE-L and SLO are percentages; brackets are pointwise 95\% paired profile-group bootstrap intervals (10,000 draws). TTFT is the mean of six per-group empirical P95s. VIN energy is mean complete replay energy per group, excluding shared model-load/calibration and between-method checkpoints; a missing group energy makes the method mean missing. Refreshes are mean counts per group. Publication delay is the mean over completed refreshes; no publication is shown as a dash. Work may differ across policies.}",
        r"\label{tab:author_online}",r"\begin{tabular}{@{}lcccccc@{}}",r"\toprule",
        r"Method & ROUGE-L [95\% CI] & SLO (\%) & P95 TTFT (ms) & VIN (kJ) & Refreshes & Publication (s) \\",r"\midrule",
        *lines,r"\bottomrule",r"\end{tabular}",r"\end{table*}",""])


def analyze(receipt,asset_dir,smoke):
    payload=json.loads(receipt.read_text()); expected=source_and_data(payload,smoke)
    assets=AssetAudit(asset_dir); events=assets.journal(payload)
    prep,_=assets.energy(payload["preparation"])
    runs=payload["runs"]; check(len(runs)==len(expected)*5,"incomplete group/method matrix")
    expected_order=[]
    for gi,group in enumerate(expected):
        expected_order.extend((group,m) for m in METHODS[gi%5:]+METHODS[:gi%5])
    check([(r["profile_group_id"],r["method"]) for r in runs]==expected_order,"predetermined group/method order changed")
    chunks=split_events(events,runs); rows=[]
    for run,chunk in zip(runs,chunks):
        energy,samples=assets.energy(run["energy"])
        row=validate_run(run,chunk,expected[run["profile_group_id"]],payload,samples)
        row["energy_kj"]=None if energy is None else energy/1000
        if energy is not None and samples:
            check(samples[0][0]<=chunk[0]["monotonic_s"] and samples[-1][0]>=chunk[-2]["monotonic_s"],"full replay CSV does not cover active events")
        rows.append(row)
    for group in expected:
        peers=[r for r in rows if r["profile_group_id"]==group]
        for peer in peers[1:]:
            for key in ("source_ids","articles","golds","arrival_offsets_s"): check(peer[key]==peers[0][key],"methods do not share episodes/arrivals: "+key)
    if smoke: check(any(r["method"]=="Fixed16-Pair" and r["refreshes"]>=1 for r in rows),"mandatory smoke Fixed16 refresh gate failed")
    article_occurrences=defaultdict(list)
    for group,items in expected.items():
        for q in items: article_occurrences[" ".join(q["article"].lower().split())].append((group,q["source_id"]))
    duplicates=[v for v in article_occurrences.values() if len(v)>1]
    if not smoke:
        check(sum(map(len,article_occurrences.values()))==144 and len(article_occurrences)==143
              and len(duplicates)==1 and len(duplicates[0])==2
              and duplicates[0][0][0]==duplicates[0][1][0],"declared retained within-packet duplicate audit differs")
    result=dict(valid=True,smoke_only=smoke,schema="tierserve_online_analysis_v1",source_tree_sha256=payload.get("source_tree_sha256"),
        receipt_sha256=sha(receipt),renderer_sha256=sha(__file__),profile_group_ids=list(expected),
        n_profile_groups=len(expected),source_episodes=24*len(expected),method_responses=120*len(expected),
        rows=rows,preparation_energy_j=prep,asset_audit=assets.records,warnings=assets.warnings,
        article_audit=dict(unique_normalized_articles=len(article_occurrences),duplicate_source_episodes=duplicates),
        maximum_peak_cuda_gib=max(r["peak_cuda_gib"] for r in rows),
        sampling=dict(unit="profile_group",resamples=0 if smoke else NBOOT,seed=SEED,
            note="Smoke overlaps the full cohort; conditional version/phase summaries are not causal quality comparisons."))
    if not smoke: result["methods"],result["prespecified_contrasts"]=summarize(rows,list(expected))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=PAPER / "results/online_full_r0.json")
    parser.add_argument("--asset-dir", type=Path, default=PAPER / "results/online_full_r0_assets")
    args = parser.parse_args()
    result = analyze(args.receipt, args.asset_dir, False)
    output = PAPER / "outputs"
    (output / "tables").mkdir(parents=True, exist_ok=True)
    (output / "online.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output / "tables/tab_author_online.tex").write_text(table(result))
    print("Wrote online results to", output)


if __name__ == "__main__":
    main()
