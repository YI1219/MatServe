#!/usr/bin/env python3
"""Replay generated headlines with feedback revealed after each response.

Records arrivals, adapter publications and board-energy samples. GPU runs must
use the existing qrun dispatcher on Thor.
"""
from __future__ import annotations
import argparse
import copy
import glob
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import random
import signal
import statistics
import time
from datetime import datetime, timezone
from online_state import OracleEpisode, OnlineState, ArrivalDriver, SysfsEnergy, EventJournal

METHODS=("RAG@4","RAG@8","Fixed16-Pair","RentKnee-Pair","RentKnee-Branch")
SENSOR_GLOB="/sys/bus/i2c/devices/2-0044/hwmon/hwmon*/power1_input"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,payload):
    temp=path.with_suffix(path.suffix+".writing")
    temp.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n"); temp.replace(path)


def persist_event(payload,journal,kind,**record):
    event=dict(kind=kind,monotonic_s=time.perf_counter(),**record)
    journal.append(event)
    payload["events"].append(event)


def guard_prefix(out):
    if list(out.parent.glob(out.stem+"*")):
        raise FileExistsError("new output prefix required; never overwrite or resume")


def bare_headline(article):
    return "Write one concise news headline.\nArticle: "+article[:750]+"\nHeadline:"


def pack_cards(tok,article,edits,indices,budget=None,prompt_cap=1792):
    bare=bare_headline(article)
    base=len(tok(bare,add_special_tokens=True).input_ids)
    if base>prompt_cap:
        raise ValueError("bare article exceeds declared prompt cap")
    selected=[]; prompt=bare; budget_excluded=[]; prompt_cap_excluded=[]
    for i in indices:
        candidates=selected+[i]
        cards=["Article: "+edits[j]["article"][:240]+"\nCorrected headline: "+edits[j]["pos"] for j in reversed(candidates)]
        trial="Previous oracle corrections:\n"+"\n\n".join(cards)+"\n\n"+bare
        total=len(tok(trial,add_special_tokens=True).input_ids)
        if total<=prompt_cap and (budget is None or max(0,total-base)<=budget):
            selected=candidates; prompt=trial
        else:
            if budget is not None and max(0,total-base)>budget:
                budget_excluded.append(edits[i]["source_id"])
            if total>prompt_cap:
                prompt_cap_excluded.append(edits[i]["source_id"])
    total=len(tok(prompt,add_special_tokens=True).input_ids)
    return dict(prompt=prompt,prompt_tokens=total,base_prompt_tokens=base,added_tokens=max(0,total-base),
                added_token_budget=budget,prompt_cap=prompt_cap,
                selected_source_ids=[edits[j]["source_id"] for j in selected],
                requested_source_ids=[edits[j]["source_id"] for j in indices],
                budget_excluded_source_ids=budget_excluded,prompt_cap_excluded_source_ids=prompt_cap_excluded)


class ReplayController:
    """Actual arrivals and feedback are handled even while fixed refresh work runs."""
    def __init__(self,episodes,arrivals,backend,method,meter,emit,clock=time.perf_counter,sleep=time.sleep):
        self.episodes,self.backend,self.method,self.meter,self.emit=episodes,backend,method,meter,emit
        self.clock,self.sleep=clock,sleep
        self.state=OnlineState(len(episodes)); self.start=clock()
        self.driver=ArrivalDriver(arrivals,self.start,clock)
        self.queries,self.publications,self.refreshes=[],[],[]
        self.write_cost=backend.initial_write_cost

    def serve(self,index,arrival):
        episode=self.episodes[index]
        public=episode.public()
        frontier=self.state.committed_frontier
        answer=self.backend.answer(public,self.state,self.method,arrival)
        # Prompt construction and generation above have no access to the target.
        episode.record_prediction(answer["prediction"],answer["generated_ids"])
        self.emit("generation_complete",method=self.method,source_id=public["source_id"],
                  source_frontier=self.state.served,committed_frontier=frontier,**copy.deepcopy(answer))
        gold=episode.gold.strip()
        pair,charge,feedback=self.backend.feedback(public,answer,gold,self.state,self.method)
        self.state.append_feedback(index,pair,charge)
        record=dict(answer,source_id=public["source_id"],query_index=index,article=public["article"],
            date=public["date"],gold=gold,feedback=feedback,oracle="gold_headline_oracle_correction",
            feedback_monotonic_s=self.clock(),source_frontier_before_response=index,
            committed_frontier_at_read=frontier,committed_edit_frontier_at_read=self.state.committed_edits,
            phi_after_feedback=self.state.phi)
        self.queries.append(record)
        self.emit("oracle_feedback",method=self.method,record=record)

    def poll(self,published=False,work=None):
        if published and self.state.active is not None:
            if work!={"optimizer_updates":7,"training_pairs":28}:
                raise RuntimeError("publication work contract violated")
            ticket=copy.deepcopy(self.state.active)
            self.backend.publish(self.state.version+1)
            self.state.publish(ticket)
            stamp=self.clock()
            cost=self.meter.energy_between(ticket["launch_monotonic_s"],stamp) if self.meter else None
            if cost is not None:
                self.write_cost=max(cost,1e-6)
            event=dict(version=self.state.version,source_frontier=ticket["source_frontier"],
                edit_frontier=ticket["edit_frontier"],
                selected_source_ids=ticket["selected_source_ids"],publication_monotonic_s=stamp,
                refresh_s=stamp-ticket["launch_monotonic_s"],work=work,
                pending_phi=self.state.phi,pending_source_ids=[e["source_id"] for e in self.state.unflushed()],
                refresh_window_vin_j=cost,updated_write_threshold=self.write_cost)
            self.publications.append(event); self.emit("publication",method=self.method,record=event)
        self.driver.drain(self.serve)

    def ready(self):
        if self.driver.done or self.state.active is not None or self.method.startswith("RAG"):
            return False
        if self.method=="Fixed16-Pair":
            return len(self.state.edits)-self.state.committed_edits>=16
        return len(self.state.edits)>=4 and bool(self.state.unflushed()) and self.state.phi>=self.write_cost

    def run(self):
        while not self.driver.done:
            self.poll()
            if self.ready():
                selected,selection=self.backend.select(self.state,self.method)
                ticket=self.state.launch(selected)
                # Attach the launch timestamp to the live frozen ticket, not a moving prefix.
                ticket["launch_monotonic_s"]=self.clock()
                self.state.active["launch_monotonic_s"]=ticket["launch_monotonic_s"]
                self.emit("refresh_start",method=self.method,ticket=copy.deepcopy(ticket),selection=selection)
                meta=self.backend.refresh(ticket,self.poll)
                if meta.get("oom") or meta.get("optimizer_updates")!=7 or meta.get("training_pairs")!=28:
                    raise RuntimeError("refresh failed/OOM/incomplete fixed work")
                if self.state.active is not None:
                    raise RuntimeError("engine failed to invoke final publication callback")
                self.refreshes.append(meta)
                self.emit("refresh_complete",method=self.method,record=meta)
            elif not self.driver.done:
                self.sleep(max(0.,self.driver.next_time()-self.clock()))
        if self.state.active is not None or len(self.queries)!=len(self.episodes):
            raise RuntimeError("incomplete query/refresh replay")
        return dict(method=self.method,n_queries=len(self.queries),queries=self.queries,
            edit_count=len(self.state.edits),publications=self.publications,refreshes=self.refreshes,
            pending_phi=self.state.phi,pending_source_ids=[e["source_id"] for e in self.state.unflushed()],
            committed_frontier=self.state.committed_frontier,
            committed_edit_frontier=self.state.committed_edits,
            wall_s=self.clock()-self.start,arrival_offsets_s=self.driver.offsets,
            actual_arrival_rate_qps=(len(self.episodes)-1)/(self.driver.offsets[-1]-self.driver.offsets[0]))


class DenseBackend:
    def __init__(self,model,tok,sp,seed,ttft_budget,tpot_budget):
        import torch
        from core import reset_peft_lora
        from snapshots import TrainableSnapshot
        self.model,self.tok,self.sp,self.seed=model,tok,sp,seed
        self.ttft_budget,self.tpot_budget=ttft_budget,tpot_budget
        self.embeddings={}; self.base_margins={}; self.raw_margins=[]; self.peak_gib=0.
        reset_peft_lora(model,seed=seed)
        self.committed=TrainableSnapshot(model,"committed-v0")
        self.version=0
        self.initial_write_cost=max(1e-6,28*sp.write_joule_per_pair)
        self.routing_budget=min(1408,max(0,int(sp.l_crit_tokens)))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def monitor(self):
        import torch
        if torch.cuda.is_available():
            self.peak_gib=max(self.peak_gib,torch.cuda.max_memory_allocated()/1024**3)
            if self.peak_gib>=8.:
                raise RuntimeError("prospective online replay peak exceeds 8 GiB")

    def generate(self,prompt,arrival,use_adapter=True):
        import torch
        from contextlib import nullcontext
        model,tok=self.model,self.tok
        ids=tok(prompt,add_special_tokens=True).input_ids
        if len(ids)>1792:
            raise RuntimeError("prompt cap violated; no silent truncation")
        was_training=model.training; model.eval()
        generated=[]; emitted=[]; token_times=[]; first=None; begin=time.perf_counter()
        with torch.no_grad(), (nullcontext() if use_adapter else model.disable_adapter()):
            inp=torch.tensor([ids],dtype=torch.long,device=next(model.parameters()).device)
            output=model(input_ids=inp,use_cache=True); past=output.past_key_values
            for step in range(32):
                nxt=output.logits[:,-1:,:].argmax(dim=-1)
                token=int(nxt.item())
                if torch.cuda.is_available(): torch.cuda.synchronize()
                token_times.append(time.perf_counter()); emitted.append(token)
                if first is None:
                    first=token_times[0]
                if token==tok.eos_token_id:
                    break
                generated.append(token)
                if step==31: break
                output=model(input_ids=nxt,past_key_values=past,use_cache=True); past=output.past_key_values
            if torch.cuda.is_available(): torch.cuda.synchronize()
            end=time.perf_counter()
            del output,past,inp,nxt
        if was_training: model.train()
        raw=tok.decode(generated,skip_special_tokens=True)
        # Use the stripped response shown to the client as the DPO negative.
        prediction=raw.strip()
        end=time.perf_counter()
        intervals=max(0,len(emitted)-1)
        tpot=1000*(token_times[-1]-first)/intervals if intervals else 0.
        ttft=1000*(first-arrival)
        return dict(prediction=prediction,raw_prediction=raw,generated_ids=generated,
            all_emitted_ids=emitted,emitted_token_count=len(emitted),token_monotonic_s=token_times,
            inter_token_intervals=intervals,single_emitted_token_tpot_policy="zero; TTFT-only",
            prompt=prompt,prompt_input_ids=ids,
            prompt_tokens=len(ids),arrival_monotonic_s=arrival,generation_start_monotonic_s=begin,
            first_token_monotonic_s=first,response_monotonic_s=end,
            ttft_ms=ttft,tpot_ms=tpot,queue_wait_ms=max(0.,1000*(begin-arrival)),
            generation_s=end-begin,generated_tokens=len(generated),
            joint_slo=ttft<=self.ttft_budget and tpot<=self.tpot_budget,
            ttft_budget_ms=self.ttft_budget,tpot_budget_ms=self.tpot_budget)

    def centered(self,edits,query_embedding=None):
        import torch
        from planning import center_embeddings_zero_param
        raw=torch.stack([self.embeddings[e["source_id"]] for e in edits])
        if query_embedding is None:
            return center_embeddings_zero_param(raw,raw)[0]
        centered,query=center_embeddings_zero_param(raw,raw,query_embedding.unsqueeze(0))
        return centered,query[0]

    def answer(self,public,state,method,arrival):
        from core import embed_texts,select_mmr
        from planning import compute_null_similarity_quantile
        from baselines import select_lamp_bm25
        source_ids=[]; query_embedding=None
        if method.startswith("Rent"):
            with self.model.disable_adapter():
                query_embedding=embed_texts(self.model,self.tok,[public["article"]],max_len=192)[0].detach()
            candidates=state.unflushed()
            if len(candidates)>=2:
                centered,query=self.centered(candidates,query_embedding)
                threshold=compute_null_similarity_quantile(centered,q=.5)
                indices=select_mmr(query,centered,k=min(4,len(candidates)),lam=.5,sim_floor=threshold)[0]
            else:
                indices=[]
            packed=pack_cards(self.tok,public["article"],candidates,indices,self.routing_budget)
            mode="delta_gate_mmr"
        else:
            candidates=state.edits
            k=8 if method=="RAG@8" else 4
            indices=select_lamp_bm25(public["article"],[{"request":e["article"]} for e in candidates],k) if candidates else []
            packed=pack_cards(self.tok,public["article"],candidates,indices)
            mode="bm25"
        routing_complete=time.perf_counter()
        # Snapshot activation is graph-free and its switching time is included in TTFT.
        with self.committed.activate(self.model):
            result=self.generate(packed["prompt"],arrival,use_adapter=not method.startswith("RAG"))
        result.update({k:v for k,v in packed.items() if k!="prompt"})
        result.update(adapter_read_version=f"committed-v{state.version}" if not method.startswith("RAG") else "base",
                      retrieval_mode=mode,available_source_ids=[e["source_id"] for e in candidates],
                      routing_complete_monotonic_s=routing_complete)
        if query_embedding is not None:
            self.embeddings[public["source_id"]]=query_embedding
        self.monitor()
        return result

    def feedback(self,public,answer,gold,state,method):
        from core import mean_logp_no_grad
        from planning import rouge_l_f1
        prediction=answer["prediction"]
        edited=prediction!=gold
        negative=prediction if prediction else "[EMPTY_HEADLINE]"
        feedback=dict(oracle="gold_headline_oracle_correction",edited=edited,
            actual_prediction=prediction,negative_text=negative,empty_prediction=not bool(prediction),
            gold_source_id=public["source_id"]+":gold",
            negative_source_id=public["source_id"]+(":actual_generation" if prediction else ":synthetic_empty_marker"),
            negative_is_synthetic_empty_marker=not bool(prediction),
            rouge_l=rouge_l_f1(prediction,gold))
        pair=None
        charge=answer["added_tokens"]*self.sp.joule_per_ctx_tok if method.startswith("Rent") else 0.
        feedback["read_charge_j"]=charge
        if edited:
            pair=dict(source_id=public["source_id"],article=public["article"],request=answer["prompt"],
                      training_prompt=answer["prompt"],
                      pos=gold,neg=negative,actual_prediction_ids=answer["generated_ids"])
            if method.startswith("Rent"):
                with self.committed.activate(self.model):
                    pos=mean_logp_no_grad(self.model,self.tok,answer["prompt"],gold,max_prompt=512,max_comp=64)
                    neg=mean_logp_no_grad(self.model,self.tok,answer["prompt"],negative,max_prompt=512,max_comp=64)
                with self.model.disable_adapter():
                    rp=mean_logp_no_grad(self.model,self.tok,answer["prompt"],gold,max_prompt=512,max_comp=64)
                    rn=mean_logp_no_grad(self.model,self.tok,answer["prompt"],negative,max_prompt=512,max_comp=64)
                if any(x is None for x in (pos,neg,rp,rn)):
                    raise RuntimeError("unscorable on-policy oracle correction")
                margin=pos-neg; self.raw_margins.append(margin)
                scale=max(statistics.pstdev(self.raw_margins),1e-4)
                z=max(-40.,min(40.,-margin/scale))
                surprise=min(2.,math.log1p(math.exp(z))/math.log(2.))
                overflow=bool(answer["budget_excluded_source_ids"])
                surprise_charge=self.sp.write_joule_per_pair*surprise*(2 if overflow else 1)
                charge+=surprise_charge
                self.base_margins[public["source_id"]]=rp-rn
                feedback.update(raw_margin=margin,base_margin=rp-rn,surprise_charge_j=surprise_charge,
                                margin_scale=scale,surprise=surprise,overflow_weight=2 if overflow else 1,
                                feedback_score_prompt_cap=512)
        feedback["maintenance_charge_j"]=charge
        self.monitor()
        return pair,charge,feedback

    def select(self,state,method):
        from planning import auto_dedilute_kneedle_select
        if method=="Fixed16-Pair":
            indices=list(range(max(0,len(state.edits)-16),len(state.edits)))
            return indices,dict(rule="recent16",n=len(indices))
        centered=self.centered(state.edits)
        margins=[self.base_margins[e["source_id"]] for e in state.edits]
        return auto_dedilute_kneedle_select(state.edits,margins,centered)

    def refresh(self,ticket,poll):
        from snapshots import TrainableSnapshot
        from core import reset_peft_lora
        from execution import run_step_boundary_preempt_engine,run_adjoint_microtxn_engine
        from execution import pair_training_prompt
        from core import _encode_prompt_comp
        encoded=[]
        for pair in ticket["pairs"]:
            prompt=pair_training_prompt(pair)
            pos=_encode_prompt_comp(self.tok,prompt,pair["pos"],512,64)
            neg=_encode_prompt_comp(self.tok,prompt,pair["neg"],512,64)
            if pos is None or neg is None:
                raise RuntimeError("unscorable exact on-policy training prompt")
            encoded.append(dict(source_id=pair["source_id"],training_prompt=prompt,
                encoded_prompt_ids=pos[0],positive_completion_ids=pos[1],negative_completion_ids=neg[1]))
        # Save the published adapter before training the next candidate.
        self.committed=TrainableSnapshot(self.model,f"committed-v{self.version}")
        reset_peft_lora(self.model,seed=self.seed)
        self.monitor()
        engine=run_adjoint_microtxn_engine if self.method.endswith("Branch") else run_step_boundary_preempt_engine
        kwargs=dict(n_steps=7,grad_accum=4,lr=7.5e-5,max_p_train=512,max_c_train=64,gen_toks=32,
                    drain_future_arrivals=False,sft_alpha=.45,reset_lora=False,seed=self.seed,
                    precomputed_refs=None,foreground_callback=poll)
        if engine is run_step_boundary_preempt_engine:
            kwargs["yield_granularity"]="pair"
        _,meta=engine(self.model,self.tok,ticket["pairs"],[],[],**kwargs)
        meta["actual_encoded_training_inputs"]=encoded
        meta["training_exposure_source_ids"]=[p["source_id"] for p in (ticket["pairs"]*28)[:28]]
        self.monitor()
        return meta

    def publish(self,version):
        from snapshots import TrainableSnapshot
        self.version=version; self.committed=TrainableSnapshot(self.model,f"committed-v{version}")


def load_packets(directory,groups,queries):
    manifest=json.loads((directory/"_manifest.json").read_text())[6:12]
    if len(manifest)!=6:
        raise ValueError("next six unused manifest groups unavailable")
    paths=[directory/(row["author_id"]+".json") for row in manifest[:groups]]
    packets=[]
    for path in paths:
        packet=json.loads(path.read_text())
        indexed=list(enumerate(packet["history_pairs"]))
        if any(not isinstance(item.get("date"),str) or len(item["date"])!=10 for _,item in indexed):
            raise ValueError("profile date unavailable")
        ordered=sorted(indexed,key=lambda x:(x[1]["date"],x[0]))[:queries]
        if len(ordered)!=queries:
            raise ValueError("insufficient historical query episodes")
        packets.append((packet["author_id"],[dict(source_id=f"{packet['author_id']}:history:{item.get('ep_idx',i)}",
            article=item["article"],target=item["pos"],date=item["date"]) for i,item in ordered]))
    return paths,packets


def audit_article_overlap(directory,packets):
    """Report fixed-cohort overlap; never select, discard, or reorder a query."""
    def normalized_hash(text):
        return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()
    references={"history_all":{},"l1_history_last60":{},"heldout":{}}
    manifest=json.loads((directory/"_manifest.json").read_text())[:6]
    reference_paths=[]
    for entry in manifest:
        path=directory/(entry["author_id"]+".json"); reference_paths.append(path)
        packet=json.loads(path.read_text())
        for role,items in (("history_all",packet["history_pairs"]),
                           ("l1_history_last60",packet["history_pairs"][-60:]),("heldout",packet["heldout_queries"])):
            for i,item in enumerate(items):
                key=normalized_hash(item["article"])
                references[role].setdefault(key,[]).append(f"{entry['author_id']}:{role}:{item.get('ep_idx',item.get('id',i))}")
    occurrences={}; query_rows=[]; tie_counts={}
    for group,items in packets:
        tie_counts[group]=sum(a["date"]==b["date"] for a,b in zip(items,items[1:]))
        for item in items:
            key=normalized_hash(item["article"])
            occurrences.setdefault(key,[]).append((group,item["source_id"]))
            query_rows.append(dict(profile_group_id=group,source_id=item["source_id"],normalized_article_sha256=key,
                overlap={role:refs.get(key,[]) for role,refs in references.items()}))
    within=0; across=0; duplicates=[]
    for key,values in occurrences.items():
        if len(values)>1:
            duplicates.append(dict(normalized_article_sha256=key,occurrences=values))
        for i,a in enumerate(values):
            for b in values[i+1:]:
                if a[0]==b[0]: within+=1
                else: across+=1
    return dict(normalization="lowercase; collapse all whitespace; SHA-256; full prepared article",
        profile_groups=len(packets),query_items=len(query_rows),unique_normalized_articles=len(occurrences),
        within_group_duplicate_pairs=within,across_group_duplicate_pairs=across,
        overlap_item_counts={role:sum(bool(row["overlap"][role]) for row in query_rows) for role in references},
        same_day_adjacent_query_pairs=tie_counts,chronology="date-only; stable original order for equal dates",
        duplicate_articles=duplicates,query_overlap_records=query_rows,
        first6_reference_data_sha256={p.name:digest(p) for p in reference_paths},
        decision="report only; no cohort filtering or replacement")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lamp-dir",type=Path,required=True); ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--groups",type=int,default=6); ap.add_argument("--queries",type=int,default=24)
    ap.add_argument("--max-wall-s",type=int,required=True); ap.add_argument("--seed",type=int,default=20261112)
    ap.add_argument("--preflight",action="store_true")
    args=ap.parse_args()
    smoke=(args.groups,args.queries)==(1,24)
    if (args.groups,args.queries) not in ((1,24),(6,24)) or not 0<args.max_wall_s<=(600 if smoke else 2400):
        raise ValueError("only declared 1x24 smoke or 6x24 full recipe within budget")
    guard_prefix(args.out)
    paths,packets=load_packets(args.lamp_dir,args.groups,args.queries)
    _,audit_packets=load_packets(args.lamp_dir,6,24)
    from planning import V2_MODEL_PATHS
    model_path=Path(V2_MODEL_PATHS["dense_qwen15b"])
    sidecars=[p for p in model_path.glob("*.json") if p.is_file()]
    if not (model_path/"config.json").exists(): raise FileNotFoundError("model config missing")
    sensors=sorted(glob.glob(SENSOR_GLOB)); sensor=Path(sensors[0]) if sensors else Path("/unavailable/INA238/VIN/power1_input")
    sensor_probe=dict(path=str(sensor),unit="uW",raw=None,error=None)
    try:
        sensor_probe["raw"]=int(sensor.read_text().strip())
        sensor_probe["watts"]=sensor_probe["raw"]/1e6
        for name in ("name","power1_label","in1_label"):
            p=sensor.parent/name
            if p.exists(): sensor_probe[name]=p.read_text().strip()
    except Exception as exc: sensor_probe["error"]=repr(exc)
    payload=dict(schema="tierserve_online_oracle_lamp_v1",ok=False,args={**vars(args),"lamp_dir":str(args.lamp_dir),"out":str(args.out)},
        started_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(p.relative_to(ROOT)):digest(p) for folder in ("src", "experiments") for p in sorted((ROOT/folder).glob("*.py"))},
        data_sha256={p.name:digest(p) for p in [args.lamp_dir/"_manifest.json",*paths]},
        model_path=str(model_path),model_sidecar_sha256={p.name:digest(p) for p in sidecars},
        profile_groups=[p[0] for p in packets],manifest_positions=[6,6+args.groups],
        oracle="gold_headline_oracle_correction_not_human_preference",sensor_preflight=sensor_probe,
        methods=METHODS,nominal_load=.34,analysis_unit="profile_group",events=[],runs=[],preparation=None)
    payload["smoke_gate"]=dict(required_fixed16_completed_refreshes=1,queries_per_method=24,
        updates_per_refresh=7,pair_exposures_per_refresh=28) if smoke else None
    payload["cohort_relation"]="smoke uses the first full-replay profile group; not independent confirmation"
    payload["prospective_full_cohort_article_audit"]=audit_article_overlap(args.lamp_dir,audit_packets)
    payload["source_tree_sha256"]=hashlib.sha256(json.dumps(payload["source_sha256"],sort_keys=True,separators=(",",":")).encode()).hexdigest()
    if args.preflight:
        print(json.dumps(payload,indent=2)); return
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.with_suffix(".provenance.json").open("x") as handle: json.dump(payload,handle,indent=2)
    journal=EventJournal(args.out.with_suffix(".events.jsonl"))
    payload["event_journal"]=journal.metadata()
    started=time.perf_counter(); prep=None
    def timeout(*_): raise TimeoutError("prospective wall budget exhausted")
    signal.signal(signal.SIGALRM,timeout); signal.alarm(args.max_wall_s)
    def emit(kind,**record):
        persist_event(payload,journal,kind,**record)
    try:
        import torch
        import transformers,peft
        from core import set_seed
        from planning import load_v2_backbone,run_specprobe_calibration
        from execution import generate_coupled_coserve_arrivals
        if not torch.cuda.is_available(): raise RuntimeError("qrun-serialized Thor GPU execution required")
        torch.cuda.set_per_process_memory_fraction(.90)
        set_seed(args.seed)
        payload["versions"]=dict(torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__)
        with SysfsEnergy(sensor,args.out.with_suffix(".preparation_power.csv")) as prep:
            model,tok,load_info=load_v2_backbone("dense_qwen15b")
            for module in model.modules():
                if isinstance(module,torch.nn.Dropout): module.p=0.
            sp=run_specprobe_calibration(model,tok,backbone="dense_qwen15b",sync_from_s1=False)
            probe=DenseBackend(model,tok,sp,args.seed,350.,20.)
            neutral=bare_headline("A public library opened a new reading room for local residents.")
            # Warm up and calibrate without using episode target titles.
            for _ in range(2): probe.generate(neutral,time.perf_counter(),False)
            unloaded=[probe.generate(neutral,time.perf_counter(),False) for _ in range(5)]
        payload["preparation"]=prep.result
        percentile=lambda xs: sorted(xs)[math.ceil(.95*len(xs))-1]
        ttft=max(350.,2.5*percentile([q["ttft_ms"] for q in unloaded]))
        tpot=max(20.,1.5*percentile([q["tpot_ms"] for q in unloaded]))
        mean_service=statistics.fmean(q["response_monotonic_s"]-q["arrival_monotonic_s"] for q in unloaded)
        capacity=1/max(mean_service,1e-6)
        payload.update(load_info=load_info,specprobe=sp.__dict__,unloaded=unloaded,
            ttft_budget_ms=ttft,tpot_budget_ms=tpot,service_capacity_qps=capacity,
            learning_recipe=dict(objective="DPO",beta=.1,sft_alpha=.45,lora_r=8,lora_alpha=16,
                updates=7,grad_accum=4,lr=7.5e-5,prompt_cap=512,completion_cap=64,
                generation_tokens=32,generation_prompt_cap=1792,dropout=0.,rematerialize_each_refresh=True))
        for gi,(group_id,items) in enumerate(packets):
            arrivals=generate_coupled_coserve_arrivals(args.queries,.34*capacity,mean_service*1000,args.seed+gi,False)
            # One fixed order per declared group; cyclic shift is predetermined, not outcome selected.
            order=METHODS[gi%5:]+METHODS[:gi%5]
            for method in order:
                meter=SysfsEnergy(sensor,args.out.with_name(args.out.stem+f".g{gi}.{method}.power.csv"))
                run=dict(profile_group_id=group_id,method=method,complete=False,energy=None)
                payload["runs"].append(run)
                try:
                    with meter:
                        backend=DenseBackend(model,tok,sp,args.seed,ttft,tpot); backend.method=method
                        episodes=[OracleEpisode(x["source_id"],x["article"],x["target"],x["date"]) for x in items]
                        controller=ReplayController(episodes,arrivals,backend,method,meter,emit)
                        result=controller.run(); backend.monitor()
                        result["realized_offered_load"]=result["actual_arrival_rate_qps"]/capacity
                        result["peak_cuda_gib"]=backend.peak_gib
                        result["mean_rouge_l"]=statistics.fmean(q["feedback"]["rouge_l"] for q in result["queries"])
                        result["joint_slo_fraction"]=statistics.fmean(q["joint_slo"] for q in result["queries"])
                        result["mean_rouge_l_times_joint_slo"]=statistics.fmean(q["feedback"]["rouge_l"]*q["joint_slo"] for q in result["queries"])
                        result["empty_prediction_count"]=sum(q["feedback"]["empty_prediction"] for q in result["queries"])
                        result["synthetic_empty_negative_count"]=sum(q["feedback"]["negative_is_synthetic_empty_marker"] for q in result["queries"])
                        run.update(result=result,complete=True)
                finally:
                    run["energy"]=meter.result
                emit("method_complete",profile_group_id=group_id,method=method)
                payload["event_journal"]=journal.metadata()
                atomic(args.out.with_suffix(".partial.json"),payload)
        if len(payload["runs"])!=args.groups*5 or any(not r["complete"] or r["result"]["n_queries"]!=24 for r in payload["runs"]):
            raise RuntimeError("incomplete five-method/query contract")
        if smoke and not any(r["method"]=="Fixed16-Pair" and r["result"]["refreshes"] for r in payload["runs"]):
            raise RuntimeError("smoke gate failed: no completed Fixed16-Pair refresh")
        payload.update(ok=True,wall_time_s=time.perf_counter()-started)
        payload["event_journal"]=journal.metadata()
        atomic(args.out,payload)
    except BaseException as exc:
        if prep is not None and prep.result is not None: payload["preparation"]=prep.result
        payload.update(ok=False,error=repr(exc),wall_time_s=time.perf_counter()-started)
        payload["event_journal"]=journal.metadata()
        atomic(args.out.with_suffix(".partial.json"),payload)
        atomic(args.out.with_suffix(".failed.json"),payload)
        raise
    finally:
        signal.alarm(0)
        journal.close()


if __name__=="__main__": main()
