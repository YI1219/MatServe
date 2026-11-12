#!/usr/bin/env python3
"""Run the fixed retrieval and coreset controls without resuming prior outputs.

GPU runs must use the existing qrun dispatcher on Thor.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

K_VALUES = (0, 1, 2, 4, 8)
SEED = 20261112
MAX_P_TRAIN, MAX_C = 384, 96
MAX_P_READ = 2304
STEPS, GRAD_ACCUM = 6, 4


def partition_user(user):
    """Keep first occurrences within roles; cross-role ids use different namespaces."""
    unique, duplicate_ids, seen_events, seen_requests = [], [], set(), set()
    for original_i, pair in enumerate(user.get('train_k160', [])):
        event = pair.get('ep_idx')
        request = pair.get('request', '')
        if (event is not None and event in seen_events) or (request and request in seen_requests):
            duplicate_ids.append(original_i)
            continue
        if event is not None: seen_events.add(event)
        if request: seen_requests.add(request)
        unique.append((original_i, pair))
    ep_ids = [p['ep_idx'] for _, p in unique if p.get('ep_idx') is not None]
    if any(a > b for a, b in zip(ep_ids, ep_ids[1:])):
        raise ValueError('history ep_idx is not nondecreasing in stored order')
    if len(unique) < 24:
        raise ValueError(f'need 24 distinct history events/requests; available={len(unique)}')
    history = unique[:24]
    held, rejected, held_events, held_requests = [], [], set(), set()
    for original_i, pair in enumerate(user.get('heldout', [])):
        event, request = pair.get('ep_idx'), pair.get('request', '')
        reason = None
        # Canonical heldout extraction enumerates a separate [200:400] slice.
        # Numeric ep_idx values therefore cannot be equated across roles.
        if request and request in seen_requests: reason = 'history_request_overlap'
        elif event is not None and event in held_events: reason = 'duplicate_heldout_event'
        elif request and request in held_requests: reason = 'duplicate_heldout_request'
        if reason:
            rejected.append(dict(original_index=original_i, reason=reason))
            continue
        if event is not None: held_events.add(event)
        if request: held_requests.add(request)
        held.append((original_i, pair))
    if len(held) < 8:
        raise ValueError(f'need 8 distinct nonoverlapping heldout items; available={len(held)}')
    metadata = dict(history_original_indices=[i for i, _ in history],
        profile_original_indices=[i for i, _ in history[:20]],
        validation_original_indices=[i for i, _ in history[20:24]],
        heldout_original_indices=[i for i, _ in held[:8]],
        history_duplicates_removed=duplicate_ids, heldout_exclusions=rejected,
        n_unique_history=len(unique), n_clean_heldout=len(held),
        episode_id_scope='role-local: training and heldout ep_idx namespaces differ',
        cross_role_disjointness='preparer disjoint prefix/tail slices plus exact-request exclusion; not numeric-id equality',
        replay_order='stored episode order; timestamps not independently verified')
    return ([p for _, p in history[:20]], [p for _, p in history[20:]],
            [p for _, p in held[:8]], metadata)


def causal_split(user):
    return partition_user(user)[:3]


def choose_validation_k(scores):
    if not scores:
        raise ValueError('empty validation scores')
    for k, values in scores.items():
        if k < 0 or len(values) != 4 or any(not math.isfinite(x) for x in values):
            raise ValueError('validation requires four finite scores per k')
    return min(scores, key=lambda k: (-sum(scores[k]) / 4, k))


def matched_coresets(n, knee_ids, margins, seed=SEED):
    if not knee_ids or len(set(knee_ids)) != len(knee_ids) or any(i < 0 or i >= n for i in knee_ids):
        raise ValueError('invalid KneeCore ids')
    if len(margins) != n or any(not math.isfinite(x) for x in margins):
        raise ValueError('invalid utility margins')
    m = len(knee_ids)
    return dict(KneeCore=list(knee_ids), RandomM=random.Random(seed).sample(range(n), m),
        RecentM=list(range(n-m, n)),
        MarginUtilityM=sorted(range(n), key=lambda i: (margins[i], i))[:m])


def training_schedule(ids, steps=STEPS, grad_accum=GRAD_ACCUM):
    if not ids or steps <= 0 or grad_accum <= 0:
        raise ValueError('training requires nonempty selection and positive work')
    return [ids[i % len(ids)] for i in range(steps * grad_accum)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_receipt(path, payload):
    tmp = path.with_suffix('.writing.json')
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def score_pair(model, tok, prompt, pair, max_prompt, use_adapter=False):
    import torch
    from core import _encode_prompt_comp, sum_and_mean_logp_no_grad
    ep = _encode_prompt_comp(tok, prompt, pair['pos'], max_prompt, MAX_C)
    en = _encode_prompt_comp(tok, prompt, pair['neg'], max_prompt, MAX_C)
    if ep is None or en is None:
        raise ValueError('unscorable pair: no padding or silent skip allowed')
    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad(), (contextlib.nullcontext() if use_adapter else model.disable_adapter()):
        sp, lp = sum_and_mean_logp_no_grad(model, tok, prompt, pair['pos'], max_prompt, MAX_C)
        sn, ln = sum_and_mean_logp_no_grad(model, tok, prompt, pair['neg'], max_prompt, MAX_C)
    torch.cuda.synchronize()
    if any(x is None or not math.isfinite(float(x)) for x in (sp, sn, lp, ln)):
        raise ValueError('nonfinite or missing log probability')
    return dict(margin=float(lp-ln), accuracy=int(lp > ln),
        positive_mean_logp=float(lp), negative_mean_logp=float(ln),
        ref_tuple=[float(sp), float(sn), len(ep[1]), len(en[1])],
        prompt_tokens=len(ep[0]), constructed_prompt_tokens=len(tok(prompt, add_special_tokens=True).input_ids),
        positive_tokens=len(ep[1]), negative_tokens=len(en[1]),
        scoring_s=time.perf_counter()-t0, logprob_calls=2,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())


def measured_phase(tag, callback, power_dir):
    import torch
    from core import TegraWatch
    started = time.perf_counter()
    with TegraWatch(tag, log_dir=power_dir) as watch:
        value = callback()
        torch.cuda.synchronize()
    if watch.result.get('vin_mw_mean', 0.0) <= 0:
        raise RuntimeError('board VIN telemetry missing; no zero-energy substitution')
    return value, dict(wall_s=time.perf_counter()-started, board_power=watch.result,
        power_log=str(watch.path),
        energy_boundary='board VIN during this phase, including scoring; no decode generation')


def memory_guard(backbone):
    import torch
    from core import mem_snapshot
    snapshot = mem_snapshot()
    limit = 12 if backbone == 'dense_qwen15b' else 80
    if torch.cuda.max_memory_allocated() / 2**30 >= limit:
        raise RuntimeError(f'CUDA allocation exceeded prospective {limit} GiB cap: {snapshot}')
    return snapshot


def retrieval_user(args, user, split, model, tok, spec, power_dir, progress=None):
    from core import embed_texts, format_bare_prompt, format_vanilla_rag_prompt, select_mmr, select_topk_cosine
    from planning import build_capacity_bounded_prompt, center_embeddings_zero_param, compute_null_similarity_quantile
    profile, validation, held, split_meta = split
    uid = user['prefix']
    def embed_all():
        p = embed_texts(model, tok, [x['request'] for x in profile], max_len=192)
        v = embed_texts(model, tok, [x['request'] for x in validation], max_len=192)
        h = embed_texts(model, tok, [x['request'] for x in held], max_len=192)
        return center_embeddings_zero_param(p, p, v, h)
    (p_emb, v_emb, h_emb), embedding_cost = measured_phase(uid+'_retrieval_embeddings', embed_all, power_dir)
    gate = compute_null_similarity_quantile(p_emb, q=0.50)
    def dense_prompt(pair, q_emb, k):
        ids, _ = select_topk_cosine(q_emb, p_emb, k)
        pr = format_bare_prompt(pair['request']) if k == 0 else format_vanilla_rag_prompt(
            [profile[j] for j in reversed(ids)], pair['request'], max_ex_chars=1400)
        return pr, ids
    def validation_scores():
        rows = {}
        for k in K_VALUES:
            rows[k] = []
            for i, pair in enumerate(validation):
                pr, ids = dense_prompt(pair, v_emb[i], k)
                row = score_pair(model, tok, pr, pair, MAX_P_READ)
                row.update(validation_index=i, selected_profile_ids=ids)
                rows[k].append(row)
        return rows
    val, validation_cost = measured_phase(uid+'_validation', validation_scores, power_dir)
    chosen = choose_validation_k({k: [r['margin'] for r in rows] for k, rows in val.items()})
    validation_cost.update(pair_scoring_calls=20, logprob_calls=40)
    methods = {}
    result = dict(uid=uid, split=split_meta, embedding_cost=embedding_cost,
        validation=val, validation_cost=validation_cost, chosen_k=chosen, methods=methods)
    if progress: progress(result)
    for k in K_VALUES:
        def evaluate_dense(k=k):
            rows = []
            for i, pair in enumerate(held):
                pr, ids = dense_prompt(pair, h_emb[i], k)
                row = score_pair(model, tok, pr, pair, MAX_P_READ)
                row.update(heldout_index=i, ep_idx=pair.get('ep_idx'), selected_profile_ids=ids)
                rows.append(row)
            return rows
        rows, cost = measured_phase(uid+'_dense_k'+str(k), evaluate_dense, power_dir)
        methods['DenseK'+str(k)] = dict(k=k, items=rows, cost=cost)
        if progress: progress(result)
    for name, budget in [('SpecProbeGateMMR', int(spec.l_crit_tokens)), ('Fixed1408GateMMR', 1408)]:
        def evaluate_bounded(budget=budget):
            rows = []
            for i, pair in enumerate(held):
                ids, sim = select_mmr(h_emb[i], p_emb, k=8, lam=0.50, sim_floor=gate)
                pr, packed, added = build_capacity_bounded_prompt(tok, [profile[j] for j in ids],
                    pair['request'], l_crit_tokens=budget)
                if added > budget:
                    raise RuntimeError('strict context packing contract violated')
                row = score_pair(model, tok, pr, pair, MAX_P_READ)
                row.update(heldout_index=i, ep_idx=pair.get('ep_idx'), candidate_profile_ids=ids,
                    n_packed=packed, added_context_tokens=added, max_similarity=sim)
                rows.append(row)
            return rows
        rows, cost = measured_phase(uid+'_'+name, evaluate_bounded, power_dir)
        methods[name] = dict(budget_tokens=budget, gate_floor=gate, items=rows, cost=cost)
        if progress: progress(result)
    # An alias of already-observed held-out evaluations, not a second model run.
    methods['ValidationChosenK'] = dict(chosen_k=chosen, reused_method='DenseK'+str(chosen),
        items=methods['DenseK'+str(chosen)]['items'], incremental_scoring_calls=0)
    base = methods['DenseK0']['items']
    for method in methods.values():
        for i, row in enumerate(method['items']): row['delta_margin_vs_bare'] = row['margin']-base[i]['margin']
        method['mean_margin'] = sum(r['margin'] for r in method['items'])/8
        method['accuracy'] = sum(r['accuracy'] for r in method['items'])/8
    result['memory'] = memory_guard(args.backbone)
    return result


def coreset_user(args, user, split, model, tok, power_dir, progress=None):
    from core import embed_texts, format_bare_prompt, reset_peft_lora
    from planning import auto_dedilute_kneedle_select, center_embeddings_zero_param
    from execution import run_adjoint_microtxn_engine
    profile, validation, held, split_meta = split
    history = profile + validation
    uid = user['prefix']
    reset_peft_lora(model, seed=SEED)
    def references():
        return [score_pair(model, tok, format_bare_prompt(p['request']), p, MAX_P_TRAIN) for p in history]
    ref_rows, reference_cost = measured_phase(uid+'_reference_cache', references, power_dir)
    base_held, base_cost = measured_phase(uid+'_bare_heldout', lambda: [
        score_pair(model, tok, format_bare_prompt(p['request']), p, MAX_P_TRAIN) for p in held], power_dir)
    def select_all():
        raw = embed_texts(model, tok, [p['request'] for p in history], max_len=192)
        embs = center_embeddings_zero_param(raw, raw)[0]
        margins = [r['margin'] for r in ref_rows]
        knee, metadata = auto_dedilute_kneedle_select(history, margins, embs)
        selections = matched_coresets(24, knee, margins, seed=SEED)
        fixed, _ = auto_dedilute_kneedle_select(history, margins, embs, fixed_m=10)
        selections.update(FixedM10=fixed, Full24=list(range(24)))
        return selections, metadata
    (selections, knee_metadata), selection_cost = measured_phase(uid+'_coreset_selection', select_all, power_dir)
    methods = {}
    # Cycle the fixed order by user index to reduce systematic order confounding.
    names = list(selections)
    shift = args.user_index % len(names)
    names = names[shift:] + names[:shift]
    result = dict(uid=uid, split=split_meta, reference_items=ref_rows, reference_cost=reference_cost,
        base_heldout=base_held, base_evaluation_cost=base_cost, selection_cost=selection_cost,
        knee_metadata=knee_metadata, execution_order=names, methods=methods)
    if progress: progress(result)
    for name in names:
        selected = sorted(selections[name])
        pairs, refs = [history[i] for i in selected], [ref_rows[i]['ref_tuple'] for i in selected]
        reset_peft_lora(model, seed=SEED)
        _, execution = run_adjoint_microtxn_engine(model, tok, pairs, [], [],
            n_steps=STEPS, grad_accum=GRAD_ACCUM, lr=1e-4,
            max_p_train=MAX_P_TRAIN, max_c_train=MAX_C, gen_toks=4,
            drain_future_arrivals=False, reset_lora=False, seed=SEED,
            precomputed_refs=refs, sft_alpha=0.10)
        import shutil
        train_power_path = power_dir/(uid+'_'+name+'_train.log')
        shutil.copyfile('/tmp/deltaview_tegra/adjoint_microtxn_dpo.log', train_power_path)
        execution['power_log'] = str(train_power_path)
        execution['power_log_sha256'] = digest(train_power_path)
        if execution.get('tegra', {}).get('vin_mw_mean', 0.0) <= 0:
            raise RuntimeError('training board VIN telemetry missing')
        if execution.get('oom') or execution['training_pairs'] != 24 or execution['optimizer_updates'] != 6:
            raise RuntimeError('matched optimizer work contract violated')
        if execution['reference_pairs_computed'] != 0 or execution['reference_cache_hits'] != len(selected):
            raise RuntimeError('warm reference cache contract violated')
        items, evaluation_cost = measured_phase(uid+'_'+name+'_heldout', lambda: [
            score_pair(model, tok, format_bare_prompt(p['request']), p, MAX_P_TRAIN, use_adapter=True)
            for p in held], power_dir)
        exposed = training_schedule(selected)
        exposure = []
        for i in exposed:
            r = ref_rows[i]
            exposure.append(dict(history_id=i, prompt_tokens=r['prompt_tokens'],
                positive_tokens=r['positive_tokens'], negative_tokens=r['negative_tokens']))
        for i, row in enumerate(items):
            row.update(heldout_index=i, ep_idx=held[i].get('ep_idx'), delta_margin_vs_bare=row['margin']-base_held[i]['margin'])
        methods[name] = dict(selected_ids=selected, selection_ranking=selections[name], m=len(selected),
            execution=execution, items=items, evaluation_cost=evaluation_cost,
            pair_exposure=exposure,
            objective_exposure_tokens=sum(2*r['prompt_tokens']+r['positive_tokens']+r['negative_tokens'] for r in exposure),
            executor_forward_exposure_tokens=sum(3*r['prompt_tokens']+r['positive_tokens']+2*r['negative_tokens'] for r in exposure),
            mean_margin=sum(r['margin'] for r in items)/8, accuracy=sum(r['accuracy'] for r in items)/8,
            memory=memory_guard(args.backbone))
        if progress: progress(result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--suite', choices=['retrieval', 'coreset'], required=True)
    ap.add_argument('--backbone', choices=['dense_qwen15b', 'hybrid_qwen27b'], required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--users', type=int, default=6)
    ap.add_argument('--max-wall-s', type=int)
    args = ap.parse_args()
    if args.users <= 0: ap.error('--users must be positive')
    if args.max_wall_s is None: args.max_wall_s = 3600 if args.backbone == 'dense_qwen15b' else 7200
    if args.max_wall_s <= 0: ap.error('--max-wall-s must be positive')
    out = Path(args.out)
    companions = [out, *(out.with_suffix(s) for s in ['.partial.json', '.failed.json', '.provenance.json', '.writing.json'])]
    if any(p.exists() for p in companions): raise FileExistsError('new prefix required; no overwrite or resume')
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    payload = dict(schema='tierserve_author_controls_v1', ok=False,
        started_utc=datetime.now(timezone.utc).isoformat(), args=vars(args).copy(),
        source_sha256={str(p.relative_to(ROOT)):digest(p) for folder in ("src", "experiments") for p in sorted((ROOT/folder).glob("*.py"))},
        protocol=dict(seed=SEED, history=24, retrieval_profile=20, validation=4, heldout=8,
            k_values=list(K_VALUES), fixed_gate_budget=1408, train_max_prompt=MAX_P_TRAIN,
            max_completion=MAX_C, retrieval_max_prompt=MAX_P_READ, steps=STEPS,
            grad_accum=GRAD_ACCUM, lr=1e-4, sft_alpha=0.10,
            scope='exploratory matched controls on existing cohort; no untouched-confirmation claim',
            utility_baseline='ascending base preference margin, not formal EL2N'), users=[])
    # Exclusive provenance creation reserves the entire output prefix.
    with out.with_suffix('.provenance.json').open('x') as f: json.dump(payload, f, indent=2)
    def timeout_handler(*_): raise TimeoutError('prospective wall budget exhausted')
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.max_wall_s)
    try:
        root = Path(args.data_dir)
        manifest_path = root/'_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        payload['data_sha256'] = {'_manifest.json': digest(manifest_path)}
        eligible, skipped = [], []
        for row in manifest:
            path = root/(row['prefix']+'.json')
            payload['data_sha256'][path.name] = digest(path)
            user = json.loads(path.read_text())
            try: split = partition_user(user)
            except ValueError as exc:
                skipped.append(dict(uid=row['prefix'], reason=str(exc)))
                continue
            eligible.append((user, split))
        payload.update(eligible_user_ids=[u['prefix'] for u, _ in eligible], skipped_users=skipped)
        write_receipt(out.with_suffix('.provenance.json'), payload)
        if len(eligible) < args.users:
            raise ValueError(f'requested {args.users} eligible users, available {len(eligible)}; explicitly request a feasible count')
        import torch, transformers, peft
        from core import reset_peft_lora
        from planning import load_v2_backbone, run_specprobe_calibration
        if not torch.cuda.is_available(): raise RuntimeError('GPU controls require a serialized Thor qrun job')
        torch.cuda.set_per_process_memory_fraction(0.90)
        payload['versions'] = dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__)
        model, tok, payload['load_info'] = load_v2_backbone(args.backbone)
        payload['model_config'] = json.loads(model.config.to_json_string())
        payload['model_config_sha256'] = hashlib.sha256(model.config.to_json_string().encode()).hexdigest()
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout): module.p = 0.0
        memory_guard(args.backbone)
        spec = None
        power_dir = out.parent/(out.stem+'_power')
        if args.suite == 'retrieval':
            # SpecProbe starts its own TegraWatch; an outer watch would be stopped by it.
            t_spec = time.perf_counter()
            spec = run_specprobe_calibration(model, tok, backbone=args.backbone, sync_from_s1=False)
            payload.update(specprobe=spec.__dict__, specprobe_cost=dict(
                wall_s=time.perf_counter()-t_spec, board_energy_j=None,
                energy_boundary='full calibration energy unmeasured; internal read-high rail sample is not full calibration energy'))
        write_receipt(out.with_suffix('.provenance.json'), payload)
        for i, (user, split) in enumerate(eligible[:args.users]):
            args.user_index = i
            reset_peft_lora(model, seed=SEED)
            def publish(record):
                if len(payload['users']) == i: payload['users'].append(record)
                else: payload['users'][i] = record
                payload['wall_time_s'] = time.perf_counter()-started
                write_receipt(out.with_suffix('.partial.json'), payload)
            result = (retrieval_user(args, user, split, model, tok, spec, power_dir, publish)
                if args.suite == 'retrieval' else coreset_user(args, user, split, model, tok, power_dir, publish))
            publish(result)
            payload['wall_time_s'] = time.perf_counter()-started
            write_receipt(out.with_suffix('.partial.json'), payload)
            print('AUTHOR_CONTROLS', args.suite, args.backbone, i+1, user['prefix'], flush=True)
        payload.update(ok=True, wall_time_s=time.perf_counter()-started, memory_final=memory_guard(args.backbone))
        write_receipt(out, payload)
    except BaseException as exc:
        payload.update(ok=False, error=repr(exc), wall_time_s=time.perf_counter()-started)
        write_receipt(out.with_suffix('.failed.json'), payload)
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
