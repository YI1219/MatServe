#!/usr/bin/env python3
"""Build retrieval and coreset tables and figures from completed control receipts."""
from pathlib import Path
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stream_executor import ci, latex_mean_ci, LABELS

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results'
ROW=chr(92)*2
CORE=['KneeCore','RandomM','RecentM','MarginUtilityM','FixedM10','Full24']
RETR=['DenseK0','DenseK1','DenseK2','DenseK4','DenseK8','ValidationChosenK','SpecProbeGateMMR','Fixed1408GateMMR']
NAMES={'KneeCore':r'\dedilute','RandomM':'Random, same $M$','RecentM':'Recent, same $M$',
    'MarginUtilityM':'Margin utility, same $M$','FixedM10':'Fixed $M=10$','Full24':'Full history',
    'DenseK0':'Bare query','DenseK1':'Top-1','DenseK2':'Top-2','DenseK4':'Top-4','DenseK8':'Top-8',
    'ValidationChosenK':'Validation-selected $k$','SpecProbeGateMMR':'SpecProbe + gate/MMR',
    'Fixed1408GateMMR':'1,408 tokens + gate/MMR'}


def read(name):
    p=DATA/name
    d=json.loads(p.read_text())
    if not d.get('ok') or len(d['users'])!=d['args']['users']:
        raise ValueError('Incomplete controls: '+name)
    return d


def mean_delta(user, method):
    return float(np.mean([r['delta_margin_vs_bare'] for r in user['methods'][method]['items']]))


def tables(receipts, suite):
    order=CORE if suite=='coreset' else RETR
    caption=(r'Matched coreset controls: six histories, six updates/24 pair exposures per variant, common initialization and warm references. '
        r'Intervals resample histories. Objective tokens count each prompt twice; pair exposure is matched, token exposure is not. '
        r'Training VIN energy excludes shared reference construction, selection and evaluation.'
        if suite=='coreset' else
        r'Retrieval controls: six histories, 20-event profile, four separate validation events and eight later test items. '
        r'Intervals resample histories. Energy measures pair scoring, not answer generation; validation/calibration are separate. '
        r'SpecProbe and the fixed-budget row use the same gate/MMR.')
    headers=(r'Method & Mean $M$ & Margin gain [95\% CI] & Accuracy & Train kJ & Objective (k tok.) '
        if suite=='coreset' else
        r'Method & Mean $k$ / packed & Margin gain [95\% CI] & Accuracy & Prompt tokens & Score kJ ')
    lines=[r'\begin{table*}[t]',r'\centering',r'\caption{'+caption+'}',r'\label{tab:author_'+suite+'}',
        r'\footnotesize',r'\setlength{\tabcolsep}{5pt}',r'\begin{tabular}{@{}lrrrrr@{}}',r'\toprule',headers+ROW,r'\midrule']
    analysis=[]
    for d in receipts:
        b=d['args']['backbone']; users=d['users']
        lines += [r'\multicolumn{6}{@{}l}{\textbf{'+LABELS[b]+f", {len(users)} histories"+r'}} '+ROW]
        rows=[]
        for method in order:
            delta=[mean_delta(u,method) for u in users]
            accuracy=np.mean([u['methods'][method]['accuracy'] for u in users])
            summary=dict(backbone=b,method=method,n_histories=len(users),margin_gain_ci=ci(delta),accuracy=float(accuracy))
            if suite=='coreset':
                m=np.mean([u['methods'][method]['m'] for u in users])
                energy=np.mean([u['methods'][method]['execution']['tegra']['vin_j'] for u in users])/1000
                tokens=np.mean([u['methods'][method]['objective_exposure_tokens'] for u in users])/1000
                summary.update(mean_m=float(m),train_kj=float(energy),objective_exposure_k_tokens=float(tokens),
                    executor_forward_exposure_k_tokens=float(np.mean([u['methods'][method]['executor_forward_exposure_tokens'] for u in users])/1000),
                    knee_minus_method_ci=ci([mean_delta(u,'KneeCore')-mean_delta(u,method) for u in users]))
                cells=f"{NAMES[method]} & {m:.1f} & {latex_mean_ci(delta)} & {accuracy:.3f} & {energy:.2f} & {tokens:.1f} "
            else:
                def actual(u):
                    row=u['methods'][method]
                    return u['methods'][row['reused_method']] if 'reused_method' in row else row
                packed=np.mean([np.mean([item.get('n_packed',actual(u).get('k',0)) for item in actual(u)['items']]) for u in users])
                tokens=np.mean([np.mean([r['prompt_tokens'] for r in actual(u)['items']]) for u in users])
                energy=np.mean([actual(u)['cost']['board_power']['vin_j'] for u in users])/1000
                summary.update(mean_packed=float(packed),mean_prompt_tokens=float(tokens),pair_scoring_kj=float(energy),
                    specprobe_minus_method_ci=ci([mean_delta(u,'SpecProbeGateMMR')-mean_delta(u,method) for u in users]))
                cells=f"{NAMES[method]} & {packed:.1f} & {latex_mean_ci(delta)} & {accuracy:.3f} & {tokens:.0f} & {energy:.2f} "
            lines.append(cells+ROW); rows.append(summary)
        if suite=='retrieval':
            cost=np.mean([u['validation_cost']['board_power']['vin_j'] for u in users])
            lines.append(r'\multicolumn{6}{@{}l}{'+f"Validation scoring: mean {cost:.1f} J/history; synthetic calibration: {d['specprobe_cost']['wall_s']:.1f} s/backbone (energy not fully measured)."+r'} '+ROW)
        analysis+=rows;lines.append(r'\midrule')
    lines[-1]=r'\bottomrule';lines += [r'\end{tabular}',r'\end{table*}']
    (ROOT/'outputs/tables'/('tab_author_'+suite+'.tex')).write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,len(receipts),figsize=(7.16,2.6),squeeze=False)
    for j,d in enumerate(receipts):
        ax=axes[0,j]; users=d['users']; means=[]; errors=[]
        for method in order:
            m,lo,hi=ci([mean_delta(u,method) for u in users]);means.append(m);errors.append([m-lo,hi-m])
        colors=['#0072B2' if m in ['KneeCore','SpecProbeGateMMR'] else '#8C8C8C' for m in order]
        ax.bar(range(len(order)),means,color=colors)
        ax.errorbar(range(len(order)),means,yerr=np.array(errors).T,fmt='none',ecolor='#333333',capsize=1.5,lw=.6)
        labels=(['Knee','Random','Recent','Utility','Fixed10','Full24'] if suite=='coreset' else ['k=0','1','2','4','8','Val-k','Probe','1408'])
        ax.set_xticks(range(len(order)),labels,rotation=35,ha='right')
        ax.set_title(LABELS[d['args']['backbone']]);ax.set_ylabel('Margin gain (history 95% CI)')
        ax.axhline(0,color='#555555',lw=.5);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(ROOT/'outputs/figures'/('author_'+suite+'.pdf'));fig.savefig(ROOT/'outputs/figures'/('author_'+suite+'.png'),dpi=200);plt.close(fig)
    return analysis


def main():
    (ROOT / "outputs/tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs/figures").mkdir(parents=True, exist_ok=True)
    ap=argparse.ArgumentParser();ap.add_argument("--allow-partial-cohorts",action="store_true");args=ap.parse_args()
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','STIXGeneral'],
        'font.size':8,'pdf.fonttype':42,'ps.fonttype':42})
    missing=[f'{model}_{suite}_c1.json' for suite in ['retrieval','coreset'] for model in ['dense','hybrid'] if not (DATA/f'{model}_{suite}_c1.json').exists()]
    if missing and not args.allow_partial_cohorts: raise FileNotFoundError('Pending controls: '+', '.join(missing))
    summary={'pending':missing}
    for suite in ['retrieval','coreset']:
        receipts=[read(f'{model}_{suite}_c1.json') for model in ['dense','hybrid'] if (DATA/f'{model}_{suite}_c1.json').exists()]
        if receipts: summary[suite]=tables(receipts,suite)
    (ROOT/'outputs/controls.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
