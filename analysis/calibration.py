"""Plot recorded context-retention and prefill-cost calibration curves."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results'

def main():
    (ROOT / "outputs/tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs/figures").mkdir(parents=True, exist_ok=True)
    parser=argparse.ArgumentParser()
    parser.add_argument("--allow-partial-cohorts",action="store_true")
    parser.add_argument("--hybrid-from-control",action="store_true",
                        help="Use the completed GDN C1 retrieval calibration, identified separately from R1.")
    args=parser.parse_args()
    names_by_key = {'dense_qwen15b':'dense_stream_20u_r1.json',
        'mamba_falcon15b':'mamba_stream_6u_r1.json','hybrid_qwen27b':'hybrid_stream_6u_r1.json'}
    if args.hybrid_from_control:
        names_by_key['hybrid_qwen27b']='hybrid_retrieval_c1.json'
    payloads={}
    manifest={}
    pending=[]
    phases={}
    for key,name in names_by_key.items():
        path=SOURCE/name
        if not path.exists():
            pending.append(name)
            if args.allow_partial_cohorts: continue
            raise FileNotFoundError("Pending fresh calibration: "+name)
        d=json.loads(path.read_text())
        control=key=='hybrid_qwen27b' and args.hybrid_from_control
        backbone=d['args']['backbone'] if control else d['result']['backbone']
        if not d.get('ok') or backbone!=key:
            raise ValueError('Incomplete or mismatched fresh calibration: '+name)
        payloads[key]=d['specprobe'] if control else d['result']['specprobe']
        phases[key]='C1 retrieval-control calibration' if control else 'R1 recorded-order replay calibration'
        manifest[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','STIXGeneral'],
        'font.size':8,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.16,2.45))
    if not payloads: raise ValueError("No complete calibration receipts")
    keys=list(payloads)
    labels={'dense_qwen15b':'Dense 1.5B','mamba_falcon15b':'Mamba 1.5B','hybrid_qwen27b':'GDN 27B'}
    palette={'dense_qwen15b':'#332288','mamba_falcon15b':'#D55E00','hybrid_qwen27b':'#CC79A7'}
    names=[labels[k] for k in keys];colors=[palette[k] for k in keys]
    for key,name,col in zip(keys,names,colors):
        sp=payloads[key]
        label=f"{name} ({sp['l_crit_tokens']} tok.)"
        axes[0].plot(sp['grid_lengths'],sp['composite_fidelity'],label=label,color=col,marker='o',markersize=3,lw=1.1)
        axes[1].plot(sp['grid_lengths'],sp['prefill_ms'],color=col,marker='o',markersize=3,lw=1.1)
    axes[0].axhline(1/np.sqrt(2),color='#888888',ls=':',lw=1)
    axes[0].set(xlabel='Distractor prefix (tokens)',ylabel='Synthetic-anchor fidelity',ylim=(0,1.05),title='(a) Context-budget calibration')
    axes[1].set(xlabel='Distractor prefix (tokens)',ylabel='Prefill latency (ms)',yscale='log',title='(b) Prefill cost')
    for ax in axes: ax.spines[['top','right']].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,fontsize=7)
    fig.tight_layout(rect=(0,0,1,.85))
    for ext in ['pdf','png']: fig.savefig(ROOT/'outputs/figures'/('author_calibration.'+ext),dpi=200)
    plt.close(fig)
    audit={'sources_sha256':manifest,'pending':pending,'calibration_source_studies':phases,
        'boundary':'Actual calibration observations from completed, explicitly named studies; no archived or reconstructed curves. A C1 calibration does not stand in for a policy replay.'}
    (ROOT/'outputs/calibration.json').write_text(json.dumps(audit,indent=2)+'\n')

if __name__=='__main__': main()
