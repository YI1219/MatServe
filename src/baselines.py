"""Retrieval, summary-memory and coreset baselines."""
from __future__ import annotations
from collections import Counter
import math
import re
from typing import Sequence
import torch
from core import format_bare_prompt, word_edit_ratio


def _tokenize_bm25(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())



def select_lamp_bm25(query: str, corpus: Sequence[dict], k: int, k1: float = 1.5, b: float = 0.75) -> list[int]:
    """LaMP-BM25 (Salemi et al., ACL 2024): Okapi BM25 sparse personal history retriever."""
    if k <= 0 or not corpus:
        return []
    docs = [_tokenize_bm25(c["request"]) for c in corpus]
    n_docs = len(docs)
    avgdl = sum(len(d) for d in docs) / max(n_docs, 1)
    df: Counter[str] = Counter()
    for d in docs:
        for w in set(d):
            df[w] += 1
    q_toks = _tokenize_bm25(query)
    scores = []
    for idx, d in enumerate(docs):
        tf = Counter(d)
        dl = len(d)
        sc = 0.0
        for qt in q_toks:
            if tf[qt] == 0:
                continue
            idf = math.log(1.0 + (n_docs - df[qt] + 0.5) / (df[qt] + 0.5))
            denom = tf[qt] + k1 * (1.0 - b + b * (dl / max(avgdl, 1e-6)))
            sc += idf * (tf[qt] * (k1 + 1.0)) / max(denom, 1e-8)
        scores.append((sc, idx))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [idx for _, idx in scores[: min(k, n_docs)]]



def synthesize_memgpt_persona_block(prefix_pairs: Sequence[dict], max_examples: int = 3) -> str:
    """Build a bounded style-summary prompt from response lengths, formats and edits.
    
    This local summary baseline is inspired by MemGPT (Packer et al., 2024).
    """
    if not prefix_pairs:
        return ""
    ratios, edits, styles = [], [], Counter()
    for p in prefix_pairs:
        wn = max(len(p["neg"].split()), 1)
        wp = max(len(p["pos"].split()), 1)
        ratios.append(wp / wn)
        edits.append(word_edit_ratio(p["pos"], p["neg"]))
        pos = p["pos"]
        if "\n-" in pos or "\n•" in pos or pos.lstrip().startswith(("- ", "• ")):
            styles["bulleted-list"] += 1
        elif "\n1." in pos or pos.lstrip().startswith("1."):
            styles["numbered-steps"] += 1
        elif wp <= 25:
            styles["terse-direct"] += 1
        else:
            styles["structured-prose"] += 1
    med_ratio = sorted(ratios)[len(ratios) // 2]
    dom_style = styles.most_common(1)[0][0]
    mean_edit = sum(edits) / len(edits)
    direction = (
        f"compresses assistant drafts to {med_ratio:.2f}x length"
        if med_ratio < 0.75
        else (f"expands drafts to {med_ratio:.2f}x detail" if med_ratio > 1.25 else f"preserves ~{med_ratio:.2f}x length")
    )
    # Pick top-2 highest-edit representative anchors for core memory
    ranked = sorted(prefix_pairs, key=lambda x: word_edit_ratio(x["pos"], x["neg"]), reverse=True)[:max_examples]
    snips = []
    for ex in ranked:
        rq = ex["request"].strip()[:220]
        ps = ex["pos"].strip()[:360]
        snips.append(f"Request: {rq}\nUser's response:\n{ps}")
    core = (
        f"[Core Persona Memory: User {direction}, prefers {dom_style} formatting "
        f"(mean rewrite ratio={mean_edit:.2f}).]\n\n"
        + "\n\n".join(snips)
    )
    return core



def format_memgpt_prompt(core_block: str, query: str) -> str:
    if not core_block:
        return format_bare_prompt(query)
    return f"{core_block}\n\nRequest: {query.strip()}\n\nUser's response:\n\n"



def select_coreset_el2n(mean_margins: Sequence[float | None], m: int = 18) -> list[int]:
    """Select high-error preference pairs without a diversity term.
    
    Local EL2N-style baseline following Paul et al. (NeurIPS 2021).
    """
    n = len(mean_margins)
    if n <= m:
        return list(range(n))
    scored = []
    for i, bm in enumerate(mean_margins):
        err = 1.0 / (1.0 + math.exp(max(-20.0, min(20.0, (bm if bm is not None else 0.0) / 0.25))))
        scored.append((err, i))
    scored.sort(key=lambda x: x[0], reverse=True)
    return sorted(idx for _, idx in scored[:m])



def select_coreset_craig_facility_location(pair_embs: torch.Tensor, m: int = 18) -> list[int]:
    """Select embedding medoids by facility-location coverage, without edit utility.
    
    Local CRAIG-style baseline following Mirzasoleiman et al. (ICML 2020).
    """
    n = pair_embs.shape[0]
    if n <= m or pair_embs.ndim != 2 or pair_embs.shape[1] <= 1:
        return list(range(min(n, m)))
    sim_mat = torch.clamp(pair_embs @ pair_embs.t(), min=0.0).tolist()
    cov = [0.0] * n
    selected: list[int] = []
    sel_set: set[int] = set()
    for _ in range(m):
        best_i, best_gain = -1, -1e9
        for i in range(n):
            if i in sel_set:
                continue
            gain = sum(max(0.0, sim_mat[v][i] - cov[v]) for v in range(n))
            if gain > best_gain:
                best_gain, best_i = gain, i
        if best_i < 0:
            break
        selected.append(best_i)
        sel_set.add(best_i)
        for v in range(n):
            if sim_mat[v][best_i] > cov[v]:
                cov[v] = sim_mat[v][best_i]
    return sorted(selected)

