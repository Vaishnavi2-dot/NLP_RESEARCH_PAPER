# -*- coding: utf-8 -*-
"""Cross-encoder reranking experiment.

Retrieval is the binding constraint on AVeriTeC (hybrid Recall@5 = 0.415), so the
question is whether a reranker over a deeper candidate pool lifts it. Protocol:
retrieve top-POOL with hybrid (RRF of BM25 + dense), rerank with a cross-encoder,
keep top-5, and compare Recall@5 / Precision@5 / Evidence-F1@5 against the
un-reranked top-5 on exactly the same claims.

Reuses the notebook's cached document embeddings so nothing is re-encoded.
Deterministic; writes truthlens_outputs/rerank_eval.csv.
"""
import json, math, os, re, sys, time, heapq
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CACHE = os.path.join(DATA, "cache")
OUT = os.path.join(HERE, "truthlens_outputs")
SEED = 42
POOL = 50
K = 5
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

STOP = set("a an the is are was were be been being of to in on at for with by from as "
           "that this it its and or not no".split())


def bm25_tokens(s):
    return [w for w in re.findall(r"\w+", s.lower()) if w not in STOP and len(w) > 1]


class BM25:
    def __init__(self, texts, ids, k1=1.5, b=0.75):
        self.k1, self.b, self.ids = k1, b, list(ids)
        docs = [bm25_tokens(t) for t in texts]
        self.N = len(docs)
        self.dl = [len(d) for d in docs]
        self.avgdl = sum(self.dl) / max(1, self.N)
        df = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.postings = defaultdict(list)
        for i, d in enumerate(docs):
            for t, f in Counter(d).items():
                self.postings[t].append((i, f))

    def search(self, q, k):
        scores = defaultdict(float)
        for t in bm25_tokens(q):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, f in self.postings.get(t, ()):
                dl = self.dl[i]
                scores[i] += idf * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return heapq.nlargest(k, ((s, i) for i, s in scores.items()))


def hybrid_pool(bm25, doc_vecs, qv, k, rrf_k=60, cand=200):
    """Same RRF fusion as HybridRetriever: returns k document indices."""
    sims = doc_vecs @ qv
    n = min(cand, sims.shape[0])
    part = np.argpartition(-sims, n - 1)[:n]
    dense_order = part[np.argsort(-sims[part])]
    dense_rank = {int(i): r + 1 for r, i in enumerate(dense_order)}
    bm = bm25.search(" ".join([]), 0) if False else None
    return dense_rank, sims


def run(name, claims, claim_texts, corpus_texts, gold_sets, emb_path, ce):
    print(f"\n=== {name} ===", flush=True)
    doc_vecs = np.load(emb_path).astype(np.float32)
    doc_vecs /= np.maximum(np.linalg.norm(doc_vecs, axis=1, keepdims=True), 1e-9)
    assert doc_vecs.shape[0] == len(corpus_texts), \
        f"embedding/corpus mismatch {doc_vecs.shape[0]} vs {len(corpus_texts)}"
    print(f"corpus {len(corpus_texts)} docs | claims {len(claim_texts)}", flush=True)

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=DEV)
    qvs = enc.encode(claim_texts, normalize_embeddings=True, batch_size=256,
                     convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    bm25 = BM25(corpus_texts, list(range(len(corpus_texts))))
    print("bm25 index built", flush=True)

    base_r, base_p, base_f = [], [], []
    rr_r, rr_p, rr_f = [], [], []
    t0 = time.time()
    for qi, (gold, qv) in enumerate(zip(gold_sets, qvs)):
        if not gold:
            continue
        # ---- hybrid pool of POOL candidates (RRF over bm25 + dense) ----------
        sims = doc_vecs @ qv
        n = min(200, sims.shape[0])
        part = np.argpartition(-sims, n - 1)[:n]
        dense_order = part[np.argsort(-sims[part])]
        dense_rank = {int(i): r + 1 for r, i in enumerate(dense_order)}
        bm_res = bm25.search(claim_texts[qi], max(POOL, 50))
        bm_rank = {int(i): r + 1 for r, (s, i) in enumerate(bm_res)}
        cand = set(dense_rank) | set(bm_rank)
        fused = []
        for idx in cand:
            s = 0.0
            if idx in bm_rank:
                s += 1.0 / (60 + bm_rank[idx])
            if idx in dense_rank:
                s += 1.0 / (60 + dense_rank[idx])
            fused.append((idx, s))
        pool = [i for i, _ in sorted(fused, key=lambda x: -x[1])[:POOL]]

        def score(top):
            hits = len(set(top) & gold)
            return hits / len(gold), hits / K, 2 * hits / (K + len(gold))

        r, p, f = score(pool[:K])
        base_r.append(r); base_p.append(p); base_f.append(f)

        # ---- rerank the pool with the cross-encoder --------------------------
        pairs = [(claim_texts[qi], corpus_texts[i]) for i in pool]
        ce_scores = ce.predict(pairs, batch_size=128, show_progress_bar=False)
        order = np.argsort(-np.asarray(ce_scores))
        top = [pool[i] for i in order[:K]]
        r, p, f = score(top)
        rr_r.append(r); rr_p.append(p); rr_f.append(f)

        if (qi + 1) % 100 == 0:
            print(f"  {qi+1} claims | {time.time()-t0:.0f}s", flush=True)

    def agg(rs, ps, fs, label):
        return dict(dataset=name, system=label, n_claims=len(rs),
                    recall_at_5=round(float(np.mean(rs)), 4),
                    precision_at_5=round(float(np.mean(ps)), 4),
                    evidence_f1_at_5=round(float(np.mean(fs)), 4))

    rows = [agg(base_r, base_p, base_f, f"hybrid top-{K} (baseline)"),
            agg(rr_r, rr_p, rr_f, f"hybrid top-{POOL} + cross-encoder rerank -> top-{K}")]
    for r in rows:
        print("  ", r, flush=True)
    print(f"  Recall@5 {rows[0]['recall_at_5']:.4f} -> {rows[1]['recall_at_5']:.4f} "
          f"({rows[1]['recall_at_5'] - rows[0]['recall_at_5']:+.4f})", flush=True)
    return rows


if __name__ == "__main__":
    import torch
    DEV = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", DEV)
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(RERANKER, device=DEV, max_length=384)
    print("reranker:", RERANKER, flush=True)

    all_rows = []

    # ---- AVeriTeC ----------------------------------------------------------
    av_dev = json.load(open(os.path.join(DATA, "averitec_dev_records.json"), encoding="utf-8"))
    av_corpus = [json.loads(l) for l in open(os.path.join(DATA, "averitec_corpus.jsonl"), encoding="utf-8")]
    all_rows += run("AVeriTeC", av_dev, [c["claim"] for c in av_dev],
                    [d["text"] for d in av_corpus],
                    [set(c["evidence_ids"]) for c in av_dev],
                    os.path.join(CACHE, "emb_averitec_minilm_7022.npy"), ce)

    # ---- FEVER (expanded corpus) -------------------------------------------
    fe = json.load(open(os.path.join(DATA, "fever_sample.json"), encoding="utf-8"))
    fe_corpus = [json.loads(l) for l in open(
        os.path.join(DATA, "fever_wiki_corpus_expanded.jsonl"), encoding="utf-8")]
    id2idx = {d["id"]: i for i, d in enumerate(fe_corpus)}
    gold_sets = [{id2idx[g] for g in c["gold_evidence_ids"] if g in id2idx} for c in fe]
    all_rows += run("FEVER (expanded)", fe, [c["claim"] for c in fe],
                    [d["text"] for d in fe_corpus], gold_sets,
                    os.path.join(CACHE, "emb_fever_expanded_minilm_207089.npy"), ce)

    import pandas as pd
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUT, "rerank_eval.csv"), index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nwrote {os.path.join(OUT, 'rerank_eval.csv')}")
