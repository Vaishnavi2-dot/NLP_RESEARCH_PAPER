# A per-document Python loop in retrieval is fine at 7k docs and fatal at 200k

> **Rule:** dense scoring is one matmul against a pre-normalised matrix plus `argpartition` — never a Python loop of `cosine()` calls, and never `list.index()` in a query path.

## What was wrong

The original `HybridRetriever.search()` did, **per query**:

```python
dense_sim = {i: cosine(qv, self.doc_vecs[i]) for i in range(len(self.ids))}  # 200k cosine() calls
dense_scores = sorted(...)                                                   # full sort of 200k
ranked = sorted(((self.ids.index(i_d), sc) for sc, i_d in bm25_res), ...)    # O(n) scan each
# hybrid branch then looped over ALL ids to build the fusion table
```

At the demo scale (30 docs) and even at 6,934 it was invisible. Extrapolated to the
207,089-sentence corpus it was ~75 min for FEVER alone — the kind of thing that reads
as "the cell hung".

## What it became

- pre-normalise the doc matrix once, so cosine == dot: `sims = self.doc_vecs @ qv`
- top-k via `np.argpartition(-sims, n-1)[:n]`, then sort only those n
- `id2idx` dict instead of `list.index()`
- RRF fusion over the **union of the two candidate lists**, not all docs.
  Exact for the returned top-k as long as `CAND >> k`: a doc outside the dense
  top-CAND and outside the BM25 list scores at most `1/(rrf_k + CAND)`, below
  anything ranked near the top. `CAND = 200`, `k = 5`.

Result: 207,089 docs, 600 claims x 3 modes in **~9 s per mode**.

## Pure-Python BM25 was never the problem

Measured on the same 207k corpus: tokenise 1.0 s, inverted index 1.4 s
(168k vocab), **3 ms/query**. An inverted index with `heapq.nlargest` for top-k is
entirely adequate at this scale — the dense side was the whole cost.

## Related metric bug

`precision@k` was computed as `hits / len(top)`. BM25 legitimately returns fewer
than k documents when no query term is in the vocabulary, which (a) raises
`ZeroDivisionError` when it returns none, and (b) rewards a retriever that returns
one lucky document over one that returns k. **Precision@k divides by k.**
