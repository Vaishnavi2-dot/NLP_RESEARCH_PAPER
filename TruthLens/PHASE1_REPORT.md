# TruthLens — Phase 1 Completion Report

**Scope:** build the end-to-end research notebook and run the real experiments needed to
test the TruthLens framework from the project synopsis. **Phase 2 (the paper) has not
been started.**

All numbers below were produced by executing `TruthLens_Research_Project.ipynb` end to
end (42/42 code cells, no errors) on real public benchmarks. Nothing is synthetic. The
demonstration corpus in Parts 2–9 exists only to validate the pipeline and every one of
its artifacts is written with a `demo_` prefix.

---

## 1. Starting point vs. what Phase 1 added

The inherited project had a well-structured notebook and three working data-prep
scripts, but **it had never been executed**: every cell had `execution_count: None`,
there was no `data/cache/`, and `truthlens_outputs/` held only demonstration-mode
artifacts. The three "REAL EXPERIMENT" sections were unrun code, while the closing
section already asserted they were *"Completed (real data)"* and quoted timings for
hardware the runs had never happened on.

The prior attempt stalled on hardware: its log measured mDeBERTa NLI at **0.9 pairs/s**
on a Windows CPU, so the author downgraded to a 6-layer MiniLM and still never ran the
experiments.

Phase 1 therefore did the following.

| | |
|---|---|
| **Ported** | Windows-hardcoded paths in all 4 scripts → repo-relative; Python 3.11 venv; pinned `requirements.txt` |
| **Fixed the bottleneck** | MPS device selection. The FEVER-trained `DeBERTa-v3-large` runs at **69 pairs/s** on this M1 Pro vs 5.3 on CPU — and 167× the prior machine's mDeBERTa rate. Model tier now follows the accelerator and is part of every cache key |
| **Scaled the retriever** | `HybridRetriever` scored every document in a Python loop per query (plus `list.index()`); fine at 7k docs, ~75 min at 207k. Vectorised to one matmul + `argpartition` → **9 s** per mode |
| **Ran the experiments** | FEVER, AVeriTeC, FactDrill — all three, with baselines, ablations, bootstrap CIs and McNemar tests |
| **Added** | FEVER distractor corpus (207k sentences), temporal robustness (11.7), conflict-rule study (11.4b), real evidence graph + source provenance (11.8), majority-class baselines |
| **Corrected** | Three protocol problems and two correctness bugs — §5 and §6 |

---

## 2. Experimental setup

| | |
|---|---|
| **FEVER** | 600 dev claims (200/label, `SEED=42`), 397 with gold evidence. Evidence corpus in two settings: gold-pages-only (6,934 sents) and **expanded** (207,089 sents, +45,667 random distractor pages). Sys1 trained on 20k train claims |
| **AVeriTeC** | All 500 dev claims, 4-way labels incl. *Conflicting Evidence*. 7,022-doc shared QA-evidence corpus with real source URLs. Sys1 trained on the 3,068-claim train split. 59 India-located claims reported separately |
| **FactDrill** | 13,796 items (8,732 EN + 5,064 HI) from the open Zenodo deposit; claim→fact-check retrieval, 400 queries/language |
| **Models** | NLI `DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (MPS, fp16) · encoder `paraphrase-multilingual-MiniLM-L12-v2` |
| **Hardware** | Apple M1 Pro (8-core CPU / 14-core GPU), 32 GB, macOS 26.6, MPS. Python 3.11.16, torch 2.14.0, transformers 5.17.0 |
| **Runtime** | Cold run ≈ 27 min (6 min of it encoding the 207k corpus); fully cached re-run **108 s** |

**Systems** (the synopsis's four baselines, plus two reference points and ablations):
`Sys-Maj` majority class · `Sys0` binary text-only · `Sys1` text-only 4-way (synopsis
baseline 1) · `Sys2` retrieval + surface heuristic, no NLI (synopsis baseline 2,
"standard vector RAG") · `Sys3` retrieval + NLI top-1 (baseline 3) · `Sys4` full
TruthLens (baseline 4).

---

## 3. Results

### 3.1 Verdict classification — FEVER (600 claims, 3-class, **expanded 207k corpus**)

| System | Accuracy | Macro-F1 |
|---|---|---|
| Sys-Maj (majority = Supported) | 0.333 | 0.167 |
| Sys1 text-only classifier | 0.440 | 0.421 |
| Sys2 retrieval + heuristic (no NLI) | 0.540 | 0.504 |
| Sys3 retrieval + NLI (top-1) | 0.660 | 0.663 |
| **Sys4 TruthLens (full)** | **0.723** [0.687, 0.758] | **0.722** [0.685, 0.756] |
| — ablation: BM25-only | 0.710 | 0.712 |
| — ablation: dense-only | 0.667 | 0.669 |

A clean monotone ladder across exactly the progression the synopsis proposed:
text-only → retrieval → +NLI → +aggregation. **+0.283 accuracy over the text-only
baseline**, and this is on the *hard* corpus. McNemar: Sys4 vs Sys3 *p* = 0.00089;
Sys4 vs Sys1 *p* < 1e-5. Hybrid retrieval beats both single retrievers.

### 3.2 Verdict classification — AVeriTeC (500 claims, 4-class, 61 % *Refuted*)

| System | Accuracy | Macro-F1 |
|---|---|---|
| Sys-Maj (majority = Refuted) | 0.610 | 0.189 |
| Sys1 text-only classifier | 0.618 | 0.293 |
| Sys2 retrieval + heuristic (no NLI) | 0.212 | 0.176 |
| Sys3 retrieval + NLI (top-1) | 0.324 | 0.252 |
| **Sys4 TruthLens (full)** | 0.472 [0.430, 0.514] | **0.334** [0.293, 0.373] |
| — ablation: no conflict label | 0.498 | 0.331 |
| — ablation: dense-only | 0.484 | 0.342 |

**This result is mixed and must not be read off the accuracy column.** Sys1's accuracy
(0.618) is statistically indistinguishable from always predicting *Refuted* (0.610) — it
is a majority predictor, which is why its macro-F1 collapses to 0.293. On macro-F1, the
metric that survives the class imbalance, the ordering is **Sys4 (0.334) > Sys1 (0.293)
> Sys-Maj (0.189)**.

The honest summary: on AVeriTeC the evidence pipeline makes real distinctions that a
text classifier cannot, but it does **not** beat a majority-exploiting baseline on raw
accuracy. The bottleneck is retrieval — Recall@5 is only 0.415 (§3.3), so the NLI layer
is frequently reasoning over evidence that does not contain the answer.

*India subset (n = 59):* Sys1 0.678 / 0.234, Sys4 0.475 / 0.312 — the same pattern.

### 3.3 Evidence retrieval

| Dataset | Setting | corpus | BM25 R@5 | Dense R@5 | **Hybrid R@5** |
|---|---|---|---|---|---|
| FEVER | gold-pages-only | 6,934 | 0.764 | 0.724 | **0.788** |
| FEVER | **expanded** | 207,089 | 0.627 | 0.590 | **0.669** |
| AVeriTeC | QA shared corpus | 7,022 | 0.398 | 0.411 | **0.415** |

Hybrid (RRF of BM25 + dense) wins in every setting — the synopsis's hybrid-retrieval
hypothesis holds. Precision@5 and Evidence-F1@5 are in
`consolidated_retrieval_eval.csv`.

### 3.4 Multilingual retrieval — FactDrill (hit@5, 400 queries/language)

| Setting | | BM25 | Dense | Hybrid |
|---|---|---|---|---|
| leaky *(artefact)* | EN | 1.000 | 0.973 | 0.990 |
| leaky *(artefact)* | HI | 0.993 | 0.878 | 0.980 |
| **de-leaked** | **EN** | **0.798** | 0.428 | 0.725 |
| **de-leaked** | **HI** | **0.640** | 0.348 | 0.635 |

Two real findings once the leakage is removed (§5.2): a clear **English → Hindi gap**
(BM25 0.798 → 0.640), and — unlike FEVER and AVeriTeC — **BM25 beats hybrid here**. The
multilingual MiniLM encoder is weak on long Indian-language fact-check articles (dense
0.428 EN / 0.348 HI), and RRF fusion with a weak dense arm drags hybrid below pure
lexical retrieval. That is a concrete, actionable result for the Indian-language scope.

### 3.5 Conflict-aware verdicts — does the headline novelty earn its place?

Thresholds selected on **600 AVeriTeC train claims**, then applied to dev.

| Conflict rule | Accuracy | Macro-F1 | Conflicting-F1 | # predicted (gold = 38) |
|---|---|---|---|---|
| off | 0.498 | 0.331 | 0.000 | 0 |
| blunt (any ent+con co-occurrence) | 0.472 | 0.334 | 0.094 | 26 |
| gated (train-selected) | 0.474 | 0.336 | 0.095 | 25 |

**A negative result, reported as such.** Conflict-awareness buys +0.003 macro-F1 for
−0.024 accuracy, and gating does not rescue it — the train-selected thresholds land
essentially on the blunt rule, and McNemar blunt vs gated gives *p* = 1.0.
Conflicting-F1 of ~0.09 means the class is barely detected. Detecting cherry-picked or
genuinely conflicting evidence needs a different mechanism than "the NLI head disagreed
across retrieved passages" — this is the clearest open problem Phase 2 inherits.

### 3.6 Temporal robustness (AVeriTeC, real URL-derived dates)

*T1 — characterisation.* Claim dates parsed for **500/500**; 2,954/7,022 evidence docs
carry a recoverable date (2,207 archive upper bounds, 747 exact publication dates).
Only **9.0 %** of gold evidence provably pre-dates its claim; **362/437** claims have
none at all; median claim→evidence gap **+107 days**.

*T2/T3 — ablations.*

| Condition | Accuracy | Macro-F1 |
|---|---|---|
| A Sys4, unrestricted evidence (depth-30 pool) | 0.466 | 0.332 |
| B minus *provably* post-claim evidence (T2) | 0.462 | 0.329 |
| C *provably* contemporaneous only (T3, **bound**) | 0.308 | 0.213 |

T2 is −0.004 (McNemar *p* = 0.48): removing only leakage we can *prove* changes almost
nothing, because just 16 gold items have a two-sided publication date. The test is
**underpowered by date coverage**, not evidence of no leakage. T3 is −0.158
(*p* < 1e-5) but is dominated by forced abstentions and is reported as a **bound on
contemporaneous verification**, not a system score. AVeriTeC is a retrospective corpus;
real-time verification cannot be measured properly on it.

### 3.7 Narrative clustering, evidence graph, provenance

*Narrative clustering* (AVeriTeC dev, unsupervised, vs real labels): best k = 13,
silhouette 0.111, **NMI vs location 0.349**, vs claim-type 0.116, vs verdict 0.076.
Clusters track geography far more than they track claim type or truth status — weak
absolute structure, honestly reported.

*Evidence graph* (real): 1,975 nodes / 1,974 edges over `claim ↔ evidence ↔ source`,
exported to `averitec_evidence_graph.gexf`.

*Source provenance:* 998 evidence links across **488 distinct domains**; top-10 domains
account for only 19.0 % — the corpus is not source-concentrated. But
**47.4 % of claims rest on a single source domain** (mean 1.79 distinct domains/claim),
which directly substantiates the source-diversity risk the synopsis raises.

---

## 4. Does the synopsis's research question hold?

> *Can an evidence-aware, claim-level NLP pipeline with source comparison and
> event-level narrative analysis improve factual verification over conventional text
> classification or similarity-only retrieval?*

**On FEVER, decisively yes** — 0.723 vs 0.440 accuracy over text-only, monotone across
the component ladder, significant under McNemar, on a 207k-sentence corpus.

**On AVeriTeC, only partially** — better than text-only on macro-F1 (0.334 vs 0.293)
but below a majority-class baseline on accuracy, with retrieval (R@5 0.415) as the
binding constraint.

**The conflict-awareness contribution is not yet supported** (§3.5), and the temporal
contribution **cannot be properly tested on AVeriTeC** (§3.6). Both should be presented
in Phase 2 as open problems with measurements attached, not as wins.

---

## 5. Protocol problems found — and how each was handled

Each would have yielded a plausible-looking but misleading number.

1. **FEVER's gold-pages-only corpus inflates retrieval.** Every distractor came from a
   page already known relevant. Adding 200k random distractor sentences drops hybrid
   Recall@5 **0.788 → 0.669**. Both settings are now reported; verdicts run on the hard one.
2. **FactDrill's `claim` is copied from its own article** — the claim's first 120 chars
   appear verbatim in 89.8 % of gold articles, so retrieval was largely duplicate
   detection (leaky BM25 hit@5 = **1.000**). A de-leaked corpus removes the claim span
   from its own article (verbatim leakage 89.8 % → 4.7 %, 0 documents emptied); both
   settings reported, de-leaked is the result.
3. **AVeriTeC evidence is retrospective** and its dates are mostly *archive snapshots*,
   which are one-sided: they can prove pre-dating, never post-dating. The analysis
   asserts only what the dates support and labels the rest `unverified`.

## 6. Correctness bugs found and fixed

- **`bootstrap_ci` resampled `y_true` and `y_pred` independently**, destroying the
  pairing. Every CI in the notebook was reporting the random-pairing rate: FEVER Sys4
  read `0.333 [0.295, 0.375]` while the same predictions scored **0.72** in
  `classification_report` ten lines away. Now one index vector per replicate; verified
  against synthetic predictions of known accuracy.
- **`precision@k` divided by `len(top)`**, not `k` — raised `ZeroDivisionError` when
  BM25 matched no query term, and rewarded returning one lucky document over k.
- Placeholder `source_url` values (literal `"metadata"`) were being counted as the
  most-cited publisher domain.

## 7. Experiments that ran

Recorded live in `truthlens_outputs/experiment_registry.json` — an experiment appears
there only because its cell executed. 12 entries: FEVER retrieval (2 settings) and
verdict classification; AVeriTeC retrieval, 4-way verdicts, India subset, conflict-rule
study, narrative clustering, T1 timing characterisation, T2/T3 temporal ablations,
evidence graph + provenance; FactDrill characterisation and EN/HI retrieval (2
settings).

## 8. Experiments that could **not** run, and why

| # | Not run | Why | To close it |
|---|---|---|---|
| 1 | FEVEROUS | Table/structured evidence needs the FEVEROUS DB + linearisation pipeline | Separate build; claim loader is trivial, evidence side is not |
| 2 | FactDrill verdict classification | **No structured verdict labels** in the deposit; verdicts sit in free text. Pattern-matching titles would be constructing supervision and scoring against it | Annotate a stratified sample (protocol in notebook 14.5), report κ |
| 3 | Full-Wikipedia FEVER retrieval | 207k-sentence corpus, not the 5.4M-page index | FAISS/ANN index; the sampler already streams the dump |
| 4 | AVeriTeC knowledge store | Multi-GB per-claim search dumps; released QA evidence used instead | Makes AVeriTeC retrieval open-web rather than shared-corpus |
| 5 | AVerImaTeC (image-text) | Out of Phase-1 scope per the synopsis MVP (text-first) | Phase 5 in the synopsis roadmap |
| 6 | Multilingual **NLI** evaluation | FactDrill has no verdict labels, so entailment cannot be scored in Hindi. Multilingual capability is evaluated on **retrieval** only | Labelled Hindi claim–evidence pairs |
| 7 | Claim extraction on real data | No public benchmark annotates atomic claims for social posts; P/R/F1 = 1.00 in Part 3 is **demonstration data only** | Annotate the synopsis's custom Indian event dataset |
| 8 | Human evaluation of justifications | Needs human raters | Template at `truthlens_outputs/human_eval_template.csv` |
| 9 | Gold "recycled content" temporal labels | No benchmark ships them | §3.6 measures a verifiable proxy instead |

## 9. Deliverables

| Item | Where |
|---|---|
| Executed notebook | `TruthLens_Research_Project.ipynb` (42/42 cells, no errors) |
| Source code | `build_notebook.py` (source of truth), `prep_*.py` ×4, `run_notebook.py`, `bench_nli_devices.py` |
| Dataset acquisition | `README.md` + notebook Part 14.2 (exact URLs, sizes, licences) |
| Result tables | `truthlens_outputs/*.csv` (12 files) |
| Figures | `truthlens_outputs/*.png` (10 files) + `averitec_evidence_graph.gexf` |
| Metrics | `*_eval.csv`, `averitec_provenance.json`, `averitec_temporal_characterisation.json` |
| Ran / could-not-run | `experiment_registry.json`, notebook 14.1 + §7–8 above |
| Hardware & software | `run_environment.json` (captured at runtime), `requirements.txt` |
| Engineering learnings | `learnings/` (7 notes) |

## 10. What Phase 2 should be careful about

1. **Do not quote the AVeriTeC accuracy column without the majority baseline** beside it.
2. **Do not present conflict-awareness as a validated contribution** — §3.5 says otherwise.
3. **Always state the corpus size next to a retrieval number**; the FEVER pair (0.788 vs
   0.669) is the same system on the same claims.
4. **The FactDrill leaky numbers are not results.** Only the de-leaked rows are.
5. **Retrieval is the binding constraint on AVeriTeC**, not the NLI layer — the most
   valuable next experiment is a stronger retriever (cross-encoder reranking), not a
   bigger NLI model.
