# TruthLens — Phase 1

**An Evidence-Aware NLP Framework for Claim Verification and Narrative Analysis in Indian Social Media**

Phase 1 is the reproducible research notebook plus the real experiments that test the
framework proposed in the project synopsis. Every headline number comes from a public
benchmark and is produced by executing the notebook end to end.

**Status:** Phase 1 complete — notebook runs **44/44 cells with no errors**, 14
experiments recorded.
**Phase 2 (writing the research paper) has not been started.** Nothing here is paper prose.

- Full results and interpretation → **`PHASE1_REPORT.md`**
- Non-obvious engineering findings → **`learnings/`** (indexed in `CLAUDE.md`)
- Machine-readable results → **`truthlens_outputs/*.csv`**

---

## 1. Quick start

```bash
# Python 3.11 or 3.12 ONLY. torch has no wheels for 3.13/3.14, and the system
# python on macOS is 3.14 — using it will fail at install time.
uv venv --python 3.11 .venv && . .venv/bin/activate
uv pip install -r requirements.txt

# 1. download datasets into data/   (URLs in §4)
# 2. prepare them — idempotent, deterministic (SEED=42), safe to re-run
python prep_fever.py          # 600 dev claims + gold-page evidence corpus
python prep_fever_expand.py   # + 45,667 random distractor pages -> 207k sentences
python prep_averitec.py       # claim records + QA evidence corpus (keeps source URLs)
python prep_factdrill.py      # consolidate xlsx -> records + article corpus

# 3. execute the notebook end to end
python run_notebook.py
```

Results land in `truthlens_outputs/`. Heavy computation is cached in `data/cache/`;
set `FORCE_RERUN = True` in the config cell to recompute from scratch.

**Runtime:** cold run ≈ 27 min (about 6 min of that is encoding the 207k-sentence
corpus, then cached) plus ≈ 8 min for the reranking sections. A fully cached re-run is
**108 s**.

## 2. Repository layout

| Path | What it is |
|---|---|
| `build_notebook.py` | **Source of truth for the notebook.** Edit this, never the `.ipynb` |
| `TruthLens_Research_Project.ipynb` | Generated notebook, 63 cells, with executed outputs |
| `run_notebook.py` | Headless executor — streams cell output, safe against interrupt |
| `prep_fever.py` · `prep_fever_expand.py` · `prep_averitec.py` · `prep_factdrill.py` | Dataset preparation |
| `exp_rerank.py` | Standalone cross-encoder reranking experiment |
| `bench_nli_devices.py` | NLI throughput per model per device (decides the model tier) |
| `bench_nli.py` · `bench_minilm_nli.py` · `preload_models.py` | **Superseded** — kept for provenance, headers say why |
| `requirements.txt` | Pinned environment |
| `learnings/` | 7 notes on things that were not what they looked like |
| `truthlens_outputs/` | Tables, figures, JSON. `demo_*` = demonstration data, **not results** |
| `PHASE1_REPORT.md` | Completion report: full tables, limitations, what ran |

## 3. How to read the results — please read this before quoting any number

Five traps. Each one will produce a confident, wrong statement if ignored.

1. **`demo_*` files are not research results.** Parts 2–9 of the notebook run an 18-post
   demonstration corpus purely to validate that the pipeline works end to end. Its
   claim-extraction F1 of 1.00 is meaningless as a research figure. Real results come
   only from Parts 10–12.
2. **On AVeriTeC, read macro-F1, not accuracy.** The dev set is 61 % *Refuted*, so a
   classifier that always says "Refuted" scores 0.610 accuracy. The text-only baseline
   scores 0.618 — it is essentially that predictor. Always quote the `Sys-Maj` row
   alongside.
3. **FEVER retrieval has two corpus settings** and they differ by 0.118 Recall@5. A
   retrieval number without its corpus size is not a result. Verdict experiments use the
   hard (207k) corpus.
4. **FactDrill has leaky and de-leaked settings.** The leaky rows (BM25 hit@5 = 1.000)
   are an artefact of the claim being copied out of its own article; only the
   **de-leaked** rows are reportable.
5. **The strict contemporaneous temporal condition (T3) is a bound, not a score.** It is
   dominated by forced abstentions and measures how retrospective AVeriTeC is, not how
   good the system is.

### Headline results

**FEVER** — 600 dev claims, 3-class, expanded 207,089-sentence corpus:

| System | Accuracy | Macro-F1 |
|---|---|---|
| Sys-Maj majority class | 0.333 | 0.167 |
| Sys1 text-only classifier | 0.440 | 0.421 |
| Sys2 retrieval + heuristic (no NLI) | 0.540 | 0.504 |
| Sys3 retrieval + NLI (top-1) | 0.660 | 0.663 |
| **Sys4 TruthLens** | **0.723** [0.687, 0.758] | **0.722** |
| **Sys4 + cross-encoder rerank** | **0.753** | **0.750** |

**AVeriTeC** — 500 dev claims, 4-class incl. *Conflicting Evidence*:

| System | Accuracy | Macro-F1 |
|---|---|---|
| Sys-Maj majority class (*Refuted*) | 0.610 | 0.189 |
| Sys1 text-only classifier | 0.618 | 0.293 |
| Sys3 retrieval + NLI (top-1) | 0.324 | 0.252 |
| **Sys4 TruthLens** | 0.472 [0.430, 0.514] | **0.334** |
| **Sys4 + cross-encoder rerank** | 0.488 | 0.342 |

**Retrieval (Recall@5):**

| Dataset | corpus | BM25 | Dense | Hybrid | + rerank |
|---|---|---|---|---|---|
| FEVER gold-pages-only | 6,934 | 0.764 | 0.724 | 0.788 | — |
| FEVER expanded | 207,089 | 0.627 | 0.590 | 0.669 | **0.761** |
| AVeriTeC | 7,022 | 0.398 | 0.411 | 0.415 | **0.485** |
| FactDrill de-leaked EN | 13,796 | **0.825** | 0.470 | 0.765 | — |
| FactDrill de-leaked HI | 13,796 | 0.660 | 0.388 | **0.670** | — |

Hybrid wins on FEVER and AVeriTeC. On FactDrill it is **mixed**: BM25 clearly wins in
English (0.825 vs 0.765), while in Hindi hybrid edges ahead (0.670 vs 0.660). The
multilingual encoder is much weaker on long Indian-language articles (dense 0.470 EN /
0.388 HI), so RRF fusion with a weak dense arm buys little and in English actively
costs. The **English → Hindi gap** (0.825 → 0.660 on BM25) is the multilingual finding,
and it is consistent across all three retrievers.

**Other measured results:** conflict-aware verdicts (Conflicting-F1 ≈ 0.09 — a negative
result), temporal robustness (only 9 % of AVeriTeC gold evidence provably pre-dates its
claim, median gap +107 days), narrative clustering (silhouette 0.111, NMI vs location
0.349), source provenance (488 domains, 47.4 % of claims rest on a single domain).
All detailed in `PHASE1_REPORT.md`.

## 4. Datasets

All three are public. **Nothing is synthetic.** Download into `data/`:

| Dataset | Source | Size used |
|---|---|---|
| FEVER | `https://fever.ai/download/fever/` — `shared_task_dev.jsonl`, `train.jsonl`, `wiki-pages.zip` (1.7 GB) | 600 dev claims, 207,089-sentence corpus |
| AVeriTeC | `https://raw.githubusercontent.com/MichSchli/AVeriTeC/main/data/{train,dev}.json` | 500 dev + 3,068 train, 7,022 evidence docs |
| FactDrill | Zenodo DOI `10.5281/zenodo.5854856` — open access, 32 files ≈ 50 MB | 13,796 items (8,732 EN + 5,064 HI) |

> **AVeriTeC gotcha:** the commonly cited mirror `huggingface.co/chenxwh/AVeriTeC` is
> **gated** and returns HTTP 401 to an anonymous client. Use the GitHub URLs above —
> the files are byte-identical.

Licences: FEVER CC BY-SA / research use · AVeriTeC CC BY-NC 4.0 · FactDrill per its
Zenodo record. Datasets are **not** committed (see `.gitignore`); everything excluded is
re-downloadable or regenerable by the deterministic prep scripts.

## 5. Hardware and models

Reference run: **Apple M1 Pro** (8-core CPU / 14-core GPU), 32 GB, macOS 26.6, **MPS**.
Captured per-run to `truthlens_outputs/run_environment.json`.

The notebook detects the accelerator and picks the NLI model from it:

| Device | NLI model | Throughput |
|---|---|---|
| CUDA / MPS | `DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (MNLI + **FEVER** + ANLI) | 69 pairs/s |
| bare CPU | `cross-encoder/nli-MiniLM2-L6-H768` | 352 pairs/s (weaker model) |

The large model runs at only 5.3 pairs/s on CPU, which is why the tier follows the
device. **The tier is printed at startup and forms part of every cache key**, so a CPU
run's numbers can never be silently mixed with an accelerated run's.

Encoder: `paraphrase-multilingual-MiniLM-L12-v2` (needed for Hindi).
Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`.

## 6. Working on this safely

- **Edit `build_notebook.py`, never the `.ipynb`.** The generator is the source of
  truth; it round-trips all cells exactly. Regenerate with `python build_notebook.py`.
- **Never regenerate the notebook while a run is in flight.** The runner writes the
  executed notebook at the end and will clobber your regeneration. Conversely, an
  interrupted run deliberately does *not* write back — see
  `learnings/notebook-runner-clobber-hazard.md` for the failure mode this prevents.
- **Verify your change is actually in the `.ipynb` before running** (`grep -c` for a
  token from your edit). This catches the stale-notebook trap above.
- **Cache keys embed the run mode and NLI tier.** Changing model tier correctly
  invalidates results. Delete `data/cache/` or set `FORCE_RERUN = True` to force
  recomputation. Embeddings are cached separately as `.npy` and keyed by corpus size.
- **Read `learnings/` before touching retrieval, statistics, or dataset prep.** Seven
  notes, each one a bug or artefact that already cost time once.

## 7. Known issues and limitations

Nine documented items in `PHASE1_REPORT.md` §8. The ones most likely to matter:

- **FactDrill verdict classification is not possible** — the deposit has no structured
  verdict labels, and deriving them from article titles would be constructing
  supervision and then scoring against it. Annotation protocol is in notebook §14.5.
- **Multilingual NLI is not evaluated** — no Hindi verdict labels exist, so multilingual
  capability is measured on the **retrieval** layer only.
- **Claim extraction is evaluated on demonstration data only** — no public benchmark
  annotates atomic claims for social posts.
- **FEVER retrieval is not the full 5.4M-page shared-task setting**, and AVeriTeC uses
  the released QA evidence rather than the full knowledge store.
- **Human evaluation has not been conducted** — template at
  `truthlens_outputs/human_eval_template.csv`.

## 8. Git

Work lives on branch `phase1/evidence-aware-experiments` →
[PR #1](https://github.com/Vaishnavi2-dot/NLP_RESEARCH_PAPER/pull/1).

> **Two-account note (macOS dev machine):** this repo is owned by `Vaishnavi2-dot` and
> the remote uses the `github-personal` SSH alias. Both a work and a personal GitHub
> account are configured in `gh`; the **work** account is active by default and cannot
> open PRs here. Use `gh auth switch --user <personal>` first, then switch back.

## 9. Phase 2 — open decision

Phase 2 is the research paper. Before writing, one framing decision is outstanding,
because the results point two ways:

- **Framework paper.** FEVER supports it strongly (0.440 → 0.753 accuracy over
  text-only, monotone across the component ladder, significant under McNemar). But
  conflict-awareness — a headline contribution in the synopsis — **is not supported by
  the data** (Conflicting-F1 ≈ 0.09), and temporal robustness cannot be properly tested
  on AVeriTeC because its evidence is retrospective.
- **Methodology paper.** The three protocol problems found in Phase 1 (FEVER corpus
  inflation, FactDrill's 89.8 % claim leakage, AVeriTeC's retrospectivity) are findings
  about benchmarks that other groups are actively publishing on, and all three are
  quantified here. The framework then becomes the vehicle rather than the claim.

This is a call for the project team and faculty mentor, not something the code decides.
`PHASE1_REPORT.md` §10 lists the specific claims to avoid overstating either way.
