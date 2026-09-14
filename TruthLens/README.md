# TruthLens — Phase 1

**An Evidence-Aware NLP Framework for Claim Verification and Narrative Analysis in Indian Social Media**

Phase 1 is the reproducible research notebook and the real experiments that test the
framework proposed in the project synopsis. It is not a demo: every headline number
comes from a public benchmark, and the code that produced it runs end to end from a
clean checkout.

> **Phase 2 (writing the research paper) has not been started.** Nothing in this repo
> is paper prose.

---

## Quick start

```bash
# Python 3.11 or 3.12 — NOT 3.13/3.14 (no torch wheels)
uv venv --python 3.11 .venv && . .venv/bin/activate
uv pip install -r requirements.txt

# 1. download the datasets into data/  (see "Datasets" below)
# 2. prepare them (idempotent, deterministic, SEED=42)
python prep_fever.py
python prep_fever_expand.py
python prep_averitec.py
python prep_factdrill.py

# 3. execute the notebook end to end
python run_notebook.py
```

Artifacts land in `truthlens_outputs/`. Heavy results are cached in `data/cache/`;
set `FORCE_RERUN = True` in the config cell to recompute.

## Repository layout

| Path | What it is |
|---|---|
| `build_notebook.py` | **Source of truth for the notebook.** Edit this, never the `.ipynb`. |
| `TruthLens_Research_Project.ipynb` | Generated notebook, with executed outputs |
| `run_notebook.py` | Headless executor (streams cell output; safe against interrupt) |
| `prep_fever.py` | Samples 600 FEVER dev claims + gold-page evidence corpus |
| `prep_fever_expand.py` | Adds ~46k random distractor pages → 207k-sentence corpus |
| `prep_averitec.py` | AVeriTeC claim records + QA evidence corpus (keeps source URLs) |
| `prep_factdrill.py` | Consolidates FactDrill xlsx → records + article corpus |
| `requirements.txt` | Pinned environment |
| `learnings/` | Non-obvious findings from building this (read before changing things) |
| `truthlens_outputs/` | Result tables, figures, JSON. `demo_*` = demonstration data only |

## Datasets

All three are public. **Nothing is synthetic**; the small demonstration corpus in
Parts 2–9 exists only to validate the pipeline and is labelled as such everywhere,
including in its output filenames (`demo_*`).

| Dataset | Source | Size used |
|---|---|---|
| FEVER | `https://fever.ai/download/fever/` (`shared_task_dev.jsonl`, `train.jsonl`, `wiki-pages.zip` 1.7 GB) | 600 dev claims, 207,089-sentence corpus |
| AVeriTeC | `https://raw.githubusercontent.com/MichSchli/AVeriTeC/main/data/{train,dev}.json` | 500 dev + 3,068 train, 7,022 evidence docs |
| FactDrill | Zenodo DOI `10.5281/zenodo.5854856` (open access, 32 files ≈ 50 MB) | 13,796 items (8,732 EN + 5,064 HI) |

**AVeriTeC note:** the widely-cited `huggingface.co/chenxwh/AVeriTeC` mirror is **gated**
and returns HTTP 401 anonymously. The GitHub URLs above are the ones that actually
reproduce; the local copies are byte-identical to them.

Licences: FEVER CC BY-SA / research use · AVeriTeC CC BY-NC 4.0 · FactDrill per its
Zenodo record.

## Hardware

Reference run: Apple M1 Pro (8-core CPU / 14-core GPU), 32 GB, macOS 26.6, **MPS**. The notebook detects the
accelerator and picks the NLI tier from it:

- **CUDA/MPS** → `DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (MNLI + **FEVER** + ANLI trained)
- **bare CPU** → `cross-encoder/nli-MiniLM2-L6-H768`, because the large model runs at
  ~5 pairs/s there

The tier is printed at startup and forms part of every cache key, so a CPU run's
numbers can never be silently mixed with an accelerated run's. Full cold run ≈ 27 min
on the reference machine (≈ 6 min of that is encoding the 207k-sentence corpus, which
is then cached).

## Three protocol problems found in Phase 1

Each would have produced a real-looking but misleading number. Each is measured and
reported in both settings rather than quietly fixed or ignored.

1. **FEVER's cheap corpus inflates retrieval.** Building the evidence corpus from gold
   pages only makes every distractor topically relevant. Hybrid Recall@5 falls
   **0.788 → 0.669** once 200k random distractor sentences are added.
2. **FactDrill's `claim` is copied out of its own article** — 89.8% verbatim overlap —
   so claim-to-fact-check retrieval is largely duplicate detection (leaky BM25 hit@5
   reaches **1.00**). A de-leaked corpus, with the claim span removed from its own gold
   article, is reported alongside it.
3. **AVeriTeC's evidence is overwhelmingly retrospective** — only **9%** of gold
   evidence provably pre-dates its claim, median gap **+107 days**. Temporal
   experiments therefore separate what the dates can and cannot prove, and the strict
   contemporaneous condition is reported as a *bound*, not a score.

See `learnings/` for the full write-up of each.

## Results

See `PHASE1_REPORT.md` for the complete tables, and `truthlens_outputs/*.csv` for the
machine-readable versions.
