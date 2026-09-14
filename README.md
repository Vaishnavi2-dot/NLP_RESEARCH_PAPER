# NLP_RESEARCH_PAPER

Research repository for **TruthLens** — *An Evidence-Aware NLP Framework for Claim
Verification and Narrative Analysis in Indian Social Media*.

## Where things are

| | |
|---|---|
| **[`TruthLens/README.md`](TruthLens/README.md)** | **Start here.** Setup, datasets, results, how to read them, how to work on it safely |
| [`TruthLens/PHASE1_REPORT.md`](TruthLens/PHASE1_REPORT.md) | Phase 1 completion report — full tables, limitations, what ran and what could not |
| [`TruthLens/learnings/`](TruthLens/learnings/) | 7 notes on non-obvious findings; read before changing retrieval, statistics or dataset prep |
| `TruthLens_Project_Synopsis.docx` | The original project specification |

## Project status

**Phase 1 (build the notebook + run the real experiments): complete.** The notebook
executes end to end — 44/44 cells, no errors — over FEVER, AVeriTeC and FactDrill.
14 experiments are recorded in `truthlens_outputs/experiment_registry.json`.

**Phase 2 (write the research paper): not started.** There is an open framing decision
first — see `TruthLens/README.md` §9.

## Headline result

On FEVER (600 dev claims, 3-class, over a 207,089-sentence evidence corpus), the
evidence-aware pipeline reaches **0.723 accuracy** (0.753 with cross-encoder reranking)
against **0.440** for a text-only classifier, improving monotonically across the
component ladder the synopsis proposed and significant under McNemar's test.

On AVeriTeC the picture is deliberately reported as mixed: better than text-only on
macro-F1 (0.334 vs 0.293) but below a majority-class baseline on accuracy, with
retrieval as the binding constraint.

## Three things a reader should know before quoting any number

1. Files named `demo_*` come from an 18-post demonstration corpus used only to validate
   the pipeline. They are **not** research results.
2. On AVeriTeC, read **macro-F1, not accuracy** — the dev set is 61 % *Refuted*, so a
   majority predictor scores 0.610.
3. Retrieval numbers are meaningless without their corpus size, and two datasets are
   reported in two settings each (FEVER: gold-pages-only vs expanded; FactDrill: leaky
   vs de-leaked). Only the harder setting of each pair is a result.

`TruthLens/README.md` §3 explains all five such traps.

## Reproducing

Python **3.11 or 3.12** (torch has no wheels for 3.13/3.14):

```bash
cd TruthLens
uv venv --python 3.11 .venv && . .venv/bin/activate
uv pip install -r requirements.txt
# download datasets (URLs in TruthLens/README.md §4), then:
python prep_fever.py && python prep_fever_expand.py
python prep_averitec.py && python prep_factdrill.py
python run_notebook.py
```

Datasets, caches and the virtualenv are not committed (~3.5 GB); everything excluded is
re-downloadable or regenerable by the deterministic (`SEED=42`) prep scripts.
