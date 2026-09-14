# The FEVER "gold pages only" corpus inflates Recall@5 by ~12 points

> **Rule:** report the evidence-corpus size and composition next to every retrieval metric — a Recall@k with no stated haystack is not a result.

## What happened

`prep_fever.py` built the evidence corpus from the sentences of the sampled claims'
**gold evidence pages only**: 6,934 sentences from 435 pages. Every distractor came
from a page already known to be relevant, so the retriever never had to reject an
unrelated topic.

Measured on the same 397 claims, same retrievers, only the corpus differing:

| setting | corpus | BM25 R@5 | dense R@5 | hybrid R@5 |
|---|---|---|---|---|
| gold-pages-only | 6,934 sents / 435 pages | 0.764 | 0.724 | 0.788 |
| expanded | 207,089 sents / 46,102 pages | 0.627 | 0.590 | **0.669** |

**Hybrid Recall@5 drops 0.788 → 0.669 (−0.118).** That gap is pure protocol artefact.
Neither is the full shared-task setting (5.4M pages), so the honest move is to report
both and name the limitation.

- Hybrid (RRF of BM25 + dense) beats both components in **both** settings — the
  ranking of methods survived, only the absolute numbers moved.
- Build the harder corpus by streaming random distractor pages out of
  `wiki-pages.zip` (`prep_fever_expand.py`, seeded). No need to extract the 7 GB
  of shards — `zipfile.ZipFile.open()` streams each shard fine.

## Gotcha: `__MACOSX` entries in the dump zip

`wiki-pages.zip` contains `__MACOSX/wiki-pages/._wiki-NNN.jsonl` resource-fork
entries alongside the 109 real shards. Filtering only on `.endswith(".jsonl")`
picks them up and they are **binary**, so `json.loads` fails on the first line.
Always filter `not name.startswith("__MACOSX")`.
