# AVeriTeC: the HF mirror is gated, and its evidence is overwhelmingly retrospective

> **Rule:** cite `github.com/MichSchli/AVeriTeC` for AVeriTeC downloads — the `huggingface.co/chenxwh/AVeriTeC` mirror is gated and returns 401 anonymously.

## Provenance

- `https://huggingface.co/api/datasets/chenxwh/AVeriTeC` -> **HTTP 401** (gated).
  Instructions citing it are not reproducible by a fresh clone.
- `https://raw.githubusercontent.com/MichSchli/AVeriTeC/main/data/{train,dev}.json`
  -> **200**, anonymous. Byte-identical to the local copies
  (train 10,184,813 B / dev 1,785,475 B), so this is the real source.
- Splits: train 3,068 / dev 500. Labels are 4-way and include
  `Conflicting Evidence/Cherrypicking` — the only public benchmark with that class.

## Evidence has no date field — but URLs do

Answer objects carry only `answer`, `answer_type`, `source_url`, `source_medium`,
`cached_source_url`. Dates must be recovered from the URL, in two kinds that are
**not** equally strong:

- **path date** (`nypost.com/2020/10/30/...`) = actual publication date -> **two-sided**:
  can prove pre-dating *and* post-dating. 747 of 7,022 corpus docs.
- **archive snapshot** (`web.archive.org/web/20201101145631/...`) = the page existed by
  then, but may be far older -> **one-sided**: can prove pre-dating only. 2,207 docs.

Inferring "this evidence post-dates the claim" from an archive snapshot is wrong —
it reports when the Internet Archive happened to crawl, not when the page was written.

`claim_date` is present on all 500 dev claims but in four different paddings of
`D-M-YYYY` (`30-9-2020`, `31-10-2020`, `9-10-2020`, `9-9-2020`). A strict
`%d-%m-%Y` parse silently gets only 131/500; a flexible regex gets all 500.

## The finding: AVeriTeC is a retrospective corpus

Over 437 dev claims with gold evidence (998 items):

- only **9.0%** of gold evidence provably pre-dates its claim
- **362/437** claims have *no* provably pre-dating evidence at all
- median gap between evidence date and claim date: **+107 days**
- 76.1% of dated evidence items are dated after the claim

Consequence for experiment design: a "contemporaneous evidence only" condition is
**degenerate** here — the system is forced to abstain on most claims, so it measures
the benchmark's retrospectivity rather than the system's quality. Report it as a
bound, and run the well-powered ablation (drop only *provably* post-claim evidence)
as the actual result.
