# FactDrill's `claim` column is copied out of the article — retrieval on it is duplicate detection

> **Rule:** strip the claim span from its own gold article before scoring claim-to-fact-check retrieval on FactDrill, and report the un-stripped number only as a labelled upper bound.

## The problem

In the FactDrill Zenodo deposit the `claim` column is largely *extracted from* the
article `content`. Measured over all 13,788 items with a claim of >= 30 characters:

| relationship between claim and its own gold article | share |
|---|---|
| article **starts with** the claim's first 120 chars | 68.7% |
| claim's first 120 chars appear **elsewhere** in the article | 21.1% |
| neither | 10.2% |

So **89.8%** of queries are a verbatim substring of the document they are supposed to
retrieve. "Given the claim, retrieve its fact-check article" is then mostly exact
string matching: BM25 wins trivially, hit@5 saturates, and the number says nothing
about whether the retriever is any good. Publishing it as evidence-retrieval
performance would be reporting an artefact.

Worth noting the ordering trap: `records[i]` and `corpus[i]` are appended in the same
loop of `prep_factdrill.py`, so index alignment is correct and the gold id
`FD{i:05d}` is right — the data pipeline is fine. The flaw is in the *task*, not the code.

## The fix

Build two corpora and report both:

- **leaky** — as-is, kept only so the inflation is visible.
- **de-leaked** — the claim span removed from its own gold article, so a match must be
  earned against the fact-check's analysis text.

Removal is done on a whitespace/case-normalised copy to locate the span, then the
corresponding slice of the *raw* text is cut (walk both strings to map normalised
offsets back to raw offsets). Guard against emptying a document: if the result is
under 100 chars, keep the original.

Measured effect of de-leaking on the corpus:

- articles changed: 11,804 / 13,796 (85.6%)
- mean article length: 2,584 -> 1,838 chars
- residual verbatim leakage: **89.8% -> 4.7%**
- documents emptied: 0

## The general lesson

Before scoring any "retrieve the document for this query" task built from a scraped
corpus, check whether the query field was *derived from* the target field. It usually
was. One `substring` check over the pairs is cheap and can save a whole bogus result
table.
