# -*- coding: utf-8 -*-
"""AVeriTeC preparation: build claim records + shared evidence corpus from the
QA evidence of the dev (test) and train splits.
Outputs:
  data/averitec_dev_records.json   - 500 dev claims with gold labels + evidence doc ids
  data/averitec_train_records.json - train split for the text-only baseline
  data/averitec_corpus.jsonl       - evidence corpus (QA answers with real source URLs)
Deterministic; skipped if outputs exist."""
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = os.environ.get("TRUTHLENS_DATA") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data")
OUT_DEV = os.path.join(DATA, "averitec_dev_records.json")
OUT_TRAIN = os.path.join(DATA, "averitec_train_records.json")
OUT_CORPUS = os.path.join(DATA, "averitec_corpus.jsonl")

if os.path.exists(OUT_DEV) and os.path.exists(OUT_CORPUS):
    print("AVeriTeC preparation already done - skipping")
    sys.exit(0)

LABEL_MAP = {"Supported": "Supported", "Refuted": "Refuted",
             "Not Enough Evidence": "Not Enough Evidence",
             "Conflicting Evidence/Cherrypicking": "Conflicting Evidence"}

def build_records(path, corpus, corpus_index, is_dev):
    records = []
    with open(path, encoding="utf-8") as f:
        claims = json.load(f)
    for ci, c in enumerate(claims):
        label = LABEL_MAP.get(c.get("label"))
        if label is None:
            continue
        ev_ids = []
        for q in (c.get("questions") or []):
            for a in (q.get("answers") or []):
                text = (a.get("answer") or "").strip()
                if len(text) < 15 or a.get("answer_type") == "Unanswerable":
                    continue
                doc = dict(
                    text=text[:2000],
                    question=q.get("question", ""),
                    source_url=a.get("source_url") or "",
                    source_medium=a.get("source_medium") or "",
                    claim_index=ci)
                key = (doc["text"], doc["source_url"])
                if key not in corpus_index:
                    corpus.append(doc)
                    corpus_index[key] = len(corpus) - 1
                ev_ids.append(corpus_index[key])
        records.append(dict(
            id=ci, claim=c["claim"], label=label,
            claim_date=c.get("claim_date"),
            location=c.get("location_ISO_code"),
            claim_types=[t for t in (c.get("claim_types") or [])],
            speaker=c.get("speaker"),
            reporting_source=c.get("reporting_source"),
            evidence_ids=ev_ids))
    return records

corpus, corpus_index = [], {}
train_records = build_records(os.path.join(DATA, "averitec_train.json"), corpus, corpus_index, False)
dev_records = build_records(os.path.join(DATA, "averitec_dev.json"), corpus, corpus_index, True)

with open(OUT_DEV, "w", encoding="utf-8") as f:
    json.dump(dev_records, f, ensure_ascii=False)
with open(OUT_TRAIN, "w", encoding="utf-8") as f:
    json.dump(train_records, f, ensure_ascii=False)
with open(OUT_CORPUS, "w", encoding="utf-8") as f:
    for d in corpus:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

from collections import Counter
print("train claims:", len(train_records), Counter(r["label"] for r in train_records))
print("dev claims:", len(dev_records), Counter(r["label"] for r in dev_records))
print("corpus evidence docs:", len(corpus))
print("dev IN-located:", sum(1 for r in dev_records if r["location"] == "IN"))
print("avg evidence docs/claim:", sum(len(r["evidence_ids"]) for r in dev_records) / len(dev_records))
