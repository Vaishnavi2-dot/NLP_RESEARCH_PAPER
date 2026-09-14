# -*- coding: utf-8 -*-
"""FactDrill preparation: consolidate the Zenodo xlsx files into
  data/factdrill_records.jsonl  - one record per fact-checked item
  data/factdrill_corpus.jsonl   - fact-check articles as evidence documents
Schema notes (from the ICWSM 2022 paper and the xlsx files themselves):
  claim = the (usually false) social media claim text
  content = the fact-check article text
  No structured verdict label column exists -> verdict classification on
  FactDrill is NOT run (documented limitation); we use FactDrill for
  (a) real dataset characterization, (b) claim-to-fact-check retrieval in EN+HI.
Deterministic; skipped if outputs exist."""
import json, os, sys, glob, ast
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = os.environ.get("TRUTHLENS_DATA") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data")
FD_DIR = os.path.join(DATA, "factdrill")
OUT_REC = os.path.join(DATA, "factdrill_records.jsonl")
OUT_CORPUS = os.path.join(DATA, "factdrill_corpus.jsonl")

if os.path.exists(OUT_REC):
    print("FactDrill preparation already done - skipping")
    sys.exit(0)

def clean_text(x):
    if not isinstance(x, str) or not x.strip():
        return ""
    s = x.strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            parts = ast.literal_eval(s)
            if isinstance(parts, list):
                s = " ".join(str(p) for p in parts)
        except Exception:
            pass
    return " ".join(s.split())

records, corpus = [], []
for path in sorted(glob.glob(os.path.join(FD_DIR, "*.xlsx"))):
    site = os.path.basename(path).replace(".xlsx", "")
    lang = site.rsplit("_", 1)[-1]
    try:
        df = pd.read_excel(path)
    except Exception as e:
        print("skip", site, e)
        continue
    for _, row in df.iterrows():
        claim = clean_text(row.get("claim"))
        content = clean_text(row.get("content"))
        title = clean_text(row.get("title"))
        if len(content) < 100:
            continue
        rec = dict(site=site, lang=lang, unique_id=clean_text(row.get("unique_id")),
                   title=title, publish_date=str(row.get("publish_date")),
                   claim=claim, link=clean_text(row.get("link")))
        records.append(rec)
        corpus.append(dict(id=f"FD{len(corpus):05d}", text=content[:4000],
                           site=site, lang=lang, link=rec["link"],
                           title=title, publish_date=rec["publish_date"]))

with open(OUT_REC, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(OUT_CORPUS, "w", encoding="utf-8") as f:
    for c in corpus:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

from collections import Counter
print("items:", len(records), "| corpus docs:", len(corpus))
print("by language:", Counter(r["lang"] for r in records).most_common())
print("by site:", Counter(r["site"] for r in records).most_common(8))
print("with claim text:", sum(1 for r in records if len(r["claim"]) > 20))
