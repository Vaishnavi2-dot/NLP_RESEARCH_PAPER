# -*- coding: utf-8 -*-
"""FEVER corpus expansion: add randomly sampled DISTRACTOR Wikipedia pages to the
gold-page evidence corpus built by prep_fever.py.

Why this exists
---------------
prep_fever.py builds a corpus from the sentences of the *gold evidence pages only*
(6,934 sentences / 435 pages). Retrieval then only has to find the gold sentence
among sentences of pages that are already known to be relevant, so Recall@k is
optimistic and is not comparable to the FEVER shared-task setting (5.4M pages).

This script produces a second, harder corpus by adding N_DISTRACTOR_PAGES pages
sampled uniformly at random from the full June-2017 dump, excluding gold pages.
Both corpora are kept so the notebook can report retrieval in both settings and
be explicit about which number is which.

Outputs:
  data/fever_wiki_corpus_expanded.jsonl  - gold-page sentences + distractor sentences
  data/fever_expand_manifest.json        - sizes, seed, page counts (for the paper)

Deterministic given SEED; skipped if outputs already exist.
"""
import json, os, random, sys, zipfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = os.environ.get("TRUTHLENS_DATA") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data")
GOLD_CORPUS = os.path.join(DATA, "fever_wiki_corpus.jsonl")
WIKI_ZIP = os.path.join(DATA, "wiki-pages.zip")
OUT_CORPUS = os.path.join(DATA, "fever_wiki_corpus_expanded.jsonl")
OUT_MANIFEST = os.path.join(DATA, "fever_expand_manifest.json")

SEED = 42
# 50k distractor pages ~= 250k extra sentences: large enough that retrieval must
# genuinely discriminate, small enough that dense encoding stays tractable
# (250k x 384 floats ~= 380 MB) on a 32 GB laptop.
N_DISTRACTOR_PAGES = int(os.environ.get("FEVER_DISTRACTOR_PAGES", "50000"))
MIN_SENT_CHARS = 30          # skip stubs/fragments that can never be evidence

if os.path.exists(OUT_CORPUS) and os.path.exists(OUT_MANIFEST):
    print("FEVER corpus expansion already done - skipping")
    sys.exit(0)

if not os.path.exists(GOLD_CORPUS):
    sys.exit(f"missing {GOLD_CORPUS} - run prep_fever.py first")
if not os.path.exists(WIKI_ZIP):
    sys.exit(f"missing {WIKI_ZIP} - download from https://fever.ai/download/fever/wiki-pages.zip")


def page_sentences(page_rec):
    """FEVER wiki 'lines' field: '<idx>\\t<sentence>\\t<tokens...>' per line."""
    out = {}
    for line in page_rec.get("lines", "").split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            out[idx] = parts[1]
    return out


# ---- 1. load the existing gold-page corpus and remember its pages -----------
gold_rows = [json.loads(l) for l in open(GOLD_CORPUS, encoding="utf-8")]
gold_pages = {r["page"] for r in gold_rows}
print(f"gold corpus: {len(gold_rows)} sentences from {len(gold_pages)} pages")

# ---- 2. pass 1: enumerate every page id in the dump -------------------------
# Two passes keep peak memory to the id list rather than the whole dump.
print("pass 1/2: indexing dump page ids ...", flush=True)
all_ids = []
with zipfile.ZipFile(WIKI_ZIP) as z:
    names = sorted(n for n in z.namelist()
               if n.endswith(".jsonl") and not n.startswith("__MACOSX"))
    print(f"  dump shards: {len(names)}")
    for name in names:
        with z.open(name) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(b"\xef\xbb\xbf"):
                    line = line[3:]
                # cheap id extraction: avoid json.loads for every one of 5.4M lines
                if line.startswith(b'{"id": "'):
                    end = line.find(b'"', 8)
                    if end > 0:
                        pid = line[8:end].decode("utf-8", "replace")
                        if pid:
                            all_ids.append(pid)
                        continue
                try:
                    pid = json.loads(line)["id"]
                except Exception:
                    continue
                if pid:
                    all_ids.append(pid)
print(f"  total pages in dump: {len(all_ids)}")

# ---- 3. choose distractor pages deterministically ---------------------------
candidates = [p for p in all_ids if p not in gold_pages]
random.seed(SEED)
n_take = min(N_DISTRACTOR_PAGES, len(candidates))
want = set(random.sample(candidates, n_take))
print(f"sampled {len(want)} distractor pages (seed={SEED}) from {len(candidates)} candidates")
del all_ids, candidates

# ---- 4. pass 2: pull the sentences of the sampled pages ---------------------
print("pass 2/2: extracting distractor sentences ...", flush=True)
extra, seen_pages = [], 0
with zipfile.ZipFile(WIKI_ZIP) as z:
    for name in sorted(n for n in z.namelist()
                   if n.endswith(".jsonl") and not n.startswith("__MACOSX")):
        with z.open(name) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(b"\xef\xbb\xbf"):
                    line = line[3:]
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = rec.get("id")
                if pid not in want:
                    continue
                seen_pages += 1
                for idx, text in page_sentences(rec).items():
                    if len(text) >= MIN_SENT_CHARS:
                        extra.append(dict(id=f"{pid}::S{idx}", text=text, page=pid))
        if seen_pages >= len(want):
            break
print(f"  distractor pages found: {seen_pages} | distractor sentences: {len(extra)}")

# ---- 5. write the expanded corpus -------------------------------------------
gold_ids = {r["id"] for r in gold_rows}
extra = [r for r in extra if r["id"] not in gold_ids]      # never shadow a gold sentence
with open(OUT_CORPUS, "w", encoding="utf-8") as f:
    for r in gold_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for r in extra:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

manifest = dict(
    seed=SEED,
    gold_sentences=len(gold_rows), gold_pages=len(gold_pages),
    distractor_pages_requested=N_DISTRACTOR_PAGES,
    distractor_pages_found=seen_pages, distractor_sentences=len(extra),
    total_sentences=len(gold_rows) + len(extra),
    min_sentence_chars=MIN_SENT_CHARS,
    source="https://fever.ai/download/fever/wiki-pages.zip (June 2017 dump)")
with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=1)

print(json.dumps(manifest, indent=1))
print(f"wrote {OUT_CORPUS}")
