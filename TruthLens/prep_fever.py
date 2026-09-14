# -*- coding: utf-8 -*-
"""FEVER preparation: sample 600 dev claims (200/label, seed 42), extract gold
evidence sentences from the wiki-pages dump. Produces:
  data/fever_sample.json       - sampled claims with gold evidence sentences
  data/fever_wiki_corpus.jsonl - sentence-level evidence corpus (gold pages only)
Deterministic given seed; skipped if outputs already exist."""
import json, os, random, sys, zipfile
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = os.environ.get("TRUTHLENS_DATA") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data")
SAMPLE_PATH = os.path.join(DATA, "fever_sample.json")
CORPUS_PATH = os.path.join(DATA, "fever_wiki_corpus.jsonl")
N_PER_LABEL = 200
SEED = 42

if os.path.exists(SAMPLE_PATH) and os.path.exists(CORPUS_PATH):
    print("FEVER preparation already done - skipping")
    sys.exit(0)

# ---- 1. sample claims from the labelled dev set ------------------------------
rows = [json.loads(l) for l in open(os.path.join(DATA, "fever_shared_task_dev.jsonl"),
                                    encoding="utf-8")]
random.seed(SEED)
by_label = {}
for r in rows:
    by_label.setdefault(r["label"], []).append(r)
sample = []
for label, items in by_label.items():
    random.shuffle(items)
    sample.extend(items[:N_PER_LABEL])
print("sampled:", {l: sum(1 for s in sample if s["label"] == l) for l in by_label})

# ---- 2. collect gold wiki page ids -------------------------------------------
need_pages = set()
for s in sample:
    for ev_set in s["evidence"]:
        for ann in ev_set:
            page = ann[2]
            if page:
                need_pages.add(page)
print("gold wiki pages needed:", len(need_pages))

# ---- 3. stream the wiki dump, keep only needed pages ------------------------
kept = {}
bad_lines = 0
with zipfile.ZipFile(os.path.join(DATA, "wiki-pages.zip")) as z:
    names = [n for n in z.namelist() if n.endswith(".jsonl")]
    print("dump files:", names)
    for name in names:
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
                    bad_lines += 1
                    continue
                pid = rec.get("id")
                if pid in need_pages:
                    kept[pid] = rec
print("pages extracted:", len(kept), "/", len(need_pages), "| skipped bad lines:", bad_lines)

# ---- 4. build sentence-level corpus + gold evidence text --------------------
def page_sentences(page_rec):
    sents = {}
    for line in page_rec.get("lines", "").split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            sents[idx] = parts[1]
    return sents

corpus = []          # dicts: id, text, page
claim_records = []
missing_pages, missing_sents = set(), set()
for s in sample:
    gold_sents, gold_ids = [], set()
    if s["label"] != "NOT ENOUGH INFO":
        for ev_set in s["evidence"]:
            ok = True
            set_sents = []
            for ann in ev_set:
                page, sidx = ann[2], ann[3]
                if page not in kept:
                    ok = False
                    missing_pages.add(page)
                    break
                sents = page_sentences(kept[page])
                if sidx not in sents:
                    ok = False
                    missing_sents.add((page, sidx))
                    break
                doc_id = f"{page}::S{sidx}"
                set_sents.append((doc_id, sents[sidx], page))
            if ok:
                for d, t, p in set_sents:
                    gold_ids.add(d)
                    gold_sents.append(dict(id=d, text=t, page=p))
    claim_records.append(dict(
        id=s["id"], claim=s["claim"], label=s["label"],
        gold_evidence_ids=sorted(gold_ids), gold_evidence_text=gold_sents[:5]))
    for d, t, p in gold_sents:
        corpus.append(dict(id=d, text=t, page=p))

# corpus = all sentences of gold pages (evidence + distractors from same pages)
gold_doc_ids = {c["id"] for c in corpus}
for pid, rec in kept.items():
    for idx, text in page_sentences(rec).items():
        doc_id = f"{pid}::S{idx}"
        if doc_id not in gold_doc_ids:
            corpus.append(dict(id=doc_id, text=text, page=pid))
            gold_doc_ids.add(doc_id)

with open(SAMPLE_PATH, "w", encoding="utf-8") as f:
    json.dump(claim_records, f, ensure_ascii=False)
with open(CORPUS_PATH, "w", encoding="utf-8") as f:
    for c in corpus:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

n_with_ev = sum(1 for c in claim_records if c["gold_evidence_ids"])
print(f"claims: {len(claim_records)} | with gold evidence: {n_with_ev}")
print(f"corpus sentences: {len(corpus)} from {len(set(c['page'] for c in corpus))} pages")
print(f"missing pages: {len(missing_pages)} | missing sentence ids: {len(missing_sents)}")
print("labels:", {l: sum(1 for c in claim_records if c['label']==l) for l in set(c['label'] for c in claim_records)})
