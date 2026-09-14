# -*- coding: utf-8 -*-
# SUPERSEDED -- kept for provenance only.
# This script dates from the original CPU-only (Windows) attempt at this project
# and its conclusions no longer hold: it measured mDeBERTa at ~0.9 pairs/sec and
# concluded the large NLI models were unusable. On MPS the same model runs at
# 150 pairs/sec. Use bench_nli_devices.py instead, and see
# learnings/nli-throughput-mps-vs-cpu.md.
"""Benchmark mDeBERTa NLI on CPU with realistic settings."""
import sys, time, torch
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
torch.set_num_threads(14)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
mdl = AutoModelForSequenceClassification.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
mdl.eval()

# realistic FEVER-style pairs: single-sentence premises, short hypotheses
premises = ["Colin Kaepernick is an American football quarterback who played for the San Francisco 49ers of the National Football League.",
            "Roald Dahl was a British novelist, short story writer, poet, screenwriter, and fighter pilot.",
            "Greville Janner was a British politician, barrister and writer who was a Labour Member of Parliament for 47 years."] * 16
hypotheses = ["Colin Kaepernick became a starting quarterback during the 49ers 63rd season.",
              "Roald Dahl was a novelist.",
              "Greville Janner was never elected to Parliament." ] * 16

def bench(model, name, batch=32, max_length=128, n=48):
    t0 = time.time()
    for i in range(0, n, batch):
        inp = tok(premises[i:i+batch], hypotheses[i:i+batch], truncation=True,
                  max_length=max_length, padding=True, return_tensors="pt")
        with torch.no_grad():
            model(**inp)
    dt = time.time() - t0
    print(f"{name}: {n} pairs in {dt:.1f}s -> {n/dt:.1f} pairs/sec", flush=True)
    return n/dt

print("threads:", torch.get_num_threads(), flush=True)
fp32 = bench(mdl, "fp32 maxlen128")

q = torch.ao.quantization.quantize_dynamic(mdl, {torch.nn.Linear}, dtype=torch.qint8)
int8 = bench(q, "int8-dynamic maxlen128")

# even shorter for single sentences
def bench_short(model, name, max_length=64, n=48, batch=32):
    t0 = time.time()
    for i in range(0, n, batch):
        inp = tok(premises[i:i+batch], hypotheses[i:i+batch], truncation=True,
                  max_length=max_length, padding=True, return_tensors="pt")
        with torch.no_grad():
            model(**inp)
    print(f"{name}: {n/(time.time()-t0):.1f} pairs/sec", flush=True)
bench_short(q, "int8 maxlen64", max_length=64)
