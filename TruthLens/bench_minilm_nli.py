# -*- coding: utf-8 -*-
# SUPERSEDED -- kept for provenance only.
# This script dates from the original CPU-only (Windows) attempt at this project
# and its conclusions no longer hold: it measured mDeBERTa at ~0.9 pairs/sec and
# concluded the large NLI models were unusable. On MPS the same model runs at
# 150 pairs/sec. Use bench_nli_devices.py instead, and see
# learnings/nli-throughput-mps-vs-cpu.md.
"""Benchmark MiniLM2 NLI as a faster CPU alternative."""
import sys, time, torch
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
torch.set_num_threads(12)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModelForSequenceClassification.from_pretrained(MODEL)
mdl.eval()
print("labels:", mdl.config.id2label, flush=True)

premises = ["Colin Kaepernick is an American football quarterback who played for the San Francisco 49ers of the National Football League.",
            "Roald Dahl was a British novelist, short story writer, poet, screenwriter, and fighter pilot.",
            "Greville Janner was a British politician, barrister and writer who was a Labour Member of Parliament for 47 years."] * 32
hypotheses = ["Colin Kaepernick became a starting quarterback during the 49ers 63rd season.",
              "Roald Dahl was a novelist.",
              "Greville Janner was never elected to Parliament."] * 32

# warmup
inp = tok(premises[:8], hypotheses[:8], truncation=True, max_length=128, padding=True, return_tensors="pt")
with torch.no_grad():
    mdl(**inp)

n, batch = 96, 32
t0 = time.time()
for i in range(0, n, batch):
    inp = tok(premises[i:i+batch], hypotheses[i:i+batch], truncation=True,
              max_length=128, padding=True, return_tensors="pt")
    with torch.no_grad():
        mdl(**inp)
dt = time.time() - t0
print(f"MiniLM2-NLI: {n} pairs in {dt:.1f}s -> {n/dt:.1f} pairs/sec", flush=True)

# sanity: does it judge correctly?
import numpy as np
tests = [("The Eiffel Tower is located in Paris, France.", "The Eiffel Tower is in Paris."),
         ("The Eiffel Tower is located in Paris, France.", "The Eiffel Tower is in Berlin."),
         ("Many people enjoy watching football on weekends.", "The capital of Australia is Canberra.")]
inp = tok([p for p, h in tests], [h for p, h in tests], truncation=True, max_length=128,
          padding=True, return_tensors="pt")
with torch.no_grad():
    probs = torch.softmax(mdl(**inp).logits, dim=-1).numpy()
for (p, h), pr in zip(tests, probs):
    print(f"{mdl.config.id2label[int(pr.argmax())]:14s} {pr.max():.2f}  {h[:50]}")
