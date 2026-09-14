# -*- coding: utf-8 -*-
# SUPERSEDED -- kept for provenance only.
# This script dates from the original CPU-only (Windows) attempt at this project
# and its conclusions no longer hold: it measured mDeBERTa at ~0.9 pairs/sec and
# concluded the large NLI models were unusable. On MPS the same model runs at
# 150 pairs/sec. Use bench_nli_devices.py instead, and see
# learnings/nli-throughput-mps-vs-cpu.md.
"""Pre-download models for TruthLens full mode + quick CPU speed test."""
import sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sentence_transformers import SentenceTransformer
print("downloading sentence encoder...", flush=True)
enc = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("encoder ready", flush=True)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
print("downloading NLI model...", flush=True)
tok = AutoTokenizer.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
mdl = AutoModelForSequenceClassification.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
mdl.eval()
print("NLI model ready | labels:", mdl.config.id2label, flush=True)

import torch
pairs = [("Premise text about something.", "Hypothesis text."),
         ("The Eiffel Tower is in Paris.", "The Eiffel Tower is located in Paris, France."),
         ("Cats are mammals.", "Cats are reptiles.")] * 10
t0 = time.time()
inp = tok([p for p, h in pairs], [h for p, h in pairs], truncation=True, max_length=256,
          padding=True, return_tensors="pt")
with torch.no_grad():
    out = mdl(**inp)
dt = time.time() - t0
print(f"batch of {len(pairs)} pairs in {dt:.1f}s -> {len(pairs)/dt:.1f} pairs/sec (batch=30)", flush=True)

t0 = time.time()
inp = tok([p for p, h in pairs[:32]], [h for p, h in pairs[:32]], truncation=True, max_length=256,
          padding=True, return_tensors="pt")
with torch.no_grad():
    out = mdl(**inp)
print(f"batch 32 in {time.time()-t0:.2f}s", flush=True)
print("MODELS OK", flush=True)
