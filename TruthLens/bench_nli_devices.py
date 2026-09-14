# -*- coding: utf-8 -*-
"""Benchmark NLI throughput per model per device (CPU vs MPS/CUDA).

Usage:  python bench_nli_devices.py <hf-model-id> [<hf-model-id> ...]

Reference numbers on Apple M1 Pro (8-core CPU / 14-core GPU, 32 GB),
torch 2.14.0, transformers 5.17.0, batch 32, max_length 192:
  DeBERTa-v3-large-mnli-fever-anli-ling-wanli   CPU   5.3/s -> MPS  69.2/s
  mDeBERTa-v3-base-xnli-multilingual-nli-2mil7  CPU  14.0/s -> MPS 150.0/s
  cross-encoder/nli-MiniLM2-L6-H768             CPU 352.4/s -> MPS 404.1/s

Small models barely benefit from MPS (launch-overhead bound); large ones get
10-13x. This is what decides NLI_TIER in the notebook.
"""
import time, torch, sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PREM = ["Colin Kaepernick is an American football quarterback who played for the San Francisco 49ers of the National Football League.",
        "Roald Dahl was a British novelist, short story writer, poet, screenwriter, and fighter pilot.",
        "Greville Janner was a British politician, barrister and writer who was a Labour MP for 47 years."]*32
HYP  = ["Colin Kaepernick became a starting quarterback during the 49ers 63rd season.",
        "Roald Dahl was a novelist.",
        "Greville Janner was never elected to Parliament."]*32

def bench(name, dev, n=96, batch=32, ml=192):
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSequenceClassification.from_pretrained(name).eval().to(dev)
    if dev=="mps": mdl = mdl.to(torch.float16)
    # warmup
    inp = tok(PREM[:8], HYP[:8], truncation=True, max_length=ml, padding=True, return_tensors="pt").to(dev)
    with torch.no_grad(): mdl(**inp)
    if dev=="mps": torch.mps.synchronize()
    t0=time.time()
    for i in range(0,n,batch):
        inp = tok(PREM[i:i+batch], HYP[i:i+batch], truncation=True, max_length=ml, padding=True, return_tensors="pt").to(dev)
        with torch.no_grad(): mdl(**inp)
    if dev=="mps": torch.mps.synchronize()
    dt=time.time()-t0
    print(f"  {dev:4s} fp{'16' if dev=='mps' else '32'}: {n/dt:7.1f} pairs/sec", flush=True)
    del mdl
    if dev=="mps": torch.mps.empty_cache()
    return n/dt

for m in sys.argv[1:]:
    print(f"\n### {m}", flush=True)
    for dev in ["cpu","mps"]:
        try: bench(m, dev)
        except Exception as e: print(f"  {dev}: FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
