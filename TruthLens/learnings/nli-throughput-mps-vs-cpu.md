# NLI throughput on Apple Silicon: MPS is what makes this project feasible

> **Rule:** run the NLI judge and sentence encoder on MPS (or CUDA) and pick the model tier from the device — never hardcode a small model to survive CPU.

## The measurement that changed the project

Benchmarked on Apple M1 Pro (8-core CPU / 14-core GPU, 32 GB), torch 2.14.0, transformers 5.17.0,
batch 32, max_length 192, fp16 on MPS / fp32 on CPU:

| NLI model | CPU | MPS fp16 | speedup |
|---|---|---|---|
| `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | 5.3 pairs/s | **69.2 pairs/s** | 13.1x |
| `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | 14.0 pairs/s | **150.0 pairs/s** | 10.7x |
| `cross-encoder/nli-MiniLM2-L6-H768` | 352.4 pairs/s | 404.1 pairs/s | 1.1x |

- The earlier Windows/CPU attempt at this project measured mDeBERTa at **0.9 pairs/s**
  and concluded the strong models were unusable, downgrading to a 6-layer MiniLM.
  Same model on MPS is **150 pairs/s — 167x faster**. The model choice in that
  codebase was a hardware workaround, not a research decision.
- **Small models barely benefit from MPS** (1.1x). They are launch-overhead bound, not
  compute bound. So "is MPS worth it?" has opposite answers by model size — measure,
  don't assume.
- Sentence encoding (`paraphrase-multilingual-MiniLM-L12-v2`) on MPS: **614 sentences/s**,
  so a 207k-sentence corpus encodes in ~6 min. Cache the resulting matrix as `.npy`.

## Gotchas

- **Cast logits to fp32 before softmax** when the model runs in fp16 —
  `torch.softmax(out.logits.float(), dim=-1)`. fp16 softmax can underflow.
- `torch.mps.synchronize()` before stopping a timer, or you measure queue time, not work.
- Python **3.11/3.12 only**. torch has no wheels for 3.14 (system python on this Mac);
  the whole transformer stack fails to install there.
- Tag every result cache with the model tier. A small-model CPU run and a
  strong-model MPS run must never share a cache key, or numbers silently mix.
