# A bootstrap CI must resample (gold, pred) PAIRS — two index draws silently reports chance

> **Rule:** in `bootstrap_ci`, draw one index vector per replicate and index BOTH arrays with it. Two independent draws destroys the pairing and reports the random-pairing rate.

## The bug

```python
# WRONG - two independent resamples
vals = [metric_fn(y_true[rng.integers(0, n, n)],
                  y_pred[rng.integers(0, n, n)]) for _ in range(n_boot)]

# RIGHT - one resample, applied to both
for _ in range(n_boot):
    idx = rng.integers(0, n, n)
    vals.append(metric_fn(y_true[idx], y_pred[idx]))
```

The wrong version compares gold labels from one random sample of claims against
predictions from a *different* random sample. Accuracy then measures how often two
unrelated labels coincide — the marginal collision rate, roughly `1/n_classes` for
balanced data.

## Why it is dangerous rather than obvious

It fails *quietly and plausibly*. On the balanced 600-claim FEVER sample it reported:

```
Sys4 accuracy: (0.333, 0.295, 0.375)     <- looks like a real CI
```

0.333 on a 3-class balanced task reads as "the system is at chance", which is a
believable experimental outcome. The actual accuracy from
`classification_report` on the same predictions was **0.72**. The two numbers sat
about ten lines apart in the same output and disagreed by 0.39.

Verification that catches it in one run — synthesise predictions of known accuracy:

```
true accuracy    : 0.803
fixed bootstrap  : (0.804, 0.772, 0.833)
broken bootstrap : 0.335
```

## Detection heuristics

- If a bootstrap mean lands near `1/n_classes` on balanced data, suspect the pairing
  before you suspect the model.
- Always cross-check the bootstrap mean against the point estimate computed directly.
  They should agree to within ~0.01; a large gap means the resampling is wrong.
- The same trap does **not** hit McNemar's test, which is defined on paired
  correctness vectors and so cannot be de-paired by accident.

## Blast radius

Every confidence interval in the notebook was affected (FEVER and AVeriTeC, accuracy
and macro-F1). Point estimates were never affected. Because CIs are computed from
*cached predictions*, fixing this required only a re-run of the stats cells, not a
re-run of the NLI.
