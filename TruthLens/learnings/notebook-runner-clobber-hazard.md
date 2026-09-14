# A killed nbclient run silently overwrites the notebook with stale code

> **Rule:** never write the notebook back on `KeyboardInterrupt`, and regenerate from `build_notebook.py` before every run.

## The trap

`run_notebook.py` loaded the notebook into memory, executed it, and wrote it back in a
`finally:` block. That looks safe until you interrupt a run:

1. run1 starts — loads notebook **v1** into memory.
2. You fix a bug in `build_notebook.py` and regenerate — the file on disk is now **v2**.
3. You `pkill` run1 — its `finally:` writes its in-memory **v1** back over the file.
4. You start run2 — it reads the file and executes **v1 again**.

Run2 looks like it is testing the fix. It is not. The symptom is a bug that
"won't go away" across restarts, which is the same class as debugging a backend
whose process predates the edit.

## It is not only Ctrl-C — guard every abnormal exit

The first fix only caught `KeyboardInterrupt`, and the trap sprang again within the
hour. A `finally:` block that writes unconditionally also fires when:

- the **kernel dies** (nbclient raises `DeadKernelError`) — e.g. an MPS out-of-memory
  kill during a large encode, which produces no Python traceback at all;
- the process takes **SIGPIPE**, e.g. `python run_notebook.py | head -30`;
- any other unexpected exception unwinds.

Each of those wrote a *stale* in-memory notebook over a freshly regenerated file, and
the next run then executed old code while looking like it ran the new code. The tell is
brutal to spot: the run behaves exactly as before your fix, and
`grep -c '<your new token>' the.ipynb` returns 0 while the same grep on
`build_notebook.py` returns a positive number.

## Fixes applied

- Write back **only** on clean completion, or on `CellExecutionError` (a real per-cell
  failure whose traceback is worth keeping). `KeyboardInterrupt` and a catch-all
  `except BaseException` both leave the file alone and print
  `NOT writing back ... regenerate with build_notebook.py before retrying`.
- Regenerate (`python build_notebook.py`) and **verify the fix is present in the
  `.ipynb`** (`grep -c` for a token from the change) immediately before launching.
- Because `build_notebook.py` is the source of truth, edit *it*, never the `.ipynb`.
  Confirmed the generator round-trips: all 49 cells byte-identical in source and
  metadata, so regenerating loses nothing but execution outputs.
- Do not regenerate while a run is in flight — the run's final write will clobber it.
