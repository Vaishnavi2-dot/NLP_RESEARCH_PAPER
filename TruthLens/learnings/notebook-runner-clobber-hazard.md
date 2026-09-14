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

## Fixes applied

- `except KeyboardInterrupt: write_back = False` — an interrupted run leaves the
  source notebook untouched.
- Regenerate (`python build_notebook.py`) and **verify the fix is present in the
  `.ipynb`** (`grep -c` for a token from the change) immediately before launching.
- Because `build_notebook.py` is the source of truth, edit *it*, never the `.ipynb`.
  Confirmed the generator round-trips: all 49 cells byte-identical in source and
  metadata, so regenerating loses nothing but execution outputs.
- Do not regenerate while a run is in flight — the run's final write will clobber it.
