# -*- coding: utf-8 -*-
"""Execute TruthLens_Research_Project.ipynb and write the executed copy back.

Runs the notebook headlessly with nbclient so Phase-1 results are produced by an
actual end-to-end execution rather than by hand-assembling numbers. Cell output
is streamed to stdout as it is produced so a long run can be monitored.

Usage:
  python run_notebook.py                 # execute in place
  python run_notebook.py --out X.ipynb   # write elsewhere
  python run_notebook.py --timeout 7200  # per-cell timeout (seconds)
"""
import argparse, os, sys, time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

HERE = os.path.dirname(os.path.abspath(__file__))


class StreamingClient(NotebookClient):
    """NotebookClient that prints each cell's stdout/stderr as it arrives."""

    def output(self, outs, msg, display_id, cell_index):
        res = super().output(outs, msg, display_id, cell_index)
        mtype = msg.get("msg_type")
        content = msg.get("content", {})
        if mtype == "stream":
            sys.stdout.write(content.get("text", ""))
            sys.stdout.flush()
        elif mtype == "error":
            sys.stdout.write("\n".join(content.get("traceback", [])) + "\n")
            sys.stdout.flush()
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", default=os.path.join(HERE, "TruthLens_Research_Project.ipynb"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--allow-errors", action="store_true",
                    help="keep going past a failing cell (default: stop)")
    args = ap.parse_args()
    out_path = args.out or args.nb

    nb = nbformat.read(args.nb, as_version=4)
    print(f"executing {os.path.basename(args.nb)}: {len(nb.cells)} cells "
          f"(per-cell timeout {args.timeout}s)", flush=True)

    client = StreamingClient(
        nb, timeout=args.timeout, kernel_name="python3",
        resources={"metadata": {"path": HERE}},
        allow_errors=args.allow_errors,
        record_timing=True,
    )

    # Write back ONLY on an outcome that actually produced results: a clean run, or
    # a run where a specific cell raised (worth capturing, since the traceback is the
    # useful artifact). Every other exit -- interrupt, dead kernel, SIGPIPE, any
    # unexpected exception -- leaves the file alone.
    #
    # This matters more than it looks. The in-memory notebook is whatever was loaded
    # at startup. If build_notebook.py regenerated the file in the meantime, an
    # unconditional write-back silently reverts it, and the NEXT run then executes
    # stale code while appearing to run the new version. That has already happened
    # twice here: once via Ctrl-C and once via a kernel that died mid-encode.
    t0 = time.time()
    status = 0
    write_back = False
    try:
        client.execute()
        write_back = True
    except CellExecutionError as e:
        status = 1
        write_back = True          # a real per-cell failure; keep the traceback
        print(f"\n!!! CELL EXECUTION FAILED: {e}", flush=True)
    except KeyboardInterrupt:
        status = 130
        print("\ninterrupted - source notebook left untouched", flush=True)
    except BaseException as e:     # dead kernel, SIGPIPE, anything else
        status = 2
        print(f"\n!!! ABORTED ({type(e).__name__}: {e}) - source notebook left untouched",
              flush=True)
    finally:
        if write_back:
            nbformat.write(nb, out_path)
            print(f"\nwrote {out_path} after {time.time()-t0:.0f}s", flush=True)
        else:
            print(f"NOT writing back after {time.time()-t0:.0f}s "
                  f"- regenerate with `python build_notebook.py` before retrying",
                  flush=True)

    # summarise which cells actually ran and which errored
    ran = sum(1 for c in nb.cells
              if c.cell_type == "code" and c.get("execution_count") is not None)
    errs = [i for i, c in enumerate(nb.cells)
            if any(o.get("output_type") == "error" for o in c.get("outputs", []))]
    total_code = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"code cells executed: {ran}/{total_code} | cells with errors: {errs or 'none'}")
    return status


if __name__ == "__main__":
    sys.exit(main())
