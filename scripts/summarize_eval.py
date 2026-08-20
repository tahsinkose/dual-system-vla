"""Summarise an evaluation JSONL log: success rate, step counts, subtask completion.

`eval/run_eval.py` prints this for the slice it just ran. Reading the log back gives the
same summary after the fact — for a run that was interrupted, or one whose output has
scrolled away.

One invocation of the harness evaluates one checkpoint under one conditioning, so a log
is one run and its episodes summarise as a whole.

Example::

    python scripts/summarize_eval.py outputs/eval/live/results.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.logging import read_results, summarize  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log", type=Path, help="a JSONL log written by eval/run_eval.py")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.log.is_file():
        raise SystemExit(f"no such log: {args.log}")

    results = read_results(args.log)
    if not results:
        raise SystemExit(f"no episodes in {args.log}")

    print(summarize(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
