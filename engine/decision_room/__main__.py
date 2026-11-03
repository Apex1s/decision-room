"""Command line: `python -m decision_room serve | demo | export <folder>`."""

from __future__ import annotations

import argparse
import sys
import time

from .server import serve
from .workspace import Workspace


def main() -> None:
    # Windows consoles default to cp1252; the memo and the protocol are UTF-8.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="decision_room")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve", help="JSON-RPC engine on stdio (used by the ACP agent)")
    demo = sub.add_parser("demo", help="run the full pipeline on a market and write the memo")
    demo.add_argument("--data", default="demo", help="market folder, or 'demo' for the synthetic market")
    demo.add_argument("--out", default="out")
    demo.add_argument("--draws", type=int, default=200)
    exp = sub.add_parser("export", help="write the synthetic demo market as CSV files")
    exp.add_argument("folder")
    args = ap.parse_args()

    if args.cmd == "serve":
        serve()
    elif args.cmd == "export":
        print(Workspace().export(args.folder))
    else:
        t0 = time.time()
        ws = Workspace(lambda msg: print(f"  [{time.time() - t0:5.1f}s] {msg}", file=sys.stderr))
        ws.load(args.data)
        res = ws.brief(out_dir=args.out, draws=args.draws)["memo"]
        print(res["markdown"])
        print(f"\nGrade {res['grade']} · files: {', '.join(res['files'])} · {time.time() - t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
