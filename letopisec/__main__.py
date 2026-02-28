from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from letopisec import __version__
from letopisec.server import parse_serve_config, serve


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="letopisec", description="Letopisec CF3D data upload server")
    parser.add_argument("--version", action="version", version=f"letopisec {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "serve",
        add_help=False,
        help="Run REST API server",
        description="Run REST API server",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv) if argv is not None else list(sys.argv[1:])
    parser = _build_parser()
    namespace, remaining = parser.parse_known_args(args)

    if namespace.command == "serve":
        config = parse_serve_config(remaining)
        serve(config)
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
