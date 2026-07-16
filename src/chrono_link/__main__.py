"""Run Chrono Link command modules."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m chrono_link")
    parser.add_argument("command", choices=("stream", "validate", "train", "smoke"))
    args, rest = parser.parse_known_args(argv)

    if args.command == "stream":
        from chrono_link.cli.stream import main as command
    elif args.command == "validate":
        from chrono_link.cli.validate import main as command
    elif args.command == "train":
        from chrono_link.cli.train import main as command
    else:
        from chrono_link.cli.smoke import main as command
    return command(rest)


if __name__ == "__main__":
    raise SystemExit(main())
