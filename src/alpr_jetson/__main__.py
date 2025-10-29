from __future__ import annotations

import sys
from alpr_jetson.cli.parser import build_parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    rc = args.func(args)
    sys.exit(rc)


if __name__ == "__main__":
    main()

