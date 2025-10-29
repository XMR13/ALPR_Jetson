from __future__ import annotations

import argparse

from alpr_jetson.cli.commands import e2e as cmd_e2e
from alpr_jetson.cli.commands import detector as cmd_det
from alpr_jetson.cli.commands import ocr as cmd_ocr
from alpr_jetson.cli.commands import deepstream as cmd_ds


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="alpr-jetson", description="ALPR Jetson CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    cmd_ds.add_subcommands(sub)
    cmd_ocr.add_subcommand(sub)
    cmd_det.add_subcommand(sub)
    cmd_e2e.add_subcommands(sub)

    return p

