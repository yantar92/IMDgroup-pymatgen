#!/usr/bin/env python

"""Master script to work with VASP inputs and outputs."""

import argparse
import IMDgroup.pymatgen.cli.imdg_create


def main():
    """Main routine."""
    parser = argparse.ArgumentParser(
        description="""
        imdg is a script to generate and analyze VASP inputs/outputs.
        The script is similar to pymatgen's pmg, but implements
        workflows used in the Inverse Material Design Group.

        The scripts supports several subcommands.  Type "imdg
        sub-command -h" to get help for individual sub-commands.
        """,
        epilog="Autho: Ihor Radchenko"
    )

    subparsers = parser.add_subparsers(required=True)

    parser_create = subparsers.add_parser("create")
    IMDgroup.pymatgen.cli.imdg_create.add_args(parser_create)
    parser_create.set_defaults(func=IMDgroup.pymatgen.cli.imdg_create.create)

    args = parser.parse_args()
    print(args)

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
