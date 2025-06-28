#!/usr/bin/env python3
import sys
from pathlib import Path

from pytest import MonkeyPatch

import cgenff_charmm2gmx


def test_cgenff_charmm2gmx(monkeypatch: MonkeyPatch):
    # We monkeypatch here to inject the command line arguments
    with monkeypatch.context() as m:
        m.setattr(
            sys,
            "argv",
            [
                "cgenff_charmm2gmx",
                "JZ4",
                "tests/test_data/jz4_fix.mol2",
                "tests/test_data/jz4.str",
                "tests/test_data/charmm36_ljpme-jul2022.ff",
            ],
        )

        cgenff_charmm2gmx.main()

        # Verify the files are equivalent ones generated from the Python 3.7 script

        files_to_compare = [
            Path("jz4_ini.pdb"),
            Path("jz4.itp"),
            Path("jz4.prm"),
            Path("jz4.top"),
        ]

        for file in files_to_compare:
            reference_file = "tests/output_3.7" / file
            print(f"Comparing {file}")
            assert reference_file.read_text() == file.read_text()

            # Delete the file to cleanup

            file.unlink()
