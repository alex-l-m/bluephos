import os
import shutil
from pathlib import Path

from ase.calculators.orca import ORCA, OrcaProfile
from tblite.ase import TBLite

# ----------------------------
# TBLite (in-memory calculator)
# ----------------------------


def _tblite_setup(atoms, multiplicity: int) -> None:
    """Attach a TBLite calculator with a fixed multiplicity."""
    atoms.calc = TBLite(multiplicity=multiplicity)


def tblite_singlet_setup(atoms) -> None:
    """TBLite single-point / forces setup for a singlet (multiplicity=1)."""
    _tblite_setup(atoms, multiplicity=1)


def tblite_triplet_setup(atoms) -> None:
    """TBLite single-point / forces setup for a triplet (multiplicity=3)."""
    _tblite_setup(atoms, multiplicity=3)


# ----------------------------
# ORCA (file-based calculator)
# ----------------------------

# Hardcoded ORCA settings
_ORCA_SIMPLEINPUT = "B3LYP LANL2DZ"
_ORCA_BLOCKS = """%pal
  nprocs 24
end

%maxcore 3000
"""

_ORCA_PROFILE = OrcaProfile(command=str(Path(os.environ["EBROOTORCA"]) / "orca"))


def _orca_setup(atoms, multiplicity: int) -> None:
    geometry_name = atoms.info["name"]
    if multiplicity == 1:
        job_id = f"{geometry_name}_singlet"
    elif multiplicity == 3:
        job_id = f"{geometry_name}_triplet"
    else:
        raise ValueError(f"Unexpected multiplicity: {multiplicity}")
    run_dir = Path("orca_runs") / str(job_id)
    # Delete the run directory if it already exists
    if run_dir.exists():
        shutil.rmtree(run_dir)  # deletes dir and all contents
    run_dir.mkdir(parents=True)

    atoms.calc = ORCA(
        profile=_ORCA_PROFILE,
        directory=str(run_dir),
        charge=0,
        mult=multiplicity,
        orcasimpleinput=_ORCA_SIMPLEINPUT,
        orcablocks=_ORCA_BLOCKS,
    )


def orca_singlet_setup(atoms) -> None:
    """ORCA single-point setup for a singlet (multiplicity=1)."""
    _orca_setup(atoms, multiplicity=1)


def orca_triplet_setup(atoms) -> None:
    """ORCA single-point setup for a triplet (multiplicity=3)."""
    _orca_setup(atoms, multiplicity=3)
