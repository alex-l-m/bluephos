import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Final

import torch
from ase import Atoms
from ase.calculators.orca import ORCA, OrcaProfile
from fairchem.core import FAIRChemCalculator, pretrained_mlip
from tblite.ase import TBLite


# ----------------------------
# Internal helpers / constants
# ----------------------------

_ORCA_SIMPLEINPUT: Final[str] = "B3LYP LANL2DZ"
_ORCA_BLOCKS: Final[str] = """%pal
  nprocs 24
end

%maxcore 3000
"""

_UMA_MODEL_NAME: Final[str] = "uma-s-1p1"
_UMA_TASK_NAME: Final[str] = "omol"


@lru_cache(maxsize=1)
def _uma_predictor_and_device() -> tuple[object, str]:
    """
    Load the UMA predictor once per Python process and reuse it.
    This avoids re-loading the model for singlet/triplet calls in the same run.

    Returns:
        (predictor, device_str)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = pretrained_mlip.get_predict_unit(_UMA_MODEL_NAME, device=device)
    return predictor, device


# ----------------------------
# TBLite (in-memory calculator)
# ----------------------------

def tblite_setup(charge: int, multiplicity: int, atoms: Atoms) -> None:
    """
    Attach a TBLite ASE calculator to `atoms`.

    Required inputs:
      - charge: total molecular charge (will be cast to int)
      - multiplicity: total spin multiplicity (will be cast to int)
      - atoms: ASE Atoms object (runtime checked)

    Assumptions:
      - TBLite method is whatever tblite uses by default (unless you change tblite defaults).
      - This is intended for energies (and whatever properties TBLite supports) without
        adding extra configuration knobs here.

    Serialization note:
      - The calculator object is constructed inside this function.
    """
    atoms.calc = TBLite(charge=int(charge),
                        multiplicity=int(multiplicity))


# ----------------------------
# ORCA (file-based calculator)
# ----------------------------

def orca_setup(charge: int, multiplicity: int, atoms: Atoms) -> None:
    """
    Attach an ORCA ASE calculator to `atoms`, writing files under ./orca_runs/<job_id>/.

    Required inputs:
      - charge: total molecular charge (will be cast to int)
      - multiplicity: total spin multiplicity (will be cast to int)
      - atoms: ASE Atoms object (runtime checked)

    Assumptions:
      - ORCA binary is available at: $EBROOTORCA/orca (and environment is correctly set).
      - atoms.info["name"] exists (or we fall back to chemical formula), and DOES NOT contain slashes.
      - The tuple (geometry_name, charge, multiplicity) is unique for the whole run.
        This is important because we delete any pre-existing run directory for that job_id.
      - Energy-only intent: we are NOT requesting gradients here. Do not expect forces.

    Serialization note:
      - The calculator object is constructed inside this function.
    """
    charge_i = int(charge)
    mult_i = int(multiplicity)

    geometry_name = atoms.info.get("name", atoms.get_chemical_formula())
    job_id = f"{geometry_name}_q{charge_i}_m{mult_i}"

    run_dir = Path("orca_runs") / job_id
    if run_dir.exists():
        # Assumption: job_id is unique within the run, and you control `geometry_name`,
        # so deleting an existing directory is safe and intentional.
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    orca_command = str(Path(os.environ["EBROOTORCA"]) / "orca")
    orca_profile = OrcaProfile(command=orca_command)

    atoms.calc = ORCA(
        profile=orca_profile,
        directory=str(run_dir),
        charge=charge_i,
        mult=mult_i,
        orcasimpleinput=_ORCA_SIMPLEINPUT,
        orcablocks=_ORCA_BLOCKS,
    )


# ----------------------------
# FAIRChem UMA (in-memory MLIP calculator)
# ----------------------------

def uma_small_setup(charge: int, multiplicity: int, atoms: Atoms) -> None:
    """
    Attach the FAIRChem UMA small model as an ASE calculator.

    Required inputs:
      - charge: total molecular charge (will be cast to int)
      - multiplicity: total spin multiplicity (will be cast to int)
      - atoms: ASE Atoms object (runtime checked)

    Assumptions:
      - Uses UMA small model: "uma-s-1p1"
      - Uses task_name="omol" (molecules)
      - Stores charge/spin on atoms.info as:
          atoms.info["charge"] = charge
          atoms.info["spin"]   = multiplicity
      - Device selection is automatic per process:
          "cuda" if torch.cuda.is_available() else "cpu"

    Serialization note:
      - The calculator object is constructed inside this function.
      - The model predictor is cached once per process for performance.
    """

    # FAIRChem convention used in their molecular examples: `spin` is multiplicity.
    atoms.info.update({"charge": int(charge), "spin": int(multiplicity)})

    predictor, _device = _uma_predictor_and_device()
    atoms.calc = FAIRChemCalculator(predictor, task_name=_UMA_TASK_NAME)
