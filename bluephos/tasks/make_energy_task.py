import io
from functools import partial

import ase.io
import pandas as pd
from dplutils.pipeline import PipelineTask


def energy_from_extxyz(setup_function, geometry_column: str, energy_column: str, df: pd.DataFrame) -> pd.DataFrame:
    """Compute single-point energies from extxyz geometries in df[geometry_column].

    Assumptions:
      - df[geometry_column] contains extxyz text, or "failed"/missing.
      - setup_function(atoms) attaches an ASE calculator to atoms.
    """
    error_column = f"{energy_column}_error"

    if geometry_column not in df.columns:
        raise ValueError(f"Missing geometry column: {geometry_column}")

    if energy_column not in df.columns:
        df[energy_column] = None
    if error_column not in df.columns:
        df[error_column] = None

    def one_row(row: pd.Series) -> pd.Series:
        geom = row.get(geometry_column)

        if geom is None or pd.isna(geom) or str(geom) == "failed":
            row[energy_column] = None
            row[error_column] = "missing_or_failed_geometry"
            return row

        try:
            atoms = ase.io.read(io.StringIO(str(geom)), format="extxyz")
            setup_function(atoms)
            row[energy_column] = float(atoms.get_potential_energy())
            row[error_column] = None
        except Exception as exc:
            row[energy_column] = None
            row[error_column] = f"{type(exc).__name__}: {exc}"

        return row

    return df.apply(one_row, axis=1)


def make_energy_task(setup_function, geometry_column: str, energy_column: str) -> PipelineTask:
    """Return a PipelineTask that computes energies from extxyz geometries."""
    task_name = f"energy_{energy_column}"

    # Freeze the first three args; the resulting function expects only df.
    func = partial(energy_from_extxyz, setup_function, geometry_column, energy_column)

    return PipelineTask(task_name, func, batch_size=1, num_cpus=1)
