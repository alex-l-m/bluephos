import io
from functools import partial

import ase.io
import pandas as pd
from ase import Atoms
from ase.optimize import BFGS
from dplutils.pipeline import PipelineTask


def optimize_geometry(setup_function, atoms: Atoms) -> None:
    setup_function(atoms)
    opt = BFGS(atoms, logfile=None, trajectory=None)
    opt.run(fmax=0.02)


def get_energy(setup_function, atoms: Atoms) -> float:
    setup_function(atoms)
    return float(atoms.get_potential_energy())


def optimize_from_extxyz(
    setup_function,
    in_geometry_column: str,
    out_geometry_column: str,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Optimize extxyz in df[in_geometry_column] and write extxyz to df[out_geometry_column]."""
    error_column = f"{out_geometry_column}_error"

    if in_geometry_column not in df.columns:
        raise ValueError(f"Missing geometry column: {in_geometry_column}")

    if out_geometry_column not in df.columns:
        df[out_geometry_column] = None
    if error_column not in df.columns:
        df[error_column] = None

    def one_row(row: pd.Series) -> pd.Series:
        geom = row.get(in_geometry_column)

        if geom is None or pd.isna(geom) or str(geom) == "failed":
            row[out_geometry_column] = "failed"
            row[error_column] = "missing_or_failed_geometry"
            return row

        try:
            atoms = ase.io.read(io.StringIO(str(geom)), format="extxyz")
            optimize_geometry(setup_function, atoms)

            buf = io.StringIO()
            ase.io.write(buf, atoms, format="extxyz")

            row[out_geometry_column] = buf.getvalue()
            row[error_column] = None
        except Exception as exc:
            row[out_geometry_column] = "failed"
            row[error_column] = f"{type(exc).__name__}: {exc}"

        return row

    return df.apply(one_row, axis=1)


def energy_from_extxyz(
    setup_function,
    geometry_column: str,
    energy_column: str,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute single-point energies from extxyz geometries in df[geometry_column]."""
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
            row[energy_column] = get_energy(setup_function, atoms)
            row[error_column] = None
        except Exception as exc:
            row[energy_column] = None
            row[error_column] = f"{type(exc).__name__}: {exc}"

        return row

    return df.apply(one_row, axis=1)


def make_energy_task(setup_function, geometry_column: str, energy_column: str) -> PipelineTask:
    task_name = energy_column
    func = partial(energy_from_extxyz, setup_function, geometry_column, energy_column)
    return PipelineTask(task_name, func, batch_size=1, num_cpus=1)


def make_optimization_task(
    setup_function,
    in_geometry_column: str,
    out_geometry_column: str,
) -> PipelineTask:
    task_name = out_geometry_column
    func = partial(optimize_from_extxyz, setup_function, in_geometry_column, out_geometry_column)
    return PipelineTask(task_name, func, batch_size=1, num_cpus=1)
