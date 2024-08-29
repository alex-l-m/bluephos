__doc__ = """"BluePhos Discovery Pipeline"""

from pathlib import Path
import pandas as pd
from dplutils import cli
from dplutils.pipeline.ray import RayStreamGraphExecutor
from bluephos.tasks.generateligandtable import GenerateLigandTableTask
from bluephos.tasks.nn import NNTask
from bluephos.tasks.optimizegeometries import OptimizeGeometriesTask
from bluephos.tasks.carbene_smiles2sdf import Smiles2SDFTask
from bluephos.tasks.dft import DFTTask


def rerun_candidate_generator(input_dir, t_ste):
    """
    Generates candidate DataFrames from parquet files in the input directory.

    Core Algorithm:
    - and 'ste' is None or its absolute value is less than t_ste,
    - and 'dft_energy_diff' is None,
    This row is then added to a new DataFrame and yielded for re-run.

    Additional Context:
    1. All valid ligand pairs should already have run through the NN process and have a 'z' score.
    2. If a row's 'ste' is None, then it's 'dft_energy_diff' should also be None.

    Args:
        input_dir (str): Directory containing input parquet files.
        t_ste (float): Threshold for 'ste'.

    Yields:
        DataFrame: A single-row DataFrame containing candidate data.
    """
    for file in Path(input_dir).glob("*.parquet"):
        df = pd.read_parquet(file)

        filtered = df[
            (df["z"].notnull())
            & ((df["ste"].isnull()) | (df["ste"].abs() < t_ste))
            & (df["dft_energy_diff"].isna())
        ]
        for _, row in filtered.iterrows():
            yield row.to_frame().transpose()

def molecule_generator(smiles_file):
    '''Read a csv file with "mol_id" and "smiles" columns and generate tables
    corresponding to individual rows'''
    df = pd.read_csv(smiles_file)
    for _, row in df.iterrows():
        yield row.to_frame().transpose()

def get_generator(smiles, input_dir, t_ste):
    """
    Get the appropriate generator based on the input directory presence.
    """
    if not input_dir:
        return lambda: molecule_generator(smiles)
    return lambda: rerun_candidate_generator(input_dir, t_ste)

def get_pipeline(
    smiles, # Path to a csv containing "mol_id" and "smiles" columns
    element_features,  # Path to the element features file
    train_stats,  # Path to the train stats file
    model_weights,  # Path to the model weights file
    input_dir=None,  # Directory containing input parquet files(rerun). Defaults to None.
    dft_package="orca",  # DFT package to use. Defaults to "orca".
    t_ste=1.9,  # Threshold for 'ste'. Defaults to None
):
    """
    Set up and return the BluePhos discovery pipeline executor
    Returns:
        RayStreamGraphExecutor: An executor for the BluePhos discovery pipeline
    """
    steps = (
        [
            Smiles2SDFTask,
            OptimizeGeometriesTask,
            DFTTask,
        ]
        if not input_dir
        else [
            OptimizeGeometriesTask,
            DFTTask,
        ]
    )
    generator = get_generator(smiles, input_dir, t_ste)
    pipeline_executor = RayStreamGraphExecutor(graph=steps, generator=generator)

    context_dict = {
        "smiles": smiles,
        "element_features": element_features,
        "train_stats": train_stats,
        "model_weights": model_weights,
        "dft_package": dft_package,
        "t_ste": t_ste,
    }

    for key, value in context_dict.items():
        pipeline_executor.set_context(key, value)
    return pipeline_executor


if __name__ == "__main__":
    ap = cli.get_argparser(description=__doc__)
    ap.add_argument("--smiles", required=False, help="CSV file containing 'mol_id' and 'smiles' columns")
    ap.add_argument("--features", required=True, help="Element feature file")
    ap.add_argument("--train", required=True, help="Train stats file")
    ap.add_argument("--weights", required=True, help="Full energy model weights")
    ap.add_argument("--input_dir", required=False, help="Directory containing input parquet files")
    ap.add_argument("--t_ste", type=float, required=False, default=1.9, help="Threshold for 'ste' (default: 1.9)")

    ap.add_argument(
        "--dft_package",
        required=False,
        default="orca",
        choices=["orca", "ase"],
        help="DFT package to use (default: orca)",
    )
    args = ap.parse_args()

    # Run the pipeline with the provided arguments
    cli.cli_run(
        get_pipeline(
            args.smiles,
            args.features,
            args.train,
            args.weights,
            args.input_dir,
            args.dft_package,
            args.t_ste,
        ),
        args,
    )
