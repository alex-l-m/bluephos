import pandas as pd
from dplutils.pipeline import PipelineTask


def read_ligand_smiles_file(_df: pd.DataFrame, ligand_smiles: str) -> pd.DataFrame:
    """Read ligands from a .smi-style text file.

    Expected format (no header): one ligand per line:
        SMILES
      or
        SMILES ligand_identifier
    """
    rows = []
    with open(ligand_smiles, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            smiles = parts[0]
            ident = parts[1] if len(parts) > 1 else f"L{len(rows)}"
            rows.append((ident, smiles))

    return pd.DataFrame(rows, columns=["ligand_identifier", "ligand_SMILES"])


ReadLigandSmilesTask = PipelineTask(
    "read_ligand_smiles_file",
    read_ligand_smiles_file,
    context_kwargs={"ligand_smiles": "ligand_smiles"},
    batch_size=1,
)
