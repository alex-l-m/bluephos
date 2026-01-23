import pandas as pd
from dplutils.pipeline import PipelineTask
from rdkit import Chem
from rdkit.Chem import AddHs, AllChem

from octahedral_embed import ligate


def _ligate_row(row: pd.Series) -> pd.Series:
    """Create a homoleptic Ir(L)3 complex for one ligand row.

    Assumes row has:
      - ligand_identifier
      - ligand_SMILES
    """
    complex_id = f"{row['ligand_identifier']}_homoleptic"
    row["complex_identifier"] = complex_id

    try:
        lig = Chem.MolFromSmiles(row["ligand_SMILES"])
        if lig is None:
            raise ValueError("MolFromSmiles failed")

        # Use independent copies in case ligate mutates inputs
        complex_mol = ligate([Chem.Mol(lig) for _ in range(3)])

        complex_mol = AddHs(complex_mol)
        AllChem.Compute2DCoords(complex_mol)
        complex_mol.SetProp("_Name", complex_id)

        row["complex_structure"] = Chem.MolToMolBlock(complex_mol)
        row["ligate_error"] = None

    except Exception as exc:
        row["complex_structure"] = None
        row["ligate_error"] = f"{type(exc).__name__}: {exc}"

    return row


def ligate_homoleptic_iridium(df: pd.DataFrame) -> pd.DataFrame:
    """Build homoleptic Ir(L)3 complexes from ligand SMILES.

    Expected input columns:
      - ligand_identifier
      - ligand_SMILES   (ligand SMILES with dummy atoms '*' at coordination sites)

    Adds output columns:
      - complex_identifier   (e.g. "<ligand_identifier>_homoleptic")
      - complex_structure    (RDKit MolBlock / SDF block for the complex, or None)
      - ligate_error         (None on success, else an error string)
    """
    return df.apply(_ligate_row, axis=1)


LigateHomolepticIrTask = PipelineTask(
    "ligate_homoleptic_iridium",
    ligate_homoleptic_iridium,
    batch_size=200,
)
