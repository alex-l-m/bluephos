from functools import reduce
import pandas as pd
import bluephos.modules.log_config as log_config
from dplutils.pipeline import PipelineTask
from rdkit import Chem
from rdkit.Chem import AddHs, AllChem, MolFromSmiles
from rdkit.Chem.rdmolops import CombineMols, Kekulize, SanitizeMol


# Setup logging and get a logger instance
logger = log_config.setup_logging(__name__)



def smiles_to_sdf(df: pd.DataFrame) -> pd.DataFrame:
    """Convert SMILES strings to SDF format and Add to Input dataframe"""

    for index, row in df.iterrows():
        mol_nohs = MolFromSmiles(row['smiles'])
        if mol_nohs is not None:
            mol = AddHs(mol_nohs)
            AllChem.Compute2DCoords(mol)
            mol.SetProp("_Name", row["mol_id"])
            mol_block = Chem.MolToMolBlock(mol)
            df.at[index, "structure"] = mol_block
        else:
            logger.warning(f"mol generation failed for index {index}, identifier {row['mol_id']}.")
    return df


Smiles2SDFTask = PipelineTask("smiles2sdf", smiles_to_sdf)
