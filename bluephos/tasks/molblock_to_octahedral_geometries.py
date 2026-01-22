import io

import ase.io
import pandas as pd
from ase import Atoms
from dplutils.pipeline import PipelineTask
from octahedral_embed import octahedral_embed
from rdkit import Chem

ISOMERS = ("fac", "mer")


def molblock_to_octahedral_geometries(df: pd.DataFrame) -> pd.DataFrame:
    """Generate fac and mer octahedral geometries from complex MolBlocks.

    Expected input columns:
      - complex_identifier
      - complex_structure   (RDKit MolBlock / SDF block as a string)

    Output: two rows per input row (fac + mer), with added columns:
      - isomer
      - isomer_identifier           (e.g. "<complex_identifier>_fac")
      - octahedral_embed_xyz        extxyz text (or "failed")
      - octahedral_embed_error      None on success, else an error string

    Notes / assumptions:
      - complex_structure is readable by RDKit (MolFromMolBlock).
      - octahedral_embed writes a conformer with 3D coordinates into the RDKit Mol.
      - We write geometry using ASE in *extxyz* format so ASE can preserve
        atoms.info["name"] on round-trip read/write.
        (Downstream: read with format="extxyz", not "xyz".)
    """
    out = []

    for _, row in df.iterrows():
        base_id = str(row["complex_identifier"])
        block = row["complex_structure"]

        mol0 = None
        base_error = None

        if block is None or pd.isna(block) or str(block) == "":
            base_error = "missing_complex_structure"
        else:
            mol0 = Chem.MolFromMolBlock(str(block), removeHs=False)
            if mol0 is None:
                base_error = "MolFromMolBlock failed"

        for iso in ISOMERS:
            new = dict(row)
            new["isomer"] = iso
            new["isomer_identifier"] = f"{base_id}_{iso}"

            if mol0 is None:
                new["octahedral_embed_xyz"] = "failed"
                new["octahedral_embed_error"] = base_error
                out.append(new)
                continue

            try:
                mol = Chem.Mol(mol0)  # copy
                mol.RemoveAllConformers()  # start clean
                mol.SetProp("_Name", new["isomer_identifier"])

                octahedral_embed(mol, isomer=iso)

                if mol.GetNumConformers() == 0:
                    raise ValueError("octahedral_embed produced no conformer")

                conf = mol.GetConformer()
                symbols = [a.GetSymbol() for a in mol.GetAtoms()]
                positions = conf.GetPositions()

                atoms = Atoms(symbols=symbols, positions=positions)
                atoms.info["name"] = new["isomer_identifier"]

                buf = io.StringIO()
                ase.io.write(buf, atoms, format="extxyz")

                new["octahedral_embed_xyz"] = buf.getvalue()
                new["octahedral_embed_error"] = None

            except Exception as exc:
                new["octahedral_embed_xyz"] = "failed"
                new["octahedral_embed_error"] = f"{type(exc).__name__}: {exc}"

            out.append(new)

    return pd.DataFrame(out)


MolblockToOctahedralGeometriesTask = PipelineTask(
    "molblock_to_octahedral_geometries",
    molblock_to_octahedral_geometries,
    batch_size=50,
    num_cpus=1,
)
