from bluephos.modules.setup_functions import tblite_setup, orca_setup, uma_small_setup

import pandas as pd
from rdkit import Chem
from ase import Atoms
from octahedral_embed import ligate, octahedral_embed


# Phenylpyridine ligand
# Use dummy atoms (*) at the chelation sites
# Use a dative bond (->) for the chelating N
ppy = Chem.MolFromSmiles("c1ccc(c(*)c1)c2cccc(n2->*)")

# Ir(ppy)3, with explicit hydrogens to prepare for embedding
irppy3_rdkit = Chem.AddHs(ligate([ppy] * 3))

# Generate the fac isomer
octahedral_embed(irppy3_rdkit, isomer="fac")

# ASE atoms
pos = irppy3_rdkit.GetConformer().GetPositions()
elem = [atom.GetSymbol() for atom in irppy3_rdkit.GetAtoms()]
irppy3 = Atoms(positions=pos, symbols=elem)
irppy3.info["name"] = "irppy3"


# Triplet energy above ground state at a fixed geometry, for all calculators.
# (Two single-point energy evaluations per method: triplet then singlet.)
results_rows = []

for setup in [tblite_setup, orca_setup, uma_small_setup]:
    method_name = setup.__name__.replace("_setup", "")

    setup(0, 3, irppy3)
    triplet = irppy3.get_potential_energy()

    setup(0, 1, irppy3)
    ground = irppy3.get_potential_energy()

    results_rows.append(
        {
            "method": method_name,
            "t_g_gap": triplet - ground,
            "triplet_energy": triplet,
            "ground_energy": ground,
        }
    )

df = pd.DataFrame(results_rows)
df.to_csv("irppy3_triplet_gaps.csv", index=False)
