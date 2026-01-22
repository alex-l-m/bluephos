import ray
from dplutils.cli import cli_run, get_argparser
from dplutils.pipeline import PipelineGraph
from dplutils.pipeline.ray import RayStreamGraphExecutor

from bluephos.modules.setup_functions import tblite_singlet_setup, tblite_triplet_setup
from bluephos.tasks.ligate_homoleptic_iridium import LigateHomolepticIrTask
from bluephos.tasks.make_energy_task import make_energy_task
from bluephos.tasks.molblock_to_octahedral_geometries import MolblockToOctahedralGeometriesTask
from bluephos.tasks.read_ligand_smiles_file import ReadLigandSmilesTask

ap = get_argparser()
ap.set_defaults(file="run.yaml")
args = ap.parse_args()

ray.init()

graph = PipelineGraph(
    [
        ReadLigandSmilesTask,
        LigateHomolepticIrTask,
        MolblockToOctahedralGeometriesTask,
        make_energy_task(tblite_singlet_setup, "octahedral_embed_xyz", "tblite_singlet_energy"),
        make_energy_task(tblite_triplet_setup, "octahedral_embed_xyz", "tblite_triplet_energy"),
    ]
)

executor = RayStreamGraphExecutor(graph, max_batches=1)

cli_run(executor, args)
