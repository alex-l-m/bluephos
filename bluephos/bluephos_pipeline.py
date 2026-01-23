import ray
import pandas as pd

from dplutils.cli import cli_run
from dplutils.pipeline import PipelineGraph
from dplutils.pipeline.ray import RayStreamGraphExecutor

from bluephos.modules.setup_functions import tblite_singlet_setup, tblite_triplet_setup
from bluephos.tasks.ligate_homoleptic_iridium import LigateHomolepticIrTask
from bluephos.tasks.make_ase_tasks import make_energy_task, make_optimization_task
from bluephos.tasks.molblock_to_octahedral_geometries import MolblockToOctahedralGeometriesTask

ray.init()

graph = PipelineGraph([
    LigateHomolepticIrTask,
    MolblockToOctahedralGeometriesTask,
    make_optimization_task(tblite_triplet_setup, 'octahedral_embed_xyz', 'tblite_triplet_optimized_xyz'),
    make_energy_task(tblite_singlet_setup, 'tblite_triplet_optimized_xyz', 'tblite_singlet_energy'),
    make_energy_task(tblite_triplet_setup, 'tblite_triplet_optimized_xyz', 'tblite_triplet_energy')
    ])

executor = RayStreamGraphExecutor(graph,
        generator=lambda: pd.read_csv("in.csv", chunksize=200),
)


cli_run(executor)
