from atomate.vasp.fireworks.core import OptimizeFW
from atomate.vasp.config import VASP_CMD, DB_FILE
from atomate.utils.utils import get_meta_from_structure

from CatFlows.dft_settings.settings import MOSurfaceSet


def Slab_FW(
    slab,
    name="",
    parents=None,
    add_slab_metadata=True,
    vasp_cmd=VASP_CMD,
    db_file=DB_FILE,
):
    """
    Function to generate a slab firework. Returns an OptimizeFW for the specified slab.

    Args:
        slab              (Slab Object)  : Slab corresponding to the slab to be calculated.
        name              (string)       : name of firework
        parents           (default: None): parent FWs
        add_slab_metadata (default: True): Whether to add slab metadata to task doc.
        vasp_cmd                         : vasp_comand
        db_file                          : Path to the dabase file

    Returns:
        Firework correspoding to slab calculation.
    """

    # DFT Method
    vasp_input_set = MOSurfaceSet(slab, bulk=False)

    # FW
    fw = OptimizeFW(
        name=name,
        structure=slab,
        max_force_threshold=None,
        vasp_input_set=vasp_input_set,
        vasp_cmd=vasp_cmd,
        db_file=db_file,
        parents=parents,
        job_type="normal",
    )

    # Add slab metadata
    if add_slab_metadata:
        parent_structure_metadata = get_meta_from_structure(slab.oriented_unit_cell)
        fw.tasks[-1]["additional_fields"].update(
            {
                "slab": slab,
                "parent_structure": slab.oriented_unit_cell,
                "parent_structure_metadata": parent_structure_metadata,
            }
        )

    return fw