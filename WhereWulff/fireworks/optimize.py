from typing import Counter
from atomate.vasp.fireworks.core import OptimizeFW
from atomate.vasp.config import VASP_CMD, DB_FILE
from atomate.utils.utils import get_meta_from_structure, env_chk
from atomate.vasp.firetasks.run_calc import RunVaspFake, RunVaspDirect, RunVaspCustodian
import os

from WhereWulff.dft_settings.settings import MOSurfaceSet
from WhereWulff.firetasks.handlers import ContinueOptimizeFW


# Dictionary that holds the paths to the VASP input
# and output files. Right now this assumes that the code is run
# in a container (/home/jovyan) with the files placed in the right folder.
# Maps fw_name to the ref_dir
ref_dirs = {
    # RuO2
    "RuO2_110 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_bulk_110",
    "RuO2_101 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_bulk_101",
    "RuO2_110 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_slab_110",
    "RuO2_101 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_slab_101",
    "RuO2-110-O_1": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_Ox_pbx_1",
    "RuO2-110-OH_1": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_OH_pbx_1",
    "RuO2-110-OH_2": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_OH_pbx_2",
    "RuO2-110-OH_3": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_OH_pbx_3",
    "RuO2-110-OH_4": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_OH_pbx_4",
    "RuO2-101-O_1": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_Ox_pbx_1",
    "RuO2-101-OH_1": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_OH_pbx_1",
    "RuO2-101-OH_2": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_OH_pbx_2",
    "RuO2-101-OH_3": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_OH_pbx_3",
    "RuO2-101-OH_4": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_OH_pbx_4",
    "RuO2-110-Ru-reference": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_reference",
    "RuO2-110-Ru-OH_0": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_OH_0",
    "RuO2-110-Ru-OH_1": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_OH_1",
    "RuO2-110-Ru-OH_2": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_OH_2",
    "RuO2-110-Ru-OH_3": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_OH_3",
    "RuO2-110-Ru-OOH_up_0": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_OOH_up_0",
    "RuO2-110-Ru-OOH_up_1": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_OOH_up_1",
    "RuO2-110-Ru-OOH_up_2": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_OOH_up_2",
    "RuO2-110-Ru-OOH_up_3": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_OOH_up_3",
    "RuO2-110-Ru-OOH_down_0": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_OOH_down_0",
    "RuO2-110-Ru-OOH_down_1": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_OOH_down_1",
    "RuO2-110-Ru-OOH_down_2": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_OOH_down_2",
    "RuO2-110-Ru-OOH_down_3": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_110_OOH_down_3",
    "RuO2-101-Ru-reference": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_reference",
    "RuO2-101-Ru-OH_0": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_OH_0",
    "RuO2-101-Ru-OH_1": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_OH_1",
    "RuO2-101-Ru-OH_2": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_OH_2",
    "RuO2-101-Ru-OH_3": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_OH_3",
    "RuO2-101-Ru-OOH_up_0": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_OOH_up_0",
    "RuO2-101-Ru-OOH_up_1": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_OOH_up_1",
    "RuO2-101-Ru-OOH_up_2": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_OOH_up_2",
    "RuO2-101-Ru-OOH_up_3": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_OOH_up_3",
    "RuO2-101-Ru-OOH_down_0": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_OOH_down_0",
    "RuO2-101-Ru-OOH_down_1": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_OOH_down_1",
    "RuO2-101-Ru-OOH_down_2": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_OOH_down_2",
    "RuO2-101-Ru-OOH_down_3": f"{os.environ['GITHUB_WORKSPACE']}/RuO2_full_jh/RuO2_101_OOH_down_3",
    # IrO2
    "IrO2_110 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_bulk_110",
    "IrO2_101 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_bulk_101",
    "IrO2_110 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_slab_110",
    "IrO2_101 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_slab_101",
    "IrO2-110-O_1": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_Ox_pbx_1",
    "IrO2-110-OH_1": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_OH_pbx_1",
    "IrO2-110-OH_2": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_OH_pbx_2",
    "IrO2-110-OH_3": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_OH_pbx_3",
    "IrO2-110-OH_4": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_OH_pbx_4",
    "IrO2-101-O_1": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_Ox_pbx_1",
    "IrO2-101-OH_1": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_OH_pbx_1",
    "IrO2-101-OH_2": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_OH_pbx_2",
    "IrO2-101-OH_3": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_OH_pbx_3",
    "IrO2-101-OH_4": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_OH_pbx_4",
    "IrO2-110-Ir-reference": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_reference",
    "IrO2-110-Ir-OH_0": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_OH_0",
    "IrO2-110-Ir-OH_1": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_OH_1",
    "IrO2-110-Ir-OH_2": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_OH_2",
    "IrO2-110-Ir-OH_3": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_OH_3",
    "IrO2-110-Ir-OOH_up_0": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_OOH_up_0",
    "IrO2-110-Ir-OOH_up_1": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_OOH_up_1",
    "IrO2-110-Ir-OOH_up_2": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_OOH_up_2",
    "IrO2-110-Ir-OOH_up_3": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_OOH_up_3",
    "IrO2-110-Ir-OOH_down_0": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_OOH_down_0",
    "IrO2-110-Ir-OOH_down_1": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_OOH_down_1",
    "IrO2-110-Ir-OOH_down_2": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_OOH_down_2",
    "IrO2-110-Ir-OOH_down_3": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_110_OOH_down_3",
    "IrO2-101-Ir-reference": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_reference",
    "IrO2-101-Ir-OH_0": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_OH_0",
    "IrO2-101-Ir-OH_1": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_OH_1",
    "IrO2-101-Ir-OH_2": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_OH_2",
    "IrO2-101-Ir-OH_3": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_OH_3",
    "IrO2-101-Ir-OOH_up_0": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_OOH_up_0",
    "IrO2-101-Ir-OOH_up_1": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_OOH_up_1",
    "IrO2-101-Ir-OOH_up_2": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_OOH_up_2",
    "IrO2-101-Ir-OOH_up_3": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_OOH_up_3",
    "IrO2-101-Ir-OOH_down_0": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_OOH_down_0",
    "IrO2-101-Ir-OOH_down_1": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_OOH_down_1",
    "IrO2-101-Ir-OOH_down_2": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_OOH_down_2",
    "IrO2-101-Ir-OOH_down_3": f"{os.environ['GITHUB_WORKSPACE']}/IrO2_full_jh/IrO2_101_OOH_down_3",
    # RuCrO4
    "CrRuO4_110 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/RuCrO4/RuCr_110_bulk",
    "CrRuO4_101 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/RuCrO4/RuCr_101_bulk",
    "CrRuO4_110 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/RuCrO4/RuCr_110_slab",
    "CrRuO4_101 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/RuCrO4/RuCr_101_slab",
    "CrRuO4-110-OH_1": f"{os.environ['GITHUB_WORKSPACE']}/RuCrO4/RuCr_110_OH_1",
    "CrRuO4-110-OH_2": f"{os.environ['GITHUB_WORKSPACE']}/RuCrO4/RuCr_110_OH_2",
    "CrRuO4-110-OH_3": f"{os.environ['GITHUB_WORKSPACE']}/RuCrO4/RuCr_110_OH_3",
    "CrRuO4-110-OH_4": f"{os.environ['GITHUB_WORKSPACE']}/RuCrO4/RuCr_110_OH_4",
    "CrRuO4-110-O_1": f"{os.environ['GITHUB_WORKSPACE']}/RuCrO4/RuCr_110_Ox_1",
    "CrRuO4-101-OH_1": f"{os.environ['GITHUB_WORKSPACE']}/RuCrO4/RuCr_101_OH_1",
    "CrRuO4-101-OH_2": f"{os.environ['GITHUB_WORKSPACE']}/RuCrO4/RuCr_101_OH_2",
    "CrRuO4-101-OH_3": f"{os.environ['GITHUB_WORKSPACE']}/RuCrO4/RuCr_101_OH_3",
    "CrRuO4-101-OH_4": f"{os.environ['GITHUB_WORKSPACE']}/RuCrO4/RuCr_101_OH_4",
    "CrRuO4-101-O_1": f"{os.environ['GITHUB_WORKSPACE']}/RuCrO4/RuCr_101_Ox_1",
    # RuTiO4
    "TiRuO4_110 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/RuTiO4/RuTi_110_bulk",
    "TiRuO4_101 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/RuTiO4/RuTi_101_bulk",
    "TiRuO4_110 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/RuTiO4/RuTi_110_slab",
    "TiRuO4_101 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/RuTiO4/RuTi_101_slab",
    "TiRuO4-110-OH_1": f"{os.environ['GITHUB_WORKSPACE']}/RuTiO4/RuTi_110_OH_1",
    "TiRuO4-110-OH_2": f"{os.environ['GITHUB_WORKSPACE']}/RuTiO4/RuTi_110_OH_2",
    "TiRuO4-110-OH_3": f"{os.environ['GITHUB_WORKSPACE']}/RuTiO4/RuTi_110_OH_3",
    "TiRuO4-110-OH_4": f"{os.environ['GITHUB_WORKSPACE']}/RuTiO4/RuTi_110_OH_4",
    "TiRuO4-110-O_1": f"{os.environ['GITHUB_WORKSPACE']}/RuTiO4/RuTi_110_Ox_1",
    "TiRuO4-101-OH_1": f"{os.environ['GITHUB_WORKSPACE']}/RuTiO4/RuTi_101_OH_1",
    "TiRuO4-101-OH_2": f"{os.environ['GITHUB_WORKSPACE']}/RuTiO4/RuTi_101_OH_2",
    "TiRuO4-101-OH_3": f"{os.environ['GITHUB_WORKSPACE']}/RuTiO4/RuTi_101_OH_3",
    "TiRuO4-101-OH_4": f"{os.environ['GITHUB_WORKSPACE']}/RuTiO4/RuTi_101_OH_4",
    "TiRuO4-101-O_1": f"{os.environ['GITHUB_WORKSPACE']}/RuTiO4/RuTi_101_Ox_1",
    # TiCrRu2Ox - 101
    "TiCr(RuO4)2_101 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/TiCrRuO_101_results/TiCrRuO_101_bulk",
    "Ti9Cr11(RuO4)20_101 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/TiCrRuO_101_results/TiCrRuO_101_slab",
    "Ti9Cr11(RuO4)20-101-OH_1": f"{os.environ['GITHUB_WORKSPACE']}/TiCrRuO_101_results/TiCrRuO_101_OH_1",
    "Ti9Cr11(RuO4)20-101-OH_2": f"{os.environ['GITHUB_WORKSPACE']}/TiCrRuO_101_results/TiCrRuO_101_OH_2",
    "Ti9Cr11(RuO4)20-101-OH_3": f"{os.environ['GITHUB_WORKSPACE']}/TiCrRuO_101_results/TiCrRuO_101_OH_3",
    "Ti9Cr11(RuO4)20-101-OH_4": f"{os.environ['GITHUB_WORKSPACE']}/TiCrRuO_101_results/TiCrRuO_101_OH_4",
    "Ti9Cr11(RuO4)20-101-O_1": f"{os.environ['GITHUB_WORKSPACE']}/TiCrRuO_101_results/TiCrRuO_101_O_1",
    # BaSrCo2O6 - 001 - OH terminated
    "BaSr(CoO3)2_001 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/BaSrCoO_001_results/BaSrCoO_001_bulk",
    "Ba5Sr5(Co6O17)2_001 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/BaSrCoO_001_results/BaSrCoO_001_slab",
    "Ba5Sr5(Co6O17)2-001-OH_1": f"{os.environ['GITHUB_WORKSPACE']}/BaSrCoO_001_results/BaSrCoO_001_OH_1",
    "Ba5Sr5(Co6O17)2-001-OH_2": f"{os.environ['GITHUB_WORKSPACE']}/BaSrCoO_001_results/BaSrCoO_001_OH_2",
    "Ba5Sr5(Co6O17)2-001-OH_3": f"{os.environ['GITHUB_WORKSPACE']}/BaSrCoO_001_results/BaSrCoO_001_OH_3",
    "Ba5Sr5(Co6O17)2-001-OH_4": f"{os.environ['GITHUB_WORKSPACE']}/BaSrCoO_001_results/BaSrCoO_001_OH_4",
    "Ba5Sr5(Co6O17)2-001-O_1": f"{os.environ['GITHUB_WORKSPACE']}/BaSrCoO_001_results/BaSrCoO_001_O_1",
    # Ba5Sr5(Co5O16)2 - 101
    "BaSr(CoO3)2_101 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/BaSrCoO_101_results/BaSrCoO_101_bulk",
    "Ba5Sr5(Co5O16)2_101 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/BaSrCoO_101_results/BaSrCoO_101_slab",
    # Ba5Ti10Sn5O32 - 101
    "BaTi2SnO6_101 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/BaSnTiO_101_results/BaSnTiO_101_bulk",
    "Ba5Ti10Sn5O32_101 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/BaSnTiO_101_results/BaSnTiO_101_slab",
    # BaTi2SnO6 - 110
    "BaTi2SnO6_110 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/BaSnTiO_110_results/BaSnTiO_110_bulk",
    "BaTi2SnO6_110 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/BaSnTiO_110_results/BaSnTiO_110_slab",
    "BaTi2SnO6-110-OH_4": f"{os.environ['GITHUB_WORKSPACE']}/BaSnTiO_110_results/BaSnTiO_110_pbx_OH_4",
    "BaTi2SnO6-110-OH_3": f"{os.environ['GITHUB_WORKSPACE']}/BaSnTiO_110_results/BaSnTiO_110_pbx_OH_3",
    "BaTi2SnO6-110-OH_2": f"{os.environ['GITHUB_WORKSPACE']}/BaSnTiO_110_results/BaSnTiO_110_pbx_OH_2",
    "BaTi2SnO6-110-OH_1": f"{os.environ['GITHUB_WORKSPACE']}/BaSnTiO_110_results/BaSnTiO_110_pbx_OH_1",
    "BaTi2SnO6-110-O_1": f"{os.environ['GITHUB_WORKSPACE']}/BaSnTiO_110_results/BaSnTiO_110_pbx_Ox",
    # FeSb2O6 - 110
    "Fe(SbO3)2_110 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/FeSbOx_110_results/FeSbOx_101_bulk",
    "Fe(SbO3)2_110 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/FeSbOx_110_results/FeSbOx_101_slab",
    # Pt - 110 # *OH - ontop
    "Pt_110 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/Pt_110_results/Pt_110_bulk",
    "Pt_110 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/Pt_110_results/Pt_110_slab",
    "Pt-110-OH": f"{os.environ['GITHUB_WORKSPACE']}/Pt_110_results/Pt_110_OH_1",
    "Pt-110-O": f"{os.environ['GITHUB_WORKSPACE']}/Pt_110_results/Pt_110_O_1",
    "Pt-110-Pt-reference": f"{os.environ['GITHUB_WORKSPACE']}/Pt_110_results/Pt-110-Pt-reference",
    "Pt-110-Pt-OOH": f"{os.environ['GITHUB_WORKSPACE']}/Pt_110_results/Pt-110-Pt-OOH_2",
    "Pt-110-Pt-O": f"{os.environ['GITHUB_WORKSPACE']}/Pt_110_results/Pt-110-Pt-O_0",
    # Pt - 110 # *O - bridge
    "Pt_110 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/Pt_110_results_br/Pt_110_bulk",
    "Pt_110 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/Pt_110_results_br/Pt_110_slab",
    "Pt-110-OH": f"{os.environ['GITHUB_WORKSPACE']}/Pt_110_results_br/Pt_110_OH",
    "Pt-110-O": f"{os.environ['GITHUB_WORKSPACE']}/Pt_110_results_br/Pt_110_O",
    "Pt-110-Pt-reference": f"{os.environ['GITHUB_WORKSPACE']}/Pt_110_results_br/Pt-110-Pt-reference",
    "Pt-110-Pt-OOH": f"{os.environ['GITHUB_WORKSPACE']}/Pt_110_results_br/Pt-110-Pt-OOH",
    "Pt-110-Pt-OH": f"{os.environ['GITHUB_WORKSPACE']}/Pt_110_results_br/Pt-110-Pt-OH",
    # Pt - 111 # clean termination - ontop
    "Pt_111 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/Pt_111_results/Pt_111_bulk",
    "Pt_111 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/Pt_111_results/Pt_111_slab",
    "Pt-111-OH": f"{os.environ['GITHUB_WORKSPACE']}/Pt_111_results/Pt_111_OH_1",
    "Pt-111-O": f"{os.environ['GITHUB_WORKSPACE']}/Pt_111_results/Pt_111_O_1",
    "Pt-111-Pt-reference": f"{os.environ['GITHUB_WORKSPACE']}/Pt_111_results/Pt-111-Pt-reference",
    "Pt-111-Pt-OOH": f"{os.environ['GITHUB_WORKSPACE']}/Pt_111_results/Pt-111-Pt-OOH_2",
    "Pt-111-Pt-OH": f"{os.environ['GITHUB_WORKSPACE']}/Pt_111_results/Pt-111-Pt-OH_7",
    "Pt-111-Pt-O": f"{os.environ['GITHUB_WORKSPACE']}/Pt_111_results/Pt-111-Pt-O_0",
    # Pt - 100 # *OH - ontop
    "Pt_100 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/Pt_100_results/Pt_100_bulk",
    "Pt_100 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/Pt_100_results/Pt_100_slab",
    "Pt-100-OH": f"{os.environ['GITHUB_WORKSPACE']}/Pt_100_results/Pt_100_OH_1",
    "Pt-100-O": f"{os.environ['GITHUB_WORKSPACE']}/Pt_100_results/Pt_100_O_1",
    "Pt-100-Pt-reference": f"{os.environ['GITHUB_WORKSPACE']}/Pt_100_results/Pt-100-Pt-reference",
    "Pt-100-Pt-OOH": f"{os.environ['GITHUB_WORKSPACE']}/Pt_100_results/Pt-100-Pt-OOH_2",
    "Pt-100-Pt-O": f"{os.environ['GITHUB_WORKSPACE']}/Pt_100_results/Pt-100-Pt-O_0",
    # Pt - 211
    "Pt_211 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/Pt_211_results/Pt_211_bulk",
    "Pt_211 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/Pt_211_results/Pt_211_slab",
    # Au - 100 # *OH termination - bridge
    "Au_100 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/Au_100_results/Au_100_bulk",
    "Au_100 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/Au_100_results/Au_100_slab",
    "Au-100-OH": f"{os.environ['GITHUB_WORKSPACE']}/Au_100_results/Au_100_OH",
    "Au-100-O": f"{os.environ['GITHUB_WORKSPACE']}/Au_100_results/Au_100_O",
    "Au-100-Au-reference": f"{os.environ['GITHUB_WORKSPACE']}/Au_100_results/Au-100-Au-reference",
    "Au-100-Au-OOH": f"{os.environ['GITHUB_WORKSPACE']}/Au_100_results/Au-100-Au-OOH",
    "Au-100-Au-O": f"{os.environ['GITHUB_WORKSPACE']}/Au_100_results/Au-100-Au-O",
    # Au - 110 # clean termination - bridge
    "Au_110 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/Au_110_results/Au_110_bulk",
    "Au_110 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/Au_110_results/Au_110_slab",
    "Au-110-OH": f"{os.environ['GITHUB_WORKSPACE']}/Au_110_results/Au_110_OH_1",
    "Au-110-O": f"{os.environ['GITHUB_WORKSPACE']}/Au_110_results/Au_110_O_1",
    "Au-110-Au-reference": f"{os.environ['GITHUB_WORKSPACE']}/Au_110_results/Au-110-Au-reference",
    "Au-110-Au-OOH": f"{os.environ['GITHUB_WORKSPACE']}/Au_110_results/Au-110-Au-OOH_4",
    "Au-110-Au-OH": f"{os.environ['GITHUB_WORKSPACE']}/Au_110_results/Au-110-Au-OH_0",
    "Au-110-Au-O": f"{os.environ['GITHUB_WORKSPACE']}/Au_110_results/Au-110-Au-O_0",
    # Au - 111 # clean termination - bridge
    "Au_111 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/Au_111_results/Au_111_bulk",
    "Au_111 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/Au_111_results/Au_111_slab",
    "Au-111-OH": f"{os.environ['GITHUB_WORKSPACE']}/Au_111_results/Au_111_OH_1",
    "Au-111-O": f"{os.environ['GITHUB_WORKSPACE']}/Au_111_results/Au_111_O_1",
    "Au-111-Au-reference": f"{os.environ['GITHUB_WORKSPACE']}/Au_111_results/Au-111-Au-reference",
    "Au-111-Au-OOH": f"{os.environ['GITHUB_WORKSPACE']}/Au_111_results/Au-111-Au-OOH_2",
    "Au-111-Au-OH": f"{os.environ['GITHUB_WORKSPACE']}/Au_111_results/Au-111-Au-OH_7",
    "Au-111-Au-O": f"{os.environ['GITHUB_WORKSPACE']}/Au_111_results/Au-111-Au-O_0",
    # Ag - 100 # *OH termination - bridge
    "Ag_100 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/Ag_100_results/Ag_100_bulk",
    "Ag_100 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/Ag_100_results/Ag_100_slab",
    "Ag-100-OH": f"{os.environ['GITHUB_WORKSPACE']}/Ag_100_results/Ag_100_OH",
    "Ag-100-O": f"{os.environ['GITHUB_WORKSPACE']}/Ag_100_results/Ag_100_O",
    "Ag-100-Ag-reference": f"{os.environ['GITHUB_WORKSPACE']}/Ag_100_results/Ag-100-Ag-reference",
    "Ag-100-Ag-OOH": f"{os.environ['GITHUB_WORKSPACE']}/Ag_100_results/Ag-100-Ag-OOH",
    "Ag-100-Ag-O": f"{os.environ['GITHUB_WORKSPACE']}/Ag_100_results/Ag-100-Ag-O",
    # Ag - 110 # *OH termination - bridge
    "Ag_110 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/Ag_110_results/Ag_110_bulk",
    "Ag_110 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/Ag_110_results/Ag_110_slab",
    "Ag-110-OH": f"{os.environ['GITHUB_WORKSPACE']}/Ag_110_results/Ag_110_OH",
    "Ag-110-O": f"{os.environ['GITHUB_WORKSPACE']}/Ag_110_results/Ag_110_O",
    "Ag-110-Ag-reference": f"{os.environ['GITHUB_WORKSPACE']}/Ag_110_results/Ag-110-Ag-reference",
    "Ag-110-Ag-OOH": f"{os.environ['GITHUB_WORKSPACE']}/Ag_110_results/Ag-110-Ag-OOH",
    "Ag-110-Ag-O": f"{os.environ['GITHUB_WORKSPACE']}/Ag_110_results/Ag-110-Ag-O",
    # Ag - 111 # clean termination - bridge
    "Ag_111 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/Ag_111_results/Ag_111_bulk",
    "Ag_111 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/Ag_111_results/Ag_111_slab",
    "Ag-111-OH": f"{os.environ['GITHUB_WORKSPACE']}/Ag_111_results/Ag_111_OH",
    "Ag-111-O": f"{os.environ['GITHUB_WORKSPACE']}/Ag_111_results/Ag_111_O",
    "Ag-111-Ag-reference": f"{os.environ['GITHUB_WORKSPACE']}/Ag_111_results/Ag-111-Ag-reference",
    "Ag-111-Ag-OOH": f"{os.environ['GITHUB_WORKSPACE']}/Ag_111_results/Ag-111-Ag-OOH",
    "Ag-111-Ag-OH": f"{os.environ['GITHUB_WORKSPACE']}/Ag_111_results/Ag-111-Ag-OH",
    "Ag-111-Ag-O": f"{os.environ['GITHUB_WORKSPACE']}/Ag_111_results/Ag-111-Ag-O",
    # Ir - 110 # *O termination - bridge
    "Ir_110 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/Ir_110_results/Ir_110_bulk",
    "Ir_110 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/Ir_110_results/Ir_110_slab",
    "Ir-110-OH": f"{os.environ['GITHUB_WORKSPACE']}/Ir_110_results/Ir_110_OH",
    "Ir-110-O": f"{os.environ['GITHUB_WORKSPACE']}/Ir_110_results/Ir_110_O",
    "Ir-110-Ir-reference": f"{os.environ['GITHUB_WORKSPACE']}/Ir_110_results/Ir-110-Ir-reference",
    "Ir-110-Ir-OOH": f"{os.environ['GITHUB_WORKSPACE']}/Ir_110_results/Ir-110-Ir-OOH",
    "Ir-110-Ir-OH": f"{os.environ['GITHUB_WORKSPACE']}/Ir_110_results/Ir-110-Ir-OH",
    # Ir - 111 # *O termination - bridge
    "Ir_111 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/Ir_111_results/Ir_111_bulk",
    "Ir_111 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/Ir_111_results/Ir_111_slab",
    "Ir-111-OH": f"{os.environ['GITHUB_WORKSPACE']}/Ir_111_results/Ir_111_OH",
    "Ir-111-O": f"{os.environ['GITHUB_WORKSPACE']}/Ir_111_results/Ir_111_O",
    "Ir-111-Ir-reference": f"{os.environ['GITHUB_WORKSPACE']}/Ir_111_results/Ir-111-Ir-reference",
    "Ir-111-Ir-OOH": f"{os.environ['GITHUB_WORKSPACE']}/Ir_111_results/Ir-111-Ir-OOH",
    "Ir-111-Ir-OH": f"{os.environ['GITHUB_WORKSPACE']}/Ir_111_results/Ir-111-Ir-OH",
    # Ir - 100 # *O termination - bridge
    "Ir_100 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/Ir_100_results/Ir_100_bulk",
    "Ir_100 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/Ir_100_results/Ir_100_slab",
    "Ir-100-OH": f"{os.environ['GITHUB_WORKSPACE']}/Ir_100_results/Ir_100_OH",
    "Ir-100-O": f"{os.environ['GITHUB_WORKSPACE']}/Ir_100_results/Ir_100_O",
    "Ir-100-Ir-reference": f"{os.environ['GITHUB_WORKSPACE']}/Ir_100_results/Ir-100-Ir-reference",
    "Ir-100-Ir-OOH": f"{os.environ['GITHUB_WORKSPACE']}/Ir_100_results/Ir-100-Ir-OOH",
    "Ir-100-Ir-OH": f"{os.environ['GITHUB_WORKSPACE']}/Ir_100_results/Ir-100-Ir-OH",
    # Pd - 110 # *OH termination - bridge
    "Pd_110 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/Pd_110_results/Pd_110_bulk",
    "Pd_110 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/Pd_110_results/Pd_110_slab",
    "Pd-110-OH": f"{os.environ['GITHUB_WORKSPACE']}/Pd_110_results/Pd_110_OH",
    "Pd-110-O": f"{os.environ['GITHUB_WORKSPACE']}/Pd_110_results/Pd_110_O",
    "Pd-110-Pd-reference": f"{os.environ['GITHUB_WORKSPACE']}/Pd_110_results/Pd-110-Pd-reference",
    "Pd-110-Pd-OOH": f"{os.environ['GITHUB_WORKSPACE']}/Pd_110_results/Pd-110-Pd-OOH",
    "Pd-110-Pd-O": f"{os.environ['GITHUB_WORKSPACE']}/Pd_110_results/Pd-110-Pd-O",
    # Pd - 100 # *O termination - bridge
    "Pd_100 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/Pd_100_results/Pd_100_bulk",
    "Pd_100 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/Pd_100_results/Pd_100_slab",
    "Pd-100-OH": f"{os.environ['GITHUB_WORKSPACE']}/Pd_100_results/Pd_100_OH_1",
    "Pd-100-O": f"{os.environ['GITHUB_WORKSPACE']}/Pd_100_results/Pd_100_O_1",
    "Pd-100-Pd-reference": f"{os.environ['GITHUB_WORKSPACE']}/Pd_100_results/Pd-100-Pd-reference",
    "Pd-100-Pd-OOH": f"{os.environ['GITHUB_WORKSPACE']}/Pd_100_results/Pd-100-Pd-OOH_4",
    "Pd-100-Pd-OH": f"{os.environ['GITHUB_WORKSPACE']}/Pd_100_results/Pd-100-Pd-OH_3",
    # Pd - 111 # clean termination - bridge
    "Pd_111 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/Pd_111_results/Pd_111_bulk",
    "Pd_111 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/Pd_111_results/Pd_111_slab",
    "Pd-111-OH": f"{os.environ['GITHUB_WORKSPACE']}/Pd_111_results/Pd_111_OH",
    "Pd-111-O": f"{os.environ['GITHUB_WORKSPACE']}/Pd_111_results/Pd_111_O",
    "Pd-111-Pd-reference": f"{os.environ['GITHUB_WORKSPACE']}/Pd_111_results/Pd-111-Pd-reference",
    "Pd-111-Pd-OOH": f"{os.environ['GITHUB_WORKSPACE']}/Pd_111_results/Pd-111-Pd-OOH",
    "Pd-111-Pd-OH": f"{os.environ['GITHUB_WORKSPACE']}/Pd_111_results/Pd-111-Pd-OH",
    "Pd-111-Pd-O": f"{os.environ['GITHUB_WORKSPACE']}/Pd_111_results/Pd-111-Pd-O",
    # Ru - 110 # *O termination - bridge
    "Ru_110 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/Ru_110_results/Ru_110_bulk",
    "Ru_110 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/Ru_110_results/Ru_110_slab",
    "Ru-110-OH": f"{os.environ['GITHUB_WORKSPACE']}/Ru_110_results/Ru_110_OH",
    "Ru-110-O": f"{os.environ['GITHUB_WORKSPACE']}/Ru_110_results/Ru_110_O",
    "Ru-110-Ru-reference": f"{os.environ['GITHUB_WORKSPACE']}/Ru_110_results/Ru-110-Ru-reference",
    "Ru-110-Ru-OOH": f"{os.environ['GITHUB_WORKSPACE']}/Ru_110_results/Ru-110-Ru-OOH",
    "Ru-110-Ru-OH": f"{os.environ['GITHUB_WORKSPACE']}/Ru_110_results/Ru-110-Ru-OH",
    # Ru - 111 # *O termination - bridge
    "Ru_111 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/Ru_111_results/Ru_111_bulk",
    "Ru_111 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/Ru_111_results/Ru_111_slab",
    "Ru-111-OH": f"{os.environ['GITHUB_WORKSPACE']}/Ru_111_results/Ru_111_OH",
    "Ru-111-O": f"{os.environ['GITHUB_WORKSPACE']}/Ru_111_results/Ru_111_O",
    "Ru-111-Ru-reference": f"{os.environ['GITHUB_WORKSPACE']}/Ru_111_results/Ru-111-Ru-reference",
    "Ru-111-Ru-OOH": f"{os.environ['GITHUB_WORKSPACE']}/Ru_111_results/Ru-111-Ru-OOH",
    "Ru-111-Ru-OH": f"{os.environ['GITHUB_WORKSPACE']}/Ru_111_results/Ru-111-Ru-OH",
    # Ru - 100 # *O termination - bridge
    "Ru_100 bulk optimization": f"{os.environ['GITHUB_WORKSPACE']}/Ru_100_results/Ru_100_bulk",
    "Ru_100 slab optimization": f"{os.environ['GITHUB_WORKSPACE']}/Ru_100_results/Ru_100_slab",
    "Ru-100-OH": f"{os.environ['GITHUB_WORKSPACE']}/Ru_100_results/Ru_100_OH",
    "Ru-100-O": f"{os.environ['GITHUB_WORKSPACE']}/Ru_100_results/Ru_100_O",
    "Ru-100-Ru-reference": f"{os.environ['GITHUB_WORKSPACE']}/Ru_100_results/Ru-100-Ru-reference",
    "Ru-100-Ru-OOH": f"{os.environ['GITHUB_WORKSPACE']}/Ru_100_results/Ru-100-Ru-OOH",
    "Ru-100-Ru-OH": f"{os.environ['GITHUB_WORKSPACE']}/Ru_100_results/Ru-100-Ru-OH",
}


def Bulk_FW(
    bulk,
    name="",
    vasp_input_set=None,
    parents=None,
    wall_time=43200,
    vasp_cmd=VASP_CMD,
    db_file=DB_FILE,
    run_fake=False,
):
    """
    Function to generate a bulk firework. Returns an OptimizeFW for the specified slab.

    Args:
        bulk              (Struct Object)   : Structure corresponding to the slab to be calculated.
        name              (string)          : name of firework
        parents           (default: None)   : parent FWs
        add_slab_metadata (default: True)   : Whether to add slab metadata to task doc.
        wall_time         (default: 43200) : 2 days in seconds
        vasp_cmd                            : vasp_comand
        db_file                             : Path to the dabase file

    Returns:
        Firework correspoding to bulk calculation.
    """
    import uuid

    # Generate a unique ID for Bulk_FW
    fw_bulk_uuid = uuid.uuid4()

    # DFT Method
    if not vasp_input_set:
        vasp_input_set = MOSurfaceSet(bulk, bulk=True)

    # FW
    fw = OptimizeFW(
        name=name,
        structure=bulk,
        max_force_threshold=None,
        vasp_input_set=vasp_input_set,
        vasp_cmd=vasp_cmd,
        db_file=db_file,
        parents=parents,
        job_type="normal",
        spec={
            "counter": 0,
            "_add_launchpad_and_fw_id": True,
            "uuid_lineage": [],
            "_pass_job_info": True,
            "uuid": fw_bulk_uuid,
            "wall_time": wall_time,
            "max_tries": 10,
            "name": name,
            "is_bulk": True,
        },
    )
    if run_fake:
        assert (
            "RuO2" in name
            or "IrO2" in name
            or "TiRuO4" in name
            or "Ti9Cr11(RuO4)20" in name
            or "TiCr(RuO4)2" in name
            or "Co" in name
            or "Ti" in name
            or "Sb" in name
            or "Au" in name
            or "Pt" in name
            or "Ag" in name
            or "Pd" in name
            or "Ir" in name
            or "Ru" in name
        )  # Hardcoded to RuO2,IrO2  inputs/outputs
        # Replace the RunVaspCustodian Firetask with RunVaspFake
        fake_directory = ref_dirs[name]
        fw.tasks[1] = RunVaspFake(ref_dir=fake_directory, check_potcar=False)
    else:
        # This is for submitting on Perlmutter, where there is an issue between custodian and the compiled vasp version
        # fw.tasks[1] = RunVaspDirect(
        #    vasp_cmd=vasp_cmd
        # )  # We run vasp without custodian (RAW)

        # Switch-off GzipDir for WAVECAR transferring
        fw.tasks[1].update({"gzip_output": False})
        # Switch-on WalltimeHandler in RunVaspCustodian
        if wall_time is not None:
            fw.tasks[1].update({"wall_time": 43200})

    # Append Continue-optimizeFW for wall-time handling and use for uuid message
    # passing
    fw.tasks.append(
        ContinueOptimizeFW(is_bulk=True, counter=0, db_file=db_file, vasp_cmd=vasp_cmd)
    )

    # Add bulk_uuid through VaspToDb
    fw.tasks[3]["additional_fields"].update({"uuid": fw_bulk_uuid})
    fw.tasks[3].update(
        {"defuse_unsuccessful": False}
    )  # Continue with the workflow in the event an SCF has not converged

    return fw


def Slab_FW(
    slab,
    name="",
    parents=None,
    vasp_input_set=None,
    add_slab_metadata=True,
    wall_time=43200,
    vasp_cmd=VASP_CMD,
    db_file=DB_FILE,
    run_fake=False,
):
    """
    Function to generate a slab firework. Returns an OptimizeFW for the specified slab.

    Args:
        slab              (Slab Object)     : Slab corresponding to the slab to be calculated.
        name              (string)          : name of firework
        parents           (default: None)   : parent FWs
        add_slab_metadata (default: True)   : Whether to add slab metadata to task doc.
        wall_time         (default: 43200) : 2 days in seconds
        vasp_cmd                            : vasp_comand
        db_file                             : Path to the dabase file

    Returns:
        Firework correspoding to slab calculation.
    """
    import uuid

    # Generate a unique ID for Slab_FW
    fw_slab_uuid = uuid.uuid4()

    # DFT Method
    if not vasp_input_set:
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
        spec={
            "counter": 0,
            "_add_launchpad_and_fw_id": True,
            "uuid_lineage": [],
            "_pass_job_info": True,
            "uuid": fw_slab_uuid,
            "wall_time": wall_time,
            "max_tries": 10,
            "name": name,
            "is_bulk": False,
        },
    )
    if run_fake:
        assert (
            "RuO2" in name
            or "IrO2" in name
            or "TiRuO4" in name
            or "Ti9Cr11(RuO4)20" in name
            or "TiCr(RuO4)2" in name
            or "Co" in name
            or "Ti" in name
            or "Sb" in name
            or "Au" in name
            or "Pt" in name
            or "Ag" in name
            or "Pd" in name
            or "Ir" in name
            or "Ru" in name
        )  # Hardcoded to RuO2,IrO2  inputs/outputs
        # Replace the RunVaspCustodian Firetask with RunVaspFake
        fake_directory = ref_dirs[name]
        fw.tasks[1] = RunVaspFake(ref_dir=fake_directory, check_potcar=False)
    else:
        # fw.tasks[1] = RunVaspDirect(vasp_cmd=vasp_cmd)
        # Switch-off GzipDir for WAVECAR transferring
        fw.tasks[1].update({"gzip_output": False})
        # Switch-on WalltimeHandler in RunVaspCustodian
        if wall_time is not None:
            fw.tasks[1].update({"wall_time": wall_time})

    # Append Continue-optimizeFW for wall-time handling
    fw.tasks.append(
        ContinueOptimizeFW(is_bulk=False, counter=0, db_file=db_file, vasp_cmd=vasp_cmd)
    )

    # Add slab_uuid through VaspToDb
    fw.tasks[3]["additional_fields"].update({"uuid": fw_slab_uuid})
    fw.tasks[3].update(
        {"defuse_unsuccessful": False}
    )  # Continue with the workflow in the event an SCF has not converged

    # Add slab metadata
    if add_slab_metadata:
        parent_structure_metadata = get_meta_from_structure(slab.oriented_unit_cell)
        fw.tasks[3]["additional_fields"].update(
            {
                "slab": slab,
                "parent_structure": slab.oriented_unit_cell,
                "parent_structure_metadata": parent_structure_metadata,
            }
        )

    return fw


def AdsSlab_FW(
    slab,
    name="",
    oriented_uuid="",
    slab_uuid="",
    ads_slab_uuid="",
    is_adslab=True,
    parents=None,
    vasp_input_set=None,
    add_slab_metadata=True,
    wall_time=43200,
    vasp_cmd=VASP_CMD,
    db_file=DB_FILE,
    run_fake=False,
):
    """
    Function to generate a ads_slab firework. Returns an OptimizeFW for the specified slab.

    Args:
        slab              (Slab Object)     : Slab corresponding to the slab to be calculated.
        name              (string)          : name of firework
        parents           (default: None)   : parent FWs
        add_slab_metadata (default: True)   : Whether to add slab metadata to task doc.
        wall_time         (default: 43200) : 2 days in seconds
        vasp_cmd                            : vasp_comand
        db_file                             : Path to the dabase file

    Returns:
        Firework correspoding to slab calculation.
    """

    # DFT Method
    if not vasp_input_set:
        vasp_input_set = MOSurfaceSet(slab, bulk=False)
    # breakpoint()

    # FW
    fw = OptimizeFW(
        name=name + "_gpu",
        structure=slab,
        max_force_threshold=None,
        vasp_input_set=vasp_input_set,
        vasp_cmd=vasp_cmd,
        db_file=db_file,
        parents=parents,
        job_type="normal",
        spec={
            "counter": 0,
            "_add_launchpad_and_fw_id": True,
            "_pass_job_info": True,
            "uuid": ads_slab_uuid,
            "uuid_lineage": [],
            "wall_time": wall_time,
            "name": name,
            "max_tries": 10,
            "is_bulk": False,
            "is_adslab": is_adslab,
            "oriented_uuid": oriented_uuid,  # adslab FW should get terminal node ids
            "slab_uuid": slab_uuid,
            "is_bulk": False,
        },
    )
    if run_fake and "-Ru-" not in name and "-Co-" not in name and "-Ti-" not in name:
        assert (
            "RuO2" in name
            or "IrO2" in name
            or "TiRuO4" in name
            or "Ti9Cr11(RuO4)20" in name
            or "Co" in name
            or "Ti" in name
            or "Pd" in name
            or "Au" in name
            or "Ag" in name
            or "Pt" in name
            or "Ir" in name
            or "Ru" in name
        )  # Hardcoded to RuO2,IrO2  inputs/outputs
        # Replace the RunVaspCustodian Firetask with RunVaspFake
        # breakpoint()
        fake_directory = ref_dirs[name.split("_")[0]]
        fw.tasks[1] = RunVaspFake(ref_dir=fake_directory, check_potcar=False)
    else:
        fw.tasks[1] = RunVaspCustodian(vasp_cmd=vasp_cmd)
        # Switch-off GzipDir for WAVECAR transferring
        # fw.tasks[1] = RunVaspDirect(vasp_cmd=vasp_cmd)
        fw.tasks[1].update({"gzip_output": False})
        # Switch-on WalltimeHandler in RunVaspCustodian
        if wall_time is not None:
            fw.tasks[1].update({"wall_time": wall_time})

    # Append Continue-optimizeFW for wall-time handling
    fw.tasks.append(
        ContinueOptimizeFW(is_bulk=False, counter=0, db_file=db_file, vasp_cmd=vasp_cmd)
    )

    # Add slab_uuid through VaspToDb
    fw.tasks[3]["additional_fields"].update({"uuid": ads_slab_uuid})
    fw.tasks[3].update(
        {"defuse_unsuccessful": False}
    )  # Continue with the workflow in the event an SCF has not converged

    # Add slab metadata
    if add_slab_metadata:
        parent_structure_metadata = get_meta_from_structure(slab.oriented_unit_cell)
        fw.tasks[3]["additional_fields"].update(
            {
                "slab": slab,
                "parent_structure": slab.oriented_unit_cell,
                "parent_structure_metadata": parent_structure_metadata,
            }
        )

    return fw
