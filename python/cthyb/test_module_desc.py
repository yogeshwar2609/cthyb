from cpp2py.wrap_generator import *

# The module
module = module_(full_name = "test_module", doc = "The cthyb solver", app_name = "solver_core")

# Imports
import pytriqs.gf
import pytriqs.operators
import pytriqs.statistics.histograms
import pytriqs.atom_diag
import solver_core

module.generate_code()
