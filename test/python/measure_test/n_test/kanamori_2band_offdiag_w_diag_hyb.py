import numpy as np
import pytriqs.utility.mpi as mpi
from pytriqs.gf.local import *
from pytriqs.operators import * 
from pytriqs.operators.util import * 
from pytriqs.archive import HDFArchive
from pytriqs.applications.impurity_solvers.cthyb import *
from pytriqs.utility.comparison_tests import *

# H_loc parameters
beta = 10.0
num_orbitals = 2
mu = 1.0
U = 2.0
J = 0.5

# Poles of delta
epsilon = 2.3

# Block structure of GF
spin_names = ('up','down')
orb_names = range(num_orbitals)
off_diag = True
gf_struct = set_operator_structure(spin_names,orb_names,off_diag)

# Construct solver
S = Solver(beta=beta, gf_struct=gf_struct, n_iw=1025, n_tau=2500)

# Hamiltonian
H = h_int_kanamori(spin_names,orb_names,
                   np.array([[0,U-3*J],[U-3*J,0]]),
                   np.array([[U,U-2*J],[U-2*J,U]]),
                   J,off_diag)

N = N_op(spin_names,orb_names,off_diag)
N_up = N_op(['up'],orb_names,off_diag)
N_down = N_op(['down'],orb_names,off_diag)

# Hybridization matrices
if off_diag: 
    V = 1.0 * np.eye(num_orbitals)
    delta_w = GfImFreq(indices = orb_names, beta=beta)
else:
    V = 1.0 * np.eye(1)
    delta_w = GfImFreq(indices = [0], beta=beta)

# Set hybridization function
delta_w << inverse(iOmega_n - epsilon) + inverse(iOmega_n + epsilon)
delta_w.from_L_G_R(V, delta_w, V)
S.G0_iw << inverse(iOmega_n + mu - delta_w)

# Parameters
p = {}
p["max_time"] = -1
p["random_name"] = ""
p["random_seed"] = 123 * mpi.rank + 567
p["length_cycle"] = 50
p["n_warmup_cycles"] = 5000
p["n_cycles"] = 50000#0
p["measure_g_l"] = True
p["move_double"] = True
p["use_norm_as_weight"] = True
if not off_diag: p["measure_density_matrix"] = True
p["measure_two_body_correlator"] = (N,False)
#p["measure_two_body_correlator"] = (N_up,False)
#p["measure_two_body_correlator"] = (N_down,False)
#p["measure_four_body_correlator"] = (S_op('z',spin_names,orb_names,True),True)

S.solve(h_int=H, **p)

print "GF density ", S.G_iw.density()
print "GF total density ", S.G_iw.total_density()

if mpi.is_master_node() and (not off_diag):
    static_observables = {'N' : N, 'N_up' : N_up, 'N_dn' : N_down, 'unity' : Operator(1.0)}
    dm = S.density_matrix
    for oname in static_observables.keys():
        val = trace_rho_op(dm,static_observables[oname],S.h_loc_diagonalization)
        print oname, val
