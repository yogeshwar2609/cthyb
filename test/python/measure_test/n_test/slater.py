import pytriqs.utility.mpi as mpi
from pytriqs.operators import *
from pytriqs.operators.util.op_struct import set_operator_structure
from pytriqs.operators.util.U_matrix import U_matrix
from pytriqs.operators.util.hamiltonians import h_int_slater
from pytriqs.operators.util.observables import N_op
from pytriqs.archive import HDFArchive
from pytriqs.applications.impurity_solvers.cthyb import *
from pytriqs.gf.local import *
from pytriqs.utility.comparison_tests import *

beta = 100.0
# H_loc parameters
L = 2 # angular momentum
U = 5.0
J = 0.1
F0 = U
F2 = J*(14.0/(1.0 + 0.63))
F4 = F2*0.63
half_bandwidth = 1.0
mu = 32.5  # 3 electrons in 5 bands

spin_names = ("up","down")
cubic_names = map(str,range(2*L+1))
U_mat = U_matrix(L, radial_integrals=[F0,F2,F4], basis="cubic")
off_diag = True

N = N_op(spin_names,cubic_names,off_diag)

# Parameters
p = {}
p["max_time"] = -1
p["random_name"] = ""
p["random_seed"] = 123 * mpi.rank + 567
p["length_cycle"] = 50
p["n_warmup_cycles"] = 5000
p["n_cycles"] = 50000
p["measure_g_l"] = True
p["move_double"] = False
if not off_diag: p["measure_density_matrix"] = True
p["use_norm_as_weight"] = True
p["measure_two_body_correlator"] = (N,False)

# Block structure of GF
gf_struct = set_operator_structure(spin_names,cubic_names,off_diag)

# Local Hamiltonian
H = h_int_slater(spin_names,cubic_names,U_mat,off_diag)

# Construct the solver
S = Solver(beta=beta, gf_struct=gf_struct, n_iw=1025, n_tau=100000)

# Set hybridization function
if off_diag:
    delta_w = GfImFreq(indices = cubic_names, beta=beta)
else: 
    delta_w = GfImFreq(indices = [0], beta=beta)
delta_w << (half_bandwidth/2.0)**2 * SemiCircular(half_bandwidth)
for name, g0 in S.G0_iw:
    g0 << inverse(iOmega_n + mu - delta_w)

S.solve(h_int=H, **p)

if mpi.is_master_node():
    print "GF density ", S.G_iw.density()
    print "GF total density ", S.G_iw.total_density()
    if not off_diag:
        static_observables = {'N' : N, 'unity' : Operator(1.0)}
        dm = S.density_matrix
        for oname in static_observables.keys():
            val = trace_rho_op(dm,static_observables[oname],S.h_loc_diagonalization)
            print oname, val
