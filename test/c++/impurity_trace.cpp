#include <triqs/test_tools/gfs.hpp>

#include <triqs/atom_diag/atom_diag.hpp>
#include <triqs/atom_diag/functions.hpp>
#include <triqs/atom_diag/gf.hpp>
#include <triqs/arrays/blas_lapack/dot.hpp>
#include <triqs/h5.hpp>

#include <triqs_cthyb/impurity_trace.hpp>

using namespace triqs::arrays;
using namespace triqs::hilbert_space;
using namespace triqs::atom_diag;
using namespace triqs::operators;

fundamental_operator_set make_fops(int n_orb) {
  fundamental_operator_set fops;
  for (int oidx : range(n_orb)) {
    fops.insert("up", oidx);
    fops.insert("dn", oidx);
  }
  return fops;
}

// 
template <typename OP>
OP make_hamiltonian(int n_orb, double mu, double U, double J) {

  auto orbs = range(n_orb);

  OP h;

  for (int o : orbs) h += -mu * (n("up", o) + n("dn", o));

  // Density-density interactions
  for (int o : orbs) h += U * n("up", o) * n("dn", o);

  for (int o1 : orbs) {
    for (int o2 : orbs) {
      if (o1 == o2) continue;
      h += (U - 2 * J) * n("up", o1) * n("dn", o2);
    }
  }

  for (int o1 : orbs)
    for (int o2 : orbs) {
      if (o2 >= o1) continue;
      h += (U - 3 * J) * n("up", o1) * n("up", o2);
      h += (U - 3 * J) * n("dn", o1) * n("dn", o2);
    }

  // spin-flip and pair-hopping
  for (int o1 : orbs) {
    for (int o2 : orbs) {
      if (o1 == o2) continue;
      h += -J * c_dag("up", o1) * c_dag("dn", o1) * c("up", o2) * c("dn", o2);
      h += -J * c_dag("up", o1) * c_dag("dn", o2) * c("up", o2) * c("dn", o1);
    }
  }
  
  return h;
}

TEST(impurity_trace, test_usage) {

  int n_orb = 3;
  auto fops = make_fops(n_orb);

  double U = 1.0;
  double J = 0.2;
  double mu = 0.5*U;
  
  auto H = make_hamiltonian<many_body_operator_real>(n_orb, mu, U, J);
  auto ad = triqs::atom_diag::atom_diag<false>(H, fops);
  std::cout << "Found " << ad.n_subspaces() << " subspaces." << std::endl;

  double beta = 1.0;
  triqs_cthyb::impurity_trace imp_trace(beta, ad, nullptr); 
  
}

MAKE_MAIN;
