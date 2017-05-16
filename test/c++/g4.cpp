#include <triqs/operators/many_body_operator.hpp>
#include <triqs/hilbert_space/fundamental_operator_set.hpp>
#include <triqs/gfs.hpp>
#include <triqs/test_tools/gfs.hpp>

#include "solver_core.hpp"

using namespace cthyb;
using triqs::operators::c;
using triqs::operators::c_dag;
using triqs::operators::n;
using namespace triqs::gfs;
using indices_type = triqs::operators::indices_t;

TEST(CtHyb, g4_measurments) {

  std::cout << "Welcome to the CTHYB solver\n";

  // Initialize mpi
  int rank = triqs::mpi::communicator().rank();

  // Parameters
  double beta = 2.0;
  double U    = 0.0;
  double mu   = 2.0;

  double V1       = 2.0;
  double V2       = 5.0;
  double epsilon1 = 0.0;
  double epsilon2 = 4.0;

  // GF structure
  enum spin { up, down };
  std::map<std::string, indices_type> gf_struct{{"up", {0}}, {"down", {0}}};
  auto n_up   = n("up", 0);
  auto n_down = n("down", 0);

  // define operators
  auto H = U * n_up * n_down;

  // Construct CTQMC solver
  solver_core solver(beta, gf_struct, 1025, 2500);

  // Set G0
  triqs::clef::placeholder<0> om_;
  auto g0_iw = gf<imfreq>{{beta, Fermion}, {1, 1}};
  g0_iw(om_) << om_ + mu - V1 * V1 / (om_ - epsilon1) - V2 * V2 / (om_ - epsilon2);
  for (int bl = 0; bl < 2; ++bl) solver.G0_iw()[bl] = triqs::gfs::inverse(g0_iw);

  // Solve parameters
  int n_cycles      = 500;
  auto p            = solve_parameters_t(H, n_cycles);
  p.random_name     = "";
  p.random_seed     = 123 * rank + 567;
  p.max_time        = -1;
  p.length_cycle    = 100;
  p.n_warmup_cycles = 1000;
  p.move_double     = false;

  p.measure_g2_tau=true;
  p.measure_g2_n_tau=20;
  //p.measure_g2_n_iwn=16;
  p.measure_g2_inu_fermionic=true;  
    
  // Solve!
  solver.solve(p);

  std::cout << "--> solver done, now writing and reading the results.\n";
  
  // Save the results
  std::string filename = "g4";

  if (rank == 0) {
    triqs::h5::file G_file(filename + ".out.h5", 'w');
    h5_write(G_file, "G_tau", solver.G_tau());
    h5_write(G_file, "G2_tau", solver.G2_tau());
    //h5_write(G_file, "G2_inu", solver.G2_inu());
  }

  if (rank == 0) {
    triqs::h5::file G_file(filename + ".ref.h5", 'r');

    g_tau_t g_tau;
    h5_read(G_file, "G_tau", g_tau);
    for( auto block_idx : range(g_tau.size()) )
      EXPECT_GF_NEAR(g_tau[block_idx], solver.G_tau()[block_idx]);

    g4_tau_t g4_tau;
    h5_read(G_file, "G2_tau", g4_tau);
    for( auto bidx1 : range(g4_tau.size1()) )
      for( auto bidx2 : range(g4_tau.size2()) )
	EXPECT_GF_NEAR(g4_tau(bidx1, bidx2), solver.G2_tau()(bidx1, bidx2));
    
    /*
    gf<imtime_poly_cube, tensor_valued<4>> g2_poly;
    h5_read(G_file, "G2_poly_up_up", g2_poly);
    EXPECT_GF_NEAR(g2_poly, solver.G2_poly()(0, 0));

    gf<cartesian_product<imtime, imtime, imtime>, tensor_valued<4>> g2_poly_tau;
    h5_read(G_file, "G2_poly_tau_up_up", g2_poly_tau);
    EXPECT_GF_NEAR(g2_poly_tau, solver.G2_poly_tau()(0, 0));
    */

  }
}
MAKE_MAIN;