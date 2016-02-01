/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2014, P. Seth, I. Krivenko, M. Ferrero and O. Parcollet
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#include "./measure_two_body_corr.hpp"

//#define DEBUG

namespace cthyb {
using namespace triqs::gfs;

measure_two_body_corr::measure_two_body_corr(qmc_data const& data, fundamental_operator_set const& fops,
                                             many_body_operator const& A, bool anticommute)
   : data(data), anticommute(anticommute) {
 z = 0;
 dens = 0.0; // cdag c
 dens2 = 0.0; // c cdag
 auto fops_size = fops.size();

 // Extract the non-zero monomials for a quadratic operator, op = \sum_ab coef_ab c^+_a c_b
 // and store in a table linking to operator indices: coef_ab = coefficients[cdag_index,c_index]
 // Note: it is important that the order is c^+ c, and not c c^+ !
 coefficients.resize(make_shape(fops_size, fops_size));
 coefficients() = 0.0;
 for (auto const& x : A) { // all terms in the operator A
  if (x.monomial.size() != 2) TRIQS_RUNTIME_ERROR << "measure_two_body_corr: only valid for quadratic operators";
  std::vector<int> linear_index;
  for (auto const& y : x.monomial) linear_index.push_back(fops[y.indices]); // all operators in the term
  coefficients(linear_index[0], linear_index[1]) = dcomplex(x.coef);
 }
}
// --------------------

void measure_two_body_corr::accumulate(mc_sign_type s) {

#ifdef DEBUG
bool PRINT_DEBUG = false;
#endif

 s *= data.atomic_reweighting;
 z += s;

 static constexpr double coef_threshold = 1.e-13;
 auto& tree = data.imp_trace.tree;
 auto n_dets = data.dets.size();

 // Return if the configuration is too small to contribute
 auto tree_size = tree.size();
 if (tree_size < 2) return;

 // Create a flat configuration from the tree in order of decreasing time (i.e. in order of tree, beta -> 0)
 // Also calculate the position of each operator in the appropriate determinant right away
 std::vector<node> flat_config;
 flat_config.reserve(tree_size);
 std::vector<int> op_index_in_det, counters(2 * n_dets, 0); // operator order in counters is c ... c, cdag ... cdag
 op_index_in_det.reserve(tree_size);
 foreach (tree, [&](node n) {
  flat_config.push_back(n);
  op_index_in_det.push_back(counters[(n->op.dagger * n_dets) + n->op.block_index]++);
 });

 // Update the cache without the Yee trick
 auto w_rw = data.imp_trace.compute();
 auto true_tr = w_rw.first * w_rw.second; // tr = norm * tr/norm = w * rw

 //FIXME
 // Which blocks contribute to the trace?
 auto blocks = data.imp_trace.contributing_blocks;

#ifdef DEBUG
 // Check partial linear matrices match root cache matrix
 data.imp_trace.check_ML_MM_MR(PRINT_DEBUG);
#endif

 // Loop on the pair of c, c^+
 for (int i = 0; i < flat_config.size() - 1; ++i) {

  // n1, n2 are the two first operators
  auto n1 = flat_config[i];
  auto n2 = flat_config[i + 1];
//  if (anticommute) {
//   if (n1->op.dagger == n2->op.dagger) continue;
//  } else {
//   if ((!n1->op.dagger) and (n2->op.dagger)) continue;
//  }
  if (n1->op.dagger == n2->op.dagger) continue; // DEBUG
  auto ind1 = op_index_in_det[i];
  auto ind2 = op_index_in_det[i + 1];

  // Ensure that n1 is dagger, n2 not
  auto swapped = (n2->op.dagger);
  if (swapped) {
   std::swap(n1, n2);
   std::swap(ind1, ind2);
  }

  // Coefficient for the accumulation
  auto coef = coefficients(n1->op.linear_index, n2->op.linear_index);
  if (std::abs(coef) < coef_threshold) continue; // Does this pair contribute?

  // Now measure!

  // --- Trace and the tree ---

#ifdef DEBUG
  // Check that the trace is correct
  data.imp_trace.check_trace_from_ML_MR(flat_config, i, PRINT_DEBUG);
#endif

  // Move the left operator of the pair to the right, could be either c or cdag, and
  // compute the trace and normalisation integral
  auto tr_int = compute_sliding_trace_integral_one_pair(flat_config, i, blocks);
  auto tr_over_int = tr_int.first / tr_int.second;
  if (!std::isfinite(tr_over_int)) {
   if ((tr_int.first < 1.e-20) and (tr_int.second < 1.e-20)) continue; //FIXME what thresholds to use for 0/0 check?
   TRIQS_RUNTIME_ERROR << "tr_over_int not finite " << tr_int.first << " " << tr_int.second ;
  }

  // --- Det ---

  // Properties corresponding to det
  // Could be in a block diagonal situation, so need to check block indices.
  // The coefficients array needs to be constructed such that this is not repeated
  if (n1->op.block_index == n2->op.block_index) {
   // Compute M_ba (a is the dagger index)
   auto M = data.dets[n1->op.block_index].inverse_matrix(ind2, ind1);
   // --- Accumulate into correlator ---
   //if (swapped) s = -s;
   //dens -= coef * s * M * tr_over_int;
   auto temp = coef * s * M * tr_over_int; //DEBUG
   if (swapped) dens2 += temp; // DEBUG
   else dens -= temp; // DEBUG
  }
 }
}

// ---------------------------------------------

void measure_two_body_corr::collect_results(triqs::mpi::communicator const& c) {
 z = mpi_all_reduce(z, c);
 dens = mpi_all_reduce(dens, c);
 dens2 = mpi_all_reduce(dens2, c);
 dens = dens / (z * data.config.beta());
 dens2 = dens2 / (z * data.config.beta());
 if (c.rank() == 0) std::cout << "density  from the sliding method " << dens << std::endl; //DEBUG
 if (c.rank() == 0) std::cout << "<c cdag> from the sliding method " << dens2 << std::endl; //DEBUG
}
}
