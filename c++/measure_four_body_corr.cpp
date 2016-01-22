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
#include "./measure_four_body_corr.hpp"

//#define DEBUG

namespace cthyb {
using namespace triqs::gfs;

measure_four_body_corr::measure_four_body_corr(qmc_data const& data, gf_view<imtime, scalar_valued> correlator,
                                               fundamental_operator_set const& fops, many_body_operator const& A, bool anticommute)
   : data(data), correlator(correlator), anticommute(anticommute) {
 z = 0;
 correlator() = 0.0;
 auto fops_size = fops.size();

 //FIXME generalize to take two different operators too
 // Extract the non-zero monomials for a quadratic operator, op = \sum_ab coef_ab c^+_a c_b
 // and store in a table linking to operator indices: coef = coefficients[cdag_index,c_index,cdag_index,c_index]
 // Note: it is important that the order is c^+ c, and not c c^+ !
 coefficients.resize(make_shape(fops_size, fops_size, fops_size, fops_size));
 coefficients_one_pair.resize(make_shape(fops_size, fops_size));
 coefficients() = 0.0;
 coefficients_one_pair() = 0.0;
 std::vector<std::vector<int>> linear_index;
 std::vector<dcomplex> coef;
 auto _ = range{};
 for (auto const& x : A) { // all terms in the operator A
  if (x.monomial.size() != 2) TRIQS_RUNTIME_ERROR << "measure_four_body_corr: only valid for quadratic operators";
  std::vector<int> temp;
  for (auto const& y : x.monomial) temp.push_back(fops[y.indices]); // all operators in the term
  linear_index.push_back(temp);
  coef.push_back(dcomplex(x.coef));
 }
 for (int i : range(linear_index.size())) {
  for (int j : range(linear_index.size())) {
   coefficients(linear_index[i][0], linear_index[i][1], linear_index[j][0], linear_index[j][1]) = coef[i] * coef[j];
  }
  coefficients_one_pair(linear_index[i][0], linear_index[i][1]) =
      max_element(abs(coefficients(linear_index[i][0], linear_index[i][1], _, _)));
 }
std::cout << coefficients << std::endl;
}
// --------------------

void measure_four_body_corr::accumulate(mc_sign_type s) {

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
 if (tree_size < 4) return;

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
// auto w_rw = data.imp_trace.compute();

 //FIXME
// // Which blocks contribute to the trace?
// auto blocks = data.imp_trace.contributing_blocks;
std::vector<int> blocks;
blocks.clear();
blocks.reserve(data.imp_trace.n_blocks);
for (auto bl : range(data.imp_trace.n_blocks)) {
 if (data.imp_trace.compute_block_table(data.imp_trace.tree.get_root(),bl) == bl) blocks.push_back(bl);
}

#ifdef DEBUG
 // Check partial linear matrices match root cache matrix
 data.imp_trace.check_ML_MM_MR(PRINT_DEBUG);
#endif

 // Loop on the first pair of c, c^+
 for (int i = 0; i < flat_config.size() - 3; ++i) {

  // n1, n2 are the two first operators
  auto n1 = flat_config[i];
  auto n2 = flat_config[i + 1];
  auto tau12 = n2->key; // Always shift first (left) op to the second (right) op in pair
  if (anticommute) {
   if (n1->op.dagger == n2->op.dagger) continue; // restrict to either cdag * c or c * cdag
  } else {
   if ((!n1->op.dagger) or (n2->op.dagger)) continue; // restrict to cdag * c only
  }
  auto ind1 = op_index_in_det[i];
  auto ind2 = op_index_in_det[i + 1];

  // Ensure that n1 is dagger, n2 not
  auto swapped12 = (n2->op.dagger);
  if (swapped12) {
   std::swap(n1, n2);
   std::swap(ind1, ind2);
  }

  // Does this pair contribute?
  if (std::abs(coefficients_one_pair(n1->op.linear_index, n2->op.linear_index)) < coef_threshold) continue;

  // Find the second pair of c, c^+
  for (int j = i + 2; j < flat_config.size() - 1; ++j) {

   // n3, n4 are the two other operators
   auto n3 = flat_config[j];
   auto n4 = flat_config[j + 1];
   auto tau34 = n4->key;
   if (anticommute) {
    if (n3->op.dagger == n4->op.dagger) continue;
   } else {
    if ((!n3->op.dagger) and (n4->op.dagger)) continue;
   }
   auto ind3 = op_index_in_det[j];
   auto ind4 = op_index_in_det[j + 1];

   // Ensure that n3 is dagger, n4 not
   auto swapped34 = (n4->op.dagger);
   if (swapped34) {
    std::swap(n3, n4);
    std::swap(ind3, ind4);
   }

   // Coefficient for the accumulation
   auto coef = coefficients(n1->op.linear_index, n2->op.linear_index, n3->op.linear_index, n4->op.linear_index);
   if (std::abs(coef) < coef_threshold) continue; // Do these 2 pairs contribute?

   // Now measure!

   // --- Trace and the tree ---

#ifdef DEBUG
   //DEBUG Check that the trace is correct
   data.imp_trace.check_trace_from_ML_MM_MR(flat_config, i, j, PRINT_DEBUG);
#endif

   // Move the left operator of the pair to the right, could be either c or cdag, and
   // compute the trace and normalisation integral
   auto tr_int = data.imp_trace.compute_sliding_trace_integral(flat_config, i, j, blocks);
   auto tr_over_int = tr_int.first / tr_int.second;
   if (!std::isfinite(tr_over_int)) {
    if ((tr_int.first < 1.e-20) and (tr_int.second < 1.e-20)) continue; // FIXME what thresholds to use for 0/0 check?
    TRIQS_RUNTIME_ERROR << "tr_over_int not finite " << tr_int.first << " " << tr_int.second;
   }

   // --- Det ---

   // Properties corresponding to det
   // Could be in a block diagonal situation, so need to check block indices.

   // For indices abcd, compute M_ba M_dc - M_da M_bc (a,c are the dagger indices)
   double MM1 = 0.0, MM2 = 0.0;
   auto b1 = n1->op.block_index;
   auto b2 = n2->op.block_index;
   auto b3 = n3->op.block_index;
   auto b4 = n4->op.block_index;

   if ((b1 == b2) && (b3 == b4)) MM1 = data.dets[b1].inverse_matrix(ind2, ind1) * data.dets[b3].inverse_matrix(ind4, ind3);
   if ((b1 == b4) && (b3 == b2)) MM2 = data.dets[b1].inverse_matrix(ind4, ind1) * data.dets[b3].inverse_matrix(ind2, ind3);

   // --- Accumulate into correlator ---

   //if ((anticommute) and (swapped12 xor swapped34)) s = -s;
   //correlator[closest_mesh_pt(double(tau12 - tau34))] += coef * s * (MM1 - MM2) * tr_over_int;
   auto accum = coef * s * (MM1 - MM2) * tr_over_int;
   auto tau = double(tau12 - tau34);
   auto tau2 = data.config.beta() - tau;
   binned_taus << tau; //DEBUG
   binned_taus << tau2; //DEBUG
   correlator[closest_mesh_pt(tau)] += accum;
   correlator[closest_mesh_pt(tau2)] += accum;

  } // Second pair
 }  // First pair
}

// ---------------------------------------------

void measure_four_body_corr::collect_results(triqs::mpi::communicator const& c) {
 z = mpi_all_reduce(z, c);
 //FIXME correlator[0] *= 2;
 //FIXME correlator[correlator.mesh().size() - 1] *= 2;
 correlator = mpi_all_reduce(correlator, c);
 correlator = correlator / (z * data.config.beta() * correlator.mesh().delta());
}
}
