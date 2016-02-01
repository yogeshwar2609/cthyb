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

using trace_t = impurity_trace::trace_t;
using block_and_matrix = impurity_trace::block_and_matrix;

measure_four_body_corr::measure_four_body_corr(qmc_data const& data, gf_view<imtime, scalar_valued> correlator,
                                               fundamental_operator_set const& fops, many_body_operator const& A, bool anticommute)
   : data(data), correlator(correlator), anticommute(anticommute) {
 z = 0;
 correlator() = 0.0;
 tree = data.imp_trace.tree;
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
 auto w_rw = data.imp_trace.compute();

 //FIXME
// // Which blocks contribute to the trace?
 auto blocks = data.imp_trace.contributing_blocks;
//std::vector<int> blocks;
//blocks.clear();
//blocks.reserve(data.imp_trace.n_blocks);
//for (auto bl : range(data.imp_trace.n_blocks)) {
// if (data.imp_trace.compute_block_table(data.imp_trace.tree.get_root(),bl) == bl) blocks.push_back(bl);
//}

#ifdef DEBUG
 // Check partial linear matrices match root cache matrix
 data.imp_trace.check_ML_MM_MR(PRINT_DEBUG);
#endif

 auto noncyclic = true;//DEBUG

 // Loop on the first pair of c^+, c 
 // i+1 can go up to second to last config
 for (int i = 0; i < flat_config.size() - 2; ++i) {

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

  // Find the second pair of c^+, c
  // j can go all the way to the rightmost operator and j+1 then loops to leftmost operator
  for (int j = i + 2; j < flat_config.size(); ++j) {

   auto is_j_last_op = (j == flat_config.size() - 1);
   if ((i == 1) and (is_j_last_op)) continue; // Otherwise have a clash of c_i = c_j+1!
   if ((noncyclic) and (is_j_last_op)) continue; //FIXME// Otherwise have a clash of c_i = c_j+1!
   auto j_plus_one = is_j_last_op ? 0 : j + 1 ; // Cycle around?
   // n3, n4 are the two other operators
   auto n3 = flat_config[j];
   auto n4 = flat_config[j_plus_one];
   auto tau34 = n4->key;
   if (anticommute) {
    if (n3->op.dagger == n4->op.dagger) continue;
   } else {
    if ((!n3->op.dagger) and (n4->op.dagger)) continue;
   }
   auto ind3 = op_index_in_det[j];
   auto ind4 = op_index_in_det[j_plus_one];

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
   // DEBUG Check that the trace is correct
   data.imp_trace.check_trace_from_ML_MM_MR(flat_config, i, j, PRINT_DEBUG); // FIXME -- generalise for cyclic trace
#endif

   // Move the left operator of the pair to the right, could be either c or cdag, and
   // compute the trace and normalisation integral
   auto tr_int =compute_sliding_trace_integral(flat_config, i, j, blocks);
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

   // if ((anticommute) and (swapped12 xor swapped34)) s = -s;
   // correlator[closest_mesh_pt(double(tau12 - tau34))] += coef * s * (MM1 - MM2) * tr_over_int;
   auto accum = coef * s * (MM1 - MM2) * tr_over_int;
   auto tau = double(tau12 - tau34);
   auto tau_reverse = data.config.beta() - tau;
   binned_taus << tau; // DEBUG
   binned_taus << tau_reverse; // DEBUG
   correlator[closest_mesh_pt(tau)] += accum;
   correlator[closest_mesh_pt(tau_reverse)] += accum;

  } // Second pair
 }  // First pair
}

// ---------------------------------------------

void measure_four_body_corr::collect_results(triqs::mpi::communicator const& c) {
 z = mpi_all_reduce(z, c);
 // FIXME correlator[0] *= 2;
 // FIXME correlator[correlator.mesh().size() - 1] *= 2;
 correlator = mpi_all_reduce(correlator, c);
 correlator = correlator / (z * data.config.beta() * correlator.mesh().delta());
}

// #################### BASE FUNCTIONS ##############################

template <int N> double pow(double x);
template <> double pow<1>(double x) { return x; }
template <> double pow<2>(double x) { return x * x; }
template <> double pow<3>(double x) { return pow<2>(x) * x; }
template <> double pow<4>(double x) { return pow<2>(x) * pow<2>(x); }

template <int N, typename A> void unique_with_multiplicity(A& a) {
 for (int r = 0; r < N - 1; ++r) {
  if (std::abs(a[r + 1] - a[r]) < 1.e-10) {
   a[r + 1].second += a[r].second;
   a[r].second = 0;
  }
 }
}

// Compute [\int_tau1^\tau2 dtau e^-H_{b_f}(tau2 - tau) * op_{b_i->b_f}(tau) * e^-H_{b_i}(tau - tau1)]
block_and_matrix integral(int b_i, time_pt tau1, time_pt tau2, op_desc const& op) {
//FIXME if (b_i == -1) return {-1, {}};
 auto b_f = data.imp_trace.get_op_block_map(op, b_i);
//FIXME if (b_f == -1) return {-1, {}};
 auto M = data.imp_trace.get_op_block_matrix(op, b_i);
 double dtau = double(tau2 - tau1);
 auto dim1 = data.imp_trace.get_block_dim(b_i);
 auto dim2 = data.imp_trace.get_block_dim(b_f);
 for (int i = 0; i < dim2; ++i) {
  auto lamb2 = data.imp_trace.get_block_eigenval(b_f, i);
  for (int j = 0; j < dim1; ++j) {
   auto lamb1 = data.imp_trace.get_block_eigenval(b_i, j);
   auto rhs = ((std::abs(lamb1 - lamb2) > 1.e-10) ? (std::exp(-dtau * lamb1) - std::exp(-dtau * lamb2)) / (lamb2 - lamb1)
                                                  : std::exp(-dtau * lamb1) * dtau);
   M(i, j) *= rhs;
  }
 }
 return {b_f, std::move(M)};
}

// Precondition: only valid for non-structurally-zero blocks
block_and_matrix measure_four_body_corr::integral(int b_i, time_pt tau_i, time_pt tau_f, op_desc const& op1, op_desc const& op2) {

 auto b0 = b_i;
 auto b1 = data.imp_trace.get_op_block_map(op1, b0);
 auto M1 = data.imp_trace.get_op_block_matrix(op1, b0);
 auto b2 = data.imp_trace.get_op_block_map(op2, b1);
 auto M2 = data.imp_trace.get_op_block_matrix(op2, b1);
 auto b_f = b2;
 auto dim0 = data.imp_trace.get_block_dim(b0);
 auto dim1 = data.imp_trace.get_block_dim(b1);
 auto dim2 = data.imp_trace.get_block_dim(b2);
 double dtau = double(tau_f - tau_i);
 auto dtau2 = pow<2>(dtau);
 for (int i2 = 0; i2 < dim2; ++i2) {
  auto lamb2 = data.imp_trace.get_block_eigenval(b2, i2);
  for (int i1 = 0; i1 < dim1; ++i1) {
   auto lamb1 = data.imp_trace.get_block_eigenval(b1, i1);
   for (int i0 = 0; i0 < dim0; ++i0) {
    auto lamb0 = data.imp_trace.get_block_eigenval(b0, i0);
    auto rhs =
        compute_evolution_integral(lamb0, lamb1, lamb2); // FIXME optimise by just changing one variable in list and passing list
    M2(i2, i0) *= rhs * M1(i1, i0) * dtau2;
   }
  }
 }
 return {b_f, std::move(M2)};
}

// Precondition: only valid for non-structurally-zero blocks
block_and_matrix measure_four_body_corr::integral(int b_i, time_pt tau_i, time_pt tau_f, op_desc const& op1, op_desc const& op2,
                                          op_desc const& op3, op_desc const& op4) {

 auto b0 = b_i;
 auto b1 = data.imp_trace.get_op_block_map(op1, b0);
 auto M1 = data.imp_trace.get_op_block_matrix(op1, b0);
 auto b2 = data.imp_trace.get_op_block_map(op2, b1);
 auto M2 = data.imp_trace.get_op_block_matrix(op2, b1);
 auto b3 = data.imp_trace.get_op_block_map(op3, b2);
 auto M3 = data.imp_trace.get_op_block_matrix(op3, b2);
 auto b4 = data.imp_trace.get_op_block_map(op4, b3);
 auto M4 = data.imp_trace.get_op_block_matrix(op4, b3);
 auto b_f = b4;
 auto dim0 = data.imp_trace.get_block_dim(b0);
 auto dim1 = data.imp_trace.get_block_dim(b1);
 auto dim2 = data.imp_trace.get_block_dim(b2);
 auto dim3 = data.imp_trace.get_block_dim(b3);
 auto dim4 = data.imp_trace.get_block_dim(b4);
 double dtau = double(tau_f - tau_i);
 auto dtau4 = pow<4>(dtau);
 for (int i4 = 0; i4 < dim4; ++i4) {
  auto lamb4 = data.imp_trace.get_block_eigenval(b4, i4);
  for (int i3 = 0; i3 < dim3; ++i3) {
   auto lamb3 = data.imp_trace.get_block_eigenval(b3, i3);
   for (int i2 = 0; i2 < dim2; ++i2) {
    auto lamb2 = data.imp_trace.get_block_eigenval(b2, i2);
    for (int i1 = 0; i1 < dim1; ++i1) {
     auto lamb1 = data.imp_trace.get_block_eigenval(b1, i1);
     for (int i0 = 0; i0 < dim0; ++i0) {
      auto lamb0 = data.imp_trace.get_block_eigenval(b0, i0);
      auto rhs = compute_evolution_integral(lamb0, lamb1, lamb2, lamb3,
                                            lamb4); // FIXME optimise by just changing one variable in list and passing list
      M4(i4, i0) *= rhs * M3(i3, i2) * M2(i2, i1) * M1(i1, i0) * dtau4;
     }
    }
   }
  }
 }
 return {b_f, std::move(M4)};
}

double measure_four_body_corr::compute_evolution_integral(double lamb1, double lamb2, double lamb3, double lamb4, double lamb5) {

 std::array<std::pair<double, int>, 5> lambda_and_mult = {{{lamb1, 1}, {lamb2, 1}, {lamb3, 1}, {lamb4, 1}, {lamb5, 1}}};

 // Sort in order of increasing lambdas
 std::sort(lambda_and_mult.begin(), lambda_and_mult.end(), [](auto a, auto b) { return a.first < b.first; });
 // Determine unique entries and their multiplicities
 unique_with_multiplicity<5>(lambda_and_mult);
 // Resort in order of decreasing multiplicities
 std::sort(lambda_and_mult.begin(), lambda_and_mult.end(), [](auto a, auto b) { return a.second > b.second; });
 // Find the first two (largest) multiplicities to uniquely identify the case
 auto max_mult_1 = lambda_and_mult[0].second;
 auto max_mult_2 = lambda_and_mult[1].second;

 int mult_sum = 0;
 for (auto const& lm : lambda_and_mult) mult_sum += lm.second;
 if (mult_sum != 3) TRIQS_RUNTIME_ERROR << "compute_evolution_integral: multiplities do not add up!";

 // FIXME could reduce this, only need to do it if multiplicity is nonzero
 auto l1 = lambda_and_mult[0].first;
 auto l2 = lambda_and_mult[1].first;
 auto l3 = lambda_and_mult[2].first;
 auto l4 = lambda_and_mult[3].first;
 auto l5 = lambda_and_mult[4].first;
 auto el1 = std::exp(l1);
 auto el2 = std::exp(l2);
 auto el3 = std::exp(l3);
 auto el4 = std::exp(l4);
 auto el5 = std::exp(l5);
 auto l12 = l1 - l2, l13 = l1 - l3, l14 = l1 - l4, l15 = l1 - l5;
 auto l23 = l2 - l3, l24 = l2 - l4, l25 = l2 - l5;
 auto l34 = l3 - l4, l35 = l3 - l5;
 auto l45 = l4 - l5;

 switch (max_mult_1) {
  case (1): // 11111
   return el1 / (l12 * l13 * l14 * l15) - el2 / (l12 * l23 * l24 * l25) + el3 / (l13 * l23 * l34 * l35) -
              el4 / (l14 * l24 * l34 * l45) + el5 / (l15 * l25 * l35 * l45);
  case (2): // 21110 or 22100
   switch (max_mult_2) {
    case (1):
     return el1 / (l12 * l13 * l14) + el2 / (pow<2>(l12) * l23 * l24) - el3 / (pow<2>(l13) * l23 * l34) +
                el4 / (pow<2>(l14) * l24 * l34) +
                (el1 * (-3 * pow<2>(l1) - l2 * l3 - l2 * l4 - l3 * l4 + 2 * l1 * (l2 + l3 + l4))) /
                    (pow<2>(l12) * pow<2>(l13) * pow<2>(l14));
    case (2):
     return el3 / (pow<2>(l13) * pow<2>(l23)) -
                (el2 * (l1 - 3 * l2 - l1 * l2 + pow<2>(l2) + 2 * l3 + l1 * l3 - l2 * l3)) / (pow<3>(l12) * pow<2>(l23)) +
                (el1 * (-3 * l1 + pow<2>(l1) + l2 - l1 * l2 + 2 * l3 - l1 * l3 + l2 * l3)) / (pow<3>(l12) * pow<2>(l13));
   }
  case (3): // 31100 or 32000
   switch (max_mult_2) {
    case (1):
     return el1 / (2. * l12 * l13) - el2 / (pow<3>(l12) * l23) + (el1 * (1 - l12)) / (pow<3>(l12) * l23) +
                el3 / (pow<3>(l13) * l23) + (el1 * (-1 + l13)) / (pow<3>(l13) * l23);
    case (2):
     return (3 * (el1 - el2)) / pow<4>(l12) - (2 * el1 + el2) / pow<3>(l12) + el1 / (2. * pow<2>(l12));
   }
  case (4): // 41000
   return (-el1 + el2) / pow<4>(l12) + el1 / pow<3>(l12) - el1 / (2. * pow<2>(l12)) + el1 / (6. * l12);
  case (5): // 50000
   return el1 / 24.;
 }
}

double measure_four_body_corr::compute_evolution_integral(double lamb1, double lamb2, double lamb3) {

 std::array<std::pair<double, int>, 3> lambda_and_mult = {{{lamb1, 1}, {lamb2, 1}, {lamb3, 1}}};

 // Sort in order of increasing lambdas
 std::sort(lambda_and_mult.begin(), lambda_and_mult.end(), [](auto a, auto b) { return a.first < b.first; });
 // Determine unique entries and their multiplicities
 unique_with_multiplicity<3>(lambda_and_mult);
 // Resort in order of decreasing multiplicities
 std::sort(lambda_and_mult.begin(), lambda_and_mult.end(), [](auto a, auto b) { return a.second > b.second; });
 // Find the largest multiplicity to uniquely identify the case
 auto max_mult = lambda_and_mult[0].second;

 int mult_sum = 0;
 for (auto const& lm : lambda_and_mult) mult_sum += lm.second;
 if (mult_sum != 3) TRIQS_RUNTIME_ERROR << "compute_evolution_integral: multiplities do not add up!";

 // FIXME could reduce this, only need to do it if multiplicity is nonzero
 auto l1 = lambda_and_mult[0].first;
 auto l2 = lambda_and_mult[0].first;
 auto l3 = lambda_and_mult[0].first;
 auto el1 = std::exp(l1);
 auto el2 = std::exp(l2);
 auto el3 = std::exp(l3);
 auto l12 = l1 - l2, l13 = l1 - l3;
 auto l23 = l2 - l3;

 switch (max_mult) {
  case (1): // 111
   return ((el1 - el2) / l12 + (-el1 + el3) / l13) / l23;
  case (2): // 210
   return (-el1 + el2) / pow<2>(l12) + el1 / l12;
  case (3): // 300
   return el1 / 2.;
 }
}

//*********************************************************************************
// Compute 1) trace for glued configuratons with op_l and op_r shifted to their
//            respective right neighbours (second operator of the chosen pair)
//         2) integral of trace for sliding times of op_l and op_r; they can
//            slide between neighbouring operators, but cannot exceed beta/0
//*********************************************************************************
std::pair<trace_t,trace_t> measure_four_body_corr::compute_sliding_trace_integral(std::vector<node> const& flat_config, int index_node_l,
                                                           int index_node_r, std::vector<int> const& blocks) {

 // Preconditions: chosen operator with index_node is always first of a pair,
 // i.e. there is at least one operator to the right of op(index_node)
 auto node_r1 = flat_config[index_node_r];
 auto tau_r1 = node_r1->key;
 auto node_r2 = flat_config[index_node_r + 1];
 auto tau_r2 = node_r2->key;
 auto node_l1 = flat_config[index_node_l];
 auto tau_l1 = node_l1->key;
 auto node_l2 = flat_config[index_node_l + 1];
 auto tau_l2 = node_l2->key;
 auto root = tree.get_root();
 auto conf_size = flat_config.size();
 trace_t sliding_trace = 0, int_trace = 0;

 auto tau1 = flat_config[index_node_r + 1]->key;
 auto tau2 = flat_config[index_node_r - 1]->key;
 auto tau3 = flat_config[index_node_l + 1]->key;
 auto tau4 = flat_config[index_node_l - 1]->key;

 // FIXME leave blocks as a input param or use contributing blocks by default?
 for (auto bl : blocks) {

  // FIXME
  // compute_matrix(root, bl);

  // size of tau piece outside first-last operators: beta - tmax + tmin ! the tree is in REVERSE order
  double dtau_beta = data.config.beta() - tree.min_key();
  double dtau_zero = double(tree.max_key());
  auto dtau_beta_zero = (is_i_first_op ? dtau_zero : dtau_beta + dtau_zero);

  // Matrix with c_dag,c operators stuck together
  // mat =     M_L * evo34 * op_l * M_M * evo12 * op_r * M_R
  // blocks:  bl<-b4       b4<-b3  b3<-b2        b2<-b1 b1<-bl
  // times:      t4            t_l=t3  t2           t_r=t1
  // Done in three pieces : (M_L * (evo34 * op_l * M_M * (evo12 * op_r * M_R)))
  auto b_mat_R = data.imp_trace.compute_M_R(root, tau_r, bl); // b_mat_R.b = b1
  auto trace_mat = data.imp_trace.evolve(tau1, tau2, get_op(node_r) * b_mat_R);
  auto b_mat_M = data.imp_trace.compute_M_M(root, tau_l, tau_r, trace_mat.b); // b_mat_M.b = b3
  trace_mat = data.imp_trace.evolve(tau3, tau4, get_op(node_l) * (b_mat_M * trace_mat));
  auto b_mat_L = data.imp_trace.compute_M_L(root, tau_l, trace_mat.b);
  trace_mat = b_mat_L * trace_mat;

  // Matrix for trace normalisation with integrals
  // mat =     M_L * int [evo4 * op_l * evo3] * M_M * int [evo2 * op_r * evo1] * M_R
  // blocks:  bl<-b4            b4<-b3        b3<-b2             b2<-b1         b1<-bl
  auto int_r = data.imp_trace.integral(b_mat_R.b, tau1, tau2, node_r->op); // \int evo2 * op_r * evo1
  auto int_l = data.imp_trace.integral(b_mat_M.b, tau3, tau4, node_l->op); // \int evo4 * op_l * evo3
  auto int_mat = b_mat_L * (int_l * (b_mat_M * (int_r * b_mat_R)));

  if ((trace_mat.b != bl) or (int_mat.b != bl))
   TRIQS_RUNTIME_ERROR << " compute_sliding_trace_integral: matrix takes b_i " << bl << " to " << trace_mat.b << " !";

  // Compute trace
  int_trace = trace(int_mat.M);
  sliding_trace = trace(trace_mat.M);
 }
 return {sliding_trace, int_trace};
}

//
// //*********************************************************************************
// // Compute 1) trace for glued configuratons with op_l and op_r shifted to their
// //            respective right neighbours (second operator of the chosen pair)
// //         2) integral of trace for sliding times of op_l and op_r; they can
// //            slide between neighbouring operators, but cannot exceed beta/0
// //*********************************************************************************
// std::pair<trace_t, trace_t> compute_sliding_trace_integral(std::vector<node> const& flat_config, int index_node_l,
//                                                            int index_node_r, std::vector<int> const& blocks) {
//
//  // Preconditions: chosen operator with index_node is always first of a pair,
//  // i.e. there is at least one operator to the right of op(index_node)
//  auto is_i_first_op = (index_node_l == 0);
//  auto node_r = flat_config[index_node_r];
//  auto tau_r = node_r->key;
//  auto node_l = flat_config[index_node_l];
//  auto tau_l = node_l->key;
//  auto root = tree.get_root();
//  auto conf_size = flat_config.size();
//  trace_t sliding_trace = 0, int_trace = 0;
//
//  // Determine if cycling around with j operator or not
//  auto is_r_last_op = (index_node_r == conf_size - 1);
//  auto r_plus_one = is_r_last_op ? 0 : index_node_r + 1 ; // Cycle around?
//
//  // If operator is the leftmost in config (closest to tau=beta), tau4 = beta
//  // then do not evolve to beta at the end!
//  // Do not check if operator is rightmost as we are always shifting first operator in a pair
//  auto tau1 = flat_config[r_plus_one]->key;
//  auto tau2 = flat_config[index_node_r - 1]->key;
//  auto tau3 = flat_config[index_node_l + 1]->key;
//  auto tau4 = (is_i_first_op ? _beta : flat_config[index_node_l - 1]->key);
//
//  if (is_r_last_op) {
//
//   for (auto bl : blocks) {
//
//    // FIXME
//    //compute_matrix(root, bl);
//
//    // Matrix with c_dag,c operators stuck together
//    // mat =      evo * op_r * M_L * evo * op_l * M_M * evo
//    // times: beta   t_r=t1  t1    t4   t_l=t3  t3    t2   0
//    // blocks:   b1  b1    bl=b4   b3    b3     b2    b1   b1
//    auto b_mat_M = compute_M_M(root, tau_l, tau_r, get_op_block_map(node_r->op, bl));
//    auto trace_mat = evolve(tau3, tau4, get_op(node_l) * b_mat_M);
//    auto b_mat_L = compute_M_L(root, tau_l, trace_mat.b);
//    assert(b_mat_L.b == bl); //DEBUG
//    trace_mat = evolve(tau1, _beta, get_op(node_r) * (b_mat_L * trace_mat));
//    trace_mat = evolve(_zero, tau2, trace_mat);
//
//    // Matrix for trace normalisation with integrals
//    // mat =   (int [evo2 * op_r * evo_zero] + int [evo_beta * op_r * evo1]) * M_L * int [evo4 * op_l * evo3] * M_M
//    // times:  t2                         zero/beta                          t1    t4                         t3   t2
//    // blocks: b1                                                          bl=b4   b3                         b2   b1
//    auto int_l = int_evo_op_evo(b_mat_M.b, tau3, tau4, node_l->op);   // \int evo4 * op_l * evo3
//    auto int_r = int_evo_op_evo(bl, tau1, tau2, node_r->op);          // \int evo2 * op_r * evo1
//    auto int_mat = int_r * (b_mat_L * (int_l * b_mat_M));
//
//    assert(trace_mat.b == get_op_block_map(node_r->op, bl));
//    assert(int_mat.b == get_op_block_map(node_r->op, bl));
//
//    // Compute trace
//    int_trace = trace(int_mat.M);
//    sliding_trace = trace(trace_mat.M);
//   }
//
//  } else {
//
//   // FIXME leave blocks as a input param or use contributing blocks by default?
//   for (auto bl : blocks) {
//
//    // FIXME
//    //compute_matrix(root, bl);
//
//    // size of tau piece outside first-last operators: beta - tmax + tmin ! the tree is in REVERSE order
//    double dtau_beta = config->beta() - tree.min_key();
//    double dtau_zero = double(tree.max_key());
//    auto dtau_beta_zero = (is_i_first_op ? dtau_zero : dtau_beta + dtau_zero);
//
//    // Matrix with c_dag,c operators stuck together
//    // mat =     M_L * evo34 * op_l * M_M * evo12 * op_r * M_R
//    // blocks:  bl<-b4       b4<-b3  b3<-b2        b2<-b1 b1<-bl
//    // times:      t4            t_l=t3  t2           t_r=t1
//    // Done in three pieces : (M_L * (evo34 * op_l * M_M * (evo12 * op_r * M_R)))
//    auto b_mat_R = compute_M_R(root, tau_r, bl); // b_mat_R.b = b1
//    auto trace_mat = evolve(tau1, tau2, get_op(node_r) * b_mat_R);
//    auto b_mat_M = compute_M_M(root, tau_l, tau_r, trace_mat.b); // b_mat_M.b = b3
//    trace_mat = evolve(tau3, tau4, get_op(node_l) * (b_mat_M * trace_mat));
//    auto b_mat_L = compute_M_L(root, tau_l, trace_mat.b);
//    trace_mat = b_mat_L * trace_mat;
//
//    // Matrix for trace normalisation with integrals
//    // mat =     M_L * int [evo4 * op_l * evo3] * M_M * int [evo2 * op_r * evo1] * M_R
//    // blocks:  bl<-b4            b4<-b3        b3<-b2             b2<-b1         b1<-bl
//    auto int_r = int_evo_op_evo(b_mat_R.b, tau1, tau2, node_r->op); // \int evo2 * op_r * evo1
//    auto int_l = int_evo_op_evo(b_mat_M.b, tau3, tau4, node_l->op); // \int evo4 * op_l * evo3
//    auto int_mat = b_mat_L * (int_l * (b_mat_M * (int_r * b_mat_R)));
//
//    if ((trace_mat.b != bl) or (int_mat.b != bl))
//     TRIQS_RUNTIME_ERROR << " compute_sliding_trace_integral: matrix takes b_i " << bl << " to " << trace_mat.b << " !";
//
//    // trace(mat * exp(- H * (beta - tmax)) * exp (- H * tmin)) to handle the piece outside of the first-last operators.
//    auto dim = get_block_dim(bl);
//    for (int u = 0; u < dim; ++u) {
//     auto evo = std::exp(-dtau_beta_zero * get_block_eigenval(bl, u));
//     int_trace += int_mat.M(u, u) * evo;
//     sliding_trace += trace_mat.M(u, u) * evo;
//    }
//   }
//  }
//  return {sliding_trace, int_trace};
// }


//***********************************************************************
// Compute 1) trace for glued configuratons and
//         2) integral of trace for sliding times of op
//***********************************************************************
std::pair<trace_t, trace_t> measure_four_body_corr::compute_sliding_trace_integral_one_pair(std::vector<node> const& flat_config, int index_node,
                                                                    std::vector<int> const& blocks) {

 auto is_first_op = (index_node == 0);
 auto n = flat_config[index_node];
 auto tau = n->key;
 auto root = tree.get_root();
 auto conf_size = flat_config.size();
 // size of tau piece outside first-last operators: beta - tmax + tmin ! the tree is in REVERSE order
 auto dtau_beta = _beta - tree.min_key();
 auto dtau_zero = tree.max_key();
 double dtau_beta_zero = (is_first_op ? double(dtau_zero) : double(dtau_beta + dtau_zero));
 trace_t sliding_trace = 0, int_trace = 0;

 // If operator is the leftmost in config (closest to tau=beta), tau4 = beta
 // then do not evolve to beta at the end!
 // Do not check if operator is rightmost as we are always shifting first operator in a pair
 auto tau1 = ((index_node + 1 == conf_size) ? _zero : flat_config[index_node + 1]->key);
 auto tau2 = ((index_node == 0) ? _beta : flat_config[index_node - 1]->key);

 for (auto bl : blocks) {

  // Matrix with c_dag,c operators stuck together
  // mat =     M_L * evo12 *  op  * M_R
  // blocks:  bl<-b2        b2<-b1 b1<-bl
  // Done in three pieces : (M_L * (evo12 * op_r * (M_R)))
  auto b_mat_R = data.imp_trace.compute_M_R(root, tau, bl); // b_mat_R.b = b1
  auto trace_mat = data.imp_trace.evolve(tau1, tau2, get_op(n) * b_mat_R);
  auto b_mat_L = data.imp_trace.compute_M_L(root, tau, trace_mat.b);
  trace_mat = b_mat_L * trace_mat;

  // Matrix for trace normalisation with integrals
  // mat =     M_L * int [evo2 * op_r * evo1] * M_R
  // blocks:  bl<-b2            b2<-b1         b1<-bl
  auto int_mat = b_mat_L * (integral(b_mat_R.b, tau1, tau2, n->op) * b_mat_R);

  if ((trace_mat.b != bl) or (int_mat.b != bl))
   TRIQS_RUNTIME_ERROR << " compute_sliding_trace_integral: matrix takes b_i " << bl << " to " << trace_mat.b << " !";

  // trace(mat * exp(- H * (beta - tmax)) * exp (- H * tmin)) to handle the piece outside of the first-last operators.
  auto dim = get_block_dim(bl);
  for (int u = 0; u < dim; ++u) {
   auto evo = std::exp(-dtau_beta_zero * get_block_eigenval(bl, u));
   int_trace += int_mat.M(u, u) * evo;
   sliding_trace += trace_mat.M(u, u) * evo;
  }
 }
 return {sliding_trace, int_trace};
}
}
