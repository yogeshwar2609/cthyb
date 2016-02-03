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

measure_four_body_corr::measure_four_body_corr(qmc_data const& data, gf_view<imfreq, scalar_valued> correlator,
                                               fundamental_operator_set const& fops, many_body_operator const& A, bool anticommute)
   : data(data), correlator(correlator), anticommute(anticommute), imp_tr(data.imp_trace), tree(data.imp_trace.tree) {
 z = 0;
 correlator() = 0.0;

 //FIXME generalize to take two different operators too
 // Extract the non-zero monomials for a quadratic operator, op = \sum_ab coef_ab c^+_a c_b
 // and store in a table linking to operator indices: coef = coefficients[cdag_index,c_index,cdag_index,c_index]
 // Note: it is important that the order is c^+ c, and not c c^+ !
 auto fops_size = fops.size();
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
 auto fc_size = flat_config.size();

#ifdef DEBUG
 // Check partial linear matrices match root cache matrix
 imp_tr.check_ML_MM_MR(PRINT_DEBUG);
#endif

 auto blocks = imp_tr.get_nonstructurally_zero_blocks();

 // Loop on the first pair of c^+, c, indexed 4,3
 // i+1 can go up to second to last config
 for (int i = 0; i < fc_size - 2; ++i) {

  // n4, n3 are the two first operators
  auto n4 = flat_config[i];
  auto n3 = flat_config[i + 1];
  if (anticommute) {
   if (n4->op.dagger == n3->op.dagger) continue; // restrict to either cdag * c or c * cdag
  } else {
   if ((!n4->op.dagger) or (n3->op.dagger)) continue; // restrict to cdag * c only
  }
  auto ind4 = op_index_in_det[i];
  auto ind3 = op_index_in_det[i + 1];

  // Ensure that n4 is dagger, n3 not
  auto swapped43 = (n3->op.dagger);
  if (swapped43) {
   std::swap(n4, n3);
   std::swap(ind4, ind3);
  }

  // Does this pair contribute?
  if (std::abs(coefficients_one_pair(n4->op.linear_index, n3->op.linear_index)) < coef_threshold) continue;

  // Find the second pair of c^+, c, indexed 2,1
  // j can go all the way to the rightmost operator and j+1 then loops to leftmost operator
  for (int j = i + 2; j < fc_size; ++j) {

   if (i == (j + 1 % fc_size)) continue; // Otherwise have a clash of c_i = c_j+1!
   // n2, n1 are the two other operators
   auto n2 = flat_config[j];
   auto n1 = flat_config[j + 1 % fc_size]; // Cycle around if necessary
   if (anticommute) {
    if (n2->op.dagger == n1->op.dagger) continue;
   } else {
    if ((!n2->op.dagger) and (n1->op.dagger)) continue;
   }
   auto ind2 = op_index_in_det[j];
   auto ind1 = op_index_in_det[j + 1 % fc_size];

   // Ensure that n2 is dagger, n1 not
   auto swapped21 = (n1->op.dagger);
   if (swapped21) {
    std::swap(n2, n1);
    std::swap(ind2, ind1);
   }

   // Coefficient for the accumulation
   auto coef = coefficients(n4->op.linear_index, n3->op.linear_index, n2->op.linear_index, n1->op.linear_index);
   if (std::abs(coef) < coef_threshold) continue; // Do these 2 pairs contribute?

   // Now measure!

   // --- Det ---

   // Properties corresponding to det
   // Could be in a block diagonal situation, so need to check block indices.

   // For indices abcd, compute M_ba M_dc - M_da M_bc (a,c are the dagger indices)
   double MM1 = 0.0, MM2 = 0.0;
   auto b4 = n4->op.block_index;
   auto b3 = n3->op.block_index;
   auto b2 = n2->op.block_index;
   auto b1 = n1->op.block_index;

   if ((b4 == b3) && (b2 == b1)) MM1 = data.dets[b4].inverse_matrix(ind3, ind4) * data.dets[b3].inverse_matrix(ind1, ind2);
   if ((b4 == b1) && (b2 == b3)) MM2 = data.dets[b4].inverse_matrix(ind1, ind4) * data.dets[b3].inverse_matrix(ind3, ind2);

   // --- Trace and the tree ---

#ifdef DEBUG
   // DEBUG Check that the trace is correct
   imp_tr.check_trace_from_ML_MM_MR(flat_config, i, j, PRINT_DEBUG); // FIXME -- generalise for cyclic trace
#endif

   if ((anticommute) and (swapped43 xor swapped21)) s = -s;
   coef *= s * (MM1 - MM2);
   // Compute the trace and normalisation integral, and accumulate into the correlator for all frequencies
   compute_sliding_trace_integral(flat_config, i, j, blocks, coef, correlator);

  } // Second pair
 }  // First pair
}

// ---------------------------------------------

void measure_four_body_corr::collect_results(triqs::mpi::communicator const& c) {
 z = mpi_all_reduce(z, c);
 correlator = mpi_all_reduce(correlator, c);
 correlator = correlator / (z * data.config.beta());
}

// #################### BASE FUNCTIONS ##############################

template <int N> double pow(double x);
template <> double pow<1>(double x) { return x; }
template <> double pow<2>(double x) { return x * x; }
template <> double pow<3>(double x) { return pow<2>(x) * x; }
template <> double pow<4>(double x) { return pow<2>(x) * pow<2>(x); }

template <int N, typename A> void unique_with_multiplicity(A& a) {
 for (int r = 0; r < N - 1; ++r) {
  if (std::abs(a[r + 1].first - a[r].first) < 1.e-10) {
   a[r + 1].second += a[r].second;
   a[r].second = 0;
  }
 }
}

// --------------------

// Compute [\int_tau_i^\tau_f dtau e^-H_{b_f}(tau_f - tau) * op_{b_i->b_f}(tau) * e^-H_{b_i}(tau - tau_i)]
 //    | -------- | ----- x ----- | -------------|
 // beta         t_f b_f     b_i t_i             0
// Precondition: only valid for non-structurally-zero blocks
block_and_matrix measure_four_body_corr::compute_normalization_integral(int b_i, time_pt tau_i, time_pt tau_f, op_desc const& op) {
 auto b_f = imp_tr.get_op_block_map(op, b_i);
 auto M = imp_tr.get_op_block_matrix(op, b_i);
 double dtau = double(tau_f - tau_i);
 auto dim1 = imp_tr.get_block_dim(b_i);
 auto dim2 = imp_tr.get_block_dim(b_f);
 for (int i2 = 0; i2 < dim2; ++i2) {
  auto lamb2 = imp_tr.get_block_eigenval(b_f, i2);
  for (int i1 = 0; i1 < dim1; ++i1) {
   auto lamb1 = imp_tr.get_block_eigenval(b_i, i1);
   auto rhs = compute_evolution_integral(lamb1,lamb2);
   M(i2, i1) *= rhs * (-dtau);
  }
 }
 return {b_f, std::move(M)};
}

// Compute the integral below between tau_i and tau_f, and return a matrix
// \int_tau_i^\tau_f dtau_n ... \int_tau_i^\tau_2 dtau_1 e^-H_{b_f}(tau_f - tau_n) * op{b_n-1->b_f}(tau_n) * ... * e^-H_{b_i}(tau_1 - tau_i)
//    | -------- | ----- x --- x ----- | -------------|
// beta         t_f                    t_i            0
//              b_f=b3      b2      b1=b_i
// Precondition: only valid for non-structurally-zero blocks
block_and_matrix measure_four_body_corr::compute_normalization_integral(int b_i, time_pt tau_i, time_pt tau_f, op_desc const& op1, op_desc const& op2) {

 auto b1 = b_i;
 auto M1 = imp_tr.get_op_block_matrix(op1, b1);
 auto b2 = imp_tr.get_op_block_map(op1, b1);
 auto M2 = imp_tr.get_op_block_matrix(op2, b2);
 auto b3 = imp_tr.get_op_block_map(op2, b2);
 auto b_f = b3;
 auto dim1 = imp_tr.get_block_dim(b1);
 auto dim2 = imp_tr.get_block_dim(b2);
 auto dim3 = imp_tr.get_block_dim(b3);
 double dtau = double(tau_f - tau_i);
 auto dtau2 = pow<2>(dtau);
 arrays::matrix<double> M(dim2, dim1);
 M() = 0.0;
 for (int i3 = 0; i3 < dim3; ++i3) {
  auto lamb3 = imp_tr.get_block_eigenval(b3, i3);
  for (int i2 = 0; i2 < dim2; ++i2) {
   auto lamb2 = imp_tr.get_block_eigenval(b2, i2);
   for (int i1 = 0; i1 < dim1; ++i1) {
    auto lamb1 = imp_tr.get_block_eigenval(b1, i1);
    auto rhs = compute_evolution_integral(lamb1, lamb2, lamb3);
    M(i3, i1) += rhs * dtau2 * M2(i3, i2) * M1(i2, i1);
   }
  }
 }
 return {b_f, M};
}

// Compute the integral below between tau_i and tau_f, and return a matrix
// \int_tau_i^\tau_f dtau_n ... \int_tau_i^\tau_2 dtau_1 e^-H_{b_f}(tau_f - tau_n) * op{b_n-1->b_f}(tau_n) * ... * e^-H_{b_i}(tau_1 - tau_i)
//    | -------- | ----- x --- x --- x --- x ---- | -------------|
// beta         t_f                               t_i            0
//              b_f=b5      b4    b3    b2     b1=b_i
// Precondition: only valid for non-structurally-zero blocks
block_and_matrix measure_four_body_corr::compute_normalization_integral(int b_i, time_pt tau_i, time_pt tau_f, op_desc const& op1, op_desc const& op2,
                                                  op_desc const& op3, op_desc const& op4) {
 auto b1 = b_i;
 auto M1 = imp_tr.get_op_block_matrix(op1, b1);
 auto b2 = imp_tr.get_op_block_map(op1, b1);
 auto M2 = imp_tr.get_op_block_matrix(op2, b2);
 auto b3 = imp_tr.get_op_block_map(op2, b2);
 auto M3 = imp_tr.get_op_block_matrix(op3, b3);
 auto b4 = imp_tr.get_op_block_map(op3, b3);
 auto M4 = imp_tr.get_op_block_matrix(op4, b4);
 auto b5 = imp_tr.get_op_block_map(op4, b4);
 auto b_f = b5;
 auto dim1 = imp_tr.get_block_dim(b1);
 auto dim2 = imp_tr.get_block_dim(b2);
 auto dim3 = imp_tr.get_block_dim(b3);
 auto dim4 = imp_tr.get_block_dim(b4);
 auto dim5 = imp_tr.get_block_dim(b5);
 double dtau = double(tau_f - tau_i);
 auto dtau4 = pow<4>(dtau);
 arrays::matrix<double> M(dim5, dim1);
 M() = 0.0;
 for (int i5 = 0; i5 < dim5; ++i5) {
  auto lamb5 = imp_tr.get_block_eigenval(b5, i5);
  for (int i4 = 0; i4 < dim4; ++i4) {
   auto lamb4 = imp_tr.get_block_eigenval(b4, i4);
   for (int i3 = 0; i3 < dim3; ++i3) {
    auto lamb3 = imp_tr.get_block_eigenval(b3, i3);
    for (int i2 = 0; i2 < dim2; ++i2) {
     auto lamb2 = imp_tr.get_block_eigenval(b2, i2);
     for (int i1 = 0; i1 < dim1; ++i1) {
      auto lamb1 = imp_tr.get_block_eigenval(b1, i1);
      auto rhs = compute_evolution_integral(lamb1, lamb2, lamb3, lamb4, lamb5);
      M(i5, i1) += rhs * dtau4 * M4(i5, i4) * M3(i4, i3) * M2(i3, i2) * M1(i2, i1);
     }
    }
   }
  }
 }
 return {b_f, M};
}

// --------------------

double measure_four_body_corr::compute_evolution_integral(double lamb1, double lamb2) {
 return ((std::abs(lamb1 - lamb2) > 1.e-10) ? (std::exp(lamb1) - std::exp(lamb2)) / (lamb2 - lamb1) : -std::exp(lamb1));
// return ((std::abs(lamb1 - lamb2) > 1.e-10) ? (std::exp(-dtau * lamb1) - std::exp(-dtau * lamb2)) / (lamb2 - lamb1)
//                                                  : std::exp(-dtau * lamb1) * dtau);
}

// --------------------

double measure_four_body_corr::compute_evolution_integral(double lamb1, double lamb2, double lamb3) {

 std::array<std::pair<double, int>, 3> lambda_and_mult = {{{lamb1, 1}, {lamb2, 1}, {lamb3, 1}}};

 // Sort in order of increasing lambdas
 std::sort(lambda_and_mult.begin(), lambda_and_mult.end(), [](auto a, auto b) { return a.first < b.first; });
 // Determine unique entries and their multiplicities
 unique_with_multiplicity<3>(lambda_and_mult);
 // Resort in order of decreasing multiplicities
 std::sort(lambda_and_mult.begin(), lambda_and_mult.end(), [](auto a, auto b) { return a.second > b.second; });
#ifdef DEBUG
 int mult_sum = 0;
 for (auto const& lm : lambda_and_mult) mult_sum += lm.second;
 if (mult_sum != 3) TRIQS_RUNTIME_ERROR << "compute_evolution_integral: multiplities do not add up!";
#endif

 // FIXME could reduce this, only need to do it if multiplicity is nonzero
 auto l1 = lambda_and_mult[0].first;
 auto l2 = lambda_and_mult[1].first;
 auto l3 = lambda_and_mult[2].first;
 auto el1 = std::exp(l1);
 auto el2 = std::exp(l2);
 auto el3 = std::exp(l3);
 auto l12 = l1 - l2, l13 = l1 - l3;
 auto l23 = l2 - l3;

 // Find the largest multiplicity to uniquely identify the case
 auto max_mult = lambda_and_mult[0].second;
 switch (max_mult) {
  case (1): // 111
   return ((el1 - el2) / l12 + (-el1 + el3) / l13) / l23;
  case (2): // 210
   return (-el1 + el2) / pow<2>(l12) + el1 / l12;
  case (3): // 300
   return el1 / 2.;
 }
 TRIQS_RUNTIME_ERROR << "compute_normalization_integral (2 op): did not match any of the cases.";
}

// --------------------

double measure_four_body_corr::compute_evolution_integral(double lamb1, double lamb2, double lamb3, double lamb4, double lamb5) {

 std::array<std::pair<double, int>, 5> lambda_and_mult = {{{lamb1, 1}, {lamb2, 1}, {lamb3, 1}, {lamb4, 1}, {lamb5, 1}}};

 // Sort in order of increasing lambdas
 std::sort(lambda_and_mult.begin(), lambda_and_mult.end(), [](auto a, auto b) { return a.first < b.first; });
 // Determine unique entries and their multiplicities
 unique_with_multiplicity<5>(lambda_and_mult);
 // Resort in order of decreasing multiplicities
 std::sort(lambda_and_mult.begin(), lambda_and_mult.end(), [](auto a, auto b) { return a.second > b.second; });
#ifdef DEBUG
 int mult_sum = 0;
 for (auto const& lm : lambda_and_mult) mult_sum += lm.second;
 if (mult_sum != 5) TRIQS_RUNTIME_ERROR << "compute_evolution_integral: multiplities do not add up!";
#endif

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

 // Find the first two (largest) multiplicities to uniquely identify the case
 auto max_mult_1 = lambda_and_mult[0].second;
 auto max_mult_2 = lambda_and_mult[1].second;
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
 TRIQS_RUNTIME_ERROR << "compute_normalization_integral (4 op): did not match any of the cases.";
}

//*********************************************************************************
// Compute 1) trace for glued configuratons with op_(l/r)2 and op_(l/r)1 stuck together
//         2) integral of trace for sliding times of op_l and op_r; they can
//            slide between neighbouring operators
//*********************************************************************************
void measure_four_body_corr::compute_sliding_trace_integral(std::vector<node> const& flat_config, int index_node_l,
                                                            int index_node_r, std::vector<int> const& blocks, double coef,
                                                            gf_view<imfreq, scalar_valued> correlator) {
 //    -----------M_M(tr1,tl2) ---------------------------------------------
 //   |                           M_M(tl1,tr2)                              |
 //   \--------->                 <----------->                 <-----------/
 //    | ------ | --- x --- x --- | --------- | --- x --- x --- | ----------|
 // beta              l2    l1                      r2    r1                0
 //             t4                t3          t2                t1
 //   bl=b7        b6    b5    b4                b3    b2    b1             bl
 // Preconditions: chosen operator with index_node is always first of a pair, with index 2,
 // i.e. there is at least one operator to the right of op(index_node)
 auto node_r1 = flat_config[index_node_r + 1];
 auto node_r2 = flat_config[index_node_r];
 auto node_l1 = flat_config[index_node_l + 1];
 auto node_l2 = flat_config[index_node_l];
 auto tau_r1 = node_r1->key;
 auto tau_r2 = node_r2->key;
 auto tau_l1 = node_l1->key;
 auto tau_l2 = node_l2->key;
 auto tau1 = flat_config[index_node_r + 2]->key;
 auto tau2 = flat_config[index_node_r - 1]->key;
 auto tau3 = flat_config[index_node_l + 2]->key;
 auto tau4 = flat_config[index_node_l - 1]->key;
 auto is_4op = (index_node_l + 2 == index_node_r);
 auto root = tree.get_root();
 trace_t sliding_trace = 0, int_trace = 0;

 for (auto bl : blocks) {

  imp_tr.compute_matrix(root, bl); // Update matrices without the Yee quick exit

  // Matrix with c_dag,c operators stuck together
  // mat = M_M(tr1,tl2) * evo_tl^t4 * op(l2) op(l1) * evo_t3^tl * M_M(tl1,tr2) * evo_tr^t2 * op(r2) op(r1) * evo_t1^tr
  //   /------ M_M_outer(tr1,tl2) -------------------------------------------\
  //   |                           M_M_inner(tl1,tr2)                        |
  //   \--------->                 <----------->                 <-----------/
  //    | ------ | --- x     x --- | --------- | --- x     x --- | ----------|
  // beta              l2 == l1                      r2 == r1                0
  //             t4                t3          t2                t1
  //   bl=b7        b6    b5    b4                b3    b2    b1             bl

  // Matrix for trace normalisation with integrals
  //     mat = M_M(tr1,tl2) * int[evo_tl^t4 * op(l2) * evo * op(l1) * evo_t3^tl] *
  //           M_M(tl1,tr2) * int[evo_tr^t2 * op(r2) * evo * op(r1) * evo_t1^tr]
  // or  mat = M_M(tr1,tl2) * int[evo^t4 * op(l2) * evo * op(l1) * evo * op(r2) * evo * op(r1) * evo_t1]
  block_and_matrix trace_mat, int_mat;
  auto b1 = imp_tr.compute_block_at_tau(root, tau_r1, bl);
  if (!is_4op) { // 2 x 2-operator integrals
   // Sliding trace
   auto b_mat_r = compute_fourier_sliding_trace(b1, tau1, tau2, node_r1->op, node_r2->op);
   auto M_M_inner = imp_tr.compute_M_M(root, tau_l1, tau_r2, b_mat_r.b);
   auto b_mat_l = compute_fourier_sliding_trace(M_M_inner.b, tau3, tau4, node_l1->op, node_l2->op);
   trace_mat = b_mat_l * M_M_inner * b_mat_r;
   // Normalization integral
   auto int_r = compute_normalization_integral(b1, tau1, tau2, node_r1->op, node_r2->op);
   auto int_l = compute_normalization_integral(M_M_inner.b, tau3, tau4, node_l1->op, node_l2->op);
   int_mat = int_l * (M_M_inner * int_r);
  } else { // 1 x 4-operator integral
   // Sliding trace
   trace_mat = compute_fourier_sliding_trace(b1, tau1, tau4, node_r1->op, node_r2->op, node_l1->op, node_l2->op);
   // Normalization integral
   int_mat = compute_normalization_integral(b1, tau1, tau4, node_r1->op, node_r2->op, node_l1->op, node_l2->op);
  }
  auto M_M_outer = imp_tr.compute_M_M(root, tau_r1, tau_l2, trace_mat.b);
  if (M_M_outer.b != b1) TRIQS_RUNTIME_ERROR << " compute_sliding_trace_integral: start and end blocks do not match.";
  trace_mat = M_M_outer * trace_mat;
  int_mat = M_M_outer * int_mat;

  // Compute trace
  sliding_trace = trace(trace_mat.M);
  int_trace = trace(int_mat.M);
  auto tr_over_int = sliding_trace / int_trace;
  if (!std::isfinite(tr_over_int)) {
   if ((sliding_trace < 1.e-20) and (int_trace < 1.e-20)) continue; // FIXME what thresholds to use for 0/0 check?
   TRIQS_RUNTIME_ERROR << "tr_over_int not finite " << sliding_trace << " " << int_trace;
  }
 }
}
}
