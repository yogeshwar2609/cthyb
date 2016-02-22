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
#include "impurity_trace.hpp"
namespace cthyb {

//***********************************************************************
// Checks related to cache integrity
//***********************************************************************

//-------------------- Cache integrity check --------------------------------

void impurity_trace::check_cache_integrity(bool print) {
#ifdef CHECK_CACHE
 static int check_counter = 0;
 ++check_counter;
 if (check_counter % 10 == 0) {
  if (print) std::cout << " ---- Cache integrity check ---- " << std::endl;
  if (print) std::cout << " check_counter = config number = " << check_counter << std::endl;
  if (print) tree.graphviz(std::ofstream("tree_cache_check"));
  foreach_subtree_first(tree, [&](node y) { this->check_cache_integrity_one_node(y, print); });
  if (print) std::cout << " ---- Cache integrity completed ---- " << std::endl;
 }
#endif
}

//--------------------- Compute block table for one subtree, using an ordered traversal of the subtree -------------------

int impurity_trace::check_one_block_table_linear(node n, int b, bool print) {

 int B = b;
 foreach_reverse(tree, n, [&](node y) {
  if (B == -1) return;
  auto BB = B;
  B = (y->delete_flag ? B : this->get_op_block_map(y->op, B));
  if (print)
   std::cout << "linear computation : " << y->key << " " << y->op.dagger << " " << y->op.linear_index << " | " << BB << " -> "
             << B << std::endl;
 });
 return B;
}

//--------------------- Compute block table for one subtree, using an ordered traversal of the subtree -------------------

matrix<double> impurity_trace::check_one_block_matrix_linear(node top, int b, bool print) {

 node p = tree.max(top);
 matrix<double> M = make_unit_matrix<double>(get_block_dim(b));
 auto _ = arrays::range();

 foreach_reverse(tree, top, [&](node n) {
  // multiply by the exponential unless it is the first call, i.e. first operator n==p
  if (n != p) {
   auto dtau = double(n->key - p->key);
   //  M <- exp * M
   auto dim = first_dim(M); // same as get_block_dim(b1);
   for (int i = 0; i < dim; ++i) M(i, _) *= std::exp(-dtau * get_block_eigenval(b, i));
   // M <- Op * M
  }
  // multiply by operator matrix unless it is delete_flag
  if (!n->delete_flag) {
   int bp = this->get_op_block_map(n->op, b);
   if (bp == -1) TRIQS_RUNTIME_ERROR << " Nasty error ";
   M = get_op_block_matrix(n->op, b) * M;
   b = bp;
  }
  p = n;
 });

 return M;
}
//-------------------- Cache integrity check for one node --------------------------------

void impurity_trace::check_cache_integrity_one_node(node n, bool print) {
 if (n == nullptr) return;
 if (print) std::cout << " ... checking cache integrity for node " << n->key << std::endl;

 // debug check : redo the linear calculation
 auto& ca = n->cache;
 for (int b = 0; b < n_blocks; ++b) {
  auto check = check_one_block_table_linear(n, b, false);
  if (ca.block_table[b] != check) {
   std::cout << " Inconsistent block table for block " << b << " : cache =  " << ca.block_table[b] << " while it should be  "
             << check << std::endl;
   check_one_block_table_linear(n, b, true);
   TRIQS_RUNTIME_ERROR << " FATAL ";
  }
 }
}

//***********************************************************************
// Checks related to partial matrix products M_L, M_M and M_R
//***********************************************************************

//-------- Check partial linear matrices M_L, M_M and M_R against root matrix ------------

void impurity_trace::check_ML_MM_MR(bool print) {

 if (print) std::cout << " ... checking ML, MM and MR against root cache matrix " << std::endl;

 // First update all cached matrices in tree *without* the Yee trick
 auto w_rw = compute();

 // Check M_R(beta) = M_L(0) = M_M(beta,0) = root->cache_matrix
 for (auto bl : contributing_blocks) { // Only loop over matrices that contribute to the trace
  auto root = tree.get_root();
  auto true_mat = get_cache_matrix(root, bl);
  auto b_mat_r = compute_M_R(root, _beta, bl);
  auto b_mat_l = compute_M_L(root, _zero, bl);
  auto b_mat_m = compute_M_M(root, _beta, _zero, bl);
  if ((b_mat_l.b != bl) or (b_mat_m.b != bl) or (b_mat_r.b != bl)) TRIQS_RUNTIME_ERROR << "check_ML_MM_MR: block mismatch!";
  if (max_element(abs(b_mat_l.M - true_mat.M)) > diff_threshold)
   std::cout << "check_ML_MM_MR: ML does not match true matrix: " << b_mat_l.M << " L " << true_mat.M << " true " << std::endl;
  if (max_element(abs(b_mat_m.M - true_mat.M)) > diff_threshold)
   std::cout << "check_ML_MM_MR: MM does not match true matrix: " << b_mat_m.M << " M " << true_mat.M << " true " << std::endl;
  if (max_element(abs(b_mat_r.M - true_mat.M)) > diff_threshold)
   std::cout << "check_ML_MM_MR: MR does not match true matrix: " << b_mat_r.M << " R " << true_mat.M << " true " << std::endl;
 }
}

//-------- Compute trace of configuration using M_L and M_R, and one operator ------------------------------

// Preconditions: chosen operator with index_node is always first of a pair, 
// i.e. there is at least one operator to the right of op(index_node)
void impurity_trace::check_trace_from_ML_MR(std::vector<node> const& flat_config, int index_node, bool print) {

 if (print) std::cout << " ... checking trace from ML and MR against trace from root cache matrix " << std::endl;

 // Update the cache fully, i.e. without the Yee trick
 auto w_rw = compute();
 auto true_trace = w_rw.first * w_rw.second; // tr = norm * tr/norm = w * rw

 auto is_first_op = (index_node == 0);
 auto n = flat_config[index_node];
 auto tau = n->key;
 auto root = tree.get_root();
 auto conf_size = flat_config.size();
 // size of tau piece outside first-last operators: beta - tmax + tmin ! the tree is in REVERSE order
 auto dtau_beta = _beta - tree.min_key();
 auto dtau_zero = tree.max_key();
 double dtau_beta_zero = (is_first_op ? double(dtau_zero) : double(dtau_beta + dtau_zero));
 trace_t reconstructed_trace = 0;

 // If operator is the leftmost in config (closest to tau=beta), tau4 = beta
 // then do not evolve to beta at the end!
 // Do not check if operator is rightmost as we are always shifting first operator in a pair
 auto tau1 = ((index_node + 1 == conf_size) ? _zero : flat_config[index_node + 1]->key);
 auto tau2 = (is_first_op ? _beta : flat_config[index_node - 1]->key);

 for (auto bl : contributing_blocks) {

  // Calculate matrix, working from right to left (tau = 0->beta)
  // mat =     M_L * evo2 *  op  * evo1 * M_R
  // blocks:  bl<-b2       b2<-b1        b1<-bl
  // Done in two pieces : (M_L * (evo2 * op_r * evo1 * M_R)))
  auto b_mat = evolve(tau, tau2, get_op(n) * evolve(tau1, tau, compute_M_R(root, tau, bl)));
  b_mat = compute_M_L(root, tau, b_mat.b) * b_mat;
  if (b_mat.b != bl) TRIQS_RUNTIME_ERROR << "check_trace_from_ML_MR: matrix takes b_i " << bl << " to " << b_mat.b << " !";

  // trace(mat * exp(- H * (beta - tmax)) * exp (- H * tmin)) to handle the piece outside of the first-last operators.
  auto dim = get_block_dim(bl);
  for (int u = 0; u < dim; ++u) reconstructed_trace += b_mat.M(u, u) * std::exp(-dtau_beta_zero * get_block_eigenval(bl, u));
 }

 if (reconstructed_trace - true_trace > diff_threshold)
  TRIQS_RUNTIME_ERROR << "check_trace_from_ML_MR: traces do not agree. true trace = " << true_trace
                      << ", while recomputed trace = " << reconstructed_trace;
}

//-------- Compute trace of configuration using M_L, M_M and M_R, and two operators -------------------------

// Preconditions: chosen operator with index_node_l/r is always first of a pair, 
// i.e. there is at least one operator to the right of op(index_node_r) but op(index_node_l) could be the leftmost
void impurity_trace::check_trace_from_ML_MM_MR(std::vector<node> const& flat_config, int index_node_l, int index_node_r,
                                               bool print) {

 if (print) std::cout << " ... checking trace from ML, MM and MR against trace from root cache matrix " << std::endl;

 // Update the cache fully, i.e. without the Yee trick
 auto w_rw = compute();
 auto true_trace = w_rw.first * w_rw.second; // tr = norm * tr/norm = w * rw

 auto is_i_first_op = (index_node_l == 0);
 auto node_r = flat_config[index_node_r];
 auto tau_r = node_r->key;
 auto node_l = flat_config[index_node_l];
 auto tau_l = node_l->key;
 auto root = tree.get_root();
 auto conf_size = flat_config.size();
 // size of tau piece outside first-last operators: beta - tmax + tmin ! the tree is in REVERSE order
 double dtau_beta = config->beta() - tree.min_key();
 double dtau_zero = double(tree.max_key());
 auto dtau_beta_zero = (is_i_first_op ? dtau_zero : dtau_beta + dtau_zero);
 trace_t reconstructed_trace = 0;

 // If operator is the leftmost in config (closest to tau=beta), tau4 = beta
 // then do not evolve to beta at the end!
 // Do not check if operator is rightmost as we are always shifting first operator in a pair
 auto tau1 = ((index_node_r + 1 == conf_size) ? _zero : flat_config[index_node_r + 1]->key);
 auto tau2 = flat_config[index_node_r - 1]->key;
 auto tau3 = flat_config[index_node_l + 1]->key;
 auto tau4 = (is_i_first_op ? _beta : flat_config[index_node_l - 1]->key);

 for (auto bl : contributing_blocks) {

  auto xx = compute_block_table_and_bound(root, bl, std::numeric_limits<double>::max());
  if (xx.first == -1) TRIQS_RUNTIME_ERROR << "null block";

  // Calculate matrix, working from right to left (tau = 0->beta)
  // mat =     M_L * evo4 * op_l * evo3 * M_M * evo2 * op_r * evo1 * M_R
  // blocks:  bl<-b4       b4<-b3        b3<-b2       b2<-b1        b1<-bl
  // times:      t4          t_l        t3   t2         t_r         t1
  // Done in three pieces : (M_L * (evo4 * op_l * evo3 * M_M * (evo2 * op_r * evo1 * M_R)))
  auto b_mat = evolve(tau_r, tau2, get_op(node_r) * evolve(tau1, tau_r, compute_M_R(root, tau_r, bl)));
  // If M_M is empty, only evolve using evo2
  auto b_mat_M = compute_M_M(root, tau_l, tau_r, b_mat.b);
  if (!b_mat_M.M.is_empty()) b_mat = evolve(tau3, tau_l, b_mat_M * b_mat);
  b_mat = evolve(tau_l, tau4, get_op(node_l) * b_mat);
  b_mat = compute_M_L(root, tau_l, b_mat.b) * b_mat;
  if (b_mat.b != bl) TRIQS_RUNTIME_ERROR << "check_trace_from_ML_MM_MR: matrix takes b_i " << bl << " to " << b_mat.b << " !";

  // trace(mat * exp(- H * (beta - tmax)) * exp (- H * tmin)) to handle the piece outside of the first-last operators.
  auto dim = get_block_dim(bl);
  for (int u = 0; u < dim; ++u) reconstructed_trace += b_mat.M(u, u) * std::exp(-dtau_beta_zero * get_block_eigenval(bl, u));
 }

 if (reconstructed_trace - true_trace > diff_threshold)
  TRIQS_RUNTIME_ERROR << "check_trace_from_ML_MM_MR: traces do not agree. true trace = " << true_trace
                      << ", while recomputed trace = " << reconstructed_trace;
}

//-------- Compute trace of configuration using only M_M and two operators -------------------------

// Preconditions: chosen operator with index_node_l/r is always first of a pair, 
// i.e. there is at least one operator to the right of op(index_node_r) but op(index_node_l) could be the leftmost
void impurity_trace::check_trace_from_MM(std::vector<node> const& flat_config, int index_node_l, int index_node_r,
                                               bool print) {

 if (print) std::cout << " ... checking trace from MM against trace from root cache matrix " << std::endl;

 // Update the cache fully, i.e. without the Yee trick
 auto w_rw = compute();
 auto true_trace = w_rw.first * w_rw.second; // tr = norm * tr/norm = w * rw

 auto fc_size = flat_config.size();
 auto cyclic_index = [&](int index) { return (index >= 0) ? index % fc_size : (index + fc_size) % fc_size; };
 auto node_r = flat_config[index_node_r];
 auto tau_r = node_r->key;
 auto node_l = flat_config[index_node_l];
 auto tau_l = node_l->key;
 auto root = tree.get_root();
 trace_t reconstructed_trace = 0;
 auto tau1 = flat_config[cyclic_index(index_node_r + 1)]->key;
 auto tau2 = flat_config[cyclic_index(index_node_r - 1)]->key;
 auto tau3 = flat_config[cyclic_index(index_node_l + 1)]->key;
 auto tau4 = flat_config[cyclic_index(index_node_l - 1)]->key;

 for (auto bl : contributing_blocks) {

  auto xx = compute_block_table_and_bound(root, bl, std::numeric_limits<double>::max());
  if (xx.first == -1) TRIQS_RUNTIME_ERROR << "null block";

  auto b1 = compute_block_at_tau(root, tau_r, bl);
  auto b1_from_MR = compute_M_R(root, tau_r, bl).b;
  if (b1 != b1_from_MR)
   TRIQS_RUNTIME_ERROR << "check_trace_from_MM: test 1: blocks do not match up: " << b1 << " and " << b1_from_MR;

  auto bl2 = bl;
  for (int i = fc_size - 1; i >= 0; --i) bl2 = get_op_block_map(flat_config[i]->op, bl2);
  if (bl2 != bl) TRIQS_RUNTIME_ERROR << "check_trace_from_MM: test 2: blocks do not match up: " << b1 << " and " << b1_from_MR;

  // Calculate matrix, working from right to left (tau = 0->beta)
  // mat =   evo1 * M_M * evo4 * op_l * evo3 * M_M * evo2 * op_r
  // blocks: b1    b1<-b4       b4<-b3        b3<-b2       b2<-b1
  // times:  t_r   t1 t4         t_l          t3 t2         t_r

  auto b_mat = evolve(tau_r, tau2, get_op_block_and_matrix(node_r->op, b1));
  // If M_M is empty, only evolve using evo2
  auto b_mat_M = compute_M_M(root, tau_l, tau_r, b_mat.b);
  if (!b_mat_M.M.is_empty()) b_mat = evolve(tau3, tau_l, b_mat_M * b_mat);
  b_mat = evolve(tau_l, tau4, get_op(node_l) * b_mat);
  b_mat_M = compute_M_M(root, tau_r, tau_l, b_mat.b);
  if (!b_mat_M.M.is_empty()) b_mat = evolve(tau1, tau_r, b_mat_M * b_mat);
  if (b_mat.b != b1) TRIQS_RUNTIME_ERROR << "check_trace_from_MM: matrix takes b_i " << b1 << " to " << b_mat.b << " !";

  reconstructed_trace += trace(b_mat.M);
 }

 if (reconstructed_trace - true_trace > diff_threshold)
  TRIQS_RUNTIME_ERROR << "check_trace_from_MM: traces do not agree. true trace = " << true_trace
                      << ", while recomputed trace = " << reconstructed_trace;
}
}
