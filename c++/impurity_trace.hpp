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
#pragma once
#include "./configuration.hpp"
#include "./atom_diag.hpp"
#include "./solve_parameters.hpp"
#include "triqs/utility/rbt.hpp"
#include "triqs/statistics/histograms.hpp"
//#define PRINT_CONF_DEBUG

using namespace triqs;
using triqs::utility::rb_tree;
using triqs::utility::rbt_insert_error;
using triqs::utility::make_time_pt_beta;
using triqs::utility::make_time_pt_zero;

namespace cthyb {

/********************************************
 Calculate the trace of the impurity problem.
 ********************************************/
class impurity_trace {

 bool use_norm_as_weight;
 bool measure_density_matrix;

 public:
 using trace_t = double;
 // using trace_t = std::complex<double>; TODO

 // construct from the config, the diagonalization of h_loc, and parameters
 impurity_trace(configuration& c, atom_diag const& h_diag, solve_parameters_t const& p);

 ~impurity_trace() {
  cancel_insert_impl();
 } // in case of an exception, we need to remove any trial nodes before cleaning the tree!

 std::pair<double, trace_t> compute(double p_yee = -1, double u_yee = 0);

 // ------- Configuration and h_loc data ----------------

 const configuration* config;                                  // config object does exist longer (temporally) than this object.
 const atom_diag* h_diag;                                      // access to the diagonalization of h_loc
 const int n_orbitals = h_diag->get_fops().size();             // total number of orbital flavours
 const int n_blocks = h_diag->n_blocks();                      //
 const int n_eigstates = h_diag->get_full_hilbert_space_dim(); // size of the hilbert space

 // ------- Trace data ----------------

 private:
 struct bool_and_matrix {
  bool is_valid;
  matrix<double> mat;
 };
 arrays::vector<bool_and_matrix> density_matrix; // density_matrix, by block, with a bool to say if it has been recomputed
 arrays::vector<bool_and_matrix> atomic_rho;     // atomic density matrix (non-normalized)
 double atomic_z;                                // atomic partition function
 double atomic_norm;                             // Frobenius norm of atomic_rho

 public:
 arrays::vector<bool_and_matrix> const& get_density_matrix() const { return density_matrix; }

 // ------------------ Cache data ----------------

 private:
 // The data stored for each node in tree
 struct cache_t {
  double dtau_l = 0, dtau_r = 0;                // difference in tau of this node and left and right sub-trees
  std::vector<int> block_table;                 // number of blocks limited to 2^15
  std::vector<arrays::matrix<double>> matrices; // partial product of operator/time evolution matrices
  std::vector<double> matrix_lnorms;            // -ln(norm(matrix))
  std::vector<bool> matrix_norm_valid;          // is the norm of the matrix still valid?
  cache_t(int n_blocks) : block_table(n_blocks), matrix_lnorms(n_blocks), matrices(n_blocks), matrix_norm_valid(n_blocks) {}
 };

 struct node_data_t {
  op_desc op;
  cache_t cache;
  node_data_t(op_desc op, int n_blocks) : op(op), cache(n_blocks) {}
  void reset(op_desc op_new) { op = op_new; }
 };

 struct block_and_matrix {
  long b;                   // Block index
  arrays::matrix<double> M; // Matrix for block b

  // Multiply two block_and_matrix objects, treating an empty matrix as the identity matrix
  friend block_and_matrix operator*(block_and_matrix const& b_mat1, block_and_matrix const& b_mat2) {
   if ((b_mat1.b == -1) or (b_mat2.b == -1)) return {-1, {}};
   if (b_mat1.M.is_empty()) return b_mat2;
   if (b_mat2.M.is_empty()) return b_mat1;
   return {b_mat1.b, b_mat1.M * b_mat2.M};
  }

  friend block_and_matrix operator*(std::pair<op_desc, const atom_diag*> const& op_hdiag, block_and_matrix const& b_mat) {
   if (b_mat.b == -1) return {-1, {}};
   auto h_diag = op_hdiag.second;
   auto const& op = op_hdiag.first;
   auto b_f = op.dagger ? h_diag->cdag_connection(op.linear_index, b_mat.b) : h_diag->c_connection(op.linear_index, b_mat.b);
   if (b_f == -1) return {-1, {}};
   auto const& op_M = op.dagger ? h_diag->cdag_matrix(op.linear_index, b_mat.b) : h_diag->c_matrix(op.linear_index, b_mat.b);
   if (b_mat.M.is_empty()) return {b_f, op_M};
   return {b_f, op_M * b_mat.M};
  }
 };

#ifdef EXT_DEBUG
#endif
 public: // FIXME make private with accessor
 using rb_tree_t = rb_tree<time_pt, node_data_t, std::greater<time_pt>>;
 using node = rb_tree_t::node;
 rb_tree_t tree; // the red black tree and its nodes

 // ---------------- Cache machinery ----------------
 void update_cache();

 private:
 // The dimension of block b
 int get_block_dim(int b) const { return h_diag->get_block_dim(b); }

 // the i-th eigenvalue of the block b
 double get_block_eigenval(int b, int i) const { return h_diag->get_eigenvalue(b, i); }

 // the minimal eigenvalue of the block b
 double get_block_emin(int b) const { return get_block_eigenval(b, 0); }

 // op, block -> image of the block by op (the operator)
 int get_op_block_map(op_desc op, int b) const {
  return (op.dagger ? h_diag->cdag_connection(op.linear_index, b) : h_diag->c_connection(op.linear_index, b));
 }

 // the matrix of n->op, from block b to its image
 matrix<double> const& get_op_block_matrix(op_desc op, int b) const {
  return (op.dagger ? h_diag->cdag_matrix(op.linear_index, b) : h_diag->c_matrix(op.linear_index, b));
 }

 // FIXME remove this function?
 // exit block and matrix for operator op at entry block b
 block_and_matrix get_op_block_and_matrix(op_desc op, int b) const {
  return (op.dagger ? block_and_matrix{h_diag->cdag_connection(op.linear_index, b), h_diag->cdag_matrix(op.linear_index, b)}
                    : block_and_matrix{h_diag->c_connection(op.linear_index, b), h_diag->c_matrix(op.linear_index, b)});
 }

 // Return op and pointer to h_diag
 std::pair<op_desc, const atom_diag*> get_op(node n) const { return {n->op, h_diag}; }

 // Recursive function for tree traversal
 public: //FIXME
 int compute_block_table(node n, int b);
 private:
 std::pair<int, double> compute_block_table_and_bound(node n, int b, double bound_threshold, bool use_threshold = true);
 block_and_matrix compute_matrix(node n, int b);

 void update_cache_impl(node n);
 void update_dtau(node n);

 bool use_norm_of_matrices_in_cache = true; // When a matrix is computed in cache, its spectral radius replaces the norm estimate

 // integrity check
 void check_cache_integrity(bool print = false);
 void check_cache_integrity_one_node(node n, bool print);
 int check_one_block_table_linear(node n, int b, bool print); // compare block table to that of a linear method (ie. no tree)
 matrix<double> check_one_block_matrix_linear(node n, int b,
                                              bool print); // compare matrix to that of a linear method (ie. no tree)

 public:
 // checks
 double diff_threshold = 1.e-8;
 void check_ML_MM_MR(bool print = false);
 void check_trace_from_ML_MR(std::vector<node> const& flat_config, int index_node, bool print = false);
 void check_trace_from_ML_MM_MR(std::vector<node> const& flat_config, int index_node_l, int index_node_r, bool print = false);

 private:
 /*************************************************************************
  *  Calculate linear matrix products from tree caches from tau_L to tau_R
  *  Compute quantities for subtree entry block b_i, and return subtree exit block b_f and matrix.
  *  Note that the time evolution up to tau = beta and tau = 0 is NOT included for M_L and M_R.
  *  This would need to be added explicitly at point of use
  -------------------------------------------------------------------------
  *  M_L: operator just R of beta to tau_L
  *  M_M: tau_L to tau_R
  *  M_R: tau_R to operator just L of 0
    For a config as follows (x and * are operators, - indicate time evolution):
    |-- * --- * * --- x ------ * - * -- * ---- * ---- x ----- * -- * -- * -|
  beta              op_L                             op_R                  0
                   tau_L                            tau_R                  0
    |   <--M_L-->              <----- M_M ----->              <-- M_R -->  |
  *************************************************************************/

 block_and_matrix get_cache_matrix(node n, int b) {
  if ((b == -1) or (n == nullptr)) return {b, {}};
  return {n->cache.block_table[b], n->cache.matrices[b]};
 }

 // Evolve M from tau1 to tau2
 // Compute e^-H(tau2-tau1) * M
 // Does not handle case where M is empty
 block_and_matrix evolve(time_pt tau1, time_pt tau2, block_and_matrix b_mat) {
  assert(double(tau1) < double(tau2));
  assert(!b_mat.M.is_empty());
  assert(b_mat.b != -1);
  auto& M = b_mat.M;
  auto dim1 = first_dim(M); // = get_block_dim(b_mat.b)
  auto dim2 = second_dim(M);
  auto _ = arrays::range();
  double dtau = double(tau2 - tau1);
  if ((dim1 == 1) and (dim2 == 1))
   M(0, 0) *= std::exp(-dtau * get_block_eigenval(b_mat.b, 0));
  else
   for (int i = 0; i < dim1; ++i) M(i, _) *= std::exp(-dtau * get_block_eigenval(b_mat.b, i));
  return {b_mat.b, std::move(M)};
 }

 // Compute [\int_tau1^\tau2 dtau e^-H_{b_f}(tau2 - tau) * op_{b_i->b_f}(tau) * e^-H_{b_i}(tau - tau1)]
 block_and_matrix int_evo_op_evo(int b_i, time_pt tau1, time_pt tau2, op_desc const& op) {
  if (b_i == -1) return {-1, {}};
  auto b_f = get_op_block_map(op, b_i);
  if (b_f == -1) return {-1, {}};
  auto M = get_op_block_matrix(op, b_i);
  double dtau = double(tau2 - tau1);
  auto dim1 = get_block_dim(b_i);
  auto dim2 = get_block_dim(b_f);
  for (int i = 0; i < dim2; ++i) {
   auto lamb2 = get_block_eigenval(b_f, i);
   for (int j = 0; j < dim1; ++j) {
    auto lamb1 = get_block_eigenval(b_i, j);
    auto rhs = ((std::abs(lamb1 - lamb2) > 1.e-10) ? (std::exp(-dtau * lamb1) - std::exp(-dtau * lamb2)) / (lamb2 - lamb1)
                                                   : std::exp(-dtau * lamb1) * dtau);
    M(i, j) *= rhs;
   }
  }
  return {b_f, std::move(M)};
 }

 // min/max of left/right. Precondition: n->left/n->right is not null
 time_pt tau_maxL(node n) const { return tree.max_key(n->left); }
 time_pt tau_minR(node n) const { return tree.min_key(n->right); }

 // FIXME why don't these work?!
 // constexpr auto compare = tree.get_compare(); //error: non-static data member declared auto
 // auto compare = std::greater<time_pt>; //error: non-static data member declared auto
 // bool compare = [](time_pt& tau1, time_pt& tau2) { return std::greater<time_pt>(tau1, tau2); };
 // bool compare = [](time_pt tau1, time_pt tau2) { return (tau1 > tau2); }; // error: expression cannot be used as a function

 // For three functions below, remember that tree.get_compare(x,y) = (x left of y in tree)

 // FIXME -- remove this?
 bool ML_MR_MM_DEBUG = false;

 // Compute matrix products of the subtree, with the time evolution between operator
 // for the nodes with key > tau (i.e. does not include the operator at key == tau).
 // Precondition : n is not null
 // NB :does NOT include time evolution outside of the tree ...
 // b_i is the entry block
 block_and_matrix compute_M_R(node n, time_pt tau, int b_i) {
  if (ML_MR_MM_DEBUG) std::cout << "MR n->key: " << n->key << " tau: " << tau << std::endl;
  assert(n != nullptr);
  if (n->key == tau) return get_cache_matrix(n->right, b_i);
  if (tree.get_compare()(n->key, tau))
   return n->right ? compute_M_R(n->right, tau, b_i) : block_and_matrix{b_i, {}}; // n->key < tau
  // n->key > tau : M_R(n->left) * evo * op(n) * evo * cache(n->right)
  auto b_mat =
      (n->right) ? get_op(n) * evolve(tau_minR(n), n->key, get_cache_matrix(n->right, b_i)) : get_op_block_and_matrix(n->op, b_i);
  if (!n->left) return b_mat;
  auto b_mat_l = compute_M_R(n->left, tau, b_mat.b);
  return b_mat_l.M.is_empty() ? b_mat : b_mat_l * evolve(n->key, tau_maxL(n), std::move(b_mat));
 }

 // Reverse as compute_M_R.
 // Compute matrix products of the subtree, with the time evolution between operator
 // for the nodes with key < tau (i.e. does not include the operator at key == tau).
 // Precondition : n is not null
 // NB :does NOT include time evolution outside of the tree ...
 // b_i is the entry block
 block_and_matrix compute_M_L(node n, time_pt tau, int b_i) {
  if (ML_MR_MM_DEBUG) std::cout << "ML n->key: " << n->key << " tau: " << tau << std::endl;
  assert(n != nullptr);
  if (n->key == tau) return get_cache_matrix(n->left, b_i);
  if (tree.get_compare()(tau, n->key))
   return n->left ? compute_M_L(n->left, tau, b_i) : block_and_matrix{b_i, {}}; // n->key > tau
  // n->key < tau : cache(n->left) * evo * op(n) * evo * M_L(n->right)
  block_and_matrix b_mat;
  if (!n->right) {
   b_mat = get_op_block_and_matrix(n->op, b_i);
  } else {
   auto b_mat_r = compute_M_L(n->right, tau, b_i);
   b_mat = b_mat_r.M.is_empty() ? get_op_block_and_matrix(n->op, b_i) : get_op(n) * evolve(tau_minR(n), n->key, b_mat_r);
  }
  if (!n->left) return b_mat;
  return get_cache_matrix(n->left, b_mat.b) * evolve(n->key, tau_maxL(n), std::move(b_mat));
 }

 // Compute matrix product of operators and time evolution from operator to right of tau_l to operator to the left of tau_r
 // Important -- does NOT include operator at tau_l and tau_r nor the time evolution to tau_l or tau_r!
 // compute_M_L(n, tau_l, b_i) => compute_M_M(n, beta, tau_l, b_i)
 // compute_M_R(n, tau_r, b_i) => compute_M_M(n, tau_r, 0, b_i)
 block_and_matrix compute_M_M(node n, time_pt tau_l, time_pt tau_r, int b_i) {
  if (ML_MR_MM_DEBUG) std::cout << "MM n->key: " << n->key << " tauL: " << tau_l << " tauR: " << tau_r << std::endl;
  if (n == nullptr) return {b_i, {}};                                                      // Cannot guarantee that n is not null
  if (!tree.get_compare()(tau_l, n->key)) return compute_M_M(n->right, tau_l, tau_r, b_i); // n->key < tau_l
  if (!tree.get_compare()(n->key, tau_r)) return compute_M_M(n->left, tau_l, tau_r, b_i);  // n->key > tau_r
  // n->key in ] tau_l, tau_r [
  block_and_matrix b_mat;
  if (!n->right) {
   b_mat = get_op_block_and_matrix(n->op, b_i);
  } else {
   auto b_mat_r = compute_M_L(n->right, tau_r, b_i);
   b_mat = b_mat_r.M.is_empty() ? get_op_block_and_matrix(n->op, b_i) : get_op(n) * evolve(tau_minR(n), n->key, b_mat_r);
  }
  if (!n->left) return b_mat;
  auto b_mat_l = compute_M_R(n->left, tau_l, b_mat.b);
  return b_mat_l.M.is_empty() ? b_mat : b_mat_l * evolve(n->key, tau_maxL(n), std::move(b_mat));
 }

 public:
 time_pt _beta = make_time_pt_beta(config->beta());
 time_pt _zero = make_time_pt_zero(config->beta());
 std::vector<int> contributing_blocks; // Which blocks contributed to the trace in the last call of compute()?

 //*********************************************************************************
 // Compute 1) trace for glued configuratons with op_l and op_r shifted to their
 //            respective right neighbours (second operator of the chosen pair)
 //         2) integral of trace for sliding times of op_l and op_r; they can
 //            slide between neighbouring operators, but cannot exceed beta/0
 //*********************************************************************************
 std::pair<trace_t, trace_t> compute_sliding_trace_integral(std::vector<node> const& flat_config, int index_node_l,
                                                            int index_node_r, std::vector<int> const& blocks) {

  // Preconditions: chosen operator with index_node is always first of a pair,
  // i.e. there is at least one operator to the right of op(index_node)
  auto is_first_config = (index_node_l == 0);
  auto node_r = flat_config[index_node_r];
  auto tau_r = node_r->key;
  auto node_l = flat_config[index_node_l];
  auto tau_l = node_l->key;
  auto root = tree.get_root();
  auto conf_size = flat_config.size();
  // size of tau piece outside first-last operators: beta - tmax + tmin ! the tree is in REVERSE order
  double dtau_beta = config->beta() - tree.min_key();
  double dtau_zero = double(tree.max_key());
  auto dtau_beta_zero = (is_first_config ? dtau_zero : dtau_beta + dtau_zero);
  trace_t sliding_trace = 0, int_trace = 0;

  // If operator is the rightmost in config (closest to tau=0), tau1 = 0
  // operator cannot actually be rightmost as we are always shifting first operator in a pair
  // If operator is the leftmost in config (closest to tau=beta), tau4 = beta
  // then do not evolve to beta at the end!
  auto tau1 = ((index_node_r + 1 == conf_size) ? _zero : flat_config[index_node_r + 1]->key);
  auto tau2 = flat_config[index_node_r - 1]->key;
  auto tau3 = flat_config[index_node_l + 1]->key;
  auto tau4 = (is_first_config ? _beta : flat_config[index_node_l - 1]->key);

  // FIXME leave blocks as a input param or use contributing blocks by default?
  for (auto bl : blocks) {

   //FIXME
   compute_matrix(root, bl);

   // Matrix with c_dag,c operators stuck together
   // mat =     M_L * evo34 * op_l * M_M * evo12 * op_r * M_R
   // blocks:  bl<-b4       b4<-b3  b3<-b2        b2<-b1 b1<-bl
   // times:      t4            t_l=t3  t2           t_r=t1
   // Done in three pieces : (M_L * (evo34 * op_l * M_M * (evo12 * op_r * M_R)))
   auto b_mat_R = compute_M_R(root, tau_r, bl); // b_mat_R.b = b1
   auto trace_mat = evolve(tau1, tau2, get_op(node_r) * b_mat_R);
   auto b_mat_M = compute_M_M(root, tau_l, tau_r, trace_mat.b); // b_mat_M.b = b3
   trace_mat = evolve(tau3, tau4, get_op(node_l) * (b_mat_M * trace_mat));
   auto b_mat_L = compute_M_L(root, tau_l, trace_mat.b);
   trace_mat = b_mat_L * trace_mat;

   // Matrix for trace normalisation with integrals
   // mat =     M_L * int [evo4 * op_l * evo3] * M_M * int [evo2 * op_r * evo1] * M_R
   // blocks:  bl<-b4            b4<-b3        b3<-b2             b2<-b1         b1<-bl
   auto int_r = int_evo_op_evo(b_mat_R.b, tau1, tau2, node_r->op); // \int evo2 * op_r * evo1
   auto int_l = int_evo_op_evo(b_mat_M.b, tau3, tau4, node_l->op); // \int evo4 * op_r * evo3
   auto int_mat = b_mat_L * (int_l * (b_mat_M * (int_r * b_mat_R)));

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


 //***********************************************************************
 // Compute 1) trace for glued configuratons and
 //         2) integral of trace for sliding times of op
 //***********************************************************************
 std::pair<trace_t, trace_t> compute_sliding_trace_integral_one_pair(std::vector<node> const& flat_config, int index_node,
                                                                     std::vector<int> const& blocks) {

  auto is_first_config = (index_node == 0);
  auto n = flat_config[index_node];
  auto tau = n->key;
  auto root = tree.get_root();
  auto conf_size = flat_config.size();
  // size of tau piece outside first-last operators: beta - tmax + tmin ! the tree is in REVERSE order
  auto dtau_beta = _beta - tree.min_key();
  auto dtau_zero = tree.max_key();
  double dtau_beta_zero = (is_first_config ? double(dtau_zero) : double(dtau_beta + dtau_zero));
  trace_t sliding_trace = 0, int_trace = 0;

  // If operator is the rightmost in config (closest to tau=0), tau1 = 0
  // operator cannot actually be rightmost as we are always shifting first operator in a pair
  // If operator is the leftmost in config (closest to tau=beta), tau2 = beta
  // then do not evolve to beta at the end!
  auto tau1 = ((index_node + 1 == conf_size) ? _zero : flat_config[index_node + 1]->key);
  auto tau2 = ((index_node == 0) ? _beta : flat_config[index_node - 1]->key);

  for (auto bl : blocks) {

   // Matrix with c_dag,c operators stuck together
   // mat =     M_L * evo12 *  op  * M_R
   // blocks:  bl<-b2        b2<-b1 b1<-bl
   // Done in three pieces : (M_L * (evo12 * op_r * (M_R)))
   auto b_mat_R = compute_M_R(root, tau, bl); // b_mat_R.b = b1
   auto trace_mat = evolve(tau1, tau2, get_op(n) * b_mat_R);
   auto b_mat_L = compute_M_L(root, tau, trace_mat.b);
   trace_mat = b_mat_L * trace_mat;

   // Matrix for trace normalisation with integrals
   // mat =     M_L * int [evo2 * op_r * evo1] * M_R
   // blocks:  bl<-b2            b2<-b1         b1<-bl
   auto int_mat = b_mat_L * (int_evo_op_evo(b_mat_R.b, tau1, tau2, n->op) * b_mat_R);

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

 private:
 /*************************************************************************
  *  Ordinary binary search tree (BST) insertion of the trial nodes
  *************************************************************************/
 // We have a set of trial nodes, which we can glue, un-glue in the tree at will.
 // This avoids allocations.

 int tree_size = 0; // size of the tree +/- the added/deleted node

 // make a new detached black node
 std::shared_ptr<rb_tree_t::node_t> make_new_node() const {
  return std::make_shared<rb_tree_t::node_t>(time_pt{}, node_data_t{{}, n_blocks}, false, 1);
 }

 // a pool of trial nodes, ready to be glued in the tree. Max 4 to allow for double insertions
 std::vector<std::shared_ptr<rb_tree_t::node_t>> trial_nodes = {make_new_node(), make_new_node(), make_new_node(),
                                                                make_new_node()};

 // for each inserted node, need to know {parent_of_node,child_is_left}
 std::vector<std::pair<node, bool>> inserted_nodes = {{nullptr, false}, {nullptr, false}, {nullptr, false}, {nullptr, false}};
 int trial_node_index = -1; // the index of the next available node in trial_nodes

 node try_insert_impl(node h, node n) { // implementation
  if (h == nullptr) return n;
  if (h->key == n->key) throw rbt_insert_error{};
  auto smaller = tree.get_compare()(n->key, h->key);
  if (smaller)
   h->left = try_insert_impl(h->left, n);
  else
   h->right = try_insert_impl(h->right, n);
  if (inserted_nodes[trial_node_index].first == nullptr) inserted_nodes[trial_node_index] = {h, smaller};
  h->modified = true;
  return h;
 }

 // unlink all glued trial nodes
 void cancel_insert_impl() {
  for (int i = 0; i <= trial_node_index; ++i) {
   auto& r = inserted_nodes[i];
   if (r.first != nullptr) (r.second ? r.first->left : r.first->right) = nullptr;
  }
  if (tree_size == trial_node_index + 1) tree.get_root() = nullptr;
 }

 /*************************************************************************
  * Node Insertion
  *************************************************************************/

 public:
 // Put a trial node at tau for operator op using an ordinary BST insertion (ie. not red black)
 void try_insert(time_pt const& tau, op_desc const& op) {
  if (trial_node_index > 3) TRIQS_RUNTIME_ERROR << "Error : more than 4 insertions ";
  auto& root = tree.get_root();
  inserted_nodes[++trial_node_index] = {nullptr, false};
  node n = trial_nodes[trial_node_index].get(); // get the next available node
  n->reset(tau, op);                            // change the time and op of the node
  root = try_insert_impl(root, n);              // insert it using a regular BST, no red black
  tree_size++;
 }

 // Remove all trial nodes from the tree
 void cancel_insert() {
  cancel_insert_impl();
  trial_node_index = -1;
  tree_size = tree.size();
  tree.clear_modified();
  check_cache_integrity();
 }

 // confirm the insertion of the nodes, with red black balance
 void confirm_insert() {
  cancel_insert_impl();                         // remove BST inserted nodes
  for (int i = 0; i <= trial_node_index; ++i) { // then reinsert the nodes in in balanced RBT
   node n = trial_nodes[i].get();
   tree.insert(n->key, {n->op, n_blocks});
  }
  trial_node_index = -1;
  update_cache();
  tree_size = tree.size();
  tree.clear_modified();
  check_cache_integrity();
 }

 /*************************************************************************
  * Node Removal
  *************************************************************************/
 private:
 std::vector<node> removed_nodes;
 std::vector<time_pt> removed_keys;

 public:
 // Find and mark as deleted the nth operator with fixed dagger and block_index
 // n=0 : first operator, n=1, second, etc...
 time_pt try_delete(int n, int block_index, bool dagger) noexcept {
  // traverse the tree, looking for the nth operator of the correct dagger, block_index
  int i = 0;
  node x = find_if(tree, [&](node no) {
   if (no->op.dagger == dagger && no->op.block_index == block_index) ++i;
   return i == n + 1;
  });
  removed_nodes.push_back(x);             // store the node
  removed_keys.push_back(x->key);         // store the key
  tree.set_modified_from_root_to(x->key); // mark all nodes on path from node to root as modified
  x->delete_flag = true;                  // mark the node for deletion
  tree_size--;
  return x->key;
 }

 // Clean all the delete flags
 void cancel_delete() {
  for (auto& n : removed_nodes) n->delete_flag = false;
  removed_nodes.clear();
  removed_keys.clear();
  tree_size = tree.size();
  tree.clear_modified();
  check_cache_integrity();
 }

 // Confirm deletion: the nodes flagged for deletion are truly deleted
 void confirm_delete() {
  for (auto& k : removed_keys) tree.delete_node(k); // CANNOT use the node here
  removed_nodes.clear();
  removed_keys.clear();
  update_cache();
  tree_size = tree.size();
  tree.clear_modified();
  check_cache_integrity();
 }

 /*************************************************************************
  * Node shift (=insertion+deletion)
  *************************************************************************/

 // No try_shift implemented for changing operator flavour, as needed in e.g. move_shift.
 // Use combination of try_insert and try_delete instead.

 // Cancel the shift
 void cancel_shift() {

  // Inserted nodes
  cancel_insert_impl();
  trial_node_index = -1;

  // Deleted nodes
  for (auto& n : removed_nodes) n->delete_flag = false;
  removed_nodes.clear();
  removed_keys.clear();

  tree_size = tree.size();
  tree.clear_modified();
  check_cache_integrity();
 }

 // Confirm the shift of the node, with red black balance
 void confirm_shift() {

  // Inserted nodes
  cancel_insert_impl();                         //  first remove BST inserted nodes
  for (int i = 0; i <= trial_node_index; ++i) { //  then reinsert the nodes used for real in rb tree
   node n = trial_nodes[i].get();
   tree.insert(n->key, {n->op, n_blocks});
  }
  trial_node_index = -1;

  // Deleted nodes
  for (auto& k : removed_keys) tree.delete_node(k); // CANNOT use the node here
  removed_nodes.clear();
  removed_keys.clear();

  // update cache only at the end
  update_cache();
  tree_size = tree.size();
  tree.clear_modified();
  check_cache_integrity();
 }

 // FIXME REMOVE THIS COMPLETELY?
 // /*************************************************************************
 //  * Node time shift (=change only time of node)
 //  *************************************************************************/
 // private:
 // std::vector<std::pair<node, time_pt>> shifted_nodes_times;
 //
 // public:
 // // These set of routines (intended for a measure) do NOT go by the same
 // // structure as those above (intended for use in a move).
 //
 // // Shift an operator to a different given tau.
 // // Note: the operator itself is unchanged!
 // void try_time_shift(node n, time_pt const& tau) {
 //  shifted_nodes_times.push_back({n, n->key});
 //  n->key = tau;
 //  tree.set_modified_from_root_to(n->key); // mark all nodes on path from node to root as modified
 // }
 //
 // // Cancel the shift
 // void cancel_time_shift() {
 //  // All this needs to do is change time on node
 //  for (auto& n_t : shifted_nodes_times) n_t.first->key = n_t.second;
 //  shifted_nodes_times.clear();
 //  tree_size = tree.size(); // should be unchanged
 //  tree.clear_modified();
 //  check_cache_integrity();
 // }
 //
 // // Confirm the shift of the node
 // // Not implemented as not needed for a measure

 private:
 // ---------------- Histograms ----------------
 struct histograms_t {

  histograms_t(int n_subspaces) : n_subspaces(n_subspaces){};
  int n_subspaces;

  // How many block non zero at root of the tree
  statistics::histogram n_block_at_root = {n_subspaces, "histo_n_block_at_root.dat"};

  // how many block kept after the truncation with the bound
  statistics::histogram n_block_kept = {n_subspaces, "histo_n_block_kept.dat"};

  // What is the dominant block in the trace computation ? Sorted by number or energy
  statistics::histogram dominant_block_bound = {n_subspaces, "histo_dominant_block_bound.dat"};
  statistics::histogram dominant_block_trace = {n_subspaces, "histo_dominant_block_trace.dat"};
  statistics::histogram_segment_bin dominant_block_energy_bound = {0, 100, 100, "histo_dominant_block_energy_bound.dat"};
  statistics::histogram_segment_bin dominant_block_energy_trace = {0, 100, 100, "histo_dominant_block_energy_trace.dat"};

  // Various ratios : trace/bound, trace/first term of the trace, etc..
  statistics::histogram_segment_bin trace_over_norm = {0, 1.5, 100, "histo_trace_over_norm.dat"};
  statistics::histogram_segment_bin trace_abs_over_norm = {0, 1.5, 100, "histo_trace_abs_over_norm.dat"};
  statistics::histogram_segment_bin trace_over_trace_abs = {0, 1.5, 100, "histo_trace_over_trace_abs.dat"};
  statistics::histogram_segment_bin trace_over_bound = {0, 1.5, 100, "histo_trace_over_bound.dat"};
  statistics::histogram_segment_bin trace_first_over_sec_term = {0, 1.0, 100, "histo_trace_first_over_sec_term.dat"};
  statistics::histogram_segment_bin trace_first_term_trace = {0, 1.0, 100, "histo_trace_first_term_trace.dat"};
 };
 std::unique_ptr<histograms_t> histo;
};
}
