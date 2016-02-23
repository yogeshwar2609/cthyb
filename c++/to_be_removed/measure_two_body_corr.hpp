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
#include <triqs/gfs.hpp>
#include "./qmc_data.hpp"

namespace cthyb {

using namespace triqs::gfs;

// Measure the two body correlator C_ab = < (c+_a c_b) >
struct measure_two_body_corr {
 using mc_sign_type = double;
 using node = impurity_trace::node;

 qmc_data const& data;
 mc_sign_type z;
 qmc_data::trace_t new_atomic_weight, new_atomic_reweighting;
 arrays::array<dcomplex, 2> coefficients;          // Coefficients of op, where op is a quadratic operator
 bool anticommute;                                 // Do the cdag and c operators anticommute?
 dcomplex dens; //DEBUG
 dcomplex dens2; //DEBUG

 measure_two_body_corr(qmc_data const& data, fundamental_operator_set const & fops, many_body_operator const & A, bool anticommute);
 void accumulate(mc_sign_type s);
 void collect_results(triqs::mpi::communicator const& c);
};
}
