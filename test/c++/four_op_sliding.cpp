#include "./measure_four_body_corr.hpp"

using namespace cthyb;

int main() {

 template <class T1, class T2> 
 void assert_close(T1 const& A, T2 const& B, double precision) {
  if (std::abs(A - B) > precision) TRIQS_RUNTIME_ERROR << "assert_close error : " << A << "\n" << B;
 }
 const double PRECISION = 1.e-6;

 // define lambdas
 double lamb1 = -1.1;
 double lamb2 = 0.001;
 double lamb3 = 0.5;
 double lamb4 = -2.3;
 double lamb5 = 5.9;

// // n = 1 tests
// assert_close(measure_four_body_corr::compute_evolution_integral(lamb1, lamb2),)
// assert_close(measure_four_body_corr::compute_evolution_integral(lamb1, lamb1),)
//
// // n = 2 tests
// assert_close(measure_four_body_corr::compute_evolution_integral(lamb1, lamb2, lamb3),)
// assert_close(measure_four_body_corr::compute_evolution_integral(lamb1, lamb1, lamb2),)
// assert_close(measure_four_body_corr::compute_evolution_integral(lamb1, lamb1, lamb1),)
//
// // n = 4 tests
//
// assert_close(measure_four_body_corr::compute_evolution_integral(lamb1, lamb2, lamb3, lamb4, lamb5),)
// assert_close(measure_four_body_corr::compute_evolution_integral(lamb1, lamb1, lamb2, lamb3, lamb4),)
// assert_close(measure_four_body_corr::compute_evolution_integral(lamb1, lamb1, lamb2, lamb2, lamb5),)
// assert_close(measure_four_body_corr::compute_evolution_integral(lamb1, lamb1, lamb1, lamb2, lamb3),)
// assert_close(measure_four_body_corr::compute_evolution_integral(lamb1, lamb1, lamb1, lamb2, lamb2),)
// assert_close(measure_four_body_corr::compute_evolution_integral(lamb1, lamb1, lamb1, lamb1, lamb2),)
// assert_close(measure_four_body_corr::compute_evolution_integral(lamb1, lamb1, lamb1, lamb1, lamb1),)
}
