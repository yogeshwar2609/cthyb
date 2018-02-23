#include <triqs/utility/first_include.hpp>
#include <measures/g2.hpp>

using namespace cthyb;

int main() {

  double beta = 2;

  tilde_p_gen gen(beta);

  time_segment seg(beta);
  time_pt tau1 = seg.make_time_pt(1.6);
  time_pt tau2 = seg.make_time_pt(0.8);

  std::cout << "tau1 - tau2 = " << (tau1 - tau2) << std::endl;
  gen.reset(tau1, tau2);
  for(int l = 0; l < 5; ++l)
    std::cout << l << ": " << gen.next() << std::endl;

  std::cout << "tau2 - tau1 = " << (tau2 - tau1) << std::endl;
  gen.reset(tau2, tau1);
  for(int l = 0; l < 5; ++l)
    std::cout << l << ": " << gen.next() << std::endl;

  return 0;
}
