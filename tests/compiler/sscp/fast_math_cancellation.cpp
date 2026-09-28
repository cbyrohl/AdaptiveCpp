// RUN: %acpp --acpp-targets=generic --acpp-dryrun -O3 -ffast-math -fno-fast-math -c %s | FileCheck %s --check-prefix=PRECISE
// RUN: %acpp --acpp-targets=generic --acpp-dryrun -O3 -ffast-math -fno-fast-math -ffast-math -c %s | FileCheck %s --check-prefix=FAST
// RUN: %acpp --acpp-targets=generic --acpp-dryrun -Ofast -fno-fast-math -c %s | FileCheck %s --check-prefix=PRECISE
// RUN: %acpp --acpp-targets=generic --acpp-dryrun -fno-fast-math -Ofast -c %s | FileCheck %s --check-prefix=FAST
// RUN: %acpp --acpp-targets=generic --acpp-dryrun -Ofast -O3 -c %s | FileCheck %s --check-prefix=PRECISE
// RUN: %acpp --acpp-targets=generic --acpp-dryrun -O3 -Ofast -c %s | FileCheck %s --check-prefix=FAST
// RUN: %acpp --acpp-targets=generic --acpp-dryrun -ffast-math -fhonor-nans -c %s | FileCheck %s --check-prefix=PRECISE
// RUN: %acpp --acpp-targets=generic --acpp-dryrun -ffast-math -fhonor-nans -fno-honor-nans -c %s | FileCheck %s --check-prefix=FAST
// FAST: -acpp-sscp-kernel-opts=fast-math
// PRECISE-NOT: -acpp-sscp-kernel-opts=fast-math
// RUN: python3 %S/fast_math_driver.py
// RUN: %acpp %s -o %t --acpp-targets=generic -O3 -ffast-math -fno-fast-math
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -Ofast -fno-fast-math
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3 -ffast-math -fno-finite-math-only
// RUN: %t | FileCheck %s

#include <sycl/sycl.hpp>
#include <iostream>
#include <limits>
#include "common.hpp"

template<class T>
bool check_classification() {
  T input[] = {T{1}, std::numeric_limits<T>::infinity(),
               -std::numeric_limits<T>::infinity(),
               std::numeric_limits<T>::quiet_NaN()};
  int output[4] = {};
  {
    sycl::buffer<T> in{input, sycl::range<1>{4}};
    sycl::buffer<int> out{output, sycl::range<1>{4}};
    auto q = get_queue();
    q.submit([&](sycl::handler& h) {
      auto a = in.template get_access<sycl::access::mode::read>(h);
      auto b = out.template get_access<sycl::access::mode::write>(h);
      h.parallel_for(sycl::range<1>{4}, [=](sycl::id<1> i) {
        b[i] = sycl::isfinite(a[i]);
      });
    }).wait_and_throw();
  }
  return output[0] && !output[1] && !output[2] && !output[3];
}

int main() {
  bool floats = check_classification<float>();
  bool doubles = check_classification<double>();
  // CHECK: float: 1 double: 1
  std::cout << "float: " << floats << " double: " << doubles << '\n';
  return floats && doubles ? 0 : 1;
}
