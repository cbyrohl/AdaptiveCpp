// REQUIRES: system-linux
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: rm -rf %t.appdb
// RUN: rm -f %t.default.ll %t.selected.ll %t.warm.ll
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_APPDB_DIR=%t.appdb ACPP_JITOPT_HOST_NO_INLINE= ACPP_S2_DUMP_IR_FINAL=%t.default.ll %t
// RUN: FileCheck %s --check-prefix=DEFAULT --input-file=%t.default.ll
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_APPDB_DIR=%t.appdb ACPP_JITOPT_HOST_NO_INLINE="selectedHelper, indexHelper, barrierHelper" ACPP_S2_DUMP_IR_FINAL=%t.selected.ll %t
// RUN: FileCheck %s --check-prefix=SELECTED --implicit-check-not=indexHelper --implicit-check-not=barrierHelper --input-file=%t.selected.ll
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_APPDB_DIR=%t.appdb ACPP_JITOPT_HOST_NO_INLINE="selectedHelper, indexHelper, barrierHelper" ACPP_S2_DUMP_IR_FINAL=%t.warm.ll %t
// RUN: test ! -e %t.warm.ll

#include <sycl/sycl.hpp>

[[gnu::noinline]] static double selectedHelper(double x) {
  return sycl::fma(x, x, 1.0);
}

[[gnu::noinline]] static unsigned indexHelper(sycl::nd_item<1> item) {
  return item.get_global_linear_id();
}

[[gnu::noinline]] static void barrierHelper(sycl::nd_item<1> item) {
  sycl::group_barrier(item.get_group());
}

int main() {
  sycl::queue queue;
  auto *values = sycl::malloc_shared<double>(8, queue);
  for (unsigned i = 0; i < 8; ++i)
    values[i] = i;

  queue.parallel_for(sycl::nd_range<1>{8, 4}, [=](sycl::nd_item<1> item) {
    const unsigned i = indexHelper(item);
    const double result = selectedHelper(values[i]);
    barrierHelper(item);
    values[i] = result;
  }).wait_and_throw();

  bool correct = true;
  for (unsigned i = 0; i < 8; ++i)
    correct &= values[i] == i * i + 1;
  sycl::free(values, queue);
  return correct ? 0 : 1;
}

// DEFAULT-NOT: selectedHelper
// DEFAULT: define {{.*}}@{{.*}}__acpp_sscp_kernel
// DEFAULT-NOT: selectedHelper

// SELECTED: call {{.*}}@{{.*}}selectedHelper
