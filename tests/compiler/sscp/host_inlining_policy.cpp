// REQUIRES: system-linux
// RUN: %acpp %s -o %t --acpp-targets=generic -O3 -ffp-contract=off
// RUN: rm -rf %t.appdb
// RUN: rm -f %t.llvm.ll %t.forced.ll %t.budget.ll %t.root.ll %t.growth.ll %t.low.ll %t.high.ll
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_APPDB_DIR=%t.appdb ACPP_JITOPT_HOST_INLINING_POLICY=llvm ACPP_S2_DUMP_IR_JIT_OPTIMIZATIONS=%t.llvm.ll %t
// RUN: FileCheck %s --check-prefix=LLVM --implicit-check-not='@{{.*}}indexHelper' --input-file=%t.llvm.ll
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_APPDB_DIR=%t.appdb ACPP_JITOPT_HOST_INLINING_POLICY=llvm ACPP_JITOPT_HOST_FORCE_INLINE=outerHelper ACPP_S2_DUMP_IR_JIT_OPTIMIZATIONS=%t.forced.ll %t
// RUN: FileCheck %s --check-prefix=FORCED --implicit-check-not='@{{.*}}outerHelper' --input-file=%t.forced.ll
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_APPDB_DIR=%t.appdb ACPP_JITOPT_HOST_INLINE_BUDGET=0 ACPP_S2_DUMP_IR_FINAL=%t.budget.ll %t
// RUN: FileCheck %s --check-prefix=BUDGET --input-file=%t.budget.ll
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_APPDB_DIR=%t.appdb ACPP_JITOPT_HOST_INLINE_BUDGET=0 ACPP_JITOPT_HOST_INLINE_ROOT_BUDGET=10 ACPP_S2_DUMP_IR_FINAL=%t.root.ll %t
// RUN: FileCheck %s --check-prefix=ROOT --input-file=%t.root.ll
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_APPDB_DIR=%t.appdb ACPP_JITOPT_HOST_INLINE_BUDGET=0 ACPP_JITOPT_HOST_INLINE_GROWTH=1 ACPP_S2_DUMP_IR_FINAL=%t.growth.ll %t
// RUN: FileCheck %s --check-prefix=GROWTH --implicit-check-not='@{{.*}}outerHelper' --input-file=%t.growth.ll
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_APPDB_DIR=%t.appdb ACPP_JITOPT_HOST_INLINING_POLICY=llvm ACPP_JITOPT_HOST_INLINE_THRESHOLD=0 ACPP_S2_DUMP_IR_FINAL=%t.low.ll %t
// RUN: FileCheck %s --check-prefix=LOW --input-file=%t.low.ll
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_APPDB_DIR=%t.appdb ACPP_JITOPT_HOST_INLINING_POLICY=llvm ACPP_JITOPT_HOST_INLINE_THRESHOLD=1000 ACPP_S2_DUMP_IR_FINAL=%t.high.ll %t
// RUN: FileCheck %s --check-prefix=HIGH --implicit-check-not='@{{.*}}heavyHelper' --input-file=%t.high.ll
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_APPDB_DIR=%t.appdb ACPP_JITOPT_HOST_INLINE_BUDGET=0 ACPP_JITOPT_HOST_ATTRIBUTOR=1 %t

#include <sycl/sycl.hpp>

#include <cmath>

[[gnu::noinline]] static double innerHelper(double x) {
  return sycl::fma(x, x, 1.0);
}

[[gnu::noinline]] static double outerHelper(double x, unsigned iterations) {
  for (unsigned i = 0; i < iterations; ++i) {
    x = innerHelper(x);
    x = sycl::fma(x, 0.5, 0.25);
  }
  return x;
}

[[gnu::noinline]] static double heavyHelper(double x, double y) {
#pragma clang loop unroll(full)
  for (int i = 1; i <= 30; ++i)
    x = sycl::fma(x, y, static_cast<double>(i));
  return x;
}

[[gnu::noinline]] static unsigned indexHelper(sycl::nd_item<1> item) {
  return item.get_global_linear_id();
}

int main() {
  sycl::queue queue;
  auto *values = sycl::malloc_shared<double>(8, queue);
  auto *heavy = sycl::malloc_shared<double>(8, queue);
  for (unsigned i = 0; i < 8; ++i)
    values[i] = 0.5;

  queue.parallel_for(sycl::nd_range<1>{8, 4}, [=](sycl::nd_item<1> item) {
    const unsigned i = indexHelper(item);
    const double x = values[i];
    values[i] = outerHelper(x, i) + innerHelper(x) + innerHelper(x + 1.0);
    heavy[i] = heavyHelper(x, x + 1.0) + heavyHelper(x + 1.0, x);
  }).wait_and_throw();

  double heavyA = 0.5;
  double heavyB = 1.5;
  for (int i = 1; i <= 30; ++i) {
    heavyA = std::fma(heavyA, 1.5, static_cast<double>(i));
    heavyB = std::fma(heavyB, 0.5, static_cast<double>(i));
  }
  bool correct = true;
  for (unsigned i = 0; i < 8; ++i) {
    double x = 0.5;
    for (unsigned j = 0; j < i; ++j)
      x = std::fma(std::fma(x, x, 1.0), 0.5, 0.25);
    correct &= values[i] == x + 1.25 + 3.25;
    correct &= heavy[i] == heavyA + heavyB;
  }
  sycl::free(values, queue);
  sycl::free(heavy, queue);
  return correct ? 0 : 1;
}

// LLVM: call {{.*}}@{{.*}}outerHelper
// FORCED: call {{.*}}@{{.*}}innerHelper

// BUDGET-LABEL: define {{.*}}@{{.*}}outerHelper
// BUDGET: call {{.*}}@{{.*}}innerHelper

// ROOT-LABEL: define {{.*}}@{{.*}}outerHelper
// ROOT-NOT: call {{.*}}@{{.*}}innerHelper
// ROOT: ret double
// ROOT-LABEL: define {{.*}}@{{.*}}__acpp_sscp_kernel
// ROOT-DAG: call {{.*}}@{{.*}}outerHelper
// ROOT-DAG: call {{.*}}@{{.*}}innerHelper

// GROWTH: call {{.*}}@{{.*}}innerHelper

// LOW: call {{.*}}@{{.*}}heavyHelper
// HIGH: define {{.*}}@{{.*}}__acpp_sscp_kernel
