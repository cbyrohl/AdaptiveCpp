// clang-format off
// REQUIRES: system-linux
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: rm -rf %t.appdb
// RUN: rm -f %t.default.ll %t.o1.ll %t.o3.ll %t.warm.ll
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_APPDB_DIR=%t.appdb ACPP_JITOPT_HOST_PRE_CBS_OPT_LEVEL= ACPP_S2_DUMP_IR_FULL_OPTIMIZATIONS=%t.default.ll %t
// RUN: test -s %t.default.ll
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_APPDB_DIR=%t.appdb ACPP_JITOPT_HOST_PRE_CBS_OPT_LEVEL=1 ACPP_S2_DUMP_IR_FULL_OPTIMIZATIONS=%t.o1.ll %t
// RUN: test ! -e %t.o1.ll
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_APPDB_DIR=%t.appdb ACPP_JITOPT_HOST_PRE_CBS_OPT_LEVEL=3 ACPP_S2_DUMP_IR_FULL_OPTIMIZATIONS=%t.o3.ll %t
// RUN: test -s %t.o3.ll
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_APPDB_DIR=%t.appdb ACPP_JITOPT_HOST_PRE_CBS_OPT_LEVEL=3 ACPP_S2_DUMP_IR_FULL_OPTIMIZATIONS=%t.warm.ll %t
// RUN: test ! -e %t.warm.ll
// RUN: not env ACPP_VISIBILITY_MASK=omp ACPP_APPDB_DIR=%t.appdb ACPP_JITOPT_HOST_PRE_CBS_OPT_LEVEL=2 %t 2>&1 | FileCheck %s
// clang-format on

#include <sycl/sycl.hpp>

#include <exception>
#include <iostream>

int main() {
  constexpr unsigned group_size = 4;
  constexpr unsigned size = 2 * group_size;
  sycl::queue queue{[](sycl::exception_list errors) {
    if (!errors.empty())
      std::rethrow_exception(errors.front());
  }};
  auto *values = sycl::malloc_shared<unsigned>(size, queue);

  try {
    queue
        .submit([&](sycl::handler &cgh) {
          sycl::local_accessor<unsigned, 1> scratch{group_size, cgh};
          cgh.parallel_for(sycl::nd_range<1>{size, group_size},
                           [=](sycl::nd_item<1> item) {
                             const unsigned index = item.get_global_linear_id();
                             const unsigned lane = item.get_local_linear_id();
                             scratch[lane] = index;
                             sycl::group_barrier(item.get_group());
                             values[index] = scratch[(lane + 1) % group_size];
                           });
        })
        .wait_and_throw();
  } catch (const sycl::exception &error) {
    std::cerr << error.what() << '\n';
    sycl::free(values, queue);
    return 1;
  }

  bool correct = true;
  for (unsigned i = 0; i < size; ++i)
    correct &= values[i] == i / group_size * group_size + (i + 1) % group_size;
  sycl::free(values, queue);
  return correct ? 0 : 1;
}

// CHECK: ACPP_JITOPT_HOST_PRE_CBS_OPT_LEVEL must be 1 or 3
