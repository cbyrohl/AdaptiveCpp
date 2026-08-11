// REQUIRES: metal
// RUN: %acpp %s -x c++ %S/metal_direct_call_signature_impl.cpp.in -x none -o %t.o0 --acpp-targets=generic
// RUN: %t.o0 | FileCheck %s
// RUN: %acpp %s -x c++ %S/metal_direct_call_signature_impl.cpp.in -x none -o %t.o3 --acpp-targets=generic -O3
// RUN: %t.o3 | FileCheck %s

#include "metal_direct_call_signature.hpp"

#include <iostream>

// Keep another nominal two-float aggregate in the caller's SSCP module.
// LLVM may select such an isomorphic type while linking identified structs.
struct CallerTwoFloat {
  float first;
  float second;
};

int main() {
  sycl::queue Queue;
  float *Output = sycl::malloc_shared<float>(2, Queue);

  Queue.single_task([=]() {
    CallerTwoFloat KeepTypeAlive{3.0f, 4.0f};
    const auto [First, Second] = makeMetalAggregate(40.0f);
    Output[0] = First + 0.0f * KeepTypeAlive.first;
    Output[1] = Second + 0.0f * KeepTypeAlive.second;
  }).wait();

  // CHECK: 41 42
  std::cout << Output[0] << ' ' << Output[1] << '\n';
  sycl::free(Output, Queue);
}
