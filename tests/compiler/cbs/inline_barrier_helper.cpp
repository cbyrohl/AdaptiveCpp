// RUN: %acpp %s -o %t --acpp-targets=omp --acpp-use-accelerated-cpu -O3
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: env ACPP_VISIBILITY_MASK=omp %t | FileCheck %s

#include <sycl/sycl.hpp>

#include <array>
#include <iostream>

namespace {
constexpr std::size_t LocalSize = 8;
constexpr std::size_t GroupCount = 3;

void writeAndWait(sycl::nd_item<1> Item, sycl::local_accessor<int, 1> Scratch) {
  const auto Lane = Item.get_local_linear_id();
  const auto Group = Item.get_group_linear_id();

  // This branch is uniform within a work-group.
  if (Group % 2 == 0)
    Scratch[Lane] = 100 * Group + Lane;
  else
    Scratch[Lane] = 100 * Group + LocalSize - 1 - Lane;
  sycl::group_barrier(Item.get_group());
}
} // namespace

int main() {
  std::array<int, LocalSize * GroupCount> Output{};
  sycl::queue Queue;
  {
    sycl::buffer<int, 1> Buffer{Output.data(), sycl::range<1>{Output.size()}};
    Queue.submit([&](sycl::handler &CGH) {
      auto Result = Buffer.get_access<sycl::access::mode::write>(CGH);
      sycl::local_accessor<int, 1> Scratch{sycl::range<1>{LocalSize}, CGH};
      CGH.parallel_for<class InlineBarrierHelper>(
          sycl::nd_range<1>{Output.size(), LocalSize}, [=](sycl::nd_item<1> Item) {
            writeAndWait(Item, Scratch);
            const auto Neighbor = (Item.get_local_linear_id() + 1) % LocalSize;
            Result[Item.get_global_linear_id()] = Scratch[Neighbor];
          });
    }).wait_and_throw();
  }

  for (std::size_t I = 0; I < Output.size(); ++I) {
    const auto Group = I / LocalSize;
    const auto Neighbor = (I % LocalSize + 1) % LocalSize;
    const int Expected = 100 * Group + (Group % 2 == 0 ? Neighbor : LocalSize - 1 - Neighbor);
    if (Output[I] != Expected)
      return 1;
  }
  // CHECK: all neighboring values are correct
  std::cout << "all neighboring values are correct\n";
}
