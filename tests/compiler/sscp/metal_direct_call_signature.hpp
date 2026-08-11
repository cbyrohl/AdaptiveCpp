#pragma once

#include <sycl/sycl.hpp>

#include <utility>

using MetalAggregateResult = std::pair<float, float>;

SYCL_EXTERNAL __attribute__((noinline)) MetalAggregateResult
makeMetalAggregate(float value);
