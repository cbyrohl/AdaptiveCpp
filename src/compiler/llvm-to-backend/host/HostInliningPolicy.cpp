/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
#include "hipSYCL/compiler/llvm-to-backend/host/LLVMToHost.hpp"

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/compiler/cbs/SplitterAnnotationAnalysis.hpp"
#include "hipSYCL/compiler/llvm-to-backend/GlobalInliningAttributorPass.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/MathExtras.h>

#include <cstdint>
#include <limits>

using namespace hipsycl::compiler;

namespace {

using FunctionSet = llvm::SmallPtrSet<llvm::Function *, 32>;
using CallGraph = llvm::DenseMap<llvm::Function *, llvm::SmallVector<llvm::CallBase *, 4>>;

bool referencesGlobal(const llvm::Value &V) {
  if (llvm::isa<llvm::GlobalValue>(V))
    return true;
  if (const auto *C = llvm::dyn_cast<llvm::Constant>(&V))
    return llvm::any_of(C->operands(), [](const llvm::Use &U) {
      return referencesGlobal(*U.get());
    });
  return false;
}

bool isScalarMathBuiltin(llvm::StringRef Name) {
  return llvm::StringSwitch<bool>(Name)
      .Cases("__acpp_sscp_fma_f32", "__acpp_sscp_fma_f64", true)
      .Cases("__acpp_sscp_fabs_f32", "__acpp_sscp_fabs_f64", true)
      .Cases("__acpp_sscp_fmin_f32", "__acpp_sscp_fmin_f64", true)
      .Cases("__acpp_sscp_fmax_f32", "__acpp_sscp_fmax_f64", true)
      .Default(false);
}

bool matchesPattern(llvm::StringRef Name, llvm::StringRef Patterns) {
  while (!Patterns.empty()) {
    auto [Pattern, Tail] = Patterns.split(',');
    Pattern = Pattern.trim();
    if (!Pattern.empty() && Name.contains(Pattern))
      return true;
    Patterns = Tail;
  }
  return false;
}

FunctionSet findRequiredFunctions(llvm::Module &M, CallGraph &Calls) {
  const SplitterAnnotationInfo Splitters{M};
  llvm::DenseMap<llvm::Function *, llvm::SmallVector<llvm::Function *, 4>> Callers;
  FunctionSet Required;
  llvm::SmallVector<llvm::Function *, 32> Pending;

  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    bool NeedsFlattening = F.isConvergent() || F.hasPersonalityFn() ||
                           Splitters.isSplitterFunc(&F);
    for (auto &BB : F) {
      for (auto &I : BB) {
        auto *Call = llvm::dyn_cast<llvm::CallBase>(&I);
        if (Call) {
          auto *Callee = Call->getCalledFunction();
          if (!Callee || Call->isConvergent() || Splitters.isSplitterFunc(Callee) ||
              (Callee->isDeclaration() && !Callee->isIntrinsic() &&
               !isScalarMathBuiltin(Callee->getName()))) {
            NeedsFlattening = true;
          }
          if (Callee && !Callee->isDeclaration()) {
            Calls[&F].push_back(Call);
            Callers[Callee].push_back(&F);
          }
        }
        // CBS only rewrites work-group state inside the kernel.
        NeedsFlattening |= llvm::any_of(I.operands(), [&](const llvm::Use &U) {
          return (!Call || !Call->isCallee(&U)) && referencesGlobal(*U.get());
        });
      }
    }
    if (NeedsFlattening) {
      Required.insert(&F);
      Pending.push_back(&F);
    }
  }

  while (!Pending.empty()) {
    for (auto *Caller : Callers[Pending.pop_back_val()])
      if (Required.insert(Caller).second)
        Pending.push_back(Caller);
  }
  return Required;
}

uint64_t expandedSize(llvm::Function *F, const CallGraph &Calls,
                      llvm::DenseMap<llvm::Function *, uint64_t> &Sizes) {
  constexpr auto UnknownSize = std::numeric_limits<uint64_t>::max();
  auto [Position, Inserted] = Sizes.try_emplace(F, UnknownSize);
  if (!Inserted)
    return Position->second;

  uint64_t Size = F->getInstructionCount();
  auto Edges = Calls.find(F);
  if (Edges != Calls.end()) {
    // Count each call independently. Recursive expansion saturates the estimate.
    for (auto *Call : Edges->second)
      Size = llvm::SaturatingAdd(Size, expandedSize(Call->getCalledFunction(), Calls, Sizes));
  }
  Sizes[F] = Size;
  return Size;
}

bool orderKernelRegion(llvm::Function *F, const CallGraph &Calls,
                       const FunctionSet &Forced, FunctionSet &Region,
                       FunctionSet &Active, llvm::SmallVectorImpl<llvm::Function *> &Order) {
  if (!Region.insert(F).second)
    return !Active.contains(F);

  Active.insert(F);
  auto Edges = Calls.find(F);
  bool Acyclic = true;
  if (Edges != Calls.end()) {
    for (auto *Call : Edges->second) {
      auto *Callee = Call->getCalledFunction();
      if (Forced.contains(Callee))
        Acyclic &= orderKernelRegion(Callee, Calls, Forced, Region, Active, Order);
    }
  }
  Active.erase(F);
  Order.push_back(F);
  return Acyclic;
}

bool orderKernelRegions(const FunctionSet &Kernels, const CallGraph &Calls,
                        const FunctionSet &Forced, FunctionSet &Region,
                        llvm::SmallVectorImpl<llvm::Function *> &Order) {
  FunctionSet Active;
  bool Acyclic = true;
  for (auto *Kernel : Kernels)
    Acyclic &= orderKernelRegion(Kernel, Calls, Forced, Region, Active, Order);
  return Acyclic;
}

void promoteSmallDuplication(const FunctionSet &Kernels, const CallGraph &Calls,
                             const llvm::DenseMap<llvm::Function *, uint64_t> &Sizes,
                             const FunctionSet &Excluded, unsigned Growth,
                             FunctionSet &Forced) {
  FunctionSet Region;
  llvm::SmallVector<llvm::Function *, 32> Order;
  if (!orderKernelRegions(Kernels, Calls, Forced, Region, Order))
    return;

  llvm::DenseMap<llvm::Function *, uint64_t> Copies;
  for (auto *Kernel : Kernels)
    Copies[Kernel] = 1;
  for (auto *F : llvm::reverse(Order)) {
    auto Edges = Calls.find(F);
    if (Edges == Calls.end())
      continue;
    const auto ParentCopies = Copies.lookup(F);
    for (auto *Call : Edges->second) {
      auto &CalleeCopies = Copies[Call->getCalledFunction()];
      CalleeCopies = llvm::SaturatingAdd(CalleeCopies, ParentCopies);
    }
  }

  for (const auto &[F, Count] : Copies) {
    auto Size = Sizes.find(F);
    if (Count != 0 && !Region.contains(F) && !Excluded.contains(F) &&
        Size != Sizes.end() && Size->second != std::numeric_limits<uint64_t>::max() &&
        llvm::SaturatingMultiply(Size->second, Count - 1) <= Growth)
      Forced.insert(F);
  }
}

void keepCallOutOfLine(llvm::CallBase &Call) {
  Call.addFnAttr(llvm::Attribute::NoInline);
  Call.addFnAttr(llvm::Attribute::get(Call.getContext(), HostSkipForcedInliningAttr));
}

} // namespace

bool LLVMToHostTranslator::configureInlining(llvm::Module &M) {
  if (!InlineBudget && (InlineRootBudget || InlineGrowth)) {
    registerError("LLVMToHost: root budget and growth require a host inline budget");
    return false;
  }
  if (!InlineBudget && !UseLLVMInliner && NoInlinePatterns.empty())
    return true;

  CallGraph Calls;
  FunctionSet Forced = findRequiredFunctions(M, Calls);
  FunctionSet Kernels, Excluded;
  llvm::DenseMap<llvm::Function *, uint64_t> Sizes;

  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    if (isKernelAfterFlavoring(F)) {
      Kernels.insert(&F);
      continue;
    }
    if (Forced.contains(&F) || F.hasFnAttribute(llvm::Attribute::AlwaysInline)) {
      Forced.insert(&F);
      if (matchesPattern(F.getName(), NoInlinePatterns))
        HIPSYCL_DEBUG_INFO << "LLVMToHost: Keeping normal inlining for " << F.getName().str() << "\n";
      continue;
    }
    if (matchesPattern(F.getName(), NoInlinePatterns)) {
      Excluded.insert(&F);
    } else if (matchesPattern(F.getName(), ForceInlinePatterns)) {
      Forced.insert(&F);
    } else if (InlineBudget && expandedSize(&F, Calls, Sizes) <= *InlineBudget) {
      Forced.insert(&F);
    }
  }

  if (InlineGrowth != 0)
    promoteSmallDuplication(Kernels, Calls, Sizes, Excluded, InlineGrowth, Forced);

  FunctionSet KernelRegion;
  const unsigned RootBudget = InlineRootBudget.value_or(0);
  if (RootBudget > InlineBudget.value_or(0)) {
    llvm::SmallVector<llvm::Function *, 32> Order;
    orderKernelRegions(Kernels, Calls, Forced, KernelRegion, Order);
  }

  for (auto &F : M) {
    if (F.isDeclaration() || Kernels.contains(&F))
      continue;
    if (Forced.contains(&F)) {
      // Explicit force patterns can override a frontend noinline attribute.
      if (matchesPattern(F.getName(), ForceInlinePatterns))
        F.removeFnAttr(llvm::Attribute::NoInline);
      continue;
    }
    if (!InlineBudget && !UseLLVMInliner && !Excluded.contains(&F))
      continue;

    auto Size = Sizes.find(&F);
    if (!KernelRegion.empty() && !Excluded.contains(&F) && Size != Sizes.end() &&
        Size->second <= RootBudget) {
      // The global inliner can flatten this helper inside out-of-line code.
      // Keep only its calls from the kernel region out of line.
      for (auto *User : F.users()) {
        auto *Call = llvm::dyn_cast<llvm::CallBase>(User);
        if (Call && Call->getCalledFunction() == &F && KernelRegion.contains(Call->getFunction()))
          keepCallOutOfLine(*Call);
      }
    } else {
      F.addFnAttr(HostSkipForcedInliningAttr);
      if (InlineBudget || Excluded.contains(&F))
        F.addFnAttr(llvm::Attribute::NoInline);
    }
  }
  return true;
}
