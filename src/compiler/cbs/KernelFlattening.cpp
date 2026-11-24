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
#include "hipSYCL/compiler/cbs/KernelFlattening.hpp"

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/compiler/cbs/IRUtils.hpp"
#include "hipSYCL/compiler/cbs/SplitterAnnotationAnalysis.hpp"
#include "hipSYCL/compiler/llvm-to-backend/GlobalInliningAttributorPass.hpp"

#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/IR/DiagnosticHandler.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/ValueHandle.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include <utility>

namespace {
using namespace hipsycl::compiler;
bool inlineCallsInBasicBlock(llvm::BasicBlock &BB) {
  bool LastChanged = false, Changed = false;

  do {
    LastChanged = false;
    for (auto &I : BB) {
      if (auto *CallI = llvm::dyn_cast<llvm::CallBase>(&I)) {
        auto *Callee = CallI->getCalledFunction();
        if (Callee && !Callee->hasFnAttribute(HostSkipForcedInliningAttr) &&
            !CallI->getAttributes().hasFnAttr(HostSkipForcedInliningAttr)) {
          LastChanged = hipsycl::compiler::utils::checkedInlineFunction(CallI, "[KernelFlattening]",
                                                                        HIPSYCL_DEBUG_LEVEL_INFO);
          if (LastChanged)
            break;
        }
      }
    }
    Changed |= LastChanged;
  } while (LastChanged);

  return Changed;
}

//! \pre all contained functions are non recursive!
// todo: have a recursive-ness termination
bool inlineCallsWithStockScan(llvm::Function &F) {
  bool Changed = false;
  bool LastChanged;

  do {
    LastChanged = false;
    for (auto &BB : F) {
      LastChanged = inlineCallsInBasicBlock(BB);
      Changed |= LastChanged;
      if (LastChanged)
        break;
    }
  } while (LastChanged);

  return Changed;
}

//! \pre all contained functions are non recursive!
bool inlineCallsWithWorklist(llvm::Function &F) {
  bool Changed = false;
  bool LastChanged;
  do {
    LastChanged = false;
    // Seed in reverse instruction order so callees can be considered as soon
    // as they are inlined. Value handles account for instructions erased or
    // replaced by later inlines.
    llvm::SmallVector<llvm::WeakTrackingVH, 32> Calls;
    for (auto &BB : llvm::reverse(F))
      for (auto &I : llvm::reverse(BB))
        if (auto *Call = llvm::dyn_cast<llvm::CallBase>(&I))
          Calls.emplace_back(Call);

    while (!Calls.empty()) {
      auto *Call = llvm::dyn_cast_or_null<llvm::CallBase>(static_cast<llvm::Value *>(Calls.pop_back_val()));
      if (!Call)
        continue;
      auto *Callee = Call->getCalledFunction();
      if (!Callee || Callee->hasFnAttribute(HostSkipForcedInliningAttr) ||
          Call->getAttributes().hasFnAttr(HostSkipForcedInliningAttr))
        continue;

      llvm::InlineFunctionInfo IFI;
      if (!utils::checkedInlineFunction(Call, "[KernelFlattening]", HIPSYCL_DEBUG_LEVEL_INFO, IFI))
        continue;
      LastChanged = true;

      for (auto *InlinedCall : llvm::reverse(IFI.InlinedCallSites))
        Calls.emplace_back(InlinedCall);
    }
    Changed |= LastChanged;
    // Inlining can turn an already visited indirect call into a direct call.
  } while (LastChanged);

  return Changed;
}

// Look through statically known callees. The CBS lowering recognizes a barrier
// only when the kernel itself contains a direct call to a splitter.
const llvm::Function *findHiddenBarrier(const llvm::Function &F,
                                        const SplitterAnnotationInfo &SAA) {
  llvm::SmallPtrSet<const llvm::Function *, 32> Visited;
  llvm::SmallVector<std::pair<const llvm::Function *, const llvm::Function *>, 16> Pending;

  auto EnqueueCalls = [&](const llvm::Function &Caller, const llvm::Function *RootHelper) {
    for (const auto &Inst : llvm::instructions(Caller)) {
      const auto *Call = llvm::dyn_cast<llvm::CallBase>(&Inst);
      if (!Call || (!RootHelper && utils::isBarrier(Call, SAA)))
        continue;
      const auto *Callee =
          llvm::dyn_cast<llvm::Function>(Call->getCalledOperand()->stripPointerCastsAndAliases());
      if (Callee && Visited.insert(Callee).second)
        Pending.emplace_back(Callee, RootHelper ? RootHelper : Callee);
    }
  };

  EnqueueCalls(F, nullptr);
  while (!Pending.empty()) {
    const auto [Callee, RootHelper] = Pending.pop_back_val();
    if (SAA.isSplitterFunc(Callee) || Callee->getName() == cbs::BarrierIntrinsicName)
      return RootHelper;
    EnqueueCalls(*Callee, RootHelper);
  }
  return nullptr;
}

constexpr char StockFlatteningAttr[] = "acpp.cbs.stock_flattening";

bool inlineCallsInFunction(llvm::Function &F, const SplitterAnnotationInfo &SAA,
                           bool VerifyBarriers) {
  const auto *Handler = F.getContext().getDiagHandlerPtr();
  if (Handler && Handler->HasErrors)
    return false;

  const bool UseStock = F.hasFnAttribute(StockFlatteningAttr);
  if (!UseStock) {
    if (!VerifyBarriers)
      return inlineCallsWithWorklist(F);

    // A block address can have users outside the function. Keep its original
    // blocks in that case so a rollback cannot invalidate those users.
    if (llvm::any_of(F, [](const llvm::BasicBlock &BB) { return BB.hasAddressTaken(); })) {
      llvm::errs() << "AdaptiveCpp warning: using stock kernel flattening for '" << F.getName()
                   << "' because a block address prevents rollback\n";
    } else {
      llvm::ValueToValueMapTy CloneMap;
      auto *Original = llvm::CloneFunction(&F, CloneMap);
      auto EraseOriginal = llvm::make_scope_exit([&] { Original->eraseFromParent(); });

      const bool Changed = inlineCallsWithWorklist(F);
      const auto *Helper = findHiddenBarrier(F, SAA);
      if (!Helper)
        return Changed;

      llvm::errs() << "AdaptiveCpp warning: a barrier remains in helper '" << Helper->getName()
                   << "' in kernel '" << F.getName()
                   << "' after worklist flattening; retrying with stock kernel flattening\n";

      const auto Linkage = F.getLinkage();
      F.deleteBody();
      F.setLinkage(Linkage);
      llvm::ValueToValueMapTy RestoreMap;
      for (auto &Arg : Original->args())
        RestoreMap[&Arg] = F.getArg(Arg.getArgNo());
      llvm::SmallVector<llvm::ReturnInst *, 8> Returns;
      llvm::CloneFunctionInto(&F, Original, RestoreMap,
                              llvm::CloneFunctionChangeType::LocalChangesOnly, Returns);
    }
    // SSCP can run flattening again before and after work-group lowering.
    F.addFnAttr(StockFlatteningAttr);
  }

  const bool Changed = inlineCallsWithStockScan(F);
  if (VerifyBarriers)
    if (const auto *Helper = findHiddenBarrier(F, SAA))
      F.getContext().emitError("AdaptiveCpp: barrier remains in helper '" + Helper->getName() +
                              "' in kernel '" + F.getName() + "' after stock kernel flattening");
  return Changed || !UseStock;
}
} // namespace

void hipsycl::compiler::KernelFlatteningPassLegacy::getAnalysisUsage(
    llvm::AnalysisUsage &AU) const {
  AU.addRequired<SplitterAnnotationAnalysisLegacy>();
  AU.addPreserved<SplitterAnnotationAnalysisLegacy>();
}

bool hipsycl::compiler::KernelFlatteningPassLegacy::runOnFunction(llvm::Function &F) {
  auto &SAA = getAnalysis<SplitterAnnotationAnalysisLegacy>().getAnnotationInfo();

  if (!SAA.isKernelFunc(&F))
    return false;

  return inlineCallsInFunction(F, SAA, VerifyBarriers);
}

llvm::PreservedAnalyses
hipsycl::compiler::KernelFlatteningPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &AM) {
  const auto &MAMProxy = AM.getResult<llvm::ModuleAnalysisManagerFunctionProxy>(F);
  const auto *SAA = MAMProxy.getCachedResult<SplitterAnnotationAnalysis>(*F.getParent());
  if (!SAA) {
    llvm::errs() << "SplitterAnnotationAnalysis not cached.\n";
    return llvm::PreservedAnalyses::all();
  }
  if (!SAA->isKernelFunc(&F))
    return llvm::PreservedAnalyses::all();

  if (!inlineCallsInFunction(F, *SAA, VerifyBarriers))
    return llvm::PreservedAnalyses::all();

  llvm::PreservedAnalyses PA;
  PA.preserve<SplitterAnnotationAnalysis>();
  return PA;
}

char hipsycl::compiler::KernelFlatteningPassLegacy::ID = 0;
