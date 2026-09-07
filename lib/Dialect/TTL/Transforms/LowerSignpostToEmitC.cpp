// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Passes.h"

#include <map>
#include <set>

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"

namespace mlir::tt::ttl {
#define GEN_PASS_DEF_TTLLOWERSIGNPOSTTOEMITC
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

static void createEmitCVerbatim(Location loc, StringRef value,
                                ConversionPatternRewriter &rewriter) {
  OperationState state(loc, "emitc.verbatim");
  state.addAttribute("value", rewriter.getStringAttr(value));
  rewriter.create(state);
}

// Cheap ttkernel ops that should not trigger profiling scopes. These are
// coordinate lookups that define values used throughout the kernel body.
static bool isSkippedOp(Operation *op) {
  auto name = op->getName().getStringRef();
  return name == "ttkernel.my_x" || name == "ttkernel.my_y" ||
         name == "ttkernel.my_logical_x_" || name == "ttkernel.my_logical_y_";
}

// Check if an operation or any of its nested ops are profiling-worthy
// ttkernel ops (i.e. ttkernel dialect and not on the skip list). Ops that are
// already covered by a nested signpost scope are not counted; those scopes
// will emit their own profiling markers.
static bool containsTTKernelOp(Operation *op) {
  if (op->getDialect() && op->getDialect()->getNamespace() == "ttkernel" &&
      !isSkippedOp(op)) {
    return true;
  }
  for (auto &region : op->getRegions()) {
    for (auto &block : region) {
      bool insideSignpost = false;
      for (auto &nested : block) {
        if (auto sp = dyn_cast<SignpostOp>(&nested)) {
          insideSignpost = !sp.getIsEnd();
          continue;
        }
        if (!insideSignpost && containsTTKernelOp(&nested)) {
          return true;
        }
      }
    }
  }
  return false;
}

// Check if any op between beforeOp and afterOp defines a value used after
// afterOp in the same block, or in a different block entirely.
static bool hasEscapingValues(SignpostOp beforeOp, SignpostOp afterOp) {
  Block *scopeBlock = beforeOp->getBlock();
  for (auto *op = beforeOp->getNextNode(); op != afterOp.getOperation();
       op = op->getNextNode()) {
    for (auto result : op->getResults()) {
      for (auto *user : result.getUsers()) {
        if (user->getBlock() != scopeBlock || afterOp->isBeforeInBlock(user)) {
          return true;
        }
      }
    }
  }
  return false;
}

// Walk forward from a begin signpost to find its matching end signpost in the
// same block. Returns the end op if found, nullptr otherwise. Sets
// `hasInterestingOps` if any ttkernel dialect op is found between the pair,
// including inside nested regions (e.g. scf.for bodies).
static SignpostOp findMatchingEnd(SignpostOp beginOp, bool &hasInterestingOps) {
  hasInterestingOps = false;
  StringRef beginName = beginOp.getName();

  for (auto *op = beginOp->getNextNode(); op; op = op->getNextNode()) {
    if (auto signpost = dyn_cast<SignpostOp>(op)) {
      if (signpost.getIsEnd() && signpost.getName() == beginName) {
        return signpost;
      }
    }
    if (containsTTKernelOp(op)) {
      hasInterestingOps = true;
    }
  }
  return nullptr;
}

struct SignpostLowering : OpConversionPattern<SignpostOp> {
  // Set of names whose end signpost should emit a closing brace.
  // Populated by begin signpost handlers when the pair contains ttkernel ops.
  std::set<std::string> &keptEndNames;
  mutable std::map<std::string, std::string> explicitEndNames;
  mutable unsigned nextScopeId = 0;

  SignpostLowering(MLIRContext *ctx, std::set<std::string> &keptEndNames)
      : OpConversionPattern(ctx), keptEndNames(keptEndNames) {}

  LogicalResult
  matchAndRewrite(SignpostOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    StringRef name = op.getName();

    if (!op.getIsEnd()) {
      bool hasInterestingOps = false;
      SignpostOp endOp = findMatchingEnd(op, hasInterestingOps);

      if (hasInterestingOps || name.starts_with("ttl_")) {
        if (endOp && hasEscapingValues(op, endOp)) {
          std::string identifier =
              "ttl_profile_scope_" + std::to_string(nextScopeId++);
          createEmitCVerbatim(loc,
                              "DeviceZoneBeginN(\"" + name.str() + "\", " +
                                  identifier + ");",
                              rewriter);
          explicitEndNames.emplace(name.str(), identifier);
        } else {
          createEmitCVerbatim(loc, "{", rewriter);
          createEmitCVerbatim(loc, "DeviceZoneScopedN(\"" + name.str() + "\");",
                              rewriter);
          keptEndNames.insert(name.str());
        }
      }
    } else {
      auto explicitEnd = explicitEndNames.find(name.str());
      if (explicitEnd != explicitEndNames.end()) {
        createEmitCVerbatim(loc, "DeviceZoneEnd(" + explicitEnd->second + ");",
                            rewriter);
        explicitEndNames.erase(explicitEnd);
      } else if (keptEndNames.count(name.str())) {
        createEmitCVerbatim(loc, "}", rewriter);
        keptEndNames.erase(name.str());
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct TTLLowerSignpostToEmitCPass
    : impl::TTLLowerSignpostToEmitCBase<TTLLowerSignpostToEmitCPass> {
  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    ModuleOp mod = getOperation();

    ConversionTarget target(ctx);
    target.addIllegalOp<SignpostOp>();
    target.addLegalDialect<emitc::EmitCDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    std::set<std::string> keptEndNames;

    RewritePatternSet patterns(&ctx);
    patterns.insert(std::make_unique<SignpostLowering>(&ctx, keptEndNames));

    if (failed(applyPartialConversion(mod, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
