#include "warpforge/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/APFloat.h"

#include <utility>

namespace mlir::warpforge {
namespace {

class VerifyInferencePass
    : public PassWrapper<VerifyInferencePass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyInferencePass)

  StringRef getArgument() const final { return "warpforge-verify-inference"; }
  StringRef getDescription() const final {
    return "Verify WarpForge static tensor matmul legality";
  }

  void runOnOperation() final {
    WalkResult result = getOperation().walk([&](linalg::MatmulOp op) {
      auto lhsType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
      auto rhsType = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
      auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
      if (!lhsType || !rhsType || !resultType) {
        op.emitOpError()
            << "requires tensor semantics; bufferize only after WarpForge "
               "inference optimization";
        return WalkResult::interrupt();
      }
      if (lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
          resultType.getRank() != 2) {
        op.emitOpError("requires rank-2 lhs, rhs, and result tensors");
        return WalkResult::interrupt();
      }
      if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
          !resultType.hasStaticShape()) {
        op.emitOpError()
            << "requires fully static shapes for deterministic GPU tiling; "
               "specialize dynamic dimensions before this pipeline";
        return WalkResult::interrupt();
      }

      auto elementType = dyn_cast<FloatType>(lhsType.getElementType());
      if (!elementType ||
          (elementType.getWidth() != 16 && elementType.getWidth() != 32) ||
          rhsType.getElementType() != elementType ||
          resultType.getElementType() != elementType) {
        op.emitOpError(
            "requires a uniform f16 or f32 element type on lhs, rhs, and result");
        return WalkResult::interrupt();
      }

      int64_t m = lhsType.getDimSize(0);
      int64_t k = lhsType.getDimSize(1);
      int64_t n = rhsType.getDimSize(1);
      if (rhsType.getDimSize(0) != k || resultType.getDimSize(0) != m ||
          resultType.getDimSize(1) != n) {
        op.emitOpError("has inconsistent MxK, KxN, and MxN dimensions");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

static bool isSplatFloatZero(Value value) {
  Attribute attribute;
  if (!matchPattern(value, m_Constant(&attribute)))
    return false;
  auto dense = dyn_cast<DenseFPElementsAttr>(attribute);
  return dense && dense.isSplat() && dense.getSplatValue<APFloat>().isZero();
}

class FuseBiasReluPattern : public OpRewritePattern<linalg::MaxOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MaxOp maxOp,
                                PatternRewriter &rewriter) const final {
    if (maxOp->getNumOperands() != 3 || maxOp->getNumResults() != 1)
      return failure();

    Value addValue = maxOp->getOperand(0);
    Value zeroValue = maxOp->getOperand(1);
    if (isSplatFloatZero(addValue))
      std::swap(addValue, zeroValue);
    if (!isSplatFloatZero(zeroValue))
      return rewriter.notifyMatchFailure(maxOp,
                                         "activation threshold is not zero");

    auto addOp = addValue.getDefiningOp<linalg::AddOp>();
    if (!addOp || !addOp->hasOneUse() || addOp->getNumOperands() != 3)
      return rewriter.notifyMatchFailure(
          maxOp, "expected a single-use linalg.add producer");

    Value matmulValue = addOp->getOperand(0);
    Value biasValue = addOp->getOperand(1);
    if (!matmulValue.getDefiningOp<linalg::MatmulOp>())
      std::swap(matmulValue, biasValue);
    if (!matmulValue.getDefiningOp<linalg::MatmulOp>())
      return rewriter.notifyMatchFailure(maxOp, "add does not consume a matmul");

    auto outputType = dyn_cast<RankedTensorType>(maxOp.getResult(0).getType());
    if (!outputType || outputType.getRank() != 2 ||
        matmulValue.getType() != outputType || biasValue.getType() != outputType)
      return rewriter.notifyMatchFailure(
          maxOp, "epilogue values must have one rank-2 tensor type");

    auto elementType = dyn_cast<FloatType>(outputType.getElementType());
    if (!elementType)
      return rewriter.notifyMatchFailure(maxOp,
                                         "epilogue must be floating point");

    AffineMap identity = rewriter.getMultiDimIdentityMap(2);
    SmallVector<AffineMap> indexingMaps(3, identity);
    SmallVector<utils::IteratorType> iteratorTypes(
        2, utils::IteratorType::parallel);
    auto fused = rewriter.create<linalg::GenericOp>(
        maxOp.getLoc(), outputType, ValueRange{matmulValue, biasValue},
        ValueRange{maxOp->getOperand(2)}, indexingMaps,
        iteratorTypes,
        [&](OpBuilder &builder, Location location, ValueRange arguments) {
          Value sum =
              builder.create<arith::AddFOp>(location, arguments[0], arguments[1]);
          Value zero = builder.create<arith::ConstantOp>(
              location, builder.getFloatAttr(elementType, 0.0));
          Value relu =
              builder.create<arith::MaximumFOp>(location, sum, zero);
          builder.create<linalg::YieldOp>(location, relu);
        });
    fused->setAttr("warpforge.fused_epilogue",
                   rewriter.getStringAttr("bias_relu"));
    rewriter.replaceOp(maxOp, fused.getResults());
    rewriter.eraseOp(addOp);
    return success();
  }
};

class FuseMatmulEpiloguePass
    : public PassWrapper<FuseMatmulEpiloguePass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseMatmulEpiloguePass)

  StringRef getArgument() const final {
    return "warpforge-fuse-matmul-epilogue";
  }
  StringRef getDescription() const final {
    return "Fuse matmul bias-add and ReLU epilogues";
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseBiasReluPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void registerWarpForgePasses() {
  PassRegistration<VerifyInferencePass>();
  PassRegistration<FuseMatmulEpiloguePass>();
  PassPipelineRegistration<>(
      "warpforge-simplify",
      "Run canonical folding and common-subexpression elimination",
      [](OpPassManager &manager) {
        manager.addNestedPass<func::FuncOp>(createCanonicalizerPass());
        manager.addNestedPass<func::FuncOp>(createCSEPass());
      });
}

} // namespace mlir::warpforge
