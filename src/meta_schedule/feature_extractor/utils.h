/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include <tvm/tir/transform.h>

#include <cmath>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../utils.h"

namespace tvm {
namespace tir {

/*! \brief Type for multi-dimensional index */
using MultiIndex = std::vector<PrimExpr>;
/*! \brief Type for a region */
using tvm::support::NDIntSet;
/*! \brief Vector of int64_t */
using IntVec = std::vector<int64_t>;
/*! \brief Vector of for loops */
using ForVec = std::vector<const ForNode*>;

/*!
 * \brief An unordered_map for (for, buffer) => V
 * \tparam V The value type
 */
template <class V>
using ForBufferMap = std::unordered_map<const ForNode*, std::unordered_map<const BufferNode*, V>>;

/*! \brief Given x, compute log2(|x| + 1) */
// inline double slog(double x) { return x >= 0 ? std::log2(x + 1) : std::log2(-x + 1); }
inline double slog(double x) { return x > 0 ? std::log2(x) : x; }

}  // namespace tir
}  // namespace tvm

namespace tvm {
namespace tir {
namespace transform {

/*!
 * \brief Create a pass that simplifies the IR for feature extraction
 * \return The pass created
 */
inline Pass SimplifyForFeatureExtraction() {
  class Simplifier : private StmtExprMutator {
   public:
    static Stmt Run(Stmt stmt) { return Simplifier()(std::move(stmt)); }

   private:
    static bool HasBufferLoad(const PrimExpr& expr) {
      bool found = false;
      PostOrderVisit(expr, [&found](const ObjectRef& node) {
        if (node->IsInstance<BufferLoadNode>()) {
          found = true;
        }
      });
      return found;
    }

    PrimExpr VisitExpr_(const SelectNode* node) final {
      if (HasBufferLoad(node->true_value) || HasBufferLoad(node->false_value) ||
          HasBufferLoad(node->condition)) {
        return GetRef<Select>(node);
      }
      return make_const(node->dtype, 1.0);
    }

    PrimExpr VisitExpr_(const VarNode* var) final {
      if (unit_vars_.count(GetRef<Var>(var))) {
        return make_const(var->dtype, 0.0);
      }
      return GetRef<Var>(var);
    }

    Stmt VisitStmt_(const ForNode* loop) final {
      if (is_zero(loop->min) && is_one(loop->extent) && loop->kind == ForKind::kSerial &&
          loop->annotations.empty()) {
        unit_vars_.insert(loop->loop_var);
        return VisitStmt(loop->body);
      } else {
        return StmtExprMutator::VisitStmt_(loop);
      }
    }

    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> unit_vars_;
  };
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    PrimFuncNode* n = f.CopyOnWrite();
    n->body = Simplifier::Run(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.SimplifyForFeatureExtraction", {});
}

/*!
 * \brief Create a list of passes that preprocesses the IR for feature extraction
 * \return The list of passes created
 */
inline Sequential PassListForFeatureExtraction() {
  return Sequential({
      tir::transform::RemoveWeightLayoutRewriteBlock(),
      tir::transform::SimplifyForFeatureExtraction(),
      tir::transform::LowerCrossThreadReduction(),
      tir::transform::LowerInitBlock(),
      tir::transform::PlanAndUpdateBufferAllocationLocation(),
      tir::transform::ConvertBlocksToOpaque(),
      tir::transform::UnifyThreadBinding(),
      tir::transform::CompactBufferAllocation(),
      tir::transform::LowerMatchBuffer(),
      tir::transform::Simplify(),
  });
}

}  // namespace transform
}  // namespace tir
}  // namespace tvm

namespace tvm {
namespace tir {
namespace utils {

/*!
 * \brief Given a loop, return its `pragma_auto_unroll_max_step` annotation if it exists
 * \param loop The loop to be checked
 * \return The value of `pragma_auto_unroll_max_step` if it exists, or -1 if it does not exist
 */
inline int64_t GetPragmaAutoUnroll(const ForNode* loop) {
  if (Optional<IntImm> auto_unroll = GetAnn<IntImm>(loop, tir::attr::pragma_auto_unroll_max_step)) {
    return auto_unroll.value()->value;
  }
  return -1;
}

/*!
 * \brief Given a list of loops, return the extent of the first loop if the list is not empty,
 * and the first loop has constant extent. Otherwise returns the default value given
 * \param loops The list of loops to be checked
 * \param default_value The default value to be returned if the list is empty or the first loop
 * does not have constant extent
 * \return The extent of the first loop if the list is not empty, or the first loop has constant
 * extent. Otherwise returns the default value
 */
inline int64_t FirstLoopExtent(const ForVec& loops, int64_t default_value) {
  if (!loops.empty()) {
    if (const int64_t* extent = GetLoopIntExtent(loops[0])) {
      return *extent;
    }
  }
  return default_value;
}

/*!
 * \brief Get the shape of the buffer
 * \param buffer The buffer
 * \param analyzer The analyzer
 * \return The shape of the buffer
 */
inline std::vector<int64_t> GetBufferShape(const Buffer& buffer, arith::Analyzer* analyzer) {
  int ndim = buffer->shape.size();
  std::vector<int64_t> result;
  result.reserve(ndim);
  for (const PrimExpr& i : buffer->shape) {
    if (const IntImmNode* int_imm = i.as<IntImmNode>()) {
      result.push_back(int_imm->value);
      continue;
    }
    arith::ConstIntBound bound = analyzer->const_int_bound(i);
    if (0 <= bound->max_value && bound->max_value < arith::ConstIntBound::kPosInf) {
      result.push_back(bound->max_value);
    } else {
      result.push_back(1);
    }
  }
  return result;
}

/*!
 * \brief Converts a 2-dimensional STL vector to a TVM NDArray
 * \param src The source 2-dimensional STL vector
 * \return The converted TVM NDArray
 */
inline runtime::NDArray AsNDArray(const std::vector<std::vector<double>>& src) {
  ICHECK(!src.empty());
  int n = src.size();
  int m = src[0].size();
  runtime::NDArray tgt = runtime::NDArray::Empty(
      /*shape=*/{n, m},
      /*dtype=*/DLDataType{kDLFloat, 64, 1},
      /*ctx=*/DLDevice{kDLCPU, 0});
  double* data = static_cast<double*>(tgt->data);
  for (const std::vector<double>& row : src) {
    for (double v : row) {
      *data++ = v;
    }
  }
  return tgt;
}

/*!
 * \brief Relax each of the multi-indexing pattern according to the domains bound in the analyzer,
 * and then union them into a single region
 * \param multi_index_pattern A list of multi-index pattern to be relaxed
 * \param analyzer The analyzer that contains the domain information
 * \return numel The size of the single region after union
 * \return access_shape The relaxed and unioned region
 */
inline std::tuple<int64_t, IntVec> RelaxAndUnion(const std::vector<MultiIndex>& multi_indices,
                                                 arith::Analyzer* analyzer) {
  int64_t numel = 1;
  if (multi_indices.empty()) {
    return {};
  }
  int n_indices = multi_indices.size();
  int ndim = multi_indices[0].size();
  IntVec access_shape(ndim, 0);
  for (int i = 0; i < ndim; ++i) {
    int64_t minimum = arith::ConstIntBound::kPosInf;
    int64_t maximum = arith::ConstIntBound::kNegInf;
    for (int j = 0; j < n_indices; ++j) {
      arith::ConstIntBound bound = analyzer->const_int_bound(multi_indices[j][i]);
      minimum = std::min(minimum, bound->min_value);
      maximum = std::max(maximum, bound->max_value);
    }
    numel *= maximum - minimum + 1;
    access_shape[i] = maximum - minimum + 1;
  }
  return make_tuple(numel, access_shape);
}

/*!
 * \brief Relax each of the regions according to the domains bound in the analyzer,
 * and then union them into a single region
 * \param regions The regions that the buffer is accessed
 * \param analyzer The analyzer that contains the domain information
 * \return numel The size of the single region after union
 * \return access_shape The relaxed and unioned region
 */
inline std::tuple<int64_t, IntVec> RelaxAndUnion(const std::vector<NDIntSet>& regions,
                                                 arith::Analyzer* analyzer) {
  IntVec access_shape = {};
  if (regions.empty()) {
    return make_tuple(1, access_shape);
  }
  int64_t numel = 1;
  int n_regions = regions.size();
  int ndim = regions[0].size();
  for (int i = 0; i < ndim; ++i) {
    // Calculate the union set
    Array<arith::IntSet> int_sets;
    int_sets.reserve(n_regions);
    for (int j = 0; j < n_regions; ++j) {
      int_sets.push_back(regions[j][i]);
    }
    arith::IntSet union_set = arith::Union(int_sets);
    // Update the area
    int64_t min = analyzer->const_int_bound(union_set.min())->min_value;
    int64_t max = analyzer->const_int_bound(union_set.max())->max_value;
    if (arith::ConstIntBound::kNegInf < min && max < arith::ConstIntBound::kPosInf) {
      numel *= max - min + 1;
      access_shape.push_back(max - min + 1);
    } else {
      access_shape.push_back(1);
    }
  }
  return make_tuple(numel, access_shape);
}

class CoefficientExtractor : private ExprVisitor {
 public:
  static int64_t Extract(const PrimExpr& expr, const Var& var) {
    CoefficientExtractor extractor(var);
    extractor.VisitExpr(expr);
    return (extractor.visited_var && !extractor.visited_mul && !extractor.visited_add)
               ? 1
               : (extractor.visited_var ? extractor.stride : 0);
  }

 private:
  explicit CoefficientExtractor(const Var& var)
      : var(var), stride(0), visited_var(false), visited_add(false), visited_mul(false) {}

  void VisitExpr_(const MulNode* node) override {
    ExprVisitor::VisitExpr_(node);
    if (visited_var && !visited_add) {
      if (const auto* a = node->a.as<IntImmNode>()) {
        visited_mul = true;
        stride = a->value;
      } else if (const auto* b = node->b.as<IntImmNode>()) {
        visited_mul = true;
        stride = b->value;
      }
    }
  }

  void VisitExpr_(const AddNode* node) override {
    ExprVisitor::VisitExpr_(node);
    if (visited_var && !visited_mul) {
      visited_add = true;
      stride = 1;
    }
  }

  void VisitExpr_(const VarNode* node) override {
    if (node == var.get()) {
      visited_var = true;
      stride = 2;
    }
  }

  const Var& var;
  int64_t stride;
  bool visited_var;
  bool visited_add;
  bool visited_mul;
};

/*!
 * \brief Given a list of multi-index pattern, return the minimal stride of a variable on it
 * \param multi_indices The list of multi-index pattern
 * \param buffer_stride The stride of the buffer
 * \param var The variable to be checked
 * \return The minimal stride of the variable on the multi-index pattern
 */
inline int64_t GetVarStride(const std::vector<MultiIndex>& multi_indices,
                            const IntVec& buffer_stride, const Var& var) {
  constexpr int64_t kNotFound = std::numeric_limits<int64_t>::max();
  int ndim = buffer_stride.size();
  // Calculate the min stride possible
  int64_t result = kNotFound;
  for (const MultiIndex& multi_index : multi_indices) {
    ICHECK_EQ(multi_index.size(), buffer_stride.size());
    // Find the rightest dimension that contains the given variable
    for (int i = ndim - 1; i >= 0; --i) {
      int64_t coef = CoefficientExtractor::Extract(multi_index[i], var);
      if (coef != 0) {
        result = std::min(result, std::abs(coef) * buffer_stride[i]);
        break;
      }
    }
  }
  return (result == kNotFound) ? 0 : result;
}

/*!
 * \brief Given an array of regions, return the minimal stride of a variable on it
 * \param regions The regions that the buffer is accessed
 * \param buffer_stride The stride of the buffer
 * \param var The variable to be checked
 * \return The minimal stride of the variable on the regions
 */
inline int64_t GetVarStride(const std::vector<NDIntSet>& regions, const IntVec& buffer_stride,
                            const Var& var) {
  constexpr int64_t kNotFound = std::numeric_limits<int64_t>::max();
  int ndim = buffer_stride.size();
  // Calculate the min stride possible
  int64_t result = kNotFound;
  for (const NDIntSet& region : regions) {
    ICHECK_EQ(region.size(), buffer_stride.size());
    // Find the rightest dimension that contains the given variable
    for (int i = ndim - 1; i >= 0; --i) {
      PrimExpr idx = region[i].min();
      int64_t coef = CoefficientExtractor::Extract(idx, var);
      if (coef != 0) {
        result = std::min(result, std::abs(coef) * buffer_stride[i]);
        break;
      }
    }
  }
  return (result == kNotFound) ? 0 : result;
}

/*! \brief A data structure managing loop nests */
struct LoopNest {
  int64_t prod = 1;    // The product of the extents of all the loops
  ForVec loops;        // All the loops
  IntVec auto_unroll;  // The loops with auto unroll pragma
  ForVec parallel;     // The loops whose ForKind are kParallel
  ForVec vectorize;    // The loops whose ForKind are kVectorized
  ForVec unroll;       // The loops whose ForKind are kUnrolled
  ForVec blockIdx_x;   // The loops whose ForKind are kThreadBinding to blockIdx.x
  ForVec blockIdx_y;   // The loops whose ForKind are kThreadBinding to blockIdx.y
  ForVec blockIdx_z;   // The loops whose ForKind are kThreadBinding to blockIdx.z
  ForVec threadIdx_x;  // The loops whose ForKind are kThreadBinding to threadIdx.x
  ForVec threadIdx_y;  // The loops whose ForKind are kThreadBinding to threadIdx.y
  ForVec threadIdx_z;  // The loops whose ForKind are kThreadBinding to threadIdx.z
  ForVec vthread;      // The loops whose ForKind are kThreadBinding to vthread.*

  /*!
   * \brief Push a new loop into the loop nest
   * \param loop The loop to be pushed
   * \param auto_unroll_attr The auto unroll attribute of the loop
   * \return A list of for loops that the loop is bound to
   */
  ForVec* Push(const ForNode* loop, int64_t* auto_unroll_attr) {
    if (const int64_t* extent = GetLoopIntExtent(loop)) {
      this->prod *= *extent;
    }
    this->loops.push_back(loop);
    if ((*auto_unroll_attr = utils::GetPragmaAutoUnroll(loop)) > 0) {
      this->auto_unroll.push_back(*auto_unroll_attr);
    }
    ForVec* ref_loops = nullptr;
    if (loop->kind == ForKind::kParallel) {
      ref_loops = &parallel;
    } else if (loop->kind == ForKind::kVectorized) {
      ref_loops = &vectorize;
    } else if (loop->kind == ForKind::kUnrolled) {
      ref_loops = &unroll;
    } else if (loop->kind == ForKind::kThreadBinding) {
      std::string thread_tag = loop->thread_binding.value()->thread_tag;
      if (thread_tag == "blockIdx.x") {
        ref_loops = &blockIdx_x;
      } else if (thread_tag == "blockIdx.y") {
        ref_loops = &blockIdx_y;
      } else if (thread_tag == "blockIdx.z") {
        ref_loops = &blockIdx_z;
      } else if (thread_tag == "threadIdx.x") {
        ref_loops = &threadIdx_x;
      } else if (thread_tag == "threadIdx.y") {
        ref_loops = &threadIdx_y;
      } else if (thread_tag == "threadIdx.z") {
        ref_loops = &threadIdx_z;
      } else if (support::StartsWith(thread_tag, "vthread")) {
        ref_loops = &vthread;
      } else {
        LOG(FATAL) << "ValueError: Unable to recognize thread tag: " << thread_tag;
      }
    }
    if (ref_loops != nullptr) {
      ref_loops->push_back(loop);
    }
    return ref_loops;
  }

  /*!
   * \brief Pop the last loop from the loop nest
   * \param loop The loop to be popped
   * \param ref_loops The list of for loops that the loop is bound to
   * \param auto_unroll_attr The auto unroll attribute of the loop
   */
  void Pop(const ForNode* loop, ForVec* ref_loops, int auto_unroll_attr) {
    if (ref_loops) {
      ref_loops->pop_back();
    }
    if (auto_unroll_attr > 0) {
      this->auto_unroll.pop_back();
    }
    if (const int64_t* extent = GetLoopIntExtent(loop)) {
      this->prod /= *extent;
    }
    this->loops.pop_back();
  }
};

}  // namespace utils
}  // namespace tir
}  // namespace tvm
