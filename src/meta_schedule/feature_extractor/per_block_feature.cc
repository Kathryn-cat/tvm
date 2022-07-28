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
#include <tvm/arith/int_set.h>
#include <tvm/support/parallel_for.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

#include "../../tir/schedule/utils.h"
#include "../utils.h"

namespace tvm {
namespace meta_schedule {

template <class K, class V>
using ObjMap = std::unordered_map<const K*, V>;

template <class K1, class K2, class V>
using ObjPairMap = ObjMap<K1, ObjMap<K2, V>>;

using NDIntSet = Array<arith::IntSet>;

// shifted log to incorporate the property that slog(0) = 0
inline double slog(double x) {
  if (x < 0) {
    x = -x;
  }
  return std::log2(x + 1);
}

std::ostream& operator<<(std::ostream& os, const NDIntSet& nd_int_set) {
  os << '[';
  bool is_first = true;
  for (const arith::IntSet& int_set : nd_int_set) {
    if (is_first) {
      is_first = false;
    } else {
      os << ", ";
    }
    PrimExpr min = int_set.min();
    PrimExpr max = int_set.max();
    os << min << ":" << max;
  }
  os << ']';
  return os;
}

struct DoubleNDArrayPusher {
  explicit DoubleNDArrayPusher(const std::vector<int64_t>& shape)
      : array(runtime::NDArray::Empty(/*shape=*/shape, /*dtype=*/DLDataType{kDLFloat, 64, 1},
                                      /*ctx=*/DLDevice{kDLCPU, 0})),
        back(static_cast<double*>(array->data)) {}

  template <class TIter>
  void Push(TIter begin, TIter end) {
    while (begin != end) {
      *back = *begin;
      ++back;
      ++begin;
    }
  }

  void PushRepeat(int n, double value) {
    while (n-- > 0) {
      *back = value;
      ++back;
    }
  }

  runtime::NDArray Done() {
    int64_t* shape = array->shape;
    int64_t array_size = 1;
    for (int i = 0, ndim = array->ndim; i < ndim; ++i) {
      array_size *= shape[i];
    }
    int64_t written_size = back - static_cast<double*>(array->data);
    ICHECK_EQ(array_size, written_size);
    return std::move(array);
  }

  runtime::NDArray array;
  double* back;
};

struct FeatureSet {
  // Group 1: Computation related features
  const tir::BlockRealizeNode* block_realize;
  // Group 2: Buffer access related features (per buffer)
  // Group 3: Arithmetic intensity related features
  // Group 4: Allocation related features
  // Group 5: Outer scope related features
};

#define TVM_FEATURE_INC_CNT(DType, FloatCounter, IntCounter) \
  if (DType.is_float()) {                                    \
    ++result_->FloatCounter;                                 \
  } else {                                                   \
    ++result_->IntCounter;                                   \
  }

#define TVM_FEATURE_SIMPLE(Type, Counter) \
  void VisitExpr_(const Type* op) final { \
    ++result_->Counter;                   \
    StmtExprVisitor::VisitExpr_(op);      \
  }

#define TVM_FEATURE_BINARY(Type, FloatCounter, IntCounter) \
  void VisitExpr_(const Type* op) final {                  \
    if (op->dtype.is_float()) {                            \
      ++result_->FloatCounter;                             \
    } else {                                               \
      ++result_->IntCounter;                               \
    }                                                      \
    StmtExprVisitor::VisitExpr_(op);                       \
  }

class MathOpCounter : public tir::StmtExprVisitor {};

#undef TVM_FEATURE_BINARY
#undef TVM_FEATURE_SIMPLE
#undef TVM_FEATURE_INC_CNT

class CoefficientExtractor : public tir::StmtExprVisitor {
 public:
  explicit CoefficientExtractor(const tir::Var& var)
      : var(var), stride(0), visited_var(false), visited_add(false), visited_mul(false) {}

  void VisitExpr_(const tir::MulNode* node) override {
    StmtExprVisitor::VisitExpr_(node);
    if (visited_var) {
      if (!visited_add) {
        if (const auto* a = node->a.as<IntImmNode>()) {
          visited_mul = true;
          stride = a->value;
        } else if (const auto* b = node->b.as<IntImmNode>()) {
          visited_mul = true;
          stride = b->value;
        }
      }
    }
  }

  void VisitExpr_(const tir::AddNode* node) override {
    StmtExprVisitor::VisitExpr_(node);
    if (visited_var) {
      if (!visited_mul) {
        visited_add = true;
        stride = 1;
      }
    }
  }

  void VisitExpr_(const tir::VarNode* node) override {
    if (node == var.get()) {
      visited_var = true;
      stride = 2;  // This is a magic default stride in case our approximation strategy fails
    }
  }

  static int64_t Extract(const PrimExpr& expr, const tir::Var& var) {
    CoefficientExtractor extractor(var);
    extractor.VisitExpr(expr);
    return (extractor.visited_var && !extractor.visited_mul && !extractor.visited_add)
               ? 1
               : (extractor.visited_var ? extractor.stride : 0);
  }

  const tir::Var& var;
  int64_t stride;
  bool visited_var;
  bool visited_add;
  bool visited_mul;
};

class PerBlockFeatureExtractor : public tir::StmtExprVisitor {
 public:
  static std::vector<FeatureSet> Extract(const tir::PrimFunc& func) {
    PerBlockFeatureExtractor extractor;
    extractor.VisitStmt(func->body);
    std::vector<FeatureSet> result;
    result.reserve(extractor.ordered_blocks_.size());
    for (const tir::BlockRealizeNode* realize : extractor.ordered_blocks_) {
      if (!realize->block->name_hint.empty()) {
        result.push_back(extractor.per_block_feature_.at(realize));
      }
    }
    return result;
  }

 private:
  /******** Visitors ********/
  void VisitStmt_(const tir::BlockRealizeNode* realize) override {
    if (!scopes_.empty()) {
      ordered_blocks_.push_back(realize);
    }
    scopes_.push_back(realize);
    dfs_path_.push_back(realize);
    tir::StmtExprVisitor::VisitStmt_(realize);
    dfs_path_.pop_back();
    scopes_.pop_back();
    if (scopes_.empty()) {
      return;
    }
    // Get the ancestor loops from inner to outer, up to the parent scope
    std::vector<const tir::ForNode*> loops;
    for (auto iter = dfs_path_.rbegin(); iter != dfs_path_.rend(); ++iter) {
      const tir::StmtNode* stmt = *iter;
      if (stmt->IsInstance<tir::ForNode>()) {
        loops.push_back(static_cast<const tir::ForNode*>(stmt));
      }
    }
    FeatureSet& feature = per_block_feature_[realize];
    feature.block_realize = realize;
    // Group 1: Computation related features
    // Group 2: Buffer access related features
    // Group 3: Arithmetic intensity related features
    // Group 4: Allocation related features
    // Group 5: Outer scope related features
  }

  void VisitStmt_(const tir::ForNode* loop) override {
    int64_t auto_unroll = -1;
    int64_t extent = *GetLoopIntExtent(loop);
    if (extent == -1) {
      extent = 1;
    }
    // Handling annotated loops
    std::vector<const tir::ForNode*>* ref_loops = nullptr;
    if (!loop->annotations.empty()) {
      for (const auto& ann : loop->annotations) {
        if (ann.first == "pragma_auto_unroll_max_step") {
          auto_unroll = Downcast<Integer>(ann.second)->value;
        }
      }
    }

    if (loop->kind == tir::ForKind::kParallel) {
      ref_loops = &parallel_;
    } else if (loop->kind == tir::ForKind::kVectorized) {
      ref_loops = &vectorize_;
    } else if (loop->kind == tir::ForKind::kUnrolled) {
      ref_loops = &unroll_;
    } else if (loop->kind == tir::ForKind::kThreadBinding) {
      ICHECK(loop->thread_binding.defined());
      std::string thread_tag = loop->thread_binding.value()->thread_tag;
      if (thread_tag == "blockIdx.x") {
        ref_loops = &blockIdx_x_;
      } else if (thread_tag == "blockIdx.y") {
        ref_loops = &blockIdx_y_;
      } else if (thread_tag == "blockIdx.z") {
        ref_loops = &blockIdx_z_;
      } else if (thread_tag == "threadIdx.x") {
        ref_loops = &threadIdx_x_;
      } else if (thread_tag == "threadIdx.y") {
        ref_loops = &threadIdx_y_;
      } else if (thread_tag == "threadIdx.z") {
        ref_loops = &threadIdx_z_;
      } else if (thread_tag == "vthread") {
        ref_loops = &vthread_;
      }
    }

    if (ref_loops != nullptr) {
      ref_loops->push_back(loop);
    }
    if (auto_unroll != -1) {
      auto_unroll_.push_back(auto_unroll);
    }
    outer_loop_prod_ *= extent;
    if (extent != 1 || ref_loops != nullptr) {
      dfs_path_.push_back(loop);
      loops_.push_back(loop);
      analyzer_.Bind(loop->loop_var, loop->min);
    }
    tir::StmtExprVisitor::VisitStmt_(loop);
    if (extent != 1 || ref_loops != nullptr) {
      loops_.pop_back();
      dfs_path_.pop_back();
    }
    outer_loop_prod_ /= extent;
    if (auto_unroll != -1) {
      auto_unroll_.pop_back();
    }
    if (ref_loops != nullptr) {
      ref_loops->pop_back();
    }
  }

  void VisitStmt_(const tir::BufferStoreNode* store) override {}

 private:
  static int64_t ProdLoopExtent(const std::vector<const tir::ForNode*>& loops) {
    int64_t prod = 1;
    for (const tir::ForNode* loop : loops) {
      int64_t extent = *GetLoopIntExtent(loop);
      if (extent != -1) {
        prod *= extent;
      }
    }
    return prod;
  }

  static int64_t FirstLoopExtent(const std::vector<const tir::ForNode*>& loops) {
    if (loops.empty()) {
      return 1;
    }
    int64_t extent = *GetLoopIntExtent(loops[0]);
    if (extent == -1) {
      return 1;
    }
    return extent;
  }

 private:
  /******** Data structure used in recursive visiting ********/
  /*! \brief The scope info used in recursive visiting */
  std::vector<const tir::BlockRealizeNode*> scopes_;
  /*! \brief The loop / block-realize visited up-down in the DFS path */
  std::vector<const tir::StmtNode*> dfs_path_;
  // The stacks to store different kinds of for-loops
  std::vector<const tir::ForNode*> loops_;
  std::vector<const tir::ForNode*> parallel_;
  std::vector<const tir::ForNode*> vectorize_;
  std::vector<const tir::ForNode*> unroll_;
  std::vector<const tir::ForNode*> blockIdx_x_;
  std::vector<const tir::ForNode*> blockIdx_y_;
  std::vector<const tir::ForNode*> blockIdx_z_;
  std::vector<const tir::ForNode*> threadIdx_x_;
  std::vector<const tir::ForNode*> threadIdx_y_;
  std::vector<const tir::ForNode*> threadIdx_z_;
  std::vector<const tir::ForNode*> vthread_;
  std::vector<int64_t> auto_unroll_;
  /*! \brief The persistent analyzer */
  mutable arith::Analyzer analyzer_;
  /*! \brief The product of the extents of outer loops */
  int64_t outer_loop_prod_ = 1;
  /*!
   * \brief For a specific buffer, record the regions it is acccessed under a specific loop.
   * The information is preserved across different blocks and is used for detecting serial buffer
   * reuse
   */
  ObjPairMap<tir::ForNode, tir::BufferNode, std::vector<int64_t>> buffer_touched_under_loop_;
  /*! \brief The output: features for each BlockRealizeNode */
  ObjMap<tir::BlockRealizeNode, FeatureSet> per_block_feature_;
  /*! \brief The pre-order visit order of all the BlockRealizeNodes */
  std::vector<const tir::BlockRealizeNode*> ordered_blocks_;
};

runtime::NDArray PerBlockFeature(const tir::Schedule& sch, int max_num_buffer_access_features) {}

Array<String> PerBlockFeatureNames(int max_num_buffer_access_features) {}

Array<runtime::NDArray> PerBlockFeatureBatched(const Array<tir::Schedule>& schs,
                                               int max_num_buffer_access_features) {}

}  // namespace meta_schedule
}  // namespace tvm
