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
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../utils.h"
#include "./utils.h"

namespace tvm {
namespace tir {
namespace per_block_feature {

using utils::LoopNest;

/****** Group 1: Computation related features ******/

namespace group1 {

/*! \brief Group 1 features */
struct Feature {
  /*! \brief Arithmetic features */
  struct ArithOps {
    // Float-point arithmetic features
    int64_t float_mad = 0;         // The number of float MAD (Multiply–add) ops
    int64_t float_add_sub = 0;     // The number of float add and sub ops
    int64_t float_mul = 0;         // The number of float multiply ops
    int64_t float_div_mod = 0;     // The number of float div and mod ops
    int64_t float_cmp = 0;         // The number of float comparison ops
    int64_t float_math_func = 0;   // The number of float math func calls
    int64_t float_other_func = 0;  // The number of other float func calls
    // Integer arithmetic features
    int64_t int_mad = 0;         // The number of integer MAD (Multiply–add) ops
    int64_t int_add_sub = 0;     // The number of integer add and sub ops
    int64_t int_mul = 0;         // The number of integer multiply ops
    int64_t int_div_mod = 0;     // The number of integer div and mod ops
    int64_t int_cmp = 0;         // The number of integer comparison ops
    int64_t int_math_func = 0;   // The number of integer math func calls
    int64_t int_other_func = 0;  // The number of other integer func calls
    // Other arithmetic features
    int64_t bool_op = 0;    // The number of bool ops
    int64_t select_op = 0;  // The number of select ops

    static constexpr int64_t kCount = 16;

    ArithOps() = default;
    ArithOps(const BufferStoreNode* store, int64_t prod_loop_extent);

    void Export(std::vector<double>* v) const {
      double vs[] = {
          slog(float_mad), slog(float_add_sub),   slog(float_mul),        slog(float_div_mod),
          slog(float_cmp), slog(float_math_func), slog(float_other_func),  //
          slog(int_mad),   slog(int_add_sub),     slog(int_mul),          slog(int_div_mod),
          slog(int_cmp),   slog(int_math_func),   slog(int_other_func),  //
          slog(bool_op),   slog(select_op),
      };
      v->insert(v->end(), std::begin(vs), std::end(vs));
    }
  };

  /*! \brief Loop binding features */
  struct ForKindFeature {
    enum class Pos : int {
      kPosNone = 0,           // Does not have this kind of annotation
      kPosInnerSpatial = 1,   // The annotated iterator is the innermost spatial iterator
      kPosMiddleSpatial = 2,  // The annotated iterator is a middle spatial iterator
      kPosOuterSpatial = 3,   // The annotated iterator is the outermost spatial iterator
      kPosInnerReduce = 4,    // The annotated iterator is the innermost reduce iterator
      kPosMiddleReduce = 5,   // The annotated iterator is a middle reduce iterator
      kPosOuterReduce = 6,    // The annotated iterator is the outermost reduce iterator
      kPosMixed = 7,          // The annotated iterator is a mixed space and reduce iterator
    };
    int64_t num = 0;           // The number of iterators with the annotation
    int64_t prod = 0;          // The product of the lengths of iterators with the annotation
    int64_t len = 0;           // The length of the innermost iterator with the annotation
    Pos pos = Pos::kPosMixed;  // The position of the iterators with the annotation

    static constexpr int64_t kCount = 11;

    explicit ForKindFeature(const ForVec& loops);

    void Export(std::vector<double>* v) const {
      double vs[] = {
          slog(num),
          slog(prod),
          slog(len),
          static_cast<double>(static_cast<int>(pos) == 0),
          static_cast<double>(static_cast<int>(pos) == 1),
          static_cast<double>(static_cast<int>(pos) == 2),
          static_cast<double>(static_cast<int>(pos) == 3),
          static_cast<double>(static_cast<int>(pos) == 4),
          static_cast<double>(static_cast<int>(pos) == 5),
          static_cast<double>(static_cast<int>(pos) == 6),
          static_cast<double>(static_cast<int>(pos) == 7),
      };
      v->insert(v->end(), std::begin(vs), std::end(vs));
    }
  };

  ArithOps arith_ops;           // Arithmetic features
  ForKindFeature vectorize;     // Loop binding features: kVectorize
  ForKindFeature unroll;        // Loop binding features: kUnroll
  ForKindFeature parallel;      // Loop binding features: kParallel
  bool is_gpu = false;          // If the program is running on GPU
  int64_t blockIdx_x_len = 1;   // The length of blockIdx.x
  int64_t blockIdx_y_len = 1;   // The length of blockIdx.y
  int64_t blockIdx_z_len = 1;   // The length of blockIdx.z
  int64_t threadIdx_x_len = 1;  // The length of threadIdx.x
  int64_t threadIdx_y_len = 1;  // The length of threadIdx.y
  int64_t threadIdx_z_len = 1;  // The length of threadIdx.z
  int64_t vthread_len = 1;      // The length of virtual thread

  static constexpr int64_t kCount = ArithOps::kCount + ForKindFeature::kCount * 3 + 8;

  explicit Feature(const BlockRealizeNode* block_realize, const LoopNest& loop_nest, bool is_gpu)
      : vectorize(loop_nest.vectorize), unroll(loop_nest.unroll), parallel(loop_nest.parallel) {
    if (is_gpu) {
      this->is_gpu = true;
      this->blockIdx_x_len = utils::FirstLoopExtent(loop_nest.blockIdx_x, 1);
      this->blockIdx_y_len = utils::FirstLoopExtent(loop_nest.blockIdx_y, 1);
      this->blockIdx_z_len = utils::FirstLoopExtent(loop_nest.blockIdx_z, 1);
      this->threadIdx_x_len = utils::FirstLoopExtent(loop_nest.threadIdx_x, 1);
      this->threadIdx_y_len = utils::FirstLoopExtent(loop_nest.threadIdx_y, 1);
      this->threadIdx_z_len = utils::FirstLoopExtent(loop_nest.threadIdx_z, 1);
      this->vthread_len = utils::FirstLoopExtent(loop_nest.vthread, 1);
    }
  }

  void Export(std::vector<double>* v) const {
    this->arith_ops.Export(v);
    this->vectorize.Export(v);
    this->unroll.Export(v);
    this->parallel.Export(v);
    double vs[] = {
        static_cast<double>(is_gpu),  //
        slog(blockIdx_x_len),        slog(blockIdx_y_len),  slog(blockIdx_z_len),
        slog(threadIdx_x_len),       slog(threadIdx_y_len), slog(threadIdx_z_len),
        slog(vthread_len),
    };
    v->insert(v->end(), std::begin(vs), std::end(vs));
  }
};

Feature::ArithOps::ArithOps(const BufferStoreNode* store, int64_t prod_loop_extent) {
  class ArithOpCounter : public ExprVisitor {
   public:
#define TVM_FEATURE_SIMPLE(Type, Counter)       \
  void VisitExpr_(const Type* op) final {       \
    result_.Counter += this->prod_loop_extent_; \
    ExprVisitor::VisitExpr_(op);                \
  }
#define TVM_FEATURE_BINARY(Type, FloatCounter, IntCounter) \
  void VisitExpr_(const Type* op) final {                  \
    if (op->dtype.is_float()) {                            \
      result_.FloatCounter += this->prod_loop_extent_;     \
    } else {                                               \
      result_.IntCounter += this->prod_loop_extent_;       \
    }                                                      \
    ExprVisitor::VisitExpr_(op);                           \
  }
    TVM_FEATURE_SIMPLE(AndNode, bool_op);
    TVM_FEATURE_SIMPLE(OrNode, bool_op);
    TVM_FEATURE_SIMPLE(NotNode, bool_op);
    TVM_FEATURE_SIMPLE(SelectNode, select_op);
    TVM_FEATURE_BINARY(AddNode, float_add_sub, int_add_sub);
    TVM_FEATURE_BINARY(SubNode, float_add_sub, int_add_sub);
    TVM_FEATURE_BINARY(MulNode, float_mul, int_mul);
    TVM_FEATURE_BINARY(DivNode, float_div_mod, int_div_mod);
    TVM_FEATURE_BINARY(ModNode, float_div_mod, int_div_mod);
    TVM_FEATURE_BINARY(FloorDivNode, float_div_mod, int_div_mod);
    TVM_FEATURE_BINARY(FloorModNode, float_div_mod, int_div_mod);
    TVM_FEATURE_BINARY(MaxNode, float_cmp, int_cmp);
    TVM_FEATURE_BINARY(MinNode, float_cmp, int_cmp);
    TVM_FEATURE_BINARY(EQNode, float_cmp, int_cmp);
    TVM_FEATURE_BINARY(NENode, float_cmp, int_cmp);
    TVM_FEATURE_BINARY(LTNode, float_cmp, int_cmp);
    TVM_FEATURE_BINARY(LENode, float_cmp, int_cmp);
    TVM_FEATURE_BINARY(GTNode, float_cmp, int_cmp);
    TVM_FEATURE_BINARY(GENode, float_cmp, int_cmp);
#undef TVM_FEATURE_BINARY
#undef TVM_FEATURE_SIMPLE

    void VisitExpr_(const CallNode* op) final {
      static auto op_call_effect_ = Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
      TCallEffectKind effect_kind = op_call_effect_[Downcast<Op>(op->op)];
      bool is_pure =
          effect_kind == CallEffectKind::kPure || effect_kind == CallEffectKind::kExprAnnotation;
      if (is_pure) {
        if (op->dtype.is_float()) {
          result_.float_math_func += prod_loop_extent_;
        } else {
          result_.int_math_func += prod_loop_extent_;
        }
      } else {
        if (op->dtype.is_float()) {
          result_.float_other_func += prod_loop_extent_;
        } else {
          result_.int_other_func += prod_loop_extent_;
        }
      }
      ExprVisitor::VisitExpr_(op);
    }

    int64_t prod_loop_extent_;
    ArithOps result_;
  };
  ArithOpCounter counter;
  counter.prod_loop_extent_ = prod_loop_extent;
  counter(store->value);
  *this = counter.result_;
}

Feature::ForKindFeature::ForKindFeature(const ForVec& loops) {
  if (loops.empty()) {
    this->num = 0;
    this->prod = 0;
    this->len = 0;
    this->pos = ForKindFeature::Pos::kPosNone;
  } else {
    const int64_t* last_loop_extent = GetLoopIntExtent(loops.back());
    this->num = loops.size();
    this->len = last_loop_extent ? *last_loop_extent : 1;
    this->pos = ForKindFeature::Pos::kPosMixed;
    int64_t& prod = this->prod = 1;
    for (const ForNode* loop : loops) {
      if (const int64_t* extent = GetLoopIntExtent(loop)) {
        prod *= *extent;
      }
    }
  }
}

}  // namespace group1

/*! \brief The feature extracted */
struct Feature {
  const BlockRealizeNode* block_realize = nullptr;
  // int buffer_order = -1;
  // TODO: add feature group 1-5
  std::unique_ptr<group1::Feature> group1 = nullptr;
  // std::unique_ptr<group2::Feature> group2 = nullptr;
  // std::unique_ptr<group3::Feature> group3 = nullptr;
  // std::unique_ptr<group4::Feature> group4 = nullptr;
  // std::unique_ptr<group5::Feature> group5 = nullptr;

  // bool operator<(const Feature& other) const { return buffer_order < other.buffer_order; }
};

/*! \brief The main feature extractor */
class PerBlockFeatureCollector : private StmtVisitor {
 public:
  static std::vector<Feature> Collect(bool is_gpu, int64_t cache_line_bytes,
                                      int64_t arith_intensity_curve_num_samples,
                                      const IRModule& mod) {
    PerBlockFeatureCollector collector(is_gpu, cache_line_bytes, arith_intensity_curve_num_samples);
    for (const auto& kv : mod->functions) {
      if (const PrimFuncNode* func = kv.second.as<PrimFuncNode>()) {
        collector(func->body);
        // TODO: handle allocation-related features
        // for (const auto& it : func->buffer_map) {
        //   collector.HandleBufferAlloc(it.second);
        // }
      }
    }
    std::vector<Feature> result;
    result.reserve(collector.block_features_.size());
    for (auto& it : collector.block_features_) {
      Feature& feature = it.second;
      if (feature.block_realize != nullptr) {
        // TODO: add feature group 1-5
        // ICHECK(feature.group1);
        // ICHECK(feature.group2);
        // ICHECK(feature.group3);
        // ICHECK(feature.group5);
        // if (feature.group4 == nullptr) {
        //   feature.group4 = std::make_unique<group4::Feature>();
        // }
        result.push_back(std::move(feature));
      }
    }
    // std::sort(result.begin(), result.end());
    return result;
  }

 private:
  explicit PerBlockFeatureCollector(bool is_gpu, int64_t cache_line_bytes,
                                    int64_t arith_intensity_curve_num_samples)
      : is_gpu_(is_gpu),
        cache_line_bytes_(cache_line_bytes),
        arith_intensity_curve_num_samples_(arith_intensity_curve_num_samples) {}

  void VisitStmt_(const BlockRealizeNode* block) final {
    if (!scopes_.empty()) {
      ordered_blocks_.push_back(block);
    }
    scopes_.push_back(block);
    dfs_path_.push_back(block);
    StmtVisitor::VisitStmt_(block);
    dfs_path_.pop_back();
    scopes_.pop_back();
    if (scopes_.empty()) {
      return;
    }
    // Get the ancestor loops from inner to outer, up to the parent scope
    ForVec loops;
    for (auto iter = dfs_path_.rbegin(); iter != dfs_path_.rend(); ++iter) {
      const StmtNode* stmt = *iter;
      if (stmt->IsInstance<ForNode>()) {
        loops.push_back(static_cast<const ForNode*>(stmt));
      }
    }
    Feature& feature = block_features_[block];
    feature.block_realize = block;
    // TODO: add groups 1-5
  }

  void VisitStmt_(const ForNode* loop) final {
    int64_t auto_unroll;
    ForVec* for_vec = loop_nest_.Push(loop, &auto_unroll);
    StmtVisitor::VisitStmt_(loop);
    loop_nest_.Pop(loop, for_vec, auto_unroll);
  }

  void VisitStmt_(const BufferStoreNode* store) final {}

  bool is_gpu_;
  int64_t cache_line_bytes_;
  int64_t arith_intensity_curve_num_samples_;
  arith::Analyzer analyzer_;
  BlockVec scopes_;
  BlockVec ordered_blocks_;
  PathVec dfs_path_;
  LoopNest loop_nest_ = {};
  IntVec for_touched_bytes_ = {};
  ForBufferMap<IntVec> buffer_touched_under_loop_ = {};
  std::unordered_map<const BlockRealizeNode*, Feature> block_features_ = {};
};

}  // namespace per_block_feature
}  // namespace tir
}  // namespace tvm

namespace tvm {
namespace meta_schedule {

class PerBlockFeatureNode : public FeatureExtractorNode {
 public:
  int buffers_per_block;
  int arith_intensity_curve_num_samples;
  int cache_line_bytes;
  bool extract_workload;
  int feature_vector_length;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("buffers_per_block", &buffers_per_block);
    v->Visit("arith_intensity_curve_num_samples", &arith_intensity_curve_num_samples);
    v->Visit("cache_line_bytes", &cache_line_bytes);
    v->Visit("feature_vector_length", &feature_vector_length);
  }

  void ExtractSingle(IRModule mod, bool is_gpu, std::vector<std::vector<double>>* results) {
    using namespace tvm::tir::per_block_feature;
    static transform::Sequential passes = tir::transform::PassListForFeatureExtraction();
    mod = passes(std::move(mod));
    std::vector<Feature> features = PerBlockFeatureCollector::Collect(
        is_gpu, this->cache_line_bytes, this->arith_intensity_curve_num_samples, mod);
    int n_features = features.size();
    results->resize(n_features);
    for (int i = 0; i < n_features; ++i) {
      const Feature& feature = features[i];
      std::vector<double>& result = (*results)[i];
      result.reserve(feature_vector_length);
      // TODO: add feature group 1-5
      feature.group1->Export(&result);
      // feature.group2->Export(&result, this->buffers_per_block);
      // feature.group3->Export(&result);
      // feature.group4->Export(&result, feature.group5->outer_prod);
      // feature.group5->Export(&result);
    }
  }

  Array<runtime::NDArray> ExtractFrom(const TuneContext& tune_context,
                                      const Array<MeasureCandidate>& candidates) {
    using namespace tvm::tir::per_block_feature;
    bool is_gpu = tune_context->target.value()->kind->name == "cuda";
    std::vector<runtime::NDArray> results;
    results.resize(candidates.size());
    auto f = [this, is_gpu, &candidates, &results](int, int task_id) -> void {
      const auto& candidate = candidates[task_id];
      std::vector<std::vector<double>> features;
      ExtractSingle(DeepCopyIRModule(candidate->sch->mod()), is_gpu, &features);
      results[task_id] = tir::utils::AsNDArray(features);
    };
    support::parallel_for_dynamic(0, candidates.size(), tune_context->num_threads, f);
    return results;
  }

  static constexpr const char* _type_key = "meta_schedule.PerBlockFeature";
  TVM_DECLARE_FINAL_OBJECT_INFO(PerBlockFeatureNode, FeatureExtractorNode);
};

FeatureExtractor FeatureExtractor::PerBlockFeature(int buffers_per_block,
                                                   int arith_intensity_curve_num_samples,
                                                   int cache_line_bytes, bool extract_workload) {
  using namespace tvm::tir::per_block_feature;
  ObjectPtr<PerBlockFeatureNode> n = make_object<PerBlockFeatureNode>();
  n->buffers_per_block = buffers_per_block;
  n->arith_intensity_curve_num_samples = arith_intensity_curve_num_samples;
  n->cache_line_bytes = cache_line_bytes;
  n->extract_workload = extract_workload;
  // TODO: add feature group 1-5
  // n->feature_vector_length = group1::Feature::kCount +                                  //
  //                            group2::Feature::SubFeature::kCount * buffers_per_block +  //
  //                            arith_intensity_curve_num_samples +                        //
  //                            group4::Feature::kCount +                                  //
  //                            group5::Feature::kCount;
  return FeatureExtractor(n);
}

TVM_REGISTER_NODE_TYPE(PerBlockFeatureNode);
TVM_REGISTER_GLOBAL("meta_schedule.FeatureExtractorPerBlockFeature")
    .set_body_typed(FeatureExtractor::PerBlockFeature);

}  // namespace meta_schedule
}  // namespace tvm
