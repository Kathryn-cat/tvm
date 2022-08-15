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

    void AddExpr(const PrimExpr& expr, int64_t unit) {
#define TVM_FEATURE_SIMPLE(Type, Counter) \
  void VisitExpr_(const Type* op) final { \
    result_->Counter += this->unit_;      \
    ExprVisitor::VisitExpr_(op);          \
  }
#define TVM_FEATURE_BINARY(Type, FloatCounter, IntCounter) \
  void VisitExpr_(const Type* op) final {                  \
    if (op->dtype.is_float()) {                            \
      result_->FloatCounter += this->unit_;                \
    } else {                                               \
      result_->IntCounter += this->unit_;                  \
    }                                                      \
    ExprVisitor::VisitExpr_(op);                           \
  }
      class ArithOpCounter : public ExprVisitor {
       public:
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
        void VisitExpr_(const CallNode* op) final {
          static auto op_call_effect_ = Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
          TCallEffectKind effect_kind = op_call_effect_[Downcast<Op>(op->op)];
          bool is_pure = effect_kind == CallEffectKind::kPure ||
                         effect_kind == CallEffectKind::kExprAnnotation;
          if (is_pure) {
            if (op->dtype.is_float()) {
              result_->float_math_func += unit_;
            } else {
              result_->int_math_func += unit_;
            }
          } else {
            if (op->dtype.is_float()) {
              result_->float_other_func += unit_;
            } else {
              result_->int_other_func += unit_;
            }
          }
          ExprVisitor::VisitExpr_(op);
        }

        int64_t unit_;
        ArithOps* result_;
      };
#undef TVM_FEATURE_BINARY
#undef TVM_FEATURE_SIMPLE
      ArithOpCounter counter;
      counter.unit_ = unit;
      counter.result_ = this;
      counter(expr);
    }

    // Feature::ArithOps::ArithOps(const BufferStoreNode* store, int64_t prod_loop_extent) {
    //   ArithOpCounter counter;
    //   counter.prod_loop_extent_ = prod_loop_extent;
    //   counter(store->value);
    //   *this = counter.result_;
    // }
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

namespace group2 {

/*! \brief Group 2 features */
struct Feature {
  enum class AccessType : int {
    /*! The buffer is read but not written */
    kRead = 0,
    /*! The buffer is written but not read */
    kWrite = 1,
    /*! The buffer is both read and written */
    kReadWrite = 2,
    /*! Unknown type */
    kUnknownRW = 3,
  };
  enum class ReuseType : int {
    /*! Buffer reuse because accessed on each iteration of a loop */
    kLoopMultipleRead = 0,
    /*! Buffer reuse because it is serially accessed */
    kSerialMultipleReadWrite = 1,
    /*! No buffer reuse */
    kNoReuse = 2,
  };

  struct SubFeature {
    /*! \brief The buffer this feature is for */
    const BufferNode* buffer = nullptr;
    /*! \brief The access type of the buffer */
    AccessType access_type = AccessType::kUnknownRW;
    /*! \brief The regions that the buffer is accessed */
    std::vector<NDIntSet> regions = {};
    // Access information
    /*! \brief loop_accessed_numel[i][...] means the number of elements accessed by loops[i] */
    std::vector<std::unordered_map<const BufferNode*, int64_t>> loop_accessed_numel = {};
    /*! \brief The shape of the data access */
    IntVec access_shape;
    /*! \brief The bytes that are continuously accessed */
    int64_t num_continuous_bytes = 1;
    // Stride information
    /*! \brief The min stride of the access */
    int64_t min_stride = 0;
    /*! \brief The innermost stride */
    int64_t innermost_stride = 0;
    /*! \brief The product of the non-strided loops */
    int64_t prod_non_strided_loop_extent = 0;
    // Reuse information
    /*! The type of data reuse */
    ReuseType reuse_type = ReuseType::kNoReuse;
    /*! The reuse distance in terms of number of iterations */
    double reuse_dis_iter = 0.0;
    /*! The reuse distance in terms of bytes */
    double reuse_dis_bytes = 0.0;
    /*! The reuse count */
    int64_t reuse_ct = 0;
    // Features
    /*! The touched memory in bytes */
    double bytes;
    /*! The touched unique memory in bytes */
    double unique_bytes;
    /*! The number of touched cache lines */
    double lines;
    /*! The number touched unique cache lines */
    double unique_lines;
    /*! bytes / reuse_ct */
    double bytes_d_reuse_ct;
    /*! unique_bytes / reuse_ct */
    double unique_bytes_d_reuse_ct;
    /*! lines / reuse_ct */
    double lines_d_reuse_ct;
    /*! unique_lines / reuse_ct */
    double unique_lines_d_reuse_ct;
    /*! The stride in access */
    double stride;

    static constexpr int64_t kCount = 18;

    void Export(std::vector<double>* v) const {
      double vs[] = {
          static_cast<double>(static_cast<int>(access_type) == 0),
          static_cast<double>(static_cast<int>(access_type) == 1),
          static_cast<double>(static_cast<int>(access_type) == 2),
          // FeatureSet::BufferAccess::AccessType::kUnknownRW is ignored
          slog(bytes),
          slog(unique_bytes),
          slog(lines),
          slog(unique_lines),
          static_cast<double>(static_cast<int>(reuse_type) == 0),
          static_cast<double>(static_cast<int>(reuse_type) == 1),
          static_cast<double>(static_cast<int>(reuse_type) == 2),
          slog(reuse_dis_iter),
          slog(reuse_dis_bytes),
          slog(reuse_ct),
          slog(bytes_d_reuse_ct),
          slog(unique_bytes_d_reuse_ct),
          slog(lines_d_reuse_ct),
          slog(unique_lines_d_reuse_ct),
          slog(stride),
      };
      v->insert(v->end(), std::begin(vs), std::end(vs));
    }

    static void Pad(std::vector<double>* v) { v->insert(v->end(), 18, 0.0); }

    void SetStride(const LoopNest& loop_nest, arith::Analyzer* analyzer);

    void SetReuse(const LoopNest& loop_nest,     //
                  int64_t top_loop_touch_bytes,  //
                  const ForBufferMap<IntVec>& buffer_touched_under_loop);

    void SetFeature(const LoopNest& loop_nest, int64_t cache_line_bytes);

    explicit SubFeature(const BufferNode* buffer, AccessType access_type,
                        std::vector<NDIntSet> regions, int n_loops)
        : buffer(buffer),
          access_type(access_type),
          regions(regions),
          loop_accessed_numel(n_loops) {}
  };

  void Export(std::vector<double>* v, int buffers_per_store) const {
    int n = sub_features.size();
    for (int i = 0; i < buffers_per_store; ++i) {
      if (i < n) {
        sub_features[i].Export(v);
      } else {
        SubFeature::Pad(v);
      }
    }
  }

  explicit Feature(const BlockRealizeNode* realize, const LoopNest& loop_nest,
                   int64_t cache_line_bytes, IntVec* for_touched_bytes,
                   ForBufferMap<IntVec>* buffer_touched_under_loop, arith::Analyzer* analyzer,
                   std::vector<const BlockRealizeNode*>* scopes);

  void Init(const BlockRealizeNode* realize, int n_loops, arith::Analyzer* analyzer,
            std::vector<const BlockRealizeNode*>* scopes);

  void SetRegion(const LoopNest& loop_nest,                        //
                 IntVec* for_touched_bytes,                        //
                 ForBufferMap<IntVec>* buffer_touched_under_loop,  //
                 arith::Analyzer* analyzer);

  std::vector<SubFeature> sub_features;
};

void Feature::Init(const BlockRealizeNode* realize, int n_loops, arith::Analyzer* analyzer,
                   std::vector<const BlockRealizeNode*>* scopes) {
  // Step 1. Check if each region is 'update'
  int n_reads = realize->block->reads.size();
  int n_writes = realize->block->writes.size();
  std::vector<int> is_read_update(n_reads, 0);
  std::vector<int> is_write_update(n_writes, 0);
  for (int i_r = 0; i_r < n_reads; ++i_r) {
    // Enumerate each read region
    const BufferRegion& r = realize->block->reads[i_r];
    for (int i_w = 0; i_w < n_writes; ++i_w) {
      // Enumerate each write region
      const BufferRegion& w = realize->block->writes[i_w];
      if (r->buffer.same_as(w->buffer)) {
        const Array<Range>& r_region = r->region;
        const Array<Range>& w_region = w->region;
        int ndim = r_region.size();
        // Check if `r_region` and `w_region` are exactly the same
        bool is_same = true;
        for (int i = 0; i < ndim; ++i) {
          if (!analyzer->CanProve(r_region[i]->min == w_region[i]->min) ||
              !analyzer->CanProve(r_region[i]->extent == w_region[i]->extent)) {
            is_same = false;
            break;
          }
        }
        // If so, mark it
        if (is_same) {
          is_read_update[i_r] = true;
          is_write_update[i_w] = true;
        }
      }
    }
  }
  // Step 2. Extract the block vars in the parent scope
  std::unordered_map<const VarNode*, PrimExpr> var_substitutes;
  auto f_substitute = [analyzer, &var_substitutes](const PrimExpr& expr) -> PrimExpr {
    return analyzer->Simplify(
        Substitute(expr, [&var_substitutes](const PrimExpr& expr) -> Optional<PrimExpr> {
          if (const auto* var = expr.as<VarNode>()) {
            auto it = var_substitutes.find(var);
            if (it != var_substitutes.end()) {
              return it->second;
            }
          }
          return NullOpt;
        }));
  };
  // Step 2.1. Substitute block vars of the parent scope to its min
  if (!scopes->empty()) {
    const BlockNode* parent_block = scopes->back()->block.operator->();
    for (const IterVar& block_var : parent_block->iter_vars) {
      var_substitutes[block_var->var.get()] = block_var->dom->min;
    }
  }
  // Step 2.2. Substitute block vars to its binding
  {
    ICHECK_EQ(realize->iter_values.size(), realize->block->iter_vars.size());
    int n = realize->iter_values.size();
    for (int i = 0; i < n; ++i) {
      const Var& lhs = realize->block->iter_vars[i]->var;
      const PrimExpr& rhs = realize->iter_values[i];
      var_substitutes[lhs.get()] = f_substitute(rhs);
    }
  }
  // Step 2.3. Helper to convert a TIR region into our int-set and do necessary simplification
  auto f_make_int_set = [&f_substitute](const Array<Range>& region) -> NDIntSet {
    // Helper function to do the substitution
    int ndim = region.size();
    NDIntSet result;
    result.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
      const Range& range = region[i];
      PrimExpr min = f_substitute(range->min);
      PrimExpr max = f_substitute(min + range->extent - Integer(1));
      result.push_back(arith::IntSet::Interval(min, max));
    }
    return result;
  };
  // Step 3. Apply the substitution to each tensor region
  struct BufferInfo {
    AccessType access_type = AccessType::kUnknownRW;
    std::vector<NDIntSet> regions = {};
  };
  std::unordered_map<const BufferNode*, BufferInfo> buffer_info;
  buffer_info.reserve(realize->block->reads.size() + realize->block->writes.size());
  for (int i = 0; i < n_reads; ++i) {
    // Skip those update regions
    if (is_read_update[i]) {
      continue;
    }
    const BufferRegion& region = realize->block->reads[i];
    BufferInfo& info = buffer_info[region->buffer.get()];
    info.access_type = AccessType::kRead;
    info.regions.push_back(f_make_int_set(region->region));
  }
  for (int i = 0; i < n_writes; ++i) {
    const BufferRegion& region = realize->block->writes[i];
    BufferInfo& info = buffer_info[region->buffer.get()];
    if (is_write_update[i] || info.access_type == AccessType::kRead) {
      info.access_type = AccessType::kReadWrite;
    } else {
      info.access_type = AccessType::kWrite;
    }
    info.regions.push_back(f_make_int_set(region->region));
  }

  this->sub_features.reserve(buffer_info.size());
  for (const auto& kv : buffer_info) {
    this->sub_features.emplace_back(kv.first, kv.second.access_type, std::move(kv.second.regions),
                                    n_loops);
  }
}

void Feature::SetRegion(const LoopNest& loop_nest, IntVec* for_touched_bytes,
                        ForBufferMap<IntVec>* buffer_touched_under_loop,
                        arith::Analyzer* analyzer) {
  int n_loops = loop_nest.loops.size();
  const std::vector<const ForNode*>& loops = loop_nest.loops;
  // Step 1. Initialize and bind all the loop variables to a constant
  *for_touched_bytes = IntVec(n_loops, 0);
  for (int i = 0; i < n_loops; ++i) {
    const ForNode* loop = loops[i];
    analyzer->Bind(loop->loop_var, loop->min, /*allow_override=*/true);
  }
  // Step 2. Corner case: no loops
  if (n_loops == 0) {
    // In this case, the `access_shape` is not calculated
    for (SubFeature& feature : sub_features) {
      feature.access_shape = IntVec(feature.buffer->shape.size(), 1);
    }
    return;
  }
  // Step 3. Gradually bind the loops from inner to outer,
  // calculate the area the loops touch on each buffer
  for (int i = n_loops - 1; i >= 0; --i) {
    const ForNode* loop = loops[i];
    analyzer->Bind(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent),
                   /*allow_override=*/true);
    int64_t& touched_bytes = (*for_touched_bytes)[i] = 0;
    for (SubFeature& feature : sub_features) {
      const BufferNode* buffer = feature.buffer;
      // Note: `feature.access_shape` for `i == 0` is the only one preserved,
      // while others are discarded
      std::tuple<int64_t, IntVec> res = utils::RelaxAndUnion(feature.regions, analyzer);
      int64_t numel = std::get<0>(res);
      feature.access_shape = std::get<1>(res);
      feature.loop_accessed_numel[i][buffer] = numel;
      touched_bytes += numel * buffer->dtype.bytes();
      (*buffer_touched_under_loop)[loop][buffer].push_back(numel);
    }
  }
}

void Feature::SubFeature::SetStride(const LoopNest& loop_nest, arith::Analyzer* analyzer) {
  int n_loops = loop_nest.loops.size();
  const std::vector<const ForNode*>& loops = loop_nest.loops;
  // For each buffer, we find the loop stride on it
  const BufferNode* buffer = this->buffer;
  int ndim = this->buffer->shape.size();
  IntVec buffer_shape = utils::GetBufferShape(GetRef<Buffer>(buffer), analyzer);
  // Calculate the buffer's stride from its shape
  IntVec buffer_stride(ndim);
  if (ndim >= 1) {
    buffer_stride[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
      buffer_stride[i] = buffer_stride[i + 1] * buffer_shape[i + 1];
    }
  }
  // Calculate `num_continuous_bytes`
  {
    int64_t& num_continuous_bytes = this->num_continuous_bytes = 1;
    const IntVec& access_shape = this->access_shape;
    ICHECK_EQ(access_shape.size(), buffer_shape.size());
    for (int i = ndim - 1; i >= 0; --i) {
      if (access_shape[i] == buffer_shape[i]) {
        num_continuous_bytes = buffer_shape[i] * buffer->dtype.bytes();
        break;
      }
    }
  }
  // Enumerate loops from inner to outer
  int i = 0;
  // Calculate this->min_stride
  int64_t& stride = this->min_stride = 0;
  for (i = n_loops - 1; i >= 0; --i) {
    stride = utils::GetVarStride(this->regions, buffer_stride, loops[i]->loop_var);
    if (stride != 0) {
      break;
    }
  }
  // Calculate this->innermost_stride
  this->innermost_stride = (i == n_loops - 1) ? stride : 0;
  // Calculate this->prod
  int64_t& prod = this->prod_non_strided_loop_extent = 1;
  for (int j = n_loops - 1; j > i; --j) {
    if (const int64_t* extent = GetLoopIntExtent(loops[j])) {
      prod *= *extent;
    }
  }
}

void Feature::SubFeature::SetReuse(const LoopNest& loop_nest, int64_t top_loop_touch_bytes,
                                   const ForBufferMap<IntVec>& buffer_touched_under_loop) {
  const BufferNode* buffer = this->buffer;
  // Default case: no reuse
  ReuseType& reuse_type = this->reuse_type = ReuseType::kNoReuse;
  double& reuse_dis_iter = this->reuse_dis_iter = 0;
  double& reuse_dis_bytes = this->reuse_dis_bytes = 0;
  int64_t& reuse_ct = this->reuse_ct = 0;
  // Step 3.1. Collect all `Var`s that appears in the buffer region
  std::unordered_set<const VarNode*> region_vars;
  for (const NDIntSet& region : this->regions) {
    for (const arith::IntSet& int_set : region) {
      PostOrderVisit(int_set.min(), [&region_vars](const ObjectRef& obj) -> void {
        if (const auto* var = obj.as<VarNode>()) {
          region_vars.insert(var);
        }
      });
    }
  }
  // Step 3.2. Enumerate loops from inner to outer, find the first loop with reuse
  int n_loops = loop_nest.loops.size();
  const std::vector<const ForNode*>& loops = loop_nest.loops;
  for (int i = n_loops - 1; i >= 0; --i) {
    const ForNode* loop = loops[i];
    // Case 1. Find an invariant loop, i.e. reuse with kLoopMultipleRead
    if (!region_vars.count(loop->loop_var.get())) {
      reuse_type = ReuseType::kLoopMultipleRead;
      if (const int64_t* extent = GetLoopIntExtent(loop)) {
        reuse_ct = *extent;
      } else {
        reuse_ct = 1;
      }
      reuse_dis_iter = 1;
      for (int j = n_loops - 1; j > i; --j) {
        if (const int64_t* extent = GetLoopIntExtent(loops[j])) {
          reuse_dis_iter *= *extent;
        }
      }
      reuse_dis_bytes = 0.0;
      if (i == n_loops - 1) {
        reuse_dis_bytes = top_loop_touch_bytes;
      } else {
        for (const auto& iter : buffer_touched_under_loop.at(loops[i + 1])) {
          const BufferNode* buffer = iter.first;
          const IntVec& numels = iter.second;
          int64_t numel = std::accumulate(numels.begin(), numels.end(), int64_t(0));
          reuse_dis_bytes += numel * buffer->dtype.bytes();
        }
      }
      break;
    }
    // Case 2. Find serial reuse, i.e. reuse with kSerialMultipleReadWrite
    const IntVec& touched = buffer_touched_under_loop.at(loop).at(buffer);
    if (touched.size() >= 2) {
      int64_t extent = 1;
      if (const int64_t* ext = GetLoopIntExtent(loop)) {
        extent = *ext;
      }
      reuse_type = ReuseType::kSerialMultipleReadWrite;
      reuse_ct = touched.size() - 1;
      reuse_dis_iter = *std::min_element(touched.begin(), touched.end());
      reuse_dis_bytes = 0.0;
      for (const auto& iter : buffer_touched_under_loop.at(loop)) {
        const BufferNode* buffer = iter.first;
        const IntVec& numels = iter.second;
        int64_t numel = std::accumulate(numels.begin(), numels.end(), int64_t(0));
        reuse_dis_bytes += numel * buffer->dtype.bytes();
      }
      reuse_dis_iter /= extent;
      reuse_dis_bytes /= extent;
      break;
    }
  }
}

void Feature::SubFeature::SetFeature(const LoopNest& loop_nest, int64_t cache_line_bytes) {
  int64_t dtype_bytes = this->buffer->dtype.bytes();
  this->stride = this->innermost_stride;
  this->bytes = dtype_bytes * loop_nest.prod;
  if (loop_nest.loops.empty()) {
    this->unique_bytes = 1;
    this->lines = 1;
    this->unique_lines = 1;
  } else {
    this->unique_bytes = this->loop_accessed_numel.front().at(buffer) * dtype_bytes;
    this->lines = static_cast<double>(loop_nest.prod) / this->prod_non_strided_loop_extent *
                  std::min(1.0, 1.0 * this->min_stride * dtype_bytes / cache_line_bytes);
    this->lines = std::max(1.0, this->lines);
    this->unique_lines = static_cast<double>(this->unique_bytes) /
                         std::min(cache_line_bytes, this->num_continuous_bytes);
    this->unique_lines = std::max(1.0, this->unique_lines);
  }
  double proxy_reuse_ct = this->reuse_ct > 0 ? this->reuse_ct : 0.5;
  this->bytes_d_reuse_ct = this->bytes / proxy_reuse_ct;
  this->unique_bytes_d_reuse_ct = this->unique_bytes / proxy_reuse_ct;
  this->lines_d_reuse_ct = this->lines / proxy_reuse_ct;
  this->unique_lines_d_reuse_ct = this->unique_lines / proxy_reuse_ct;
}

Feature::Feature(const BlockRealizeNode* realize, const LoopNest& loop_nest,
                 int64_t cache_line_bytes, IntVec* for_touched_bytes,
                 ForBufferMap<IntVec>* buffer_touched_under_loop, arith::Analyzer* analyzer,
                 std::vector<const BlockRealizeNode*>* scopes) {
  int n_loops = loop_nest.loops.size();
  // Step 0. Initialize data structures
  this->Init(realize, n_loops, analyzer, scopes);
  // Step 1. Calculate region-related feature
  this->SetRegion(loop_nest, for_touched_bytes, buffer_touched_under_loop, analyzer);
  // Step 2. Calculate stride-related feature
  for (auto& feature : sub_features) {
    feature.SetStride(loop_nest, analyzer);
  }
  // Step 3. Calculate reuse-related feature
  int64_t top_loop_touch_bytes = 0.0;
  if (n_loops > 0) {
    for (const SubFeature& feature : sub_features) {
      int64_t bytes = feature.buffer->dtype.bytes();
      int64_t n_buffer = feature.loop_accessed_numel[0].size();
      top_loop_touch_bytes += bytes * n_buffer;
    }
  }
  for (auto& feature : sub_features) {
    feature.SetReuse(loop_nest, top_loop_touch_bytes, *buffer_touched_under_loop);
  }
  // Step 4. Calculate rest of the features
  for (auto& feature : sub_features) {
    feature.SetFeature(loop_nest, cache_line_bytes);
  }
  // Step 5. Sort the features
  std::sort(sub_features.begin(), sub_features.end(), [](const SubFeature& a, const SubFeature& b) {
    if (a.lines != b.lines) {
      return a.lines > b.lines;
    }
    if (a.bytes != b.bytes) {
      return a.bytes > b.bytes;
    }
    return a.buffer->name < b.buffer->name;
  });
}

}  // namespace group2

namespace group3 {

/*! \brief Group 3 feature */
struct Feature {
  /*!
   * \brief See the wiki page [1] for details
   *
   * [1] https://en.wikipedia.org/wiki/Roofline_model
   */
  std::vector<double> arith_intensity_curve;

  void Export(std::vector<double>* v) const {
    v->insert(v->end(), arith_intensity_curve.begin(), arith_intensity_curve.end());
  }

  explicit Feature(int n_samples, const LoopNest& loop_nest, const IntVec& for_touched_bytes,
                   const group1::Feature::ArithOps& arith_ops)
      : arith_intensity_curve(n_samples, 0.0) {
    const std::vector<const ForNode*>& loops = loop_nest.loops;
    ICHECK_EQ(loops.size(), for_touched_bytes.size());
    int n_loops = loops.size();
    // Calculate `memory_bytes`
    std::vector<double> memory_bytes;
    memory_bytes.resize(n_loops);
    for (int i = 0; i < n_loops; ++i) {
      memory_bytes[n_loops - 1 - i] = std::log2(for_touched_bytes[i]);
    }
    // Calculate `compute_ops` and `cur_compute_ops`
    std::vector<double> compute_ops;
    double total_compute_ops = arith_ops.float_mad + arith_ops.float_add_sub + arith_ops.float_mul +
                               arith_ops.float_div_mod + arith_ops.float_cmp +
                               arith_ops.float_math_func + arith_ops.float_other_func;
    total_compute_ops /= loop_nest.prod;
    for (int i = n_loops - 1; i >= 0; --i) {
      if (const int64_t* extent = GetLoopIntExtent(loops[i])) {
        total_compute_ops *= *extent;
      }
      compute_ops.push_back(std::log2(total_compute_ops));
    }
    // Fill the feature set
    if (total_compute_ops <= 0 || compute_ops.empty()) {
      for (int i = 0; i < n_samples; ++i) {
        arith_intensity_curve[i] = 0.0;
      }
      return;
    }
    total_compute_ops = compute_ops.back();  // i.e. total_compute_ops = log2(total_compute_ops)
    int p = 0;
    for (int i = 0; i < n_samples; ++i) {
      double& result = arith_intensity_curve[i];
      double cur_compute_ops = static_cast<double>(i + 1) / n_samples * total_compute_ops;
      // Find the first `p` that `compute[p] >= total * (i + 1) / N`
      for (; p < n_loops; ++p) {
        if (compute_ops[p] >= cur_compute_ops - 1e-4) {
          break;
        }
      }
      CHECK_LT(p, n_loops);
      if (p == 0) {
        result = compute_ops[p] / memory_bytes[p];
      } else {
        double base = compute_ops[p - 1] / memory_bytes[p - 1];
        double slope =
            (compute_ops[p] / memory_bytes[p] - compute_ops[p - 1] / memory_bytes[p - 1]) /
            (compute_ops[p] - compute_ops[p - 1]);
        result = base + slope * (cur_compute_ops - compute_ops[p - 1]);
      }
    }
  }
};

}  // namespace group3

namespace group4 {

/*! \brief Group 4 feature */
struct Feature {
  // Since allocated buffers can be in local, shared, or global memory, each attribute below is
  // divided into three cases.
  // The size of allocated buffer in bytes
  int64_t alloc_size_local = 0;
  int64_t alloc_size_shared = 0;
  int64_t alloc_size_global = 0;
  // alloc_outer_prod * written_inner_prod / buffer size
  int64_t alloc_prod_local = 0;
  int64_t alloc_prod_shared = 0;
  int64_t alloc_prod_global = 0;
  // The product of lengths of loops outside the scope of the alloc * buffer size
  int64_t alloc_outer_prod_local = 0;
  int64_t alloc_outer_prod_shared = 0;
  int64_t alloc_outer_prod_global = 0;
  // The product of lengths of loops inside the scope of alloc before the buffer is written to *
  // buffer size
  int64_t written_inner_prod_local = 0;
  int64_t written_inner_prod_shared = 0;
  int64_t written_inner_prod_global = 0;

  static constexpr int64_t kCount = 12;

  void Export(std::vector<double>* v) const {
    double vs[] = {
        slog(alloc_size_local),          slog(alloc_size_shared),
        slog(alloc_size_global),         slog(alloc_prod_local),
        slog(alloc_prod_shared),         slog(alloc_prod_global),
        slog(alloc_outer_prod_local),    slog(alloc_outer_prod_shared),
        slog(alloc_outer_prod_global),   slog(written_inner_prod_local),
        slog(written_inner_prod_shared), slog(written_inner_prod_global),
    };
    v->insert(v->end(), std::begin(vs), std::end(vs));
  }

  Feature() = default;

  explicit Feature(const LoopNest& loop_nest, const BlockRealizeNode* realize,
                   arith::Analyzer* analyzer,
                   std::unordered_map<const Buffer*, int64_t>* alloc_buffer_outer_loops_) {
    for (const Buffer& buffer : realize->block->alloc_buffers) {
      (*alloc_buffer_outer_loops_)[&buffer] = loop_nest.prod;
      std::vector<int64_t> shape = utils::GetBufferShape(buffer, analyzer);
      int64_t numel = 1;
      for (int64_t x : shape) {
        numel *= x;
      }
      runtime::StorageScope storage_scope = runtime::StorageScope::Create(buffer.scope());
      switch (storage_scope.rank) {
        case runtime::StorageRank::kLocal:
          alloc_size_local += numel * buffer->dtype.bytes();
          alloc_outer_prod_local += numel * loop_nest.prod;
          break;
        case runtime::StorageRank::kShared:
          alloc_size_shared += numel * buffer->dtype.bytes();
          alloc_outer_prod_shared += numel * loop_nest.prod;
          break;
        case runtime::StorageRank::kGlobal:
          alloc_size_global += numel * buffer->dtype.bytes();
          alloc_outer_prod_global += numel * loop_nest.prod;
          break;
        default:
          break;
      }
    }

    const Array<BufferRegion> write_buffers = realize->block->writes;
    for (const BufferRegion& write_buffer : write_buffers) {
      const Buffer& buffer = write_buffer->buffer;
      std::vector<int64_t> shape = utils::GetBufferShape(buffer, analyzer);
      int64_t numel = 1;
      for (int64_t x : shape) {
        numel *= x;
      }
      int64_t outer_loops = (*alloc_buffer_outer_loops_)[&buffer];
      runtime::StorageScope storage_scope = runtime::StorageScope::Create(buffer.scope());
      switch (storage_scope.rank) {
        case runtime::StorageRank::kLocal:
          alloc_prod_local += numel * loop_nest.prod;
          written_inner_prod_local += numel * (static_cast<double>(loop_nest.prod) / outer_loops);
          break;
        case runtime::StorageRank::kShared:
          alloc_prod_shared += numel * loop_nest.prod;
          written_inner_prod_shared += numel * (static_cast<double>(loop_nest.prod) / outer_loops);
          break;
        case runtime::StorageRank::kGlobal:
          alloc_prod_global += numel * loop_nest.prod;
          written_inner_prod_global += numel * (static_cast<double>(loop_nest.prod) / outer_loops);
          break;
        default:
          break;
      }
    }
  }
};

}  // namespace group4

namespace group5 {

/*! \brief Group 5 feature */
struct Feature {
  int64_t outer_prod;        // The product of lengths of outer loops
  int num_loops;             // The number of outer loops
  int auto_unroll_max_step;  // The value of pragma "auto_unroll_max_step"

  static constexpr int64_t kCount = 3;

  void Export(std::vector<double>* v) const {
    double vs[] = {
        slog(outer_prod),
        slog(num_loops),
        slog(auto_unroll_max_step),
    };
    v->insert(v->end(), std::begin(vs), std::end(vs));
  }

  explicit Feature(const LoopNest& loop_nest) {
    this->outer_prod = loop_nest.prod;
    this->num_loops = loop_nest.loops.size();
    this->auto_unroll_max_step = loop_nest.auto_unroll.empty() ? 0 : loop_nest.auto_unroll.back();
  }
};

}  // namespace group5

namespace group6 {

/*! \brief The auxiliary feature extractor for workloads */
class WorkloadEmbeddingExtractor : private StmtVisitor {
 public:
  static std::vector<double> Extract(const IRModule& mod) {
    WorkloadEmbeddingExtractor self;
    for (const auto& kv : mod->functions) {
      if (const PrimFuncNode* func = kv.second.as<PrimFuncNode>()) {
        self(func->body);
      }
    }
    return self.embedding;
  }

 private:
  void VisitStmt_(const BlockNode* block) final {
    StmtVisitor::VisitStmt_(block);
    std::string name = block->name_hint;
    std::for_each(name.begin(), name.end(), [](char& c) { c = ::tolower(c); });
    if (name.find("softmax") != std::string::npos) {
      embedding[0] = 1.0;
    } else if ((name.find("max") != std::string::npos) || (name.find("min") != std::string::npos)) {
      embedding[1] = 1.0;
    } else if (name.find("add") != std::string::npos) {
      embedding[2] = 1.0;
    } else if (name.find("batch_matmul") != std::string::npos) {
      embedding[3] = 1.0;
    } else if (name.find("matmul") != std::string::npos) {
      embedding[4] = 1.0;
    } else if (name.find("depthwiseconv2d") != std::string::npos) {
      embedding[5] = 1.0;
    } else if (name.find("conv2d_winograd") != std::string::npos) {
      embedding[6] = 1.0;
    } else if (name.find("conv2d") != std::string::npos) {
      embedding[7] = 1.0;
    }
  }

  std::vector<double> embedding = std::vector<double>(8, 0.0);
};

/*! \brief Group 6 feature */
struct Feature {
  explicit Feature(const IRModule& mod) {
    this->feature = WorkloadEmbeddingExtractor::Extract(mod);
  }

  void Export(std::vector<double>* v) const {
    v->insert(v->end(), std::begin(feature), std::end(feature));
  }

  std::vector<double> feature;  // The workload embedding
  static constexpr int64_t kCount = 8;
};

}  // namespace group6

/*! \brief The feature extracted */
struct Feature {
  const BlockRealizeNode* block_realize = nullptr;
  std::unique_ptr<group1::Feature> group1 = nullptr;
  std::unique_ptr<group2::Feature> group2 = nullptr;
  std::unique_ptr<group3::Feature> group3 = nullptr;
  std::unique_ptr<group4::Feature> group4 = nullptr;
  std::unique_ptr<group5::Feature> group5 = nullptr;
  std::shared_ptr<group6::Feature> group6 = nullptr;
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
      }
    }
    std::vector<Feature> result;
    result.reserve(collector.block_features_.size());
    for (const BlockRealizeNode* realize : collector.ordered_blocks_) {
      Feature& feature = collector.block_features_.at(realize);
      if (feature.block_realize != nullptr) {
        ICHECK(feature.group1);
        ICHECK(feature.group2);
        ICHECK(feature.group3);
        ICHECK(feature.group4);
        ICHECK(feature.group5);
        result.push_back(std::move(feature));
      }
    }
    return result;
  }

 private:
  explicit PerBlockFeatureCollector(bool is_gpu, int64_t cache_line_bytes,
                                    int64_t arith_intensity_curve_num_samples)
      : is_gpu_(is_gpu),
        cache_line_bytes_(cache_line_bytes),
        arith_intensity_curve_num_samples_(arith_intensity_curve_num_samples) {}

  void VisitStmt_(const BlockRealizeNode* realize) final {
    if (!scopes_.empty()) {
      ordered_blocks_.push_back(realize);
    }
    Feature& feature = block_features_[realize];
    feature.block_realize = realize;
    feature.group1 = std::make_unique<group1::Feature>(realize, loop_nest_, is_gpu_);
    scopes_.push_back(realize);
    dfs_path_.push_back(realize);
    StmtVisitor::VisitStmt_(realize);
    dfs_path_.pop_back();
    scopes_.pop_back();
    if (!scopes_.empty()) {
      AddArithOpsToScope(&feature.group1->arith_ops);
    }
    feature.group2 = std::make_unique<group2::Feature>(
        realize, loop_nest_, cache_line_bytes_, &for_touched_bytes_, &buffer_touched_under_loop_,
        &analyzer_, &scopes_);
    feature.group3 =
        std::make_unique<group3::Feature>(arith_intensity_curve_num_samples_, loop_nest_,
                                          for_touched_bytes_, feature.group1->arith_ops);
    feature.group4 = std::make_unique<group4::Feature>(loop_nest_, realize, &analyzer_,
                                                       &alloc_buffer_outer_loops_);
    feature.group5 = std::make_unique<group5::Feature>(loop_nest_);
    // Erase the feature of the root block
    if (scopes_.empty()) {
      block_features_.erase(realize);
    }
  }

  void VisitStmt_(const ForNode* loop) final {
    int64_t auto_unroll;
    const int64_t* extent = GetLoopIntExtent(loop);
    ForVec* for_vec = loop_nest_.Push(loop, &auto_unroll);
    if ((extent && (*extent != 1)) || for_vec) {
      dfs_path_.push_back(loop);
      analyzer_.Bind(loop->loop_var, loop->min);
    }
    StmtVisitor::VisitStmt_(loop);
    loop_nest_.Pop(loop, for_vec, auto_unroll);
    if ((extent && (*extent != 1)) || for_vec) {
      dfs_path_.pop_back();
    }
  }

  void VisitStmt_(const BufferStoreNode* store) final {
    ICHECK(!scopes_.empty());
    // LOG(INFO) << "store = " << GetRef<BufferStore>(store)
    //           << ", loop_nest_.prod = " << loop_nest_.prod;
    group1::Feature::ArithOps arith_ops;
    arith_ops.AddExpr(store->value, loop_nest_.prod);
    AddArithOpsToScope(&arith_ops);
  }

  void AddArithOpsToScope(group1::Feature::ArithOps* arith_ops) {
    const BlockRealizeNode* scope = scopes_.back();
    // Add the arith_ops to the parent
    group1::Feature::ArithOps& parent_arith_ops = block_features_[scope].group1->arith_ops;
#define TVM_FEATURE_MATH_OP_ADD(Name) parent_arith_ops.Name += arith_ops->Name;
    TVM_FEATURE_MATH_OP_ADD(float_mad);
    TVM_FEATURE_MATH_OP_ADD(float_add_sub);
    TVM_FEATURE_MATH_OP_ADD(float_mul);
    TVM_FEATURE_MATH_OP_ADD(float_div_mod);
    TVM_FEATURE_MATH_OP_ADD(float_cmp);
    TVM_FEATURE_MATH_OP_ADD(float_math_func);
    TVM_FEATURE_MATH_OP_ADD(float_other_func);
    TVM_FEATURE_MATH_OP_ADD(int_mad);
    TVM_FEATURE_MATH_OP_ADD(int_add_sub);
    TVM_FEATURE_MATH_OP_ADD(int_mul);
    TVM_FEATURE_MATH_OP_ADD(int_div_mod);
    TVM_FEATURE_MATH_OP_ADD(int_cmp);
    TVM_FEATURE_MATH_OP_ADD(int_math_func);
    TVM_FEATURE_MATH_OP_ADD(int_other_func);
    TVM_FEATURE_MATH_OP_ADD(bool_op);
    TVM_FEATURE_MATH_OP_ADD(select_op);
#undef TVM_FEATURE_MATH_OP_ADD
  }

  bool is_gpu_;
  int64_t cache_line_bytes_;
  int64_t arith_intensity_curve_num_samples_;
  arith::Analyzer analyzer_;
  std::vector<const BlockRealizeNode*> scopes_;
  std::vector<const StmtNode*> dfs_path_;
  LoopNest loop_nest_ = {};
  IntVec for_touched_bytes_ = {};
  ForBufferMap<IntVec> buffer_touched_under_loop_ = {};
  std::unordered_map<const Buffer*, int64_t> alloc_buffer_outer_loops_ = {};
  std::unordered_map<const BlockRealizeNode*, Feature> block_features_ = {};
  std::vector<const BlockRealizeNode*> ordered_blocks_;
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
    // LOG(INFO) << "Extracting features from:\n" << tir::AsTVMScript(mod);
    std::vector<Feature> features = PerBlockFeatureCollector::Collect(
        is_gpu, this->cache_line_bytes, this->arith_intensity_curve_num_samples, mod);
    int n_features = features.size();
    results->resize(n_features);
    for (int i = 0; i < n_features; ++i) {
      const Feature& feature = features[i];
      std::vector<double>& result = (*results)[i];
      result.reserve(feature_vector_length);
      feature.group1->Export(&result);
      feature.group2->Export(&result, this->buffers_per_block);
      feature.group3->Export(&result);
      feature.group4->Export(&result);
      feature.group5->Export(&result);
    }
  }

  Array<runtime::NDArray> ExtractFrom(const TuneContext& tune_context,
                                      const Array<MeasureCandidate>& candidates) {
    using namespace tvm::tir::per_block_feature;
    bool is_gpu = tune_context->target.value()->kind->name == "cuda";
    std::vector<runtime::NDArray> results;
    results.resize(candidates.size());
    std::unique_ptr<group6::Feature> feature_group6 = nullptr;
    if (extract_workload) {
      feature_group6 = std::make_unique<group6::Feature>(tune_context->mod.value());
    }
    auto f = [this, is_gpu, &feature_group6, &candidates, &results](int, int task_id) -> void {
      const auto& candidate = candidates[task_id];
      std::vector<std::vector<double>> features;
      ExtractSingle(DeepCopyIRModule(candidate->sch->mod()), is_gpu, &features);
      if (extract_workload) {
        for (auto& feature : features) {
          feature_group6->Export(&feature);
        }
      }
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
  n->feature_vector_length = group1::Feature::kCount +                                  //
                             group2::Feature::SubFeature::kCount * buffers_per_block +  //
                             arith_intensity_curve_num_samples +                        //
                             group4::Feature::kCount +                                  //
                             group5::Feature::kCount;
  if (extract_workload) {
    n->feature_vector_length += group6::Feature::kCount;
  }
  return FeatureExtractor(n);
}

TVM_REGISTER_NODE_TYPE(PerBlockFeatureNode);
TVM_REGISTER_GLOBAL("meta_schedule.FeatureExtractorPerBlockFeature")
    .set_body_typed(FeatureExtractor::PerBlockFeature);

}  // namespace meta_schedule
}  // namespace tvm
