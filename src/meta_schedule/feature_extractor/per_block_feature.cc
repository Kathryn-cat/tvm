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
    /*! \brief A list of multi-dimensional indices used to access the buffer */
    std::vector<MultiIndex> multi_indices = {};
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
                        std::vector<MultiIndex> multi_indices, int n_loops)
        : buffer(buffer),
          access_type(access_type),
          multi_indices(multi_indices),
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
                   ForBufferMap<IntVec>* buffer_touched_under_loop, arith::Analyzer* analyzer);

  void Init(const BlockRealizeNode* realize, int n_loops);

  void SetRegion(const LoopNest& loop_nest,                        //
                 IntVec* for_touched_bytes,                        //
                 ForBufferMap<IntVec>* buffer_touched_under_loop,  //
                 arith::Analyzer* analyzer);

  std::vector<SubFeature> sub_features;
};

void Feature::Init(const BlockRealizeNode* realize, int n_loops) {
  // TODO: original implementation
  struct Info {
    AccessType access_type = AccessType::kUnknownRW;
    std::vector<MultiIndex> multi_indices;
  };
  std::unordered_map<const BufferNode*, Info> buffer_info;
  {
    Info& info = buffer_info[store->buffer.get()];
    info.access_type = AccessType::kWrite;
    info.multi_indices.push_back({store->indices.begin(), store->indices.end()});
  }
  PostOrderVisit(store->value, [&buffer_info](const ObjectRef& obj) -> void {
    if (const BufferLoadNode* load = obj.as<BufferLoadNode>()) {
      const BufferNode* buffer = load->buffer.get();
      Info& info = buffer_info[buffer];
      switch (info.access_type) {
        case AccessType::kRead:
          break;
        case AccessType::kWrite:
          info.access_type = AccessType::kReadWrite;
          break;
        case AccessType::kReadWrite:
          break;
        case AccessType::kUnknownRW:
        default:
          info.access_type = AccessType::kRead;
          break;
      }
      if (info.access_type != AccessType::kReadWrite) {
        info.multi_indices.push_back({load->indices.begin(), load->indices.end()});
      }
    }
  });
  this->sub_features.reserve(buffer_info.size());
  for (const auto& kv : buffer_info) {
    this->sub_features.emplace_back(kv.first, kv.second.access_type,
                                    std::move(kv.second.multi_indices), n_loops);
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
      int64_t numel;
      feature.access_shape = utils::RelaxAndUnion(feature.multi_indices, &numel, analyzer);
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
    stride = utils::GetVarStride(this->multi_indices, buffer_stride, loops[i]->loop_var);
    if (stride != 0) {
      break;
    }
  }
  // Calculate this->innermost_stride
  this->innermost_stride = (i == n_loops - 1) ? stride : 0;
  // Calculate this->prod
  int64_t& prod = this->prod_non_strided_loop_extent = 1;
  for (int j = n_loops - 1; j > i; --j) {
    if (const int64_t* extent = GetLoopIntExtent(loops[n_loops - 1])) {
      prod *= *extent;
    }
  }
}

void Feature::SubFeature::SetReuse(const LoopNest& loop_nest, int64_t top_loop_touch_bytes,
                                   const ForBufferMap<IntVec>& buffer_touched_under_loop) {
  const BufferNode* buffer = this->buffer;
  // Step 3.1. Collect all `Var`s that appears in the buffer region
  std::unordered_set<const VarNode*> region_vars;
  for (const MultiIndex& multi_index : this->multi_indices) {
    for (const PrimExpr& index : multi_index) {
      PostOrderVisit(index, [&region_vars](const ObjectRef& obj) -> void {
        if (const auto* var = obj.as<VarNode>()) {
          region_vars.insert(var);
        }
      });
    }
  }
  // Default case: no reuse
  ReuseType& reuse_type = this->reuse_type = ReuseType::kNoReuse;
  double& reuse_dis_iter = this->reuse_dis_iter = 0;
  double& reuse_dis_bytes = this->reuse_dis_bytes = 0;
  int64_t& reuse_ct = this->reuse_ct = 0;

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
                 ForBufferMap<IntVec>* buffer_touched_under_loop, arith::Analyzer* analyzer) {
  int n_loops = loop_nest.loops.size();
  // Step 0. Initialize data structures
  this->Init(realize, n_loops);
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

/*! \brief The feature extracted */
struct Feature {
  const BlockRealizeNode* block_realize = nullptr;
  // int buffer_order = -1;
  // TODO: add feature group 1-5
  std::unique_ptr<group1::Feature> group1 = nullptr;
  // std::unique_ptr<group2::Feature> group2 = nullptr;
  std::unique_ptr<group3::Feature> group3 = nullptr;
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

  void VisitStmt_(const BlockRealizeNode* realize) final {
    if (!scopes_.empty()) {
      ordered_blocks_.push_back(realize);
    }
    scopes_.push_back(realize);
    dfs_path_.push_back(realize);
    StmtVisitor::VisitStmt_(realize);
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
    Feature& feature = block_features_[realize];
    feature.block_realize = realize;
    if (!scopes_.empty()) {
      AddArithOpsToScope(&feature.group1->arith_ops);
    }
    feature.group1 = std::make_unique<group1::Feature>(realize, loop_nest_, is_gpu_);
    feature.group3 =
        std::make_unique<group3::Feature>(arith_intensity_curve_num_samples_, loop_nest_,
                                          for_touched_bytes_, feature.group1->arith_ops);
    // TODO: add groups 2,4,5
  }

  void VisitStmt_(const ForNode* loop) final {
    int64_t auto_unroll;
    ForVec* for_vec = loop_nest_.Push(loop, &auto_unroll);
    StmtVisitor::VisitStmt_(loop);
    loop_nest_.Pop(loop, for_vec, auto_unroll);
  }

  void VisitStmt_(const BufferStoreNode* store) final {
    ICHECK(!scopes_.empty());
    group1::Feature::ArithOps arith_ops = group1::Feature::ArithOps(store, loop_nest_.prod);
    AddArithOpsToScope(&arith_ops);
  }

  void AddArithOpsToScope(group1::Feature::ArithOps* arith_ops) {
    const BlockRealizeNode* scope = scopes_.back();
    // The product of the loops up to the parent
    int64_t prod_loop_extent = 1;
    for (auto iter = dfs_path_.rbegin(); iter != dfs_path_.rend(); ++iter) {
      const tir::StmtNode* stmt = *iter;
      if (stmt == scope) {
        break;
      }
      ICHECK(stmt->IsInstance<tir::ForNode>());
      const int64_t* extent = GetLoopIntExtent(static_cast<const tir::ForNode*>(stmt));
      if (*extent >= 1) {
        prod_loop_extent *= *extent;
      }
    }
    // Add the arith_ops to the parent
    group1::Feature::ArithOps& parent_arith_ops = block_features_[scope].group1->arith_ops;
#define TVM_FEATURE_MATH_OP_ADD(Name)                         \
  parent_arith_ops.Name = arith_ops->Name * prod_loop_extent; \
  arith_ops->Name *= outer_loop_prod_
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
  int64_t outer_loop_prod_ = 1;
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
