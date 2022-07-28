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

struct FeatureSet {
  // Group 1: Computation related features
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

runtime::NDArray PerBlockFeature(const tir::Schedule& sch, int max_num_buffer_access_features) {}

Array<String> PerBlockFeatureNames(int max_num_buffer_access_features) {}

Array<runtime::NDArray> PerBlockFeatureBatched(const Array<tir::Schedule>& schs,
                                               int max_num_buffer_access_features) {}

}  // namespace meta_schedule
}  // namespace tvm
