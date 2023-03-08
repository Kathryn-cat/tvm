/*
 * Licensed to the Apache Software Foundation(ASF) under one
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
/*!
 * \file src/relax/transform/transform_to_gemm.cc
 * \brief Transform an einsum computation to GEMM/BGEMM for dispatching.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ir/module.h>
#include <tvm/relax/attrs/linear_algebra.h>
#include <tvm/relax/attrs/manipulate.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tir {

typedef std::unordered_map<Var, int, ObjectPtrHash, ObjectPtrEqual> VarMap;
typedef std::unordered_map<Var, std::vector<int>, ObjectPtrHash, ObjectPtrEqual> RefMap;

class GEMMMatcher : public StmtExprVisitor {
 public:
  void FindTransformRules(Stmt body) { this->VisitStmt(body); }

  // assume only one block in the body needs to be transformed for dispatching
  // below are results for transformation
  Array<Integer> A_axes;
  Array<Integer> B_axes;
  Array<Integer> C_axes;

  Array<PrimExpr> A_shape;
  Array<PrimExpr> B_shape;
  Array<PrimExpr> C_shape;

 private:
  inline void GetVars(const BufferRegion region, int weight, VarMap* var_map, Array<Var>* vars,
                      Array<Integer>* shape) {
    for (const Range range : region->region) {
      const VarNode* index_ptr = range->min.as<VarNode>();
      ICHECK(index_ptr != nullptr);
      const Var var = GetRef<Var>(index_ptr);
      if (var_map->find(var) == var_map->end()) {
        var_map->insert({var, weight});
      } else {
        (*var_map)[var] += weight;
      }
      vars->push_back(var);
    }
    // assume index var bindings are simple
    for (auto expr : region->buffer->shape) {
      shape->push_back(Downcast<Integer>(expr));
    }
  }

  inline RefMap AssignOrder(const Array<Var> ref, VarMap* var_map, int predicate) {
    RefMap order_map;
    int order = 0;
    for (auto var : ref) {
      if ((*var_map)[var] == predicate) {
        order_map[var] = std::vector<int>{5 - predicate, order};
        order++;
      }
    }
    return order_map;
  }

  inline RefMap MergeMaps(RefMap* m1, RefMap* m2, RefMap* m3) {
    RefMap m = (*m1);
    m.insert(m2->begin(), m2->end());
    m.insert(m3->begin(), m3->end());
    return m;
  }

  inline void GetAxesAndShape(const Array<Var> vars, const Array<Integer> old_shape,
                              const std::vector<int> ref_size, RefMap* ref_map,
                              std::string region_name, Array<Integer>* result_axes,
                              Array<PrimExpr>* result_shape) {
    // ordered_map maps order to index to calculate permutation
    std::map<int, int> ordered_map;
    Array<PrimExpr> type1_shapes;
    Array<PrimExpr> type2_shapes;
    Array<PrimExpr> type3_shapes;

    for (int i = 0; i < static_cast<int>(vars.size()); i++) {
      const Var var = vars[i];
      int category = (*ref_map)[var][0];
      int offset = 0;
      switch (category) {
        case 1:
          type1_shapes.push_back(old_shape[i]);
          break;
        case 2:
          offset = (region_name == "A" || region_name == "C") ? ref_size[0]
                                                              : (ref_size[0] + ref_size[2]);
          type2_shapes.push_back(old_shape[i]);
          break;
        case 3:
          offset = (region_name == "B") ? ref_size[0] : (ref_size[0] + ref_size[1]);
          type3_shapes.push_back(old_shape[i]);
      }
      int key = offset + (*ref_map)[var][1];
      ordered_map[key] = i;
    }

    // find permutation axes
    if (region_name == "C") {
      // for C, the axes mapping is reversed
      std::map<int, int> reversed_map;
      int idx = 0;
      for (const auto& kv : ordered_map) {
        reversed_map[kv.second] = idx;
        idx++;
      }
      for (const auto& kv : reversed_map) {
        result_axes->push_back(kv.second);
      }
    } else {
      for (const auto& kv : ordered_map) {
        result_axes->push_back(kv.second);
      }
    }

    // find reshape rules
    if (region_name == "C") {
      // for C, we specify how the axes should be splitted
      if (type1_shapes.size() > 1 || type2_shapes.size() > 2) {
        for (auto s : type1_shapes) {
          result_shape->push_back(s);
        }
        for (auto s : type2_shapes) {
          result_shape->push_back(s);
        }
      }
    } else {
      if (type1_shapes.size() > 1 || type2_shapes.size() > 1 || type3_shapes.size() > 1) {
        Integer type2_prod = IntImm(DataType::Int(32), 1);
        Integer type3_prod = IntImm(DataType::Int(32), 1);
        if (type1_shapes.size() > 0) {
          // batch dim exists
          Integer type1_prod = IntImm(DataType::Int(32), 1);
          for (auto s : type1_shapes) {
            type1_prod *= s;
          }
          result_shape->push_back(type1_prod);
        }
        for (auto s : type2_shapes) {
          type2_prod *= s;
        }
        for (auto s : type3_shapes) {
          type3_prod *= s;
        }
        if (region_name == "A") {
          result_shape->push_back(type2_prod);
          result_shape->push_back(type3_prod);
        } else {
          result_shape->push_back(type3_prod);
          result_shape->push_back(type2_prod);
        }
      }
    }

    // check for identity permutation
    for (int i = 0; i < static_cast<int>(result_axes->size()); i++) {
      if ((*result_axes)[i] != i) {
        return;
      }
    }
    result_axes->clear();
  }

  void VisitStmt_(const BlockNode* op) final {
    if (op->reads.size() != 2 || op->writes.size() != 1) {
      return;
    }
    // map vars that appear in A/B to weight 1, in C to weight 2 (weights are accumulated)
    // we categorize all vars into three types:
    // 1. vars that appear in (A, B, C) have weight 4
    // 2. vars that appear in (A, C) or (B, C) have weight 3
    // 3. vars that appear in (A, B) have weight 2
    VarMap var_map;  // var_map maps var to weight_category
    GetVars(op->reads[0], 1, &var_map, &A_vars_, &A_old_shape_);
    GetVars(op->reads[1], 1, &var_map, &B_vars_, &B_old_shape_);
    GetVars(op->writes[0], 2, &var_map, &C_vars_, &C_old_shape_);

    // assign order of type 1, 2, 3 vars, respectively
    RefMap type1_map = AssignOrder(C_vars_, &var_map, 4);
    RefMap type2_map = AssignOrder(C_vars_, &var_map, 3);
    RefMap type3_map = AssignOrder(A_vars_, &var_map, 2);
    // ref_map maps var to (category, order)
    RefMap ref_map = MergeMaps(&type1_map, &type2_map, &type3_map);

    // number of vars of category 1, 2, 3 respectively
    std::vector<int> ref_size;
    ref_size.push_back(static_cast<int>(type1_map.size()));
    ref_size.push_back(static_cast<int>(type2_map.size()));
    ref_size.push_back(static_cast<int>(type3_map.size()));

    // output strategies to transform A, B, C to match the input and output of GEMM/BGEMM
    // A -> input_A: use permute_dims then reshape to transform into (1)(2)(3)
    // B -> input_B: use permute_dims then reshape to transform into (1)(3)(2)
    // output -> C: use reshape then permute_dims to transform (1)(2) into C
    GetAxesAndShape(A_vars_, A_old_shape_, ref_size, &ref_map, "A", &A_axes, &A_shape);
    GetAxesAndShape(B_vars_, B_old_shape_, ref_size, &ref_map, "B", &B_axes, &B_shape);
    GetAxesAndShape(C_vars_, C_old_shape_, ref_size, &ref_map, "C", &C_axes, &C_shape);
  }

  // below are attributes
  Array<Var> A_vars_;
  Array<Var> B_vars_;
  Array<Var> C_vars_;

  Array<Integer> A_old_shape_;
  Array<Integer> B_old_shape_;
  Array<Integer> C_old_shape_;
};

}  // namespace tir

namespace relax {
/*! \brief Replace the call tir einsum part with calls of reshape / permute and GEMM / BGEMM. */
class EinsumMutator : public ExprMutator {
 public:
  explicit EinsumMutator(IRModule mod) { mod_ = mod; }

  IRModule Transform() {
    for (auto& p : mod_->functions) {
      Expr func = p.second;
      if (func->IsInstance<FunctionNode>()) {
        func = this->VisitExpr(func);
      }
      builder_->AddFunction(Downcast<BaseFunc>(func), p.first->name_hint);
    }
    return builder_->GetContextIRModule();
  }

  using ExprMutator::VisitExpr_;

  inline Array<Expr> GetCallTIRArgs(Expr args) {
    if (args.as<TupleNode>()) {
      return args.as<TupleNode>()->fields;
    } else {
      return {args};
    }
  }

  Expr VisitExpr_(const CallNode* call) override {
    Expr expr = VisitExprPostOrder_(call);
    call = expr.as<CallNode>();

    static const Op& matmul_op = Op::Get("relax.matmul");
    static const Op& permute_dims_op = Op::Get("relax.permute_dims");
    static const Op& reshape_op = Op::Get("relax.reshape");

    ObjectPtr<PermuteDimsAttrs> permute_attrs;
    ObjectPtr<MatmulAttrs> matmul_attrs;

    if (call->args.size() >= 2) {
      Array<Expr> call_args = GetCallTIRArgs(call->args[1]);
      // temporary for checking if a call can be dispatched to GEMM/BGEMM
      if (call_args.size() == 2) {
        Var A_arg = Downcast<Var>(call_args[0]);
        Var B_arg = Downcast<Var>(call_args[1]);

        tir::GEMMMatcher matcher;
        GlobalVar gv = Downcast<GlobalVar>(call->args[0]);
        tir::PrimFunc prim_func = Downcast<tir::PrimFunc>(mod_->Lookup(gv));
        tir::Stmt body = prim_func->body.as<tir::BlockRealizeNode>()->block->body;
        matcher.FindTransformRules(body);

        // permute_dims / reshape on arg A
        if (matcher.A_axes.size() > 0) {
          permute_attrs = make_object<PermuteDimsAttrs>();
          permute_attrs->axes = matcher.A_axes;
          Call permute_A = Call(permute_dims_op, {A_arg}, {Attrs(permute_attrs)});
          A_arg = builder_->Emit(permute_A);
        }
        if (matcher.A_shape.size() > 0) {
          Call reshape_A = Call(reshape_op, {A_arg, ShapeExpr(matcher.A_shape)});
          A_arg = builder_->Emit(reshape_A);
        }

        // permute_dims / reshape on arg B
        if (matcher.B_axes.size() > 0) {
          permute_attrs = make_object<PermuteDimsAttrs>();
          permute_attrs->axes = matcher.B_axes;
          Call permute_B = Call(permute_dims_op, {B_arg}, {Attrs(permute_attrs)});
          B_arg = builder_->Emit(permute_B);
        }
        if (matcher.B_shape.size() > 0) {
          Call reshape_B = Call(reshape_op, {B_arg, ShapeExpr(matcher.B_shape)});
          B_arg = builder_->Emit(reshape_B);
        }

        // matmul
        matmul_attrs = make_object<MatmulAttrs>();
        Call matmul = Call(matmul_op, {A_arg, B_arg}, Attrs(matmul_attrs));

        // reshape / permute_dims on arg C
        if (matcher.C_axes.size() > 0 || matcher.C_shape.size() > 0) {
          Var C_arg = builder_->Emit(matmul);
          if (matcher.C_shape.size() > 0) {
            Call reshape_C = Call(reshape_op, {C_arg, ShapeExpr(matcher.C_shape)});
            if (matcher.C_axes.size() > 0) {
              C_arg = builder_->Emit(reshape_C);
            } else {
              return std::move(reshape_C);
            }
          }
          if (matcher.C_axes.size() > 0) {
            permute_attrs = make_object<PermuteDimsAttrs>();
            permute_attrs->axes = matcher.C_axes;
            Call permute_C = Call(permute_dims_op, {C_arg}, {Attrs(permute_attrs)});
            return std::move(permute_C);
          }
        }

        return std::move(matmul);
      }
    }

    return GetRef<Call>(call);
  }

 private:
  IRModule mod_;
};

namespace transform {
Pass Transform2GEMM() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =  //
      [=](IRModule m, PassContext pc) { return EinsumMutator(m).Transform(); };
  return CreateModulePass(/*pass_function=*/pass_func,     //
                          /*opt_level=*/0,                 //
                          /*pass_name=*/"Transform2GEMM",  //
                          /*required=*/{});
}
TVM_REGISTER_GLOBAL("relax.transform.Transform2GEMM").set_body_typed(Transform2GEMM);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
