# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""V2 parser: parses V2 printer output back to TVM IR.

Uses tvm_ffi.pyast_parser as the core engine, with TIR/Relax
surface objects and language modules plugged in.
"""

from __future__ import annotations

import tvm
from tvm_ffi.pyast_parser import IRParser, SurfaceObject


# ============================================================================
# TIR surface objects
# ============================================================================


class _PrimFuncSurface(SurfaceObject):
    """Surface object for @T.prim_func decorator."""

    def parse_function(self, parser, node):
        with parser.var_table.frame():
            # Parse params
            params = []
            for arg in node.args:
                name = arg.lhs.name
                # TODO: parse type annotations for param dtype
                var = tvm.tirx.Var(name, "int32")
                parser.var_table.define(name, var)
                params.append(var)

            # Parse body
            body_stmts = parser.visit_body(node.body)

            # Construct body — single stmt or SeqStmt
            if len(body_stmts) == 1:
                body = body_stmts[0]
            elif len(body_stmts) > 1:
                body = tvm.tirx.SeqStmt(body_stmts)
            else:
                body = tvm.tirx.Evaluate(tvm.tirx.IntImm("int32", 0))

            func_name = node.name.name
            pf = tvm.tirx.PrimFunc(params, body)
            pf = pf.with_attr("global_symbol", func_name)
            return pf


# ============================================================================
# IR surface objects
# ============================================================================


class _IRModuleSurface(SurfaceObject):
    """Surface object for @I.ir_module decorator."""

    def parse_class(self, parser, node):
        with parser.var_table.frame():
            # Pass 1: forward-declare all function GlobalVars
            for stmt in node.body:
                from tvm_ffi import pyast

                if isinstance(stmt, pyast.Function):
                    gv = tvm.ir.GlobalVar(stmt.name.name)
                    parser.var_table.define(stmt.name.name, gv)

            # Pass 2: parse function bodies
            funcs = {}
            for stmt in node.body:
                func_ir = parser.visit_stmt(stmt)
                func_name = stmt.name.name
                gv = parser.var_table.get(func_name)
                funcs[gv] = func_ir

            return tvm.ir.IRModule(funcs)


# ============================================================================
# TIR language module callables
# ============================================================================


def _tir_evaluate(value):
    """T.evaluate(value) → Evaluate(IntImm(value))"""
    if isinstance(value, int):
        return tvm.tirx.Evaluate(tvm.tirx.IntImm("int32", value))
    return tvm.tirx.Evaluate(value)


# ============================================================================
# Language modules
# ============================================================================


class _TIRModule:
    """Language module for TIR (the 'T' in `from tvm.script import tirx as T`)."""

    prim_func = _PrimFuncSurface()
    evaluate = staticmethod(_tir_evaluate)


class _IRLangModule:
    """Language module for IR (the 'I' in `from tvm.script import ir as I`)."""

    ir_module = _IRModuleSurface()
    GlobalVar = staticmethod(tvm.ir.GlobalVar)


# ============================================================================
# Public API
# ============================================================================


def make_parser() -> IRParser:
    """Create a parser configured with TIR/Relax language modules."""
    return IRParser(lang_modules={"T": _TIRModule, "I": _IRLangModule})


def parse(text: str):
    """Parse V2 printer output back to TVM IR."""
    return make_parser().parse(text)
