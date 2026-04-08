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
    """Surface object for @T.prim_func decorator.

    Also callable: @T.prim_func(private=True) calls this and returns self.
    """

    def __call__(self, *args, **kwargs):
        # @T.prim_func(private=True) — return a configured copy
        inst = _PrimFuncSurface()
        inst._private = kwargs.get("private", False)
        return inst

    _private = False

    def parse_function(self, parser, node):
        with parser.var_table.frame():
            # Parse params
            params = []
            buffer_map = {}
            for arg in node.args:
                name = arg.lhs.name
                ann = None
                if arg.annotation is not None:
                    ann = parser.eval_expr(arg.annotation)
                # If annotation is a Buffer, create handle var + buffer_map entry
                if isinstance(ann, tvm.tirx.Buffer):
                    # Set buffer name to match param name
                    ann = tvm.tirx.decl_buffer(
                        ann.shape,
                        ann.dtype,
                        name=name,
                        strides=ann.strides,
                        elem_offset=ann.elem_offset,
                    )
                    var = tvm.tirx.Var(name, "handle")
                    parser.var_table.define(name, ann)  # name resolves to Buffer
                    params.append(var)
                    buffer_map[var] = ann
                else:
                    dtype = "int32"
                    if isinstance(ann, str):
                        dtype = ann
                    var = tvm.tirx.Var(name, dtype)
                    parser.var_table.define(name, var)
                    params.append(var)

            # Set TIR dialect callbacks for body parsing
            old = (
                parser.make_assign,
                parser.make_store,
                parser.make_for,
                parser.create_var,
                parser.handle_return,
            )
            parser.make_assign = _tir_make_assign
            parser.make_store = _tir_make_store
            parser.make_for = _tir_make_for
            parser.create_var = lambda name, ann=None: tvm.tirx.Var(name, "int32")
            parser.handle_return = _tir_handle_return
            try:
                body_stmts = parser.visit_body(node.body)
            finally:
                (
                    parser.make_assign,
                    parser.make_store,
                    parser.make_for,
                    parser.create_var,
                    parser.handle_return,
                ) = old

            # Wrap non-Stmt results (e.g. return expr) in Evaluate
            wrapped = []
            for s in body_stmts:
                if isinstance(s, tvm.tirx.Stmt):
                    wrapped.append(s)
                elif s is not None:
                    wrapped.append(tvm.tirx.Evaluate(s))
            body_stmts = wrapped

            # Construct body
            if len(body_stmts) == 1:
                body = body_stmts[0]
            elif len(body_stmts) > 1:
                body = tvm.tirx.SeqStmt(body_stmts)
            else:
                body = tvm.tirx.Evaluate(tvm.tirx.IntImm("int32", 0))

            func_name = node.name.name
            pf = tvm.tirx.PrimFunc(params, body, buffer_map=buffer_map)
            if not self._private:
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
# TIR for-loop surface objects
# ============================================================================


class _ForKindSurface(SurfaceObject):
    """Surface object for T.unroll(n), T.parallel(n), T.vectorized(n), T.serial(n)."""

    def __init__(self, kind):
        self._kind = kind

    def __call__(self, extent, start=None):
        """Called as T.unroll(2) → returns instance with captured args."""
        inst = _ForKindSurfaceInstance(self._kind, extent, start)
        return inst


class _ForKindSurfaceInstance(SurfaceObject):
    """Instance created by T.unroll(2) — captures args, dispatches parse_for."""

    def __init__(self, kind, extent, start=None):
        self._kind = kind
        self._extent = extent
        self._start = start if start is not None else tvm.tirx.IntImm("int32", 0)

    def parse_for(self, parser, node):
        loop_var = tvm.tirx.Var(node.lhs.name, "int32")
        parser.var_table.define(node.lhs.name, loop_var)
        body_stmts = parser.visit_body(node.body)
        if len(body_stmts) == 1:
            body = body_stmts[0]
        else:
            body = tvm.tirx.SeqStmt(body_stmts)
        extent = (
            self._extent
            if not isinstance(self._extent, int)
            else tvm.tirx.IntImm("int32", self._extent)
        )
        start = (
            self._start
            if not isinstance(self._start, int)
            else tvm.tirx.IntImm("int32", self._start)
        )
        return tvm.tirx.For(loop_var, start, extent, int(self._kind), body)


# ============================================================================
# TIR language module callables
# ============================================================================


def _tir_handle_return(parser, node):
    """TIR return handler: `return val` → Evaluate(Call(tirx.ret, [val]))."""
    if node.value is not None:
        val = parser.eval_expr(node.value)
        op = tvm.tirx.op.Op.get("tirx.ret")
        dtype = val.dtype if hasattr(val, "dtype") else "void"
        ret_call = tvm.tirx.Call(dtype, op, [val])
        return tvm.tirx.Evaluate(ret_call)
    return None


def _tir_make_for(loop_var, start, end, step, body):
    """TIR for callback: range(n) → For(serial)."""
    if isinstance(start, int):
        start = tvm.tirx.IntImm("int32", start)
    if isinstance(end, int):
        end = tvm.tirx.IntImm("int32", end)
    extent = end - start
    if len(body) == 1:
        body_stmt = body[0]
    else:
        body_stmt = tvm.tirx.SeqStmt(body)
    return tvm.tirx.For(loop_var, start, extent, int(tvm.tirx.ForKind.SERIAL), body_stmt)


def _tir_make_assign(parser, node, rhs_val):
    """TIR assign callback: handles alloc_buffer tuple returns."""
    name = node.lhs.name
    if isinstance(rhs_val, tuple) and len(rhs_val) == 2:
        # alloc_buffer returns (buf, AllocBuffer node)
        buf, alloc_node = rhs_val
        parser.var_table.define(name, buf)
        return alloc_node
    parser.var_table.define(name, rhs_val)
    return rhs_val


def _tir_make_store(target, value, indices):
    """TIR store callback: buf[i] = val → BufferStore."""
    if isinstance(value, int):
        value = tvm.tirx.IntImm("int32", value)
    elif isinstance(value, float):
        value = tvm.tirx.FloatImm("float32", value)
    # Convert int indices to IntImm
    indices = [tvm.tirx.IntImm("int32", i) if isinstance(i, int) else i for i in indices]
    return tvm.tirx.BufferStore(target, value, indices)


def _tir_evaluate(value):
    """T.evaluate(value) → Evaluate(IntImm(value))"""
    if isinstance(value, int):
        return tvm.tirx.Evaluate(tvm.tirx.IntImm("int32", value))
    return tvm.tirx.Evaluate(value)


def _tir_alloc_buffer(shape, dtype="float32", scope=""):
    """T.alloc_buffer(shape, dtype, scope) → (Buffer, AllocBuffer node)."""
    if isinstance(shape, tuple):
        shape = list(shape)
    shape = [tvm.tirx.IntImm("int32", s) if isinstance(s, int) else s for s in shape]
    buf = tvm.tirx.decl_buffer(shape, dtype, scope=scope)
    return buf, tvm.tirx.AllocBuffer(buf, {})


def _tir_float32(value):
    """T.float32(value) → FloatImm."""
    if isinstance(value, str):
        value = float(value)  # handles "nan", "inf", etc.
    return tvm.tirx.FloatImm("float32", value)


def _tir_int64(value):
    """T.int64(value) → IntImm."""
    return tvm.tirx.IntImm("int64", value)


# ============================================================================
# Language modules
# ============================================================================


def _tir_buffer(shape, dtype="float32", scope=""):
    """T.Buffer(shape, dtype) → Buffer object (for type annotations)."""
    if isinstance(shape, tuple):
        shape = list(shape)
    shape = [tvm.tirx.IntImm("int32", s) if isinstance(s, int) else s for s in shape]
    return tvm.tirx.decl_buffer(shape, dtype, scope=scope)


def _tir_int32(value):
    """T.int32 — used as type annotation or literal constructor."""
    if isinstance(value, str):
        return value  # type annotation: just return dtype string
    return tvm.tirx.IntImm("int32", value)


class _TIRModule:
    """Language module for TIR (the 'T' in `from tvm.script import tirx as T`)."""

    prim_func = _PrimFuncSurface()
    evaluate = staticmethod(_tir_evaluate)
    alloc_buffer = staticmethod(_tir_alloc_buffer)
    float32 = staticmethod(_tir_float32)
    int64 = staticmethod(_tir_int64)
    int32 = staticmethod(_tir_int32)
    handle = "handle"  # type annotation: T.handle → "handle" dtype string
    Buffer = staticmethod(_tir_buffer)
    unroll = _ForKindSurface(tvm.tirx.ForKind.UNROLLED)
    parallel = _ForKindSurface(tvm.tirx.ForKind.PARALLEL)
    serial = _ForKindSurface(tvm.tirx.ForKind.SERIAL)
    vectorized = _ForKindSurface(tvm.tirx.ForKind.VECTORIZED)


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
