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
                # Resolve _DtypeHelper to dtype string
                if isinstance(ann, _DtypeHelper):
                    ann = ann._dtype
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

            # Store buffer_map on parser so match_buffer can populate it
            parser._tir_buffer_map = buffer_map
            parser._tir_params = params

            # Set TIR dialect callbacks for body parsing
            old = (
                parser.make_assign,
                parser.make_store,
                parser.make_for,
                parser.create_var,
                parser.handle_return,
                parser.handle_while,
            )
            parser.make_assign = _tir_make_assign
            parser.make_store = _tir_make_store
            parser.make_for = _tir_make_for
            parser.create_var = lambda name, ann=None: tvm.tirx.Var(name, "int32")
            parser.handle_return = _tir_handle_return
            parser.handle_while = _tir_handle_while
            try:
                body_stmts = parser.visit_body(node.body)
            finally:
                (
                    parser.make_assign,
                    parser.make_store,
                    parser.make_for,
                    parser.create_var,
                    parser.handle_return,
                    parser.handle_while,
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

            # Collect final buffer_map (may have been updated by match_buffer)
            buffer_map = getattr(parser, '_tir_buffer_map', buffer_map)

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


class _GridSurface(SurfaceObject):
    """Surface object for T.grid(e1, e2, ...) — sugar for nested serial For loops."""

    def __call__(self, *extents):
        return _GridSurfaceInstance(extents)


class _GridSurfaceInstance(SurfaceObject):
    """Instance created by T.grid(n, 10) — captures extents, expands to nested For."""

    def __init__(self, extents):
        self._extents = extents

    def parse_for(self, parser, node):
        from tvm_ffi import pyast

        # Extract loop variable names from Tuple LHS
        if isinstance(node.lhs, pyast.Tuple):
            var_names = [v.name for v in node.lhs.values]
        else:
            var_names = [node.lhs.name]

        if len(var_names) != len(self._extents):
            raise ValueError(
                f"T.grid has {len(self._extents)} extents but "
                f"{len(var_names)} loop variables"
            )

        # Create all loop vars and define them before parsing body
        loop_vars = []
        for name in var_names:
            var = tvm.tirx.Var(name, "int32")
            parser.var_table.define(name, var)
            loop_vars.append(var)

        # Parse body (innermost)
        body_stmts = parser.visit_body(node.body)
        if len(body_stmts) == 1:
            body = body_stmts[0]
        else:
            body = tvm.tirx.SeqStmt(body_stmts)

        # Build nested For from inside out
        for i in reversed(range(len(loop_vars))):
            extent = self._extents[i]
            if isinstance(extent, int):
                extent = tvm.tirx.IntImm("int32", extent)
            body = tvm.tirx.For(
                loop_vars[i],
                tvm.tirx.IntImm("int32", 0),
                extent,
                int(tvm.tirx.ForKind.SERIAL),
                body,
            )

        return body


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


def _tir_handle_while(parser, node):
    """TIR while handler: `while cond: body` → While(cond, body)."""
    cond = parser.eval_expr(node.cond)
    body_stmts = parser.visit_body(node.body)
    if len(body_stmts) == 1:
        body = body_stmts[0]
    else:
        body = tvm.tirx.SeqStmt(body_stmts)
    return tvm.tirx.While(cond, body)


def _tir_make_for(parser, var_name, start, end, step, body_node):
    """TIR for callback: range(n) → For(serial). Creates loop var and parses body."""
    if isinstance(start, int):
        start = tvm.tirx.IntImm("int32", start)
    if isinstance(end, int):
        end = tvm.tirx.IntImm("int32", end)
    extent = tvm.arith.Analyzer().simplify(end - start)
    # Create loop var with dtype matching extent
    ext_dtype = str(extent.dtype) if hasattr(extent, "dtype") else "int32"
    loop_var = tvm.tirx.Var(var_name, ext_dtype)
    parser.var_table.define(var_name, loop_var)
    # Parse body
    body_stmts = parser.visit_body(body_node)
    if len(body_stmts) == 1:
        body_stmt = body_stmts[0]
    else:
        body_stmt = tvm.tirx.SeqStmt(body_stmts)
    return tvm.tirx.For(loop_var, start, extent, int(tvm.tirx.ForKind.SERIAL), body_stmt)


def _tir_make_assign(parser, node, rhs_val):
    """TIR assign callback: handles alloc_buffer, match_buffer, Bind, and plain bindings."""
    name = node.lhs.name
    if isinstance(rhs_val, _MatchBufferResult):
        # match_buffer: add to buffer_map, define buffer in var_table
        buf = rhs_val.buffer
        # Set buffer name to match the LHS variable name
        buf = tvm.tirx.decl_buffer(buf.shape, buf.dtype, name=name)
        param_var = rhs_val.param_var
        # Find the matching param Var and add to buffer_map
        if hasattr(parser, '_tir_buffer_map'):
            # Find param var by identity or name
            for pv in getattr(parser, '_tir_params', []):
                if pv.same_as(param_var) or (hasattr(param_var, 'name') and pv.name == param_var.name):
                    parser._tir_buffer_map[pv] = buf
                    break
        parser.var_table.define(name, buf)
        return None  # no stmt emitted — buffer_map entry only
    if isinstance(rhs_val, tuple) and len(rhs_val) == 2:
        # alloc_buffer / decl_buffer returns (buf, AllocBuffer/DeclBuffer node)
        buf, alloc_node = rhs_val
        parser.var_table.define(name, buf)
        return alloc_node
    # If RHS is a PrimExpr (not a Stmt), create a Bind node
    if isinstance(rhs_val, tvm.tirx.PrimExpr):
        var = tvm.tirx.Var(name, rhs_val.dtype)
        parser.var_table.define(name, var)
        return tvm.tirx.Bind(var, rhs_val)
    parser.var_table.define(name, rhs_val)
    return None  # plain binding, no stmt emitted


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


def _tir_decl_buffer(shape, dtype="float32", data=None, scope=""):
    """T.decl_buffer(shape, dtype, data=...) → (Buffer, DeclBuffer node)."""
    if isinstance(shape, tuple):
        shape = list(shape)
    shape = [tvm.tirx.IntImm("int32", s) if isinstance(s, int) else s for s in shape]
    buf = tvm.tirx.decl_buffer(shape, dtype, data=data, scope=scope)
    return buf, tvm.tirx.DeclBuffer(buf)


class _MatchBufferResult:
    """Marker returned by T.match_buffer for _tir_make_assign to handle."""
    def __init__(self, param_var, buffer):
        self.param_var = param_var
        self.buffer = buffer


def _tir_match_buffer(param, shape, dtype="float32"):
    """T.match_buffer(A, (n,), 'int64') → MatchBufferResult for make_assign."""
    if isinstance(shape, tuple):
        shape = list(shape)
    shape = [tvm.tirx.IntImm("int32", s) if isinstance(s, int) else s for s in shape]
    buf = tvm.tirx.decl_buffer(shape, dtype)
    return _MatchBufferResult(param, buf)


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


class _DtypeHelper:
    """T.int32 — usable as both type annotation (bare) and literal constructor (called).

    T.int32 as annotation → evaluates to this object, parser sees non-Buffer → dtype "int32"
    T.int32(42) as call → returns IntImm("int32", 42)
    """
    def __init__(self, dtype):
        self._dtype = dtype
    def __call__(self, value):
        if isinstance(value, str):
            value = float(value) if "." in value or value in ("nan", "inf", "-inf") else int(value)
        if "float" in self._dtype:
            return tvm.tirx.FloatImm(self._dtype, float(value))
        return tvm.tirx.IntImm(self._dtype, int(value))
    def __repr__(self):
        return self._dtype


class _TIRModule:
    """Language module for TIR (the 'T' in `from tvm.script import tirx as T`)."""

    prim_func = _PrimFuncSurface()
    evaluate = staticmethod(_tir_evaluate)
    alloc_buffer = staticmethod(_tir_alloc_buffer)
    decl_buffer = staticmethod(_tir_decl_buffer)
    match_buffer = staticmethod(_tir_match_buffer)
    float32 = _DtypeHelper("float32")
    float16 = _DtypeHelper("float16")
    float64 = _DtypeHelper("float64")
    int64 = _DtypeHelper("int64")
    int32 = _DtypeHelper("int32")
    int16 = _DtypeHelper("int16")
    int8 = _DtypeHelper("int8")
    uint8 = _DtypeHelper("uint8")
    uint16 = _DtypeHelper("uint16")
    bool = _DtypeHelper("bool")
    handle = "handle"  # type annotation: T.handle → "handle" dtype string
    Buffer = staticmethod(_tir_buffer)

    @staticmethod
    def exp(val):
        """T.exp(val) → Call(tirx.exp, [val])."""
        dtype = val.dtype if hasattr(val, "dtype") else "float32"
        return tvm.tirx.Call(dtype, tvm.tirx.op.Op.get("tirx.exp"), [val])

    @staticmethod
    def Cast(dtype, val):
        """T.Cast('float16', val) → Cast(dtype, val)."""
        if isinstance(val, int):
            val = tvm.tirx.IntImm("int32", val)
        elif isinstance(val, float):
            val = tvm.tirx.FloatImm("float32", val)
        return tvm.tirx.Cast(dtype, val)
    unroll = _ForKindSurface(tvm.tirx.ForKind.UNROLLED)
    parallel = _ForKindSurface(tvm.tirx.ForKind.PARALLEL)
    serial = _ForKindSurface(tvm.tirx.ForKind.SERIAL)
    vectorized = _ForKindSurface(tvm.tirx.ForKind.VECTORIZED)
    grid = _GridSurface()


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
