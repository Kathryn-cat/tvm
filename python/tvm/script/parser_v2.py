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
                elif isinstance(ann, tvm.ir.PointerType):
                    var = tvm.tirx.Var(name, ann)
                    parser.var_table.define(name, var)
                    params.append(var)
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
                parser.handle_if,
            )
            parser.make_assign = _tir_make_assign
            parser.make_store = _tir_make_store
            parser.make_for = _tir_make_for
            parser.create_var = lambda name, ann=None: tvm.tirx.Var(name, "int32")
            parser.handle_return = _tir_handle_return
            parser.handle_while = _tir_handle_while
            parser.handle_if = _tir_handle_if
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
                    parser.handle_if,
                ) = old

            # Collect func_attr markers, sblock_alloc_buffer, and wrap
            extra_attrs = {}
            root_alloc_bufs = []
            wrapped = []
            for s in body_stmts:
                if isinstance(s, _FuncAttrMarker):
                    extra_attrs.update(s.attrs)
                elif isinstance(s, _SBlockAllocBufferMarker):
                    parser.var_table.define(s.buf.name, s.buf)
                    root_alloc_bufs.append(s.buf)
                elif isinstance(s, tvm.tirx.Stmt):
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

            # Wrap in root SBlockRealize if needed
            has_sblocks = any(
                isinstance(s, tvm.tirx.SBlockRealize)
                for s in (body_stmts if len(body_stmts) > 1 else [body])
            ) or root_alloc_bufs
            if has_sblocks:
                root_sb = tvm.tirx.SBlock(
                    [], [], [], "root", body, init=None,
                    alloc_buffers=root_alloc_bufs, match_buffers=[],
                    annotations={},
                )
                body = tvm.tirx.SBlockRealize(
                    [], tvm.tirx.IntImm("bool", 1), root_sb,
                )

            # Collect final buffer_map (may have been updated by match_buffer)
            buffer_map = getattr(parser, '_tir_buffer_map', buffer_map)

            func_name = node.name.name
            pf = tvm.tirx.PrimFunc(params, body, buffer_map=buffer_map)
            if not self._private:
                pf = pf.with_attr("global_symbol", func_name)
            for k, v in extra_attrs.items():
                pf = pf.with_attr(k, v)
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


# ============================================================================
# SBlock markers and surface objects
# ============================================================================


class _SBlockAllocBufferMarker:
    """Marker for T.sblock_alloc_buffer — collected by root sblock."""
    def __init__(self, buf, alloc_node):
        self.buf = buf
        self.alloc_node = alloc_node


class _InitMarker:
    """Marker for `with T.init(): body` — collected by sblock."""
    def __init__(self, body):
        self.body = body


class _InitSurface(SurfaceObject):
    """Surface object for `with T.init(): body`."""
    def __call__(self):
        return self
    def parse_with(self, parser, node):
        body_stmts = parser.visit_body(node.body)
        if len(body_stmts) == 1:
            body = body_stmts[0]
        else:
            body = tvm.tirx.SeqStmt(body_stmts)
        return _InitMarker(body)


class _SBlockAttrMarker:
    """Marker for T.sblock_attr({...}) — collected by sblock surface."""
    def __init__(self, attrs):
        self.attrs = attrs


class _AxisBinding:
    """Marker for T.axis.spatial(extent, value) — collected by sblock."""
    def __init__(self, extent, value, iter_type):
        self.extent = extent
        self.value = value
        self.iter_type = iter_type  # 0=spatial, 1=reduce, 2=scan, 3=opaque


class _ReadsMarker:
    """Marker for T.reads(...)."""
    def __init__(self, regions):
        self.regions = regions


class _WritesMarker:
    """Marker for T.writes(...)."""
    def __init__(self, regions):
        self.regions = regions


class _AxisModule:
    """T.axis.spatial / T.axis.reduce / T.axis.remap."""
    @staticmethod
    def spatial(extent, value):
        if isinstance(extent, int):
            extent = tvm.tirx.IntImm("int32", extent)
        return _AxisBinding(extent, value, 0)

    @staticmethod
    def reduce(extent, value):
        if isinstance(extent, int):
            extent = tvm.tirx.IntImm("int32", extent)
        return _AxisBinding(extent, value, 1)

    @staticmethod
    def scan(extent, value):
        if isinstance(extent, int):
            extent = tvm.tirx.IntImm("int32", extent)
        return _AxisBinding(extent, value, 2)

    @staticmethod
    def opaque(extent, value):
        if isinstance(extent, int):
            extent = tvm.tirx.IntImm("int32", extent)
        return _AxisBinding(extent, value, 3)


class _SBlockSurface(SurfaceObject):
    """Surface object for `with T.sblock("name"):` → SBlockRealize/SBlock."""

    def __call__(self, name):
        return _SBlockSurfaceInstance(name)


class _SBlockSurfaceInstance(SurfaceObject):
    """Instance for `with T.sblock("B") as [vi, vj]:` → parse_with."""

    def __init__(self, name):
        self._name = name

    def parse_with(self, parser, node):
        # Define as_var if present (for root block, lhs may be None)
        # Parse body — collect axis bindings, reads, writes, sblock_attr
        body_stmts = parser.visit_body(node.body)

        iter_vars = []
        iter_values = []
        reads = []
        writes = []
        annotations = {}
        alloc_buffers = []
        init_body = None
        real_body = []

        for s in body_stmts:
            if isinstance(s, _AxisBinding):
                # The axis binding was assigned: vi = T.axis.spatial(128, i0)
                # We need to find the var from make_assign
                iter_vars.append(s)
                iter_values.append(s.value)
            elif isinstance(s, _ReadsMarker):
                reads = s.regions
            elif isinstance(s, _WritesMarker):
                writes = s.regions
            elif isinstance(s, _SBlockAttrMarker):
                annotations.update(s.attrs)
            elif isinstance(s, _InitMarker):
                init_body = s.body
            elif isinstance(s, _SBlockAllocBufferMarker):
                alloc_buffers.append(s.buf)
            elif isinstance(s, tvm.tirx.Stmt):
                real_body.append(s)

        # Build IterVars
        tir_iter_vars = []
        tir_iter_values = []
        for ab in iter_vars:
            dom = tvm.ir.Range(tvm.tirx.IntImm("int32", 0), ab.extent)
            var = ab._var  # set by _tir_make_assign when it sees _AxisBinding
            iv = tvm.tirx.IterVar(dom, var, ab.iter_type, "")
            tir_iter_vars.append(iv)
            tir_iter_values.append(ab.value)

        # Build body
        if len(real_body) == 1:
            body = real_body[0]
        elif len(real_body) > 1:
            body = tvm.tirx.SeqStmt(real_body)
        else:
            body = tvm.tirx.Evaluate(tvm.tirx.IntImm("int32", 0))

        # Convert reads/writes to BufferRegion lists
        tir_reads = _to_buffer_regions(reads)
        tir_writes = _to_buffer_regions(writes)

        sb = tvm.tirx.SBlock(
            tir_iter_vars, tir_reads, tir_writes,
            self._name, body, init=init_body,
            alloc_buffers=alloc_buffers, match_buffers=[],
            annotations=annotations,
        )
        pred = tvm.tirx.IntImm("bool", 1)
        return tvm.tirx.SBlockRealize(tir_iter_values, pred, sb)


def _to_buffer_regions(regions):
    """Convert parsed buffer index expressions to BufferRegion list."""
    if not regions:
        return []
    result = []
    for r in regions:
        if isinstance(r, tvm.tirx.BufferRegion):
            result.append(r)
        elif isinstance(r, tvm.tirx.BufferLoad):
            # buf[vi, vj] → BufferRegion with point ranges
            ranges = []
            for idx in r.indices:
                ranges.append(tvm.ir.Range(idx, tvm.tirx.IntImm("int32", 1)))
            result.append(tvm.tirx.BufferRegion(r.buffer, ranges))
    return result


class _AttrSurfaceInstance(SurfaceObject):
    """Surface object for `with T.attr(node, key, value): body` → AttrStmt."""

    def __init__(self, node_val, attr_key, value):
        self._node_val = node_val
        self._attr_key = attr_key
        self._value = value

    def parse_with(self, parser, node):
        body_stmts = parser.visit_body(node.body)
        if len(body_stmts) == 1:
            body = body_stmts[0]
        else:
            body = tvm.tirx.SeqStmt(body_stmts)
        return tvm.tirx.AttrStmt(self._node_val, self._attr_key, self._value, body)


class _ForKindSurface(SurfaceObject):
    """Surface object for T.unroll(n), T.parallel(n), T.vectorized(n), T.serial(n)."""

    def __init__(self, kind):
        self._kind = kind

    def __call__(self, extent, start=None, thread=None, **kwargs):
        """Called as T.unroll(2) or T.thread_binding(4, thread='blockIdx.x')."""
        return _ForKindSurfaceInstance(self._kind, extent, start, thread=thread)


class _ForKindSurfaceInstance(SurfaceObject):
    """Instance created by T.unroll(2) — captures args, dispatches parse_for."""

    def __init__(self, kind, extent, start=None, thread=None):
        self._kind = kind
        self._extent = extent
        self._start = start if start is not None else tvm.tirx.IntImm("int32", 0)
        self._thread = thread

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
        # Thread binding: create IterVar with thread_tag
        # IterVar uses a separate "iter" var, not the loop_var
        thread_binding = None
        if self._thread is not None:
            iter_var = tvm.tirx.Var("iter", "int32")
            iv = tvm.tirx.IterVar(None, iter_var, 1, self._thread)
            thread_binding = iv
        return tvm.tirx.For(
            loop_var, start, extent, int(self._kind), body,
            thread_binding=thread_binding,
        )


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


def _tir_handle_if(parser, node):
    """TIR if handler: `if cond: then else: else` → IfThenElse."""
    cond = parser.eval_expr(node.cond)
    then_body = parser.visit_body(node.then_branch)
    if len(then_body) == 1:
        then_stmt = then_body[0]
    else:
        then_stmt = tvm.tirx.SeqStmt(then_body)
    else_stmt = None
    if node.else_branch and len(node.else_branch) > 0:
        else_body = parser.visit_body(node.else_branch)
        if len(else_body) == 1:
            else_stmt = else_body[0]
        else:
            else_stmt = tvm.tirx.SeqStmt(else_body)
    return tvm.tirx.IfThenElse(cond, then_stmt, else_stmt)


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
    if isinstance(rhs_val, _VarDeclMarker):
        var = tvm.tirx.Var(name, rhs_val.dtype)
        parser.var_table.define(name, var)
        return None
    if isinstance(rhs_val, _AxisBinding):
        var = tvm.tirx.Var(name, "int32")
        parser.var_table.define(name, var)
        rhs_val._var = var
        return rhs_val
    if isinstance(rhs_val, _SBlockAllocBufferMarker):
        # Set buffer name to match variable name
        scope = rhs_val.buf.scope() if callable(rhs_val.buf.scope) else rhs_val.buf.scope
        buf = tvm.tirx.decl_buffer(
            rhs_val.buf.shape, rhs_val.buf.dtype, name=name,
            scope=scope,
        )
        rhs_val.buf = buf
        parser.var_table.define(name, buf)
        return rhs_val  # marker collected by PrimFunc or SBlock
    if isinstance(rhs_val, _MatchBufferResult):
        # match_buffer: add to buffer_map, define buffer in var_table
        buf = rhs_val.buffer
        # Set buffer name to match the LHS variable name, preserve all params
        scope = buf.scope() if callable(buf.scope) else buf.scope
        buf = tvm.tirx.decl_buffer(
            buf.shape, buf.dtype, name=name,
            strides=list(buf.strides) if buf.strides else [],
            elem_offset=buf.elem_offset if buf.elem_offset is not None else None,
            scope=scope,
            offset_factor=buf.offset_factor,
        )
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
        # Check for type annotation on the AST node
        dtype = rhs_val.dtype
        if node.annotation is not None:
            ann = parser.eval_expr(node.annotation)
            if isinstance(ann, _DtypeHelper):
                dtype = ann._dtype
            elif isinstance(ann, tvm.ir.PointerType):
                dtype = ann
            elif isinstance(ann, str):
                dtype = ann
        var = tvm.tirx.Var(name, dtype)
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
    # Convert indices
    converted = []
    for i in indices:
        if isinstance(i, int):
            converted.append(tvm.tirx.IntImm("int32", i))
        elif isinstance(i, slice):
            # slice(start, stop) → Ramp(start, 1, lanes=stop-start)
            start = i.start if i.start is not None else 0
            stop = i.stop
            if isinstance(start, int):
                start = tvm.tirx.IntImm("int32", start)
            if isinstance(stop, int) and isinstance(i.start, (int, type(None))):
                lanes = stop - (i.start if i.start is not None else 0)
            else:
                lanes = tvm.arith.Analyzer().simplify(stop - start)
                lanes = int(lanes) if hasattr(lanes, 'value') else int(lanes)
            converted.append(tvm.tirx.Ramp(start, tvm.tirx.IntImm("int32", 1), lanes))
        else:
            converted.append(i)
    return tvm.tirx.BufferStore(target, value, converted)


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


class _FuncAttrMarker:
    """Marker for T.func_attr({...}) — collected by PrimFunc surface object."""
    def __init__(self, attrs):
        self.attrs = attrs


class _VarDeclMarker:
    """Marker for T.int32() — declares a Var with given dtype."""
    def __init__(self, dtype):
        self.dtype = dtype


class _MatchBufferResult:
    """Marker returned by T.match_buffer for _tir_make_assign to handle."""
    def __init__(self, param_var, buffer):
        self.param_var = param_var
        self.buffer = buffer


def _tir_match_buffer(param, shape, dtype="float32", scope="",
                      strides=None, offset_factor=0, elem_offset=None,
                      align=64, **kwargs):
    """T.match_buffer(A, (n,), 'int64', scope=..., strides=..., ...) → MatchBufferResult."""
    if isinstance(shape, tuple):
        shape = list(shape)
    shape = [tvm.tirx.IntImm("int32", s) if isinstance(s, int) else s for s in shape]
    if strides is not None and isinstance(strides, (list, tuple)):
        strides = [tvm.tirx.IntImm("int32", s) if isinstance(s, int) else s for s in strides]
    buf = tvm.tirx.decl_buffer(
        shape, dtype, scope=scope,
        strides=strides or [],
        elem_offset=elem_offset,
        offset_factor=offset_factor,
    )
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
    def __call__(self, *args):
        if len(args) == 0:
            # T.int32() with no args — marker for "declare a Var with this dtype"
            return _VarDeclMarker(self._dtype)
        if len(args) == 1:
            value = args[0]
            if isinstance(value, str):
                if value in ("nan", "inf", "-inf") or "." in value:
                    value = float(value)
                else:
                    return self  # dtype string like T.handle("int32") — return self
            if "float" in self._dtype:
                return tvm.tirx.FloatImm(self._dtype, float(value))
            if self._dtype == "handle":
                return self  # T.handle(something) as annotation
            return tvm.tirx.IntImm(self._dtype, int(value))
        # Multiple args: T.handle("int32", "global") → PointerType
        if self._dtype == "handle" and len(args) >= 1:
            elem_dtype = args[0]
            scope = args[1] if len(args) > 1 else ""
            return tvm.ir.PointerType(tvm.ir.PrimType(elem_dtype), scope)
        return self
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
    handle = _DtypeHelper("handle")  # T.handle as annotation; T.handle(...) returns Var
    Buffer = staticmethod(_tir_buffer)

    sblock = _SBlockSurface()
    init = _InitSurface()
    axis = _AxisModule

    @staticmethod
    def sblock_alloc_buffer(shape, dtype="float32", scope=""):
        """T.sblock_alloc_buffer(shape) → marker for root sblock."""
        if isinstance(shape, tuple):
            shape = list(shape)
        shape = [tvm.tirx.IntImm("int32", s) if isinstance(s, int) else s for s in shape]
        buf = tvm.tirx.decl_buffer(shape, dtype, scope=scope)
        return _SBlockAllocBufferMarker(buf, tvm.tirx.AllocBuffer(buf, {}))

    @staticmethod
    def sblock_attr(attrs_dict):
        """T.sblock_attr({...}) → marker collected by sblock."""
        return _SBlockAttrMarker(attrs_dict)

    @staticmethod
    def reads(*args):
        """T.reads(buf[vi, vj]) → marker."""
        return _ReadsMarker(list(args))

    @staticmethod
    def writes(*args):
        """T.writes(buf[vi, vj]) → marker."""
        return _WritesMarker(list(args))

    @staticmethod
    def func_attr(attrs_dict):
        """T.func_attr({...}) → marker collected by PrimFunc."""
        return _FuncAttrMarker(attrs_dict)

    @staticmethod
    def vscale():
        """T.vscale() → call_intrin('int32', 'tirx.vscale')."""
        return tvm.tirx.call_intrin("int32", "tirx.vscale")

    @staticmethod
    def exp(val):
        """T.exp(val) → Call(tirx.exp, [val])."""
        dtype = val.dtype if hasattr(val, "dtype") else "float32"
        return tvm.tirx.Call(dtype, tvm.tirx.op.Op.get("tirx.exp"), [val])

    @staticmethod
    def attr(node_val, attr_key, value):
        """T.attr(node, key, value) → _AttrSurfaceInstance for `with` dispatch."""
        if isinstance(node_val, int):
            node_val = tvm.tirx.IntImm("int32", node_val)
        if isinstance(value, int):
            value = tvm.tirx.IntImm("int32", value)
        return _AttrSurfaceInstance(node_val, attr_key, value)

    @staticmethod
    def if_then_else(cond, then_val, else_val):
        """T.if_then_else(cond, then, else) → Call(tirx.if_then_else, ...)."""
        if isinstance(then_val, int):
            then_val = tvm.tirx.IntImm("int32", then_val)
        if isinstance(else_val, int):
            else_val = tvm.tirx.IntImm("int32", else_val)
        return tvm.tirx.if_then_else(cond, then_val, else_val)

    @staticmethod
    def call_extern(dtype, func_name, *args):
        """T.call_extern('float32', 'deref', ...) → Call(call_extern, ...)."""
        converted = []
        for a in args:
            if isinstance(a, int):
                a = tvm.tirx.IntImm("int32", a)
            elif isinstance(a, float):
                a = tvm.tirx.FloatImm("float32", a)
            converted.append(a)
        return tvm.tirx.call_extern(dtype, func_name, *converted)

    @staticmethod
    def address_of(load):
        """T.address_of(buf[i]) → call_intrin('handle', 'tirx.address_of', load)."""
        return tvm.tirx.call_intrin("handle", "tirx.address_of", load)

    @staticmethod
    def Broadcast(value, lanes):
        """T.Broadcast(42, 8) → Broadcast(IntImm(42), 8)."""
        if isinstance(value, int):
            value = tvm.tirx.IntImm("int32", value)
        return tvm.tirx.Broadcast(value, lanes)

    @staticmethod
    def min(a, b):
        return tvm.tirx.min(a, b)

    @staticmethod
    def max(a, b):
        return tvm.tirx.max(a, b)

    @staticmethod
    def Div(a, b):
        if isinstance(a, int): a = tvm.tirx.IntImm("int32", a)
        if isinstance(b, int): b = tvm.tirx.IntImm("int32", b)
        return tvm.tirx.Div(a, b)

    @staticmethod
    def Mod(a, b):
        if isinstance(a, int): a = tvm.tirx.IntImm("int32", a)
        if isinstance(b, int): b = tvm.tirx.IntImm("int32", b)
        return tvm.tirx.Mod(a, b)

    @staticmethod
    def Select(cond, true_val, false_val):
        return tvm.tirx.Select(cond, true_val, false_val)

    @staticmethod
    def where(cond, true_val=None, false_val=None):
        if true_val is not None and false_val is not None:
            return tvm.tirx.Select(cond, true_val, false_val)
        # T.where(cond) with one arg — just the condition
        return cond

    And = staticmethod(lambda a, b: tvm.tirx.And(a, b))
    Or = staticmethod(lambda a, b: tvm.tirx.Or(a, b))
    Not = staticmethod(lambda a: tvm.tirx.Not(a))

    @staticmethod
    def assume(cond):
        return tvm.tirx.call_intrin("bool", "tirx.assume", cond)

    @staticmethod
    def Ramp(base, stride, lanes):
        if isinstance(base, int): base = tvm.tirx.IntImm("int32", base)
        if isinstance(stride, int): stride = tvm.tirx.IntImm("int32", stride)
        return tvm.tirx.Ramp(base, stride, lanes)

    @staticmethod
    def Shuffle(vectors, indices):
        return tvm.tirx.Shuffle(vectors, indices)

    @staticmethod
    def Let(value, body=None, **kwargs):
        # T.Let is complex, skip body for now
        return value

    @staticmethod
    def call_packed(func_name, *args, dtype="void"):
        return tvm.tirx.call_packed(func_name, *args, dtype=dtype)

    @staticmethod
    def call_pure_extern(dtype, func_name, *args):
        return tvm.tirx.call_pure_extern(dtype, func_name, *args)

    @staticmethod
    def call_intrin(dtype, intrin_name, *args):
        return tvm.tirx.call_intrin(dtype, intrin_name, *args)

    @staticmethod
    def target(config_dict):
        """T.target({...}) → Target object."""
        return tvm.target.Target(config_dict)

    @staticmethod
    def reinterpret(dtype, val):
        """T.reinterpret('float32', val) → reinterpret(dtype, val)."""
        if isinstance(val, int):
            val = tvm.tirx.IntImm("int32", val)
        return tvm.tirx.reinterpret(dtype, val)

    @staticmethod
    def truncmod(a, b):
        if isinstance(a, int): a = tvm.tirx.IntImm("int32", a)
        if isinstance(b, int): b = tvm.tirx.IntImm("int32", b)
        return tvm.tirx.Mod(a, b)

    @staticmethod
    def truncdiv(a, b):
        if isinstance(a, int): a = tvm.tirx.IntImm("int32", a)
        if isinstance(b, int): b = tvm.tirx.IntImm("int32", b)
        return tvm.tirx.Div(a, b)

    @staticmethod
    def FloorDiv(a, b):
        if isinstance(a, int): a = tvm.tirx.IntImm("int32", a)
        if isinstance(b, int): b = tvm.tirx.IntImm("int32", b)
        return tvm.tirx.FloorDiv(a, b)

    @staticmethod
    def FloorMod(a, b):
        if isinstance(a, int): a = tvm.tirx.IntImm("int32", a)
        if isinstance(b, int): b = tvm.tirx.IntImm("int32", b)
        return tvm.tirx.FloorMod(a, b)

    @staticmethod
    def ceildiv(a, b):
        if isinstance(a, int): a = tvm.tirx.IntImm("int32", a)
        if isinstance(b, int): b = tvm.tirx.IntImm("int32", b)
        return tvm.tirx.ceildiv(a, b)

    @staticmethod
    def abs(val):
        return tvm.tirx.abs(val)

    @staticmethod
    def sqrt(val):
        return tvm.tirx.sqrt(val)

    @staticmethod
    def log(val):
        return tvm.tirx.log(val)

    @staticmethod
    def log2(val):
        return tvm.tirx.log(val) / tvm.tirx.log(tvm.tirx.FloatImm("float32", 2.0))

    @staticmethod
    def sigmoid(val):
        return tvm.tirx.sigmoid(val)

    @staticmethod
    def tanh(val):
        return tvm.tirx.tanh(val)

    @staticmethod
    def power(base, exp):
        return tvm.tirx.power(base, exp)

    @staticmethod
    def popcount(val):
        return tvm.tirx.popcount(val)

    @staticmethod
    def shift_left(a, b):
        return a << b

    @staticmethod
    def shift_right(a, b):
        return a >> b

    @staticmethod
    def bitwise_and(a, b):
        return a & b

    @staticmethod
    def bitwise_or(a, b):
        return a | b

    @staticmethod
    def bitwise_xor(a, b):
        return a ^ b

    @staticmethod
    def bitwise_not(a):
        return ~a

    @staticmethod
    def likely(cond):
        return tvm.tirx.call_intrin("bool", "tirx.likely", cond)

    @staticmethod
    def comm_reducer(combiner_fn, identity):
        return tvm.tirx.CommReducer(combiner_fn, identity)

    @staticmethod
    def float32x4(*args):
        return _DtypeHelper("float32x4")

    @staticmethod
    def float32x8(*args):
        return _DtypeHelper("float32x8")

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
    thread_binding = _ForKindSurface(tvm.tirx.ForKind.THREAD_BINDING)
    grid = _GridSurface()


# ============================================================================
# Relax surface objects and language module
# ============================================================================


class _RelaxFuncSurface(SurfaceObject):
    """Surface object for @R.function decorator."""

    def __call__(self, *args, **kwargs):
        # @R.function(private=True) → return self
        return self

    def parse_function(self, parser, node):
        from tvm import relax

        bb = relax.BlockBuilder()
        params = []
        for arg in node.args:
            name = arg.lhs.name
            ann = None
            if arg.annotation is not None:
                ann = parser.eval_expr(arg.annotation)
            if isinstance(ann, relax.TensorStructInfo):
                var = relax.Var(name, ann)
            elif isinstance(ann, relax.StructInfo):
                var = relax.Var(name, ann)
            else:
                var = relax.Var(name)
            parser.var_table.define(name, var)
            params.append(var)

        # Parse return type annotation if present
        ret_sinfo = None
        if node.return_type is not None:
            ret_sinfo = parser.eval_expr(node.return_type)

        func_name = node.name.name

        with bb.function(func_name, params):
            # Parse body statements → emit bindings
            for stmt in node.body:
                from tvm_ffi import pyast

                if isinstance(stmt, pyast.Assign) and stmt.rhs is not None:
                    rhs_val = parser.eval_expr(stmt.rhs)
                    name = stmt.lhs.name
                    # Evaluate annotation for struct_info
                    sinfo = None
                    if stmt.annotation is not None:
                        sinfo = parser.eval_expr(stmt.annotation)
                    if isinstance(sinfo, relax.TensorStructInfo):
                        var = bb.emit(rhs_val, name)
                    else:
                        var = bb.emit(rhs_val, name)
                    parser.var_table.define(name, var)
                elif isinstance(stmt, pyast.Return):
                    if stmt.value is not None:
                        ret_val = parser.eval_expr(stmt.value)
                        bb.emit_func_output(ret_val)
                # Other statement types: skip for now

        mod = bb.get()
        func = mod[func_name]
        return func


class _RelaxTensorSInfo:
    """R.Tensor((3, 4), dtype='float32') → TensorStructInfo."""
    def __call__(self, shape=None, dtype=None, ndim=-1):
        from tvm import relax
        if isinstance(shape, tuple):
            shape = list(shape)
        return relax.TensorStructInfo(shape, dtype or "float32", ndim=ndim)


class _RelaxModule:
    """Language module for Relax (the 'R' in `from tvm.script import relax as R`)."""

    function = _RelaxFuncSurface()
    Tensor = _RelaxTensorSInfo()

    @staticmethod
    def add(x, y):
        from tvm.relax import op
        return op.add(x, y)

    @staticmethod
    def prim_value(val):
        from tvm import relax
        return relax.PrimValue(val)

    @staticmethod
    def func_attr(attrs_dict):
        return _FuncAttrMarker(attrs_dict)

    @staticmethod
    def Object():
        from tvm import relax
        return relax.ObjectStructInfo()

    @staticmethod
    def Tuple(*args):
        from tvm import relax
        return relax.TupleStructInfo(list(args))

    @staticmethod
    def shape(dims):
        from tvm import relax
        return relax.ShapeStructInfo(dims if isinstance(dims, list) else list(dims))

    @staticmethod
    def Shape(dims):
        from tvm import relax
        return relax.ShapeStructInfo(dims if isinstance(dims, list) else list(dims))

    @staticmethod
    def str(val):
        return val

    @staticmethod
    def dtype(val):
        return val

    @staticmethod
    def Prim(dtype="int64"):
        from tvm import relax
        return relax.PrimStructInfo(dtype)


class _IRLangModule:
    """Language module for IR (the 'I' in `from tvm.script import ir as I`)."""

    ir_module = _IRModuleSurface()
    GlobalVar = staticmethod(tvm.ir.GlobalVar)


# ============================================================================
# Public API
# ============================================================================


def make_parser() -> IRParser:
    """Create a parser configured with TIR/Relax language modules."""
    return IRParser(lang_modules={"T": _TIRModule, "I": _IRLangModule, "R": _RelaxModule})


def parse(text: str):
    """Parse V2 printer output back to TVM IR."""
    return make_parser().parse(text)
