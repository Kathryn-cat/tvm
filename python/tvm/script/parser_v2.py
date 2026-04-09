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


def _wrap_body_stmts(body_stmts):
    """Wrap non-Stmt items in Evaluate, filter None, return single Stmt or SeqStmt."""
    wrapped = []
    for s in body_stmts:
        if s is None:
            continue
        if isinstance(s, tvm.tirx.Stmt):
            wrapped.append(s)
        elif isinstance(s, tvm.tirx.PrimExpr):
            wrapped.append(tvm.tirx.Evaluate(s))
        else:
            # Non-TIR expr (e.g. relax.Call from GlobalVar call) — wrap as tir Call
            try:
                if hasattr(s, "op") and hasattr(s, "args"):
                    tir_call = tvm.tirx.Call("", s.op, list(s.args))
                    wrapped.append(tvm.tirx.Evaluate(tir_call))
                else:
                    wrapped.append(tvm.tirx.Evaluate(s))
            except Exception:
                wrapped.append(tvm.tirx.Evaluate(s))
    if len(wrapped) == 1:
        return wrapped[0]
    if len(wrapped) == 0:
        return tvm.tirx.Evaluate(tvm.tirx.IntImm("int32", 0))
    return tvm.tirx.SeqStmt(wrapped)


def _has_sblock_recursive(obj):
    """Check if obj or any nested child contains a SBlockRealize."""
    if isinstance(obj, tvm.tirx.SBlockRealize):
        return True
    if isinstance(obj, tvm.tirx.For):
        return _has_sblock_recursive(obj.body)
    if isinstance(obj, tvm.tirx.SeqStmt):
        return any(_has_sblock_recursive(s) for s in obj.seq)
    if isinstance(obj, tvm.tirx.AttrStmt):
        return _has_sblock_recursive(obj.body)
    if isinstance(obj, tvm.tirx.IfThenElse):
        return _has_sblock_recursive(obj.then_case) or (
            obj.else_case is not None and _has_sblock_recursive(obj.else_case)
        )
    return False


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
                parser.handle_assert,
            )
            parser.make_assign = _tir_make_assign
            parser.make_store = _tir_make_store
            parser.make_for = _tir_make_for
            parser.create_var = lambda name, ann=None: tvm.tirx.Var(name, "int32")
            parser.handle_return = _tir_handle_return
            parser.handle_while = _tir_handle_while
            parser.handle_if = _tir_handle_if
            parser.handle_assert = _tir_handle_assert
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
                    parser.handle_assert,
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
                elif isinstance(s, _MatchBufferResult):
                    pass  # already handled by _tir_make_assign buffer_map
                elif isinstance(s, tvm.tirx.Stmt):
                    wrapped.append(s)
                elif isinstance(s, tvm.tirx.PrimExpr):
                    wrapped.append(tvm.tirx.Evaluate(s))
                elif s is not None:
                    # Non-TIR expr (e.g. relax.Call from GlobalVar call)
                    try:
                        if hasattr(s, "op") and hasattr(s, "args"):
                            tir_call = tvm.tirx.Call("", s.op, list(s.args))
                            wrapped.append(tvm.tirx.Evaluate(tir_call))
                        else:
                            wrapped.append(tvm.tirx.Evaluate(s))
                    except Exception:
                        pass  # skip non-evaluatable items
            body_stmts = wrapped

            # Construct body
            if len(body_stmts) == 1:
                body = body_stmts[0]
            elif len(body_stmts) > 1:
                body = tvm.tirx.SeqStmt(body_stmts)
            else:
                body = tvm.tirx.Evaluate(tvm.tirx.IntImm("int32", 0))

            # Wrap in root SBlockRealize if needed (but not if body is already one)
            body_is_sblock = isinstance(body, tvm.tirx.SBlockRealize)
            has_sblocks = (_has_sblock_recursive(body) or root_alloc_bufs) and not body_is_sblock
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

            # Parse return type annotation
            ret_type = None
            if node.return_type is not None:
                ret_ann = parser.eval_expr(node.return_type)
                if isinstance(ret_ann, _DtypeHelper):
                    ret_type = tvm.ir.PrimType(ret_ann._dtype)
                elif isinstance(ret_ann, str):
                    ret_type = tvm.ir.PrimType(ret_ann)

            func_name = node.name.name
            pf = tvm.tirx.PrimFunc(params, body, ret_type=ret_type, buffer_map=buffer_map)
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
            # Pass 1: forward-declare all function GlobalVars with opaque FuncStructInfo.
            # GlobalVars need struct_info for call_tir/call_tir_inplace validation
            # (see example 180). Using make_node because GlobalVar() constructor
            # doesn't accept struct_info, and struct_info_ property has no setter.
            from tvm import relax
            _opaque_func_sinfo = relax.FuncStructInfo.opaque_func()
            gv_map = {}  # name → GlobalVar
            for stmt in node.body:
                from tvm_ffi import pyast

                if isinstance(stmt, pyast.Function):
                    gv = tvm.ir.make_node("ir.GlobalVar",
                                          name_hint=stmt.name.name,
                                          struct_info_=_opaque_func_sinfo)
                    parser.var_table.define(stmt.name.name, gv)
                    gv_map[stmt.name.name] = gv

            # Define the class name as a module alias so that
            # "Module.func_name" resolves to the GlobalVar.
            # The V2 printer emits "Module.func()" for cross-function calls.
            class _ModuleAlias:
                def __getattr__(self, name):
                    if name in gv_map:
                        return gv_map[name]
                    raise AttributeError(f"Module has no function '{name}'")

            class_name = node.name.name  # e.g. "Module"
            parser.var_table.define(class_name, _ModuleAlias())

            # Pass 2: parse function bodies, collect module attrs
            funcs = {}
            module_attrs = {}
            module_global_infos = {}
            for stmt in node.body:
                from tvm_ffi import pyast
                if isinstance(stmt, pyast.Function):
                    func_ir = parser.visit_stmt(stmt)
                    func_name = stmt.name.name
                    gv = gv_map[func_name]
                    funcs[gv] = func_ir
                elif isinstance(stmt, pyast.ExprStmt):
                    val = parser.eval_expr(stmt.expr)
                    if isinstance(val, _ModuleAttrsMarker):
                        module_attrs.update(val.attrs)
                    elif isinstance(val, _ModuleGlobalInfosMarker):
                        module_global_infos = val.infos
                # skip other stmts (pass, etc.)

            mod = tvm.ir.IRModule(funcs)
            if module_attrs:
                mod = mod.with_attrs(module_attrs)
            if module_global_infos:
                for key, infos in module_global_infos.items():
                    mod.update_global_info(key, infos)
            return mod


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

        # Normalize extents and create loop vars with matching dtypes
        extents = []
        loop_vars = []
        for i, name in enumerate(var_names):
            ext = self._extents[i]
            if isinstance(ext, int):
                ext = tvm.tirx.IntImm("int32", ext)
            ext_dtype = str(ext.dtype) if hasattr(ext, "dtype") else "int32"
            var = tvm.tirx.Var(name, ext_dtype)
            parser.var_table.define(name, var)
            loop_vars.append(var)
            extents.append(ext)

        # Parse body (innermost)
        body_stmts = parser.visit_body(node.body)
        body = _wrap_body_stmts(body_stmts)

        # Build nested For from inside out
        for i in reversed(range(len(loop_vars))):
            extent = extents[i]
            ext_dtype = str(extent.dtype) if hasattr(extent, "dtype") else "int32"
            body = tvm.tirx.For(
                loop_vars[i],
                tvm.tirx.IntImm(ext_dtype, 0),
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


class _WhereMarker:
    """Marker for T.where(cond) — sets predicate on SBlockRealize."""
    def __init__(self, cond):
        self.cond = cond


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
        return _AxisBinding(extent, value, 2)  # kCommReduce = 2

    @staticmethod
    def scan(extent, value):
        if isinstance(extent, int):
            extent = tvm.tirx.IntImm("int32", extent)
        return _AxisBinding(extent, value, 3)  # kOrdered = 3

    @staticmethod
    def opaque(extent, value):
        if isinstance(extent, int):
            extent = tvm.tirx.IntImm("int32", extent)
        return _AxisBinding(extent, value, 4)  # kOpaque = 4


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
        match_bufs = []
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
            elif isinstance(s, _MatchBufferResult):
                # match_buffer inside SBlock → MatchBufferRegion (see example 346)
                source = s.param_var  # BufferRegion from A[8:16, 32:64]
                if isinstance(source, tvm.tirx.BufferRegion):
                    match_bufs.append(tvm.tirx.MatchBufferRegion(s.buffer, source))
            elif isinstance(s, _WhereMarker):
                pass  # handled below
            elif isinstance(s, tvm.tirx.Stmt):
                real_body.append(s)
            elif isinstance(s, tvm.tirx.PrimExpr):
                # Bare expressions (e.g. T.call_packed) → wrap in Evaluate
                real_body.append(tvm.tirx.Evaluate(s))
            elif s is not None and hasattr(s, "op") and hasattr(s, "args"):
                # Non-TIR expr (e.g. relax.Call) → convert and wrap
                try:
                    tir_call = tvm.tirx.Call("", s.op, list(s.args))
                    real_body.append(tvm.tirx.Evaluate(tir_call))
                except Exception:
                    pass

        # Build IterVars
        tir_iter_vars = []
        tir_iter_values = []
        for ab in iter_vars:
            ext_dtype = str(ab.extent.dtype) if hasattr(ab.extent, "dtype") else "int32"
            dom = tvm.ir.Range(tvm.tirx.IntImm(ext_dtype, 0), ab.extent)
            var = ab._var  # set by _tir_make_assign when it sees _AxisBinding
            iv = tvm.tirx.IterVar(dom, var, ab.iter_type, "")
            tir_iter_vars.append(iv)
            tir_iter_values.append(ab.value)

        # Build body
        body = _wrap_body_stmts(real_body)

        # Convert reads/writes to BufferRegion lists
        tir_reads = _to_buffer_regions(reads)
        tir_writes = _to_buffer_regions(writes)

        sb = tvm.tirx.SBlock(
            tir_iter_vars, tir_reads, tir_writes,
            self._name, body, init=init_body,
            alloc_buffers=alloc_buffers, match_buffers=match_bufs,
            annotations=annotations,
        )
        # Collect predicate from T.where(cond)
        pred = tvm.tirx.IntImm("bool", 1)
        for s in body_stmts:
            if isinstance(s, _WhereMarker):
                pred = s.cond
                break
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
                idx_dtype = str(idx.dtype) if hasattr(idx, "dtype") else "int32"
                ranges.append(tvm.ir.Range.from_min_extent(idx, tvm.tirx.IntImm(idx_dtype, 1)))
            result.append(tvm.tirx.BufferRegion(r.buffer, ranges))
    return result


class _LaunchThreadSurface(SurfaceObject):
    """T.launch_thread("blockIdx.x", 128) → surface object for with-stmt."""

    def __call__(self, thread_tag_or_var, extent):
        return _LaunchThreadInstance(thread_tag_or_var, extent)


class _LaunchThreadInstance(SurfaceObject):
    """Instance for `with T.launch_thread("blockIdx.x", 128) as i:`."""

    def __init__(self, thread_tag_or_var, extent):
        self._thread_tag_or_var = thread_tag_or_var
        self._extent = extent if not isinstance(extent, int) else tvm.tirx.IntImm("int32", extent)

    def parse_with(self, parser, node):
        # Determine thread_tag and whether var was already defined
        thread_tag = self._thread_tag_or_var
        already_defined_var = None
        if isinstance(thread_tag, tvm.tirx.Var):
            # T.launch_thread(existing_var, extent) — var already in scope
            already_defined_var = thread_tag
            # Look up the IterVar's thread_tag from the var
            # For now, use a generic tag
            thread_tag = "unknown"
        elif not isinstance(thread_tag, str):
            thread_tag = str(thread_tag)

        # Infer dtype from extent
        ext_dtype = getattr(self._extent, "dtype", "int32")
        if already_defined_var is not None:
            # Var already defined: no `as` clause
            iv = tvm.tirx.IterVar(
                tvm.ir.Range(tvm.tirx.IntImm(ext_dtype, 0), self._extent),
                already_defined_var, 1, thread_tag,
            )
            body_stmts = parser.visit_body(node.body)
            if len(body_stmts) == 1:
                body = body_stmts[0]
            else:
                body = tvm.tirx.SeqStmt(body_stmts)
            return tvm.tirx.AttrStmt(iv, "thread_extent", self._extent, body)
        else:
            # New var from `as` clause
            var_name = node.lhs.name if node.lhs is not None else "v"
            var = tvm.tirx.Var(var_name, ext_dtype)
            parser.var_table.define(var_name, var)
            iv = tvm.tirx.IterVar(
                tvm.ir.Range(tvm.tirx.IntImm(ext_dtype, 0), self._extent),
                var, 1, thread_tag,
            )
            body_stmts = parser.visit_body(node.body)
            if len(body_stmts) == 1:
                body = body_stmts[0]
            else:
                body = tvm.tirx.SeqStmt(body_stmts)
            return tvm.tirx.AttrStmt(iv, "thread_extent", self._extent, body)


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

    def __call__(self, extent, start=None, thread=None, annotations=None, **kwargs):
        """Called as T.unroll(2) or T.thread_binding(4, thread='blockIdx.x')."""
        return _ForKindSurfaceInstance(self._kind, extent, start, thread=thread, annotations=annotations)


class _ForKindSurfaceInstance(SurfaceObject):
    """Instance created by T.unroll(2) — captures args, dispatches parse_for."""

    def __init__(self, kind, extent, start=None, thread=None, annotations=None):
        self._kind = kind
        self._extent = extent
        self._start = start if start is not None else tvm.tirx.IntImm("int32", 0)
        self._thread = thread
        self._annotations = annotations

    def parse_for(self, parser, node):
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
        # Infer loop var dtype from extent (e.g. T.thread_binding(T.int64(128)))
        ext_dtype = str(extent.dtype) if hasattr(extent, "dtype") else "int32"
        loop_var = tvm.tirx.Var(node.lhs.name, ext_dtype)
        parser.var_table.define(node.lhs.name, loop_var)
        body_stmts = parser.visit_body(node.body)
        body = _wrap_body_stmts(body_stmts)
        # Thread binding: create IterVar with thread_tag
        # IterVar uses a separate "iter" var, not the loop_var
        thread_binding = None
        if self._thread is not None:
            iter_var = tvm.tirx.Var("iter", ext_dtype)
            iv = tvm.tirx.IterVar(None, iter_var, 1, self._thread)
            thread_binding = iv
        return tvm.tirx.For(
            loop_var, start, extent, int(self._kind), body,
            thread_binding=thread_binding,
            annotations=self._annotations or {},
        )


# ============================================================================
# TIR language module callables
# ============================================================================


def _tir_handle_assert(parser, node):
    """TIR assert handler: `assert cond, (kind, [msgs])` → AssertStmt."""
    cond = parser.eval_expr(node.cond)
    kind = "AssertionError"
    message_parts = []
    if node.msg is not None:
        msg = parser.eval_expr(node.msg)
        if isinstance(msg, (tuple, list)) and len(msg) >= 1:
            kind = msg[0] if isinstance(msg[0], str) else str(msg[0])
            if len(msg) >= 2 and isinstance(msg[1], (list, tuple)):
                message_parts = [tvm.tirx.StringImm(str(s)) for s in msg[1]]
        elif isinstance(msg, str):
            message_parts = [tvm.tirx.StringImm(msg)]
    return tvm.tirx.AssertStmt(cond, tvm.tirx.StringImm(kind), message_parts)


def _tir_handle_return(parser, node):
    """TIR return handler: `return val` → Evaluate(Call(tirx.ret, [val]))."""
    if node.value is not None:
        val = parser.eval_expr(node.value)
        if isinstance(val, int):
            val = tvm.tirx.IntImm("int32", val)
        elif isinstance(val, float):
            val = tvm.tirx.FloatImm("float32", val)
        op = tvm.tirx.op.Op.get("tirx.ret")
        dtype = val.dtype if hasattr(val, "dtype") else "void"
        ret_call = tvm.tirx.Call(dtype, op, [val])
        return tvm.tirx.Evaluate(ret_call)
    return None


def _tir_handle_if(parser, node):
    """TIR if handler: `if cond: then else: else` → IfThenElse."""
    cond = parser.eval_expr(node.cond)
    then_stmt = _wrap_body_stmts(parser.visit_body(node.then_branch))
    else_stmt = None
    if node.else_branch and len(node.else_branch) > 0:
        else_stmt = _wrap_body_stmts(parser.visit_body(node.else_branch))
    return tvm.tirx.IfThenElse(cond, then_stmt, else_stmt)


def _tir_handle_while(parser, node):
    """TIR while handler: `while cond: body` → While(cond, body)."""
    cond = parser.eval_expr(node.cond)
    return tvm.tirx.While(cond, _wrap_body_stmts(parser.visit_body(node.body)))


def _tir_make_for(parser, var_name, start, end, step, body_node):
    """TIR for callback: range(n) → For(serial). Creates loop var and parses body."""
    if isinstance(start, int):
        start = tvm.tirx.IntImm("int32", start)
    if isinstance(end, int):
        end = tvm.tirx.IntImm("int32", end)
    if isinstance(step, int):
        step = tvm.tirx.IntImm("int32", step)
    extent = tvm.arith.Analyzer().simplify(end - start)
    ext_dtype = str(extent.dtype) if hasattr(extent, "dtype") else "int32"
    loop_var = tvm.tirx.Var(var_name, ext_dtype)
    parser.var_table.define(var_name, loop_var)
    body_stmts = parser.visit_body(body_node)
    body_stmt = _wrap_body_stmts(body_stmts)
    # Only pass step if it's not trivially 1
    step_val = step
    if isinstance(step, tvm.tirx.IntImm) and step.value == 1:
        step_val = None
    return tvm.tirx.For(loop_var, start, extent, int(tvm.tirx.ForKind.SERIAL), body_stmt, step=step_val)


def _tir_make_assign(parser, node, rhs_val):
    """TIR assign callback: handles alloc_buffer, match_buffer, Bind, and plain bindings."""
    name = node.lhs.name
    if isinstance(rhs_val, _VarDeclMarker):
        # If already pre-scanned (e.g. Relax symbolic dim var), reuse it
        try:
            existing = parser.var_table.lookup(name)
            if isinstance(existing, tvm.tirx.Var):
                return None  # already defined by pre-scan
        except Exception:
            pass
        if rhs_val.is_size_var:
            var = tvm.tirx.SizeVar(name, rhs_val.dtype)
        else:
            var = tvm.tirx.Var(name, rhs_val.dtype)
        parser.var_table.define(name, var)
        return None
    if isinstance(rhs_val, _AxisBinding):
        ext_dtype = str(rhs_val.extent.dtype) if hasattr(rhs_val.extent, "dtype") else "int32"
        var = tvm.tirx.Var(name, ext_dtype)
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
        # match_buffer: define buffer in var_table, return marker for caller to handle.
        # PrimFunc collects as buffer_map entry; SBlock collects as MatchBufferRegion.
        buf = rhs_val.buffer
        # Set buffer name to match the LHS variable name, preserve all params
        scope = buf.scope() if callable(buf.scope) else buf.scope
        buf = tvm.tirx.decl_buffer(
            buf.shape, buf.dtype, name=name,
            strides=list(buf.strides) if buf.strides else [],
            elem_offset=buf.elem_offset if buf.elem_offset is not None else None,
            scope=scope,
            data_alignment=buf.data_alignment,
            offset_factor=buf.offset_factor,
        )
        rhs_val.buffer = buf
        param_var = rhs_val.param_var
        # Add to buffer_map if at function level (PrimFunc param match_buffer)
        if hasattr(parser, '_tir_buffer_map'):
            for pv in getattr(parser, '_tir_params', []):
                if pv.same_as(param_var) or (hasattr(param_var, 'name') and pv.name == param_var.name):
                    parser._tir_buffer_map[pv] = buf
                    break
        parser.var_table.define(name, buf)
        return rhs_val  # returned so SBlock can collect as MatchBufferRegion
    if isinstance(rhs_val, tuple) and len(rhs_val) == 2:
        # alloc_buffer / decl_buffer returns (buf, AllocBuffer/DeclBuffer node)
        buf, alloc_node = rhs_val
        parser.var_table.define(name, buf)
        return alloc_node
    # Convert raw Python int/float to IntImm/FloatImm
    if isinstance(rhs_val, int):
        rhs_val = tvm.tirx.IntImm("int32", rhs_val)
    elif isinstance(rhs_val, float):
        rhs_val = tvm.tirx.FloatImm("float32", rhs_val)
    # Convert BufferRegion → BufferLoad with Ramp (vector load, e.g. A[0:4])
    if isinstance(rhs_val, tvm.tirx.BufferRegion):
        buf = rhs_val.buffer
        region = list(rhs_val.region)
        indices = []
        for r in region:
            if isinstance(r.extent, tvm.tirx.IntImm) and r.extent.value == 1:
                indices.append(r.min)
            else:
                indices.append(tvm.tirx.Ramp(r.min, tvm.tirx.IntImm(r.min.dtype, 1), r.extent))
        rhs_val = tvm.tirx.BufferLoad(buf, indices)
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
        # If rhs is a Call with empty/mismatched dtype and we have annotation dtype,
        # reconstruct the Call with the correct dtype (e.g. GlobalVar call fallback)
        if (isinstance(rhs_val, tvm.tirx.Call) and isinstance(dtype, str)
                and dtype and rhs_val.dtype != dtype):
            rhs_val = tvm.tirx.Call(dtype, rhs_val.op, list(rhs_val.args))
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
            converted.append(tvm.tirx.Ramp(start, tvm.tirx.IntImm("int32", 1), lanes))
        else:
            converted.append(i)
    return tvm.tirx.BufferStore(target, value, converted)


def _tir_evaluate(value):
    """T.evaluate(value) → Evaluate(IntImm(value))"""
    if isinstance(value, int):
        return tvm.tirx.Evaluate(tvm.tirx.IntImm("int32", value))
    return tvm.tirx.Evaluate(value)


def _tir_decl_buffer(shape, dtype="float32", data=None, scope="",
                     strides=None, elem_offset=None, align=64, offset_factor=0):
    """T.decl_buffer(shape, dtype, data=..., strides=..., elem_offset=...) → (Buffer, DeclBuffer)."""
    if isinstance(shape, tuple):
        shape = list(shape)
    shape = [tvm.tirx.IntImm("int32", s) if isinstance(s, int) else s for s in shape]
    if strides is not None and isinstance(strides, (list, tuple)):
        strides = [tvm.tirx.IntImm("int32", s) if isinstance(s, int) else s for s in strides]
    buf = tvm.tirx.decl_buffer(
        shape, dtype, data=data, scope=scope,
        strides=strides or [], elem_offset=elem_offset,
    )
    return buf, tvm.tirx.DeclBuffer(buf)


class _FuncAttrMarker:
    """Marker for T.func_attr({...}) — collected by PrimFunc surface object."""
    def __init__(self, attrs):
        self.attrs = attrs


class _ModuleAttrsMarker:
    """Marker for I.module_attrs({...}) — collected by IRModule surface object."""
    def __init__(self, attrs):
        self.attrs = attrs


class _VarDeclMarker:
    """Marker for T.int32() — declares a Var with given dtype."""
    def __init__(self, dtype, is_size_var=False):
        self.dtype = dtype
        self.is_size_var = is_size_var


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
        data_alignment=align,
        offset_factor=offset_factor,
    )
    return _MatchBufferResult(param, buf)


def _tir_alloc_buffer(shape, dtype="float32", scope="", annotations=None):
    """T.alloc_buffer(shape, dtype, scope, annotations) → (Buffer, AllocBuffer node)."""
    if isinstance(shape, tuple):
        shape = list(shape)
    shape = [tvm.tirx.IntImm("int32", s) if isinstance(s, int) else s for s in shape]
    buf = tvm.tirx.decl_buffer(shape, dtype, scope=scope)
    return buf, tvm.tirx.AllocBuffer(buf, annotations or {})


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
    def __call__(self, *args, **kwargs):
        is_size_var = kwargs.get("is_size_var", False)
        if len(args) == 0:
            # T.int32() with no args — marker for "declare a Var with this dtype"
            return _VarDeclMarker(self._dtype, is_size_var=is_size_var)
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


class _TIRModuleMeta(type):
    """Metaclass for _TIRModule to support dynamic vector dtype lookup.

    Handles T.int32x4, T.float16x8, T.uint32x4, etc. by matching the
    pattern <base_dtype>x<lanes> and returning a _DtypeHelper.
    """
    _vector_dtype_re = None

    def __getattr__(cls, name):
        import re
        if cls._vector_dtype_re is None:
            cls._vector_dtype_re = re.compile(
                r'^(int|uint|float|bfloat|bool)(8|16|32|64)x(\d+)$'
            )
        m = cls._vector_dtype_re.match(name)
        if m:
            return _DtypeHelper(name)
        raise AttributeError(f"type object '_TIRModule' has no attribute '{name}'")


class _TIRModule(metaclass=_TIRModuleMeta):
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
    uint64 = _DtypeHelper("uint64")
    bool = _DtypeHelper("bool")
    handle = _DtypeHelper("handle")  # T.handle as annotation; T.handle(...) returns Var
    # Exotic dtypes
    float8_e3m4 = _DtypeHelper("float8_e3m4")
    float8_e4m3 = _DtypeHelper("float8_e4m3")
    float8_e4m3b11fnuz = _DtypeHelper("float8_e4m3b11fnuz")
    float8_e4m3fn = _DtypeHelper("float8_e4m3fn")
    float8_e4m3fnuz = _DtypeHelper("float8_e4m3fnuz")
    float8_e5m2 = _DtypeHelper("float8_e5m2")
    float8_e5m2fnuz = _DtypeHelper("float8_e5m2fnuz")
    float8_e8m0fnu = _DtypeHelper("float8_e8m0fnu")
    float6_e2m3fn = _DtypeHelper("float6_e2m3fn")
    float6_e3m2fn = _DtypeHelper("float6_e3m2fn")
    float4_e2m1fn = _DtypeHelper("float4_e2m1fn")
    bfloat16 = _DtypeHelper("bfloat16")
    Buffer = staticmethod(_tir_buffer)

    sblock = _SBlockSurface()
    init = _InitSurface()
    launch_thread = _LaunchThreadSurface()
    axis = _AxisModule

    @staticmethod
    def sblock_alloc_buffer(shape, dtype="float32", scope="", strides=None,
                            offset_factor=0, elem_offset=None, **kwargs):
        """T.sblock_alloc_buffer(shape, ...) → marker for root sblock."""
        if isinstance(shape, tuple):
            shape = list(shape)
        shape = [tvm.tirx.IntImm("int32", s) if isinstance(s, int) else s for s in shape]
        if strides is not None and isinstance(strides, (list, tuple)):
            strides = [tvm.tirx.IntImm("int32", s) if isinstance(s, int) else s for s in strides]
        buf = tvm.tirx.decl_buffer(
            shape, dtype, scope=scope,
            strides=strides or [],
            elem_offset=elem_offset if elem_offset is not None else None,
            offset_factor=offset_factor,
        )
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
        if isinstance(val, int): val = tvm.tirx.IntImm("int32", val)
        elif isinstance(val, float): val = tvm.tirx.FloatImm("float32", val)
        return tvm.tirx.Call(val.dtype, tvm.tirx.op.Op.get("tirx.exp"), [val])

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
        # T.where(cond) with one arg — SBlock predicate marker
        return _WhereMarker(cond)

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
    def call_packed(func_name, *args, dtype="int32"):
        call_args = [func_name if isinstance(func_name, tvm.tirx.PrimExpr) else tvm.tirx.StringImm(func_name), *args]
        return tvm.tirx.Call(dtype, tvm.ir.Op.get("tirx.tvm_call_packed"), call_args)

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
        return tvm.tirx.log2(val)

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

    # --- Additional math functions ---
    @staticmethod
    def exp2(val):
        return tvm.tirx.exp2(val)

    @staticmethod
    def exp10(val):
        if isinstance(val, int): val = tvm.tirx.IntImm("int32", val)
        elif isinstance(val, float): val = tvm.tirx.FloatImm("float32", val)
        return tvm.tirx.Call(val.dtype, tvm.tirx.op.Op.get("tirx.exp10"), [val])

    @staticmethod
    def erf(val):
        return tvm.tirx.erf(val)

    @staticmethod
    def sin(val):
        return tvm.tirx.sin(val)

    @staticmethod
    def cos(val):
        return tvm.tirx.cos(val)

    @staticmethod
    def tan(val):
        return tvm.tirx.tan(val)

    @staticmethod
    def asin(val):
        return tvm.tirx.asin(val)

    @staticmethod
    def acos(val):
        return tvm.tirx.acos(val)

    @staticmethod
    def atan(val):
        return tvm.tirx.atan(val)

    @staticmethod
    def sinh(val):
        return tvm.tirx.sinh(val)

    @staticmethod
    def cosh(val):
        return tvm.tirx.cosh(val)

    @staticmethod
    def asinh(val):
        return tvm.tirx.asinh(val)

    @staticmethod
    def acosh(val):
        return tvm.tirx.acosh(val)

    @staticmethod
    def atanh(val):
        return tvm.tirx.atanh(val)

    @staticmethod
    def atan2(x1, x2):
        return tvm.tirx.atan2(x1, x2)

    @staticmethod
    def log1p(val):
        return tvm.tirx.log1p(val)

    @staticmethod
    def fabs(val):
        return tvm.tirx.abs(val)

    @staticmethod
    def ceil(val):
        return tvm.tirx.ceil(val)

    @staticmethod
    def floor(val):
        return tvm.tirx.floor(val)

    @staticmethod
    def pow(base, exp):
        return tvm.tirx.power(base, exp)

    @staticmethod
    def rsqrt(val):
        return tvm.tirx.rsqrt(val)

    @staticmethod
    def nextafter(x1, x2):
        return tvm.tirx.nextafter(x1, x2)

    @staticmethod
    def hypot(x1, x2):
        return tvm.tirx.hypot(x1, x2)

    @staticmethod
    def copysign(x1, x2):
        return tvm.tirx.copysign(x1, x2)

    @staticmethod
    def fmod(x, y):
        return tvm.tirx.fmod(x, y)

    @staticmethod
    def round(val):
        return tvm.tirx.round(val)

    @staticmethod
    def trunc(val):
        return tvm.tirx.trunc(val)

    # --- Control flow / misc intrinsics ---
    @staticmethod
    def continue_loop():
        return tvm.tirx.Evaluate(tvm.tirx.continue_loop())

    @staticmethod
    def break_loop():
        return tvm.tirx.Evaluate(tvm.tirx.break_loop())

    @staticmethod
    def undef():
        return tvm.tirx.undef()

    @staticmethod
    def clz(x):
        return tvm.tirx.clz(x)

    @staticmethod
    def tvm_storage_sync(scope):
        return tvm.tirx.tvm_storage_sync(scope)

    @staticmethod
    def call_cpacked(func_name, *args, dtype="int32"):
        call_args = [func_name if isinstance(func_name, tvm.tirx.PrimExpr) else tvm.tirx.StringImm(func_name), *args]
        return tvm.tirx.Call(dtype, tvm.ir.Op.get("tirx.tvm_call_cpacked"), call_args)

    @staticmethod
    def env_thread(thread_tag):
        """T.env_thread("threadIdx.x") → Var (used with T.launch_thread)."""
        var = tvm.tirx.Var(thread_tag, "int32")
        iv = tvm.tirx.IterVar(None, var, 1, thread_tag)
        return iv

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

    # GPU / hardware intrinsics — pass through to tvm.tirx.*
    tvm_load_matrix_sync = staticmethod(lambda *args: tvm.tirx.tvm_load_matrix_sync(*args))
    tvm_store_matrix_sync = staticmethod(lambda *args: tvm.tirx.tvm_store_matrix_sync(*args))
    tvm_mma_sync = staticmethod(lambda *args: tvm.tirx.tvm_mma_sync(*args))
    tvm_fill_fragment = staticmethod(lambda *args: tvm.tirx.tvm_fill_fragment(*args))
    tvm_access_ptr = staticmethod(lambda *args: tvm.tirx.tvm_access_ptr(*args))
    tvm_tuple = staticmethod(lambda *args: tvm.tirx.tvm_tuple(*args))
    tvm_struct_get = staticmethod(lambda *args: tvm.tirx.tvm_struct_get(*args))
    tvm_struct_set = staticmethod(lambda *args: tvm.tirx.tvm_struct_set(*args))
    ptx_ldmatrix = staticmethod(lambda *args: tvm.tirx.ptx_ldmatrix(*args))
    ptx_mma = staticmethod(lambda *args: tvm.tirx.ptx_mma(*args))
    ptx_cp_async = staticmethod(lambda *args: tvm.tirx.ptx_cp_async(*args))
    ptx_commit_group = staticmethod(lambda *args: tvm.tirx.ptx_commit_group(*args))
    ptx_wait_group = staticmethod(lambda *args: tvm.tirx.ptx_wait_group(*args))
    mma_store = staticmethod(lambda *args: tvm.tirx.mma_store(*args))
    mma_fill = staticmethod(lambda *args: tvm.tirx.mma_fill(*args))
    call_llvm_pure_intrin = staticmethod(lambda *args: tvm.tirx.call_llvm_pure_intrin(*args))
    call_llvm_intrin = staticmethod(lambda *args: tvm.tirx.call_llvm_intrin(*args))
    vectorcombine = staticmethod(lambda *args: tvm.tirx.vectorcombine(*args))
    vectorhigh = staticmethod(lambda *args: tvm.tirx.vectorhigh(*args))
    vectorlow = staticmethod(lambda *args: tvm.tirx.vectorlow(*args))
    uint32 = _DtypeHelper("uint32")
    tvm_static_handle = staticmethod(lambda: tvm.tirx.call_intrin("handle", "tirx.tvm_static_handle"))
    type_annotation = staticmethod(lambda *args: tvm.tirx.type_annotation(*args))
    tvm_stack_make_array = staticmethod(lambda *args: tvm.tirx.tvm_stack_make_array(*args))
    tvm_stack_make_shape = staticmethod(lambda *args: tvm.tirx.tvm_stack_make_shape(*args))
    tvm_stack_alloca = staticmethod(lambda *args: tvm.tirx.tvm_stack_alloca(*args))
    simdgroup_load = staticmethod(lambda *args: tvm.tirx.simdgroup_load(*args))
    simdgroup_store = staticmethod(lambda *args: tvm.tirx.simdgroup_store(*args))
    simdgroup_multiply_accumulate = staticmethod(lambda *args: tvm.tirx.simdgroup_multiply_accumulate(*args))
    get_active_lane_mask = staticmethod(lambda *args: tvm.tirx.get_active_lane_mask(*args))

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


class _RelaxMatchCastMarker:
    """Marker for R.match_cast(value, struct_info) in Relax function body."""
    def __init__(self, value, struct_info):
        self.value = value
        self.struct_info = struct_info


class _RelaxDataflowMarker:
    """Marker for R.dataflow() — triggers dataflow block in parse_function."""
    pass


class _RelaxOutputMarker:
    """Marker for R.output(var1, var2, ...) — triggers emit_output in dataflow."""
    def __init__(self, vars):
        self.vars = vars


class _RelaxFuncSurface(SurfaceObject):
    """Surface object for @R.function decorator."""

    def __init__(self):
        self._private = False
        self._pure = True

    def __call__(self, *args, private=False, pure=True, **kwargs):
        # @R.function(private=True) → return configured self
        surf = _RelaxFuncSurface()
        surf._private = private
        surf._pure = pure
        return surf

    def _process_relax_stmts(self, parser, stmts, bb, func_attrs,
                             is_dataflow=False, output_names=None):
        """Process a list of Relax body statements, emitting to BlockBuilder."""
        from tvm import relax
        from tvm_ffi import pyast

        if output_names is None:
            output_names = set()

        for stmt in stmts:
            if isinstance(stmt, pyast.Assign) and stmt.rhs is not None:
                rhs_val = parser.eval_expr(stmt.rhs)
                name = stmt.lhs.name
                if isinstance(rhs_val, _FuncAttrMarker):
                    func_attrs.update(rhs_val.attrs)
                    continue
                if isinstance(rhs_val, _VarDeclMarker):
                    # Symbolic dim var (e.g. N = T.int64()) — already pre-scanned
                    continue
                if isinstance(rhs_val, _RelaxMatchCastMarker):
                    var = bb.match_cast(rhs_val.value, rhs_val.struct_info, name)
                    parser.var_table.define(name, var)
                    continue
                # In dataflow: output names → emit_output, others → emit
                if is_dataflow and name in output_names:
                    var = bb.emit_output(rhs_val, name)
                else:
                    var = bb.emit(rhs_val, name)
                parser.var_table.define(name, var)
            elif isinstance(stmt, pyast.Return):
                if stmt.value is not None:
                    ret_val = parser.eval_expr(stmt.value)
                    bb.emit_func_output(ret_val)
            elif isinstance(stmt, pyast.With):
                # Handle `with R.dataflow(): ...`
                ctx_val = parser.eval_expr(stmt.rhs)
                if isinstance(ctx_val, _RelaxDataflowMarker):
                    # Two-pass: find output names, then emit
                    output_names = set()
                    for s in stmt.body:
                        if isinstance(s, pyast.ExprStmt):
                            call = s.expr
                            if isinstance(call, pyast.Call) and isinstance(call.callee, pyast.Attr):
                                if call.callee.name == "output":
                                    for arg in call.args:
                                        if isinstance(arg, pyast.Id):
                                            output_names.add(arg.name)
                    with bb.dataflow():
                        self._process_relax_stmts(
                            parser, stmt.body, bb, func_attrs,
                            is_dataflow=True, output_names=output_names,
                        )
                else:
                    pass  # skip unknown with blocks
            elif isinstance(stmt, pyast.ExprStmt):
                val = parser.eval_expr(stmt.expr)
                if isinstance(val, _FuncAttrMarker):
                    func_attrs.update(val.attrs)
                elif isinstance(val, _RelaxOutputMarker):
                    # R.output outside dataflow — just skip
                    pass
                elif val is not None:
                    bb.emit(val)
            # Skip other statement types

    def parse_function(self, parser, node):
        from tvm import relax
        from tvm_ffi import pyast

        bb = relax.BlockBuilder()

        # Pre-scan: forward-declare symbolic dimension variables (e.g. example 3273).
        #
        # Relax functions use symbolic dims like R.Shape(["N"]) in parameter
        # annotations, but the variable `N` is defined later in the body as
        # `N = T.int64()`. To ensure the annotation and body refer to the
        # *same* Var object (required for structural equality), we pre-scan
        # the body for `name = T.int64()` / `T.int64(is_size_var=True)`
        # declarations and create them before parsing params. This mirrors
        # the V1 parser's approach. The parser var_table reference is
        # temporarily stored on _RelaxModule so that R.Shape/R.shape can
        # convert string dims to the pre-created vars.
        for stmt in node.body:
            if isinstance(stmt, pyast.Assign) and stmt.rhs is not None:
                if isinstance(stmt.rhs, pyast.Call):
                    callee = stmt.rhs.callee
                    # Match `T.int64()`, `T.int32()`, etc. — dtype() with no positional args.
                    # Only match T.* calls (not R.* like R.print) to avoid treating
                    # function names as dtypes (see example 3205).
                    if (isinstance(callee, pyast.Attr) and len(stmt.rhs.args) == 0
                            and isinstance(callee.obj, pyast.Id)
                            and callee.obj.name == "T"):
                        name = stmt.lhs.name
                        try:
                            parser.var_table.lookup(name)
                        except Exception:
                            # Not yet defined — create it
                            dtype = callee.name  # e.g. "int64"
                            is_size_var = False
                            for k, v in zip(stmt.rhs.kwargs_keys, stmt.rhs.kwargs_values):
                                if str(k) == "is_size_var":
                                    is_size_var = True
                            if is_size_var:
                                var = tvm.tirx.SizeVar(name, dtype)
                            else:
                                var = tvm.tirx.Var(name, dtype)
                            parser.var_table.define(name, var)

        # Store parser on module so R.Shape/R.shape can resolve string dims
        _RelaxModule._parser = parser

        params = []
        for arg in node.args:
            name = arg.lhs.name
            ann = None
            if arg.annotation is not None:
                ann = parser.eval_expr(arg.annotation)
            # Convert bare _RelaxTensorSInfo → TensorStructInfo (e.g. R.Tensor without parens)
            if isinstance(ann, _RelaxTensorSInfo):
                ann = ann()
            if isinstance(ann, relax.StructInfo):
                var = relax.Var(name, ann)
            else:
                var = relax.Var(name)
            parser.var_table.define(name, var)
            params.append(var)

        # Parse return type annotation if present
        ret_sinfo = None
        if node.return_type is not None:
            ret_sinfo = parser.eval_expr(node.return_type)

        # Clear parser reference BEFORE entering BlockBuilder scope.
        # _resolve_shape_dims is only needed during param/return-type parsing;
        # leaving _parser set during BB emission causes infinite hangs because
        # BB normalization can trigger R.Tensor() calls that re-enter resolution.
        _RelaxModule._parser = None

        func_name = node.name.name
        func_attrs = {}

        with bb.function(func_name, params):
            self._process_relax_stmts(parser, node.body, bb, func_attrs)

        mod = bb.get()
        func = mod[func_name]
        # Build final attrs: start from BB attrs, strip global_symbol if private
        final_attrs = dict(func.attrs) if func.attrs else {}
        if self._private and "global_symbol" in final_attrs:
            del final_attrs["global_symbol"]
        final_attrs.update(func_attrs)
        # Reconstruct function with correct attrs and is_pure
        func = relax.Function(
            func.params, func.body,
            ret_struct_info=func.ret_struct_info,
            is_pure=self._pure,
            attrs=tvm.ir.make_node("ir.DictAttrs", **final_attrs) if final_attrs else None,
        )
        # Clean up parser reference stored for R.Shape string dim resolution
        _RelaxModule._parser = None
        return func


class _RelaxTensorSInfo:
    """R.Tensor((3, 4), dtype='float32') → TensorStructInfo."""
    def __call__(self, shape=None, dtype=None, ndim=-1, **kwargs):
        from tvm import relax
        if isinstance(shape, tuple):
            shape = list(shape)
        # Resolve string dim names to tir Vars during param parsing
        # (e.g. R.Tensor(("M", "N")) — see example 3311/317).
        # _parser is only set during param/return-type parsing, cleared before BB.
        if shape is not None and getattr(_RelaxModule, "_parser", None) is not None:
            if any(isinstance(d, str) for d in shape):
                shape = _RelaxModule._resolve_shape_dims(shape)
        # When no dtype specified and no shape, dtype="" means unknown
        # When shape specified but no dtype, dtype="" means unknown
        if dtype is None:
            dtype = ""
        return relax.TensorStructInfo(shape, dtype, ndim=ndim)


class _RelaxModuleMeta(type):
    """Metaclass for _RelaxModule to support dynamic op lookup via R.op_name."""
    def __getattr__(cls, name):
        # Try to find in tvm.relax.op
        from tvm.relax import op
        if hasattr(op, name):
            return getattr(op, name)
        raise AttributeError(f"type object '_RelaxModule' has no attribute '{name}'")


class _RelaxModule(metaclass=_RelaxModuleMeta):
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
    def _resolve_shape_dims(dims):
        """Convert string dim names to tir Vars via pre-scanned var_table.

        When R.Shape(["N"]) is used in a param annotation, "N" is a string
        that must become a tvm.tirx.Var("N", "int64") — the same object that
        `N = T.int64()` later creates. See example 3273 and the pre-scan in
        _RelaxFuncSurface.parse_function for context.
        """
        resolved = []
        for d in dims:
            if isinstance(d, str):
                # Try to resolve from parser var_table (set during parse_function)
                parser = getattr(_RelaxModule, "_parser", None)
                if parser is not None:
                    try:
                        resolved.append(parser.var_table.lookup(d))
                        continue
                    except Exception:
                        pass
                # Fallback: create a fresh int64 Var
                import tvm.tirx
                resolved.append(tvm.tirx.Var(d, "int64"))
            else:
                resolved.append(d)
        return resolved

    @staticmethod
    def shape(dims):
        """R.shape([T.int64(16)]) → ShapeExpr (value, not annotation).

        Lowercase R.shape() returns a ShapeExpr for use as an argument to ops
        like R.reshape, R.ones, R.zeros. Uppercase R.Shape() returns
        ShapeStructInfo for type annotations.
        """
        from tvm import relax
        dims = _RelaxModule._resolve_shape_dims(dims if isinstance(dims, list) else list(dims))
        return relax.ShapeExpr(dims)

    @staticmethod
    def Shape(dims=None, ndim=-1):
        from tvm import relax
        if dims is not None:
            dims = _RelaxModule._resolve_shape_dims(dims if isinstance(dims, list) else list(dims))
            return relax.ShapeStructInfo(dims)
        return relax.ShapeStructInfo(ndim=ndim)

    @staticmethod
    def str(val):
        return val

    @staticmethod
    def dtype(val):
        return val

    @staticmethod
    def Prim(dtype="int64", value=None):
        from tvm import relax
        if value is not None:
            # R.Prim(value=16) — value is a PrimExpr
            if isinstance(value, int):
                import tvm.tirx
                value = tvm.tirx.IntImm("int64", value)
            return relax.PrimStructInfo(value=value)
        return relax.PrimStructInfo(dtype)

    @staticmethod
    def dataflow():
        return _RelaxDataflowMarker()

    @staticmethod
    def output(*args):
        return _RelaxOutputMarker(list(args))

    @staticmethod
    def tuple(*args):
        from tvm import relax
        return relax.Tuple(list(args))

    @staticmethod
    def Callable(ret=None, params=None, purity=True):
        from tvm import relax
        if ret is None and params is None:
            return relax.FuncStructInfo.opaque_func()
        return relax.FuncStructInfo(params or [], ret)

    @staticmethod
    def ExternFunc(name):
        from tvm import relax
        return relax.ExternFunc(name)

    @staticmethod
    def match_cast(value, struct_info):
        """R.match_cast(value, struct_info) → MatchCast marker."""
        from tvm import relax
        # If struct_info is the _RelaxTensorSInfo class (not called), call it
        if isinstance(struct_info, _RelaxTensorSInfo):
            struct_info = struct_info()
        return _RelaxMatchCastMarker(value, struct_info)

    @staticmethod
    def call_tir(func, args, out_sinfo, tir_vars=None):
        from tvm import relax
        if isinstance(args, tuple):
            args = relax.Tuple(list(args))
        return relax.op.call_tir(func, args, out_sinfo, tir_vars=tir_vars)

    @staticmethod
    def _normalize_sinfo_args(sinfo_args):
        """Convert bare _RelaxTensorSInfo / class instances to actual StructInfo.

        When sinfo_args=(R.Tensor,), R.Tensor is the _RelaxTensorSInfo instance
        (not called), not a TensorStructInfo. Convert it. See example 716.
        """
        if not sinfo_args:
            return None
        from tvm import relax
        result = []
        for s in sinfo_args:
            if isinstance(s, _RelaxTensorSInfo):
                result.append(s())  # call with defaults → TensorStructInfo
            elif isinstance(s, relax.StructInfo):
                result.append(s)
            else:
                result.append(s)
        return result

    @staticmethod
    def call_packed(func_name, *args, sinfo_args=None, **kwargs):
        from tvm import relax
        if isinstance(func_name, str):
            func_name = relax.ExternFunc(func_name)
        sinfo = _RelaxModule._normalize_sinfo_args(sinfo_args) or [relax.ObjectStructInfo()]
        return relax.Call(
            func_name,
            list(args),
            sinfo_args=sinfo,
        )

    @staticmethod
    def call_pure_packed(func, *args, sinfo_args=None, **kwargs):
        from tvm import relax
        sinfo = _RelaxModule._normalize_sinfo_args(sinfo_args) or [relax.ObjectStructInfo()]
        return relax.op.call_pure_packed(func, *args, sinfo_args=sinfo)

    @staticmethod
    def assert_op(cond, msg=None, format=None):
        from tvm import relax
        args = [cond]
        if format is not None:
            args.append(format)
        return relax.op.assert_op(cond, format or relax.StringImm(""))

    @staticmethod
    def const(val, dtype=None):
        from tvm import relax
        import numpy as np
        if isinstance(val, (bool, int, float)):
            if dtype:
                return relax.const(np.array(val, dtype=dtype))
            return relax.const(val)
        return relax.const(val, dtype)

    @staticmethod
    def device(device_type=0, index=0):
        """R.device(device_type=1, index=0) → VDevice-like for hint_on_device."""
        return tvm.ir.VDevice(None, 0, "global")  # placeholder — actual device from context

    @staticmethod
    def StringImm(val):
        from tvm import relax
        return relax.StringImm(val)

    # --- Relax math/unary ops ---
    @staticmethod
    def _make_unary_op(op_name):
        def _op(x):
            from tvm.relax import op
            return getattr(op, op_name)(x)
        return _op

    @staticmethod
    def _make_binary_op(op_name):
        def _op(x, y):
            from tvm.relax import op
            return getattr(op, op_name)(x, y)
        return _op


class _ModuleGlobalInfosMarker:
    """Marker for I.module_global_infos({...}) — collected by IRModule."""
    def __init__(self, infos):
        self.infos = infos


class _IRLangModule:
    """Language module for IR (the 'I' in `from tvm.script import ir as I`)."""

    ir_module = _IRModuleSurface()
    GlobalVar = staticmethod(tvm.ir.GlobalVar)

    @staticmethod
    def module_attrs(attrs_dict):
        return _ModuleAttrsMarker(attrs_dict)

    @staticmethod
    def module_global_infos(infos_dict):
        return _ModuleGlobalInfosMarker(infos_dict)

    @staticmethod
    def vdevice(target_dict, vdevice_id=0, memory_scope="global"):
        target = tvm.target.Target(target_dict) if isinstance(target_dict, dict) else target_dict
        return tvm.ir.VDevice(target, vdevice_id, memory_scope)


# ============================================================================
# Public API
# ============================================================================


def make_parser() -> IRParser:
    """Create a parser configured with TIR/Relax language modules."""
    return IRParser(lang_modules={"T": _TIRModule, "I": _IRLangModule, "R": _RelaxModule})


def parse(text: str):
    """Parse V2 printer output back to TVM IR."""
    return make_parser().parse(text)
