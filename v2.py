import math
import random
import json
from contextlib import contextmanager

class Tensor:
    __slots__ = ("storage", "shape", "strides", "offset",
        "requires_grad", "grad", "_prev", "_op", "_backward",)
    grad_enabled = True

    def __init__(self, storage, shape, strides=None, offset=0, requires_grad=False, _children=(), _op=""):
        self.storage = storage
        self.shape = tuple([int(d) for d in shape])
        self.strides = (
            self._default_strides(self.shape)
            if strides is None
            else tuple([int(s) for s in strides])
        )
        self.offset = int(offset)
        self.requires_grad = requires_grad
        self.grad = None
        self._prev = tuple(_children)
        self._op = _op
        self._backward = lambda: None

        if len(self.shape) != len(self.strides):
            raise ValueError("shape and strides must have the same length")
        if any(d < 0 for d in self.shape):
            raise ValueError("shape dimensions must be non-negative")
        if any(s < 0 for s in self.strides):
            raise ValueError("negative strides not supported yet")
        if self.offset < 0:
            raise ValueError("offset must be non-negative")
        if self.numel():
            max_pos = self.offset + sum(
                (dim - 1) * stride 
                for dim, stride in zip(self.shape, self.strides)
                )
            if max_pos >= len(self.storage):
                raise ValueError("storage is too small for given shape, strides, and offset")
        
    def item(self):
        if self.shape != ():
            raise ValueError("item() only works on scalar tensors")
        return self.storage[self.offset]

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + ((-1.0) * other)

    def __rsub__(self, other):
        return Tensor._coerce(other) + ((-1.0) * self)

    def __truediv__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError("Only scalar division is supported for now")
        if other == 0:
            raise ZeroDivisionError("division by zero")
        return self * (1.0 / float(other))

    @staticmethod
    def _iter_indices(shape):
        def rec(prefix, dim):
            if dim == len(shape):
                yield tuple(prefix)
                return
            for i in range(shape[dim]):
                yield from rec(prefix + (i,), dim + 1)
        yield from rec((), 0)

    @staticmethod
    def _canon_dim(dim, ndim):
        if dim < 0:
            dim += ndim
        if dim < 0 or dim >= ndim:
            raise ValueError("Dimension out of range")
        return dim
    
    @staticmethod
    def _prod(shape):
        out = 1
        for d in shape:
            out *= d
        return out
    
    @staticmethod
    def _default_strides(shape):
        out = [0] * len(shape)
        stride = 1
        for i in range(len(shape) - 1, -1, -1):
            out[i] = stride
            stride *= shape[i]
        return tuple(out)
    
    @classmethod
    def from_nested(cls, data, requires_grad=False):
        def infer_shape(x):
            if isinstance(x, (int, float)):
                return ()
            if not isinstance(x, list):
                raise ValueError("Input must be a nested list of numbers")
            if len(x) == 0:
                return (0,)
            first_shape = infer_shape(x[0])
            for item in x:
                if infer_shape(item) != first_shape:
                    raise ValueError("All sublists must have the same shape")
            return (len(x),) + first_shape
        
        flat = []
        def flatten(x):
            if isinstance(x, list):
                for item in x:
                    flatten(item)
            else:
                flat.append(float(x))
        
        shape = infer_shape(data)
        flatten(data)
        return cls(flat, shape=shape, requires_grad=requires_grad)
    
    @classmethod
    def zeros(cls, shape, requires_grad=False):
        return cls([0.0] * cls._prod(shape), shape=shape, requires_grad=requires_grad)

    def numel(self):
        return self._prod(self.shape)
    
    def is_contiguous(self):
        return self.strides == self._default_strides(self.shape)
    
    def _storage_index(self, idx):
        if len(idx) != len(self.shape):
            raise ValueError("Index must have the same number of dimensions as shape")
        
        pos = self.offset
        for ind, dim, stride in zip(idx, self.shape, self.strides):
            if ind < 0:
                ind += dim
            if ind < 0 or ind >= dim:
                raise IndexError("Index out of bounds")
            pos += ind * stride
        return pos
    
    def get(self, *idx):
        return self.storage[self._storage_index(idx)]
    
    def tolist(self):
        def build(prefix, dim):
            if dim == len(self.shape):
                return self.get(*prefix)
            return [build(prefix + (i,), dim + 1) for i in range(self.shape[dim])]
        return build((), 0)
    
    # def contiguous(self):
    #     flat = []

    #     def collect(prefix, dim):
    #         if dim == len(self.shape):
    #             flat.append(self.get(*prefix))
    #             return
    #         for i in range(self.shape[dim]):
    #             collect(prefix + (i,), dim + 1)

    #     collect((), 0)
    #     return Tensor(flat, shape=self.shape, requires_grad=self.requires_grad)
    
    # def reshape(self, *shape):
    #     if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
    #         shape = tuple(shape[0])
    #     else:
    #         shape = tuple(shape)

    #     neg1 = [i for i, d in enumerate(shape) if d == -1]
    #     if len(neg1) > 1:
    #         raise ValueError("Only one dimension can be -1")
    #     if neg1:
    #         known = 1
    #         for d in shape:
    #             if d != -1:
    #                 if d <= 0:
    #                     raise ValueError("Dimensions must be positive or -1")
    #                 known *= d
    #         total = self.numel()
    #         if total % known != 0:
    #             raise ValueError("Total size must be divisible by known dimensions")
    #         shape = list(shape)
    #         shape[neg1[0]] = total // known
    #         shape = tuple(shape)

    #     if self._prod(shape) != self.numel():
    #         raise ValueError("Total size of new shape must be unchanged")
    #     if not self.is_contiguous():
    #         raise ValueError("Can only reshape contiguous tensors")
    #     return Tensor(
    #         self.storage,
    #         shape=shape,
    #         strides=self._default_strides(shape),
    #         offset=self.offset,
    #         requires_grad=self.requires_grad,
    #         _children=(self,),
    #         _op="reshape",
    #     )
    
    # def permute(self, *dims):
    #     if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
    #         dims = tuple(dims[0])
    #     else:
    #         dims = tuple(dims)

    #     ndim = len(self.shape)
    #     dims = tuple(self._canon_dim(d, ndim) for d in dims)

    #     if sorted(dims) != list(range(len(self.shape))):
    #         raise ValueError("Invalid permutation of dimensions")
        
    #     return Tensor(
    #         self.storage,
    #         shape=tuple(self.shape[d] for d in dims),
    #         strides=tuple(self.strides[d] for d in dims),
    #         offset=self.offset,
    #         requires_grad=self.requires_grad,
    #         _children=(self,),
    #         _op="permute",
    #     )
    
    def transpose(self, dim0, dim1):
        ndim = len(self.shape)
        dim0 = self._canon_dim(dim0, ndim)
        dim1 = self._canon_dim(dim1, ndim)
        dims = list(range(ndim))
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        return self.permute(dims)
    
    # def narrow(self, dim, start, length):
    #     dim = self._canon_dim(dim, len(self.shape))
    #     if start < 0 or length < 0 or start + length > self.shape[dim]:
    #         raise ValueError("Invalid start or length for narrow")
        
    #     new_shape = list(self.shape)
    #     new_shape[dim] = length
    #     new_offset = self.offset + start * self.strides[dim]
    #     return Tensor(
    #         self.storage,
    #         shape=tuple(new_shape),
    #         strides=self.strides,
    #         offset=new_offset,
    #         requires_grad=self.requires_grad,
    #         _children=(self,),
    #         _op="narrow",
    #     )
    
    def select(self, dim, index):
        dim = self._canon_dim(dim, len(self.shape))
        if index < 0:
            index += self.shape[dim]
        if index < 0 or index >= self.shape[dim]:
            raise ValueError("Index out of range for select")

        return Tensor(
            self.storage,
            shape=self.shape[:dim] + self.shape[dim + 1:],
            strides=self.strides[:dim] + self.strides[dim + 1:],
            offset=self.offset + index * self.strides[dim],
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="select",
        )
    
    def __repr__(self):
        return f"Tensor(shape={self.shape}, strides={self.strides}, offset={self.offset}, requires_grad={self.requires_grad})"
    
    @classmethod
    def full(cls, shape, fill_value, requires_grad=False):
        return cls([float(fill_value)] * cls._prod(shape), shape=shape, requires_grad=requires_grad)
    
    @staticmethod
    def _coerce(other):
        if isinstance(other, Tensor):
            return other
        if isinstance(other, (int, float)):
            return Tensor([float(other)], shape=(), requires_grad=False)
        raise TypeError(f"Cannot coerce {type(other).__name__} to Tensor")
    
    @staticmethod
    def _broadcast_shape(a_shape, b_shape):
        out = []
        na, nb = len(a_shape), len(b_shape)

        for i in range(1, max(na, nb) + 1):
            da = a_shape[-i] if i <= na else 1
            db = b_shape[-i] if i <= nb else 1

            if da == db:
                out_dim = da
            elif da == 1:
                out_dim = db
            elif db == 1:
                out_dim = da
            else:
                raise ValueError(f"Cannot broadcast shapes {a_shape} and {b_shape}")

            out.append(out_dim)

        return tuple(reversed(out))

    @staticmethod
    def _project_index_to_shape(out_idx, in_shape):
        pad = len(out_idx) - len(in_shape)
        idx = []
        for i, dim in enumerate(in_shape):
            oi = out_idx[pad + i]
            idx.append(0 if dim == 1 else oi)
        return tuple(idx)
    
    def _expand_to(self, shape):
        shape = tuple(shape)
        return self if self.shape == shape else self.expand(shape)

    # def __add__(self, other):
    #     other = self._coerce(other)
    #     out_shape = self._broadcast_shape(self.shape, other.shape)

    #     a = self._expand_to(out_shape)
    #     b = other._expand_to(out_shape)

    #     out = Tensor.zeros(
    #         out_shape,
    #         requires_grad=self.requires_grad or other.requires_grad,
    #     )

    #     for idx in self._iter_indices(out_shape):
    #         out.storage[out._storage_index(idx)] = a.get(*idx) + b.get(*idx)

    #     return out

    # __radd__ = __add__

    # def __mul__(self, other):
    #     other = self._coerce(other)
    #     out_shape = self._broadcast_shape(self.shape, other.shape)

    #     a = self._expand_to(out_shape)
    #     b = other._expand_to(out_shape)

    #     out = Tensor.zeros(
    #         out_shape,
    #         requires_grad=self.requires_grad or other.requires_grad,
    #     )

    #     for idx in self._iter_indices(out_shape):
    #         out.storage[out._storage_index(idx)] = a.get(*idx) * b.get(*idx)

    #     return out

    # __rmul__ = __mul__

    @staticmethod
    def _canon_axes(axis, ndim):
        if axis is None:
            return tuple(range(ndim))
        if isinstance(axis, int):
            axis = (axis,)
        axes = tuple(Tensor._canon_dim(a, ndim) for a in axis)
        if len(set(axes)) != len(axes):
            raise ValueError("duplicate axes are not allowed")
        return tuple(sorted(axes))
    
    # def sum(self, axis=None, keepdims=False):
    #     ndim = len(self.shape)
    #     axes = self._canon_axes(axis, ndim)

    #     if keepdims:
    #         out_shape = tuple(1 if i in axes else self.shape[i] for i in range(ndim))
    #     else:
    #         out_shape = tuple(self.shape[i] for i in range(ndim) if i not in axes)

    #     out = Tensor.zeros(out_shape, requires_grad=self.requires_grad)

    #     for idx in self._iter_indices(self.shape):
    #         if keepdims:
    #             out_idx = tuple(0 if i in axes else idx[i] for i in range(ndim))
    #         else:
    #             out_idx = tuple(idx[i] for i in range(ndim) if i not in axes)

    #         out.storage[out._storage_index(out_idx)] += self.get(*idx)

    #     return out
    
    def mean(self, axis=None, keepdims=False):
        ndim = len(self.shape)
        axes = self._canon_axes(axis, ndim)

        denom = 1
        for a in axes:
            denom *= self.shape[a]

        return self.sum(axis=axes, keepdims=keepdims) * (1.0 / denom)
    
    # def expand(self, *shape):
    #     if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
    #         shape = tuple(shape[0])
    #     else:
    #         shape = tuple(shape)

    #     if len(shape) < len(self.shape):
    #         raise ValueError("cannot expand to fewer dimensions")

    #     # Right-align the old shape with the new one.
    #     pad = len(shape) - len(self.shape)
    #     old_shape = (1,) * pad + self.shape
    #     old_strides = (0,) * pad + self.strides

    #     new_strides = []
    #     for old_dim, old_stride, new_dim in zip(old_shape, old_strides, shape):
    #         if new_dim < 0:
    #             raise ValueError("expand dims must be non-negative")

    #         if old_dim == new_dim:
    #             # same size: keep the old stride
    #             new_strides.append(old_stride)
    #         elif old_dim == 1:
    #             # expanded singleton dimension: stride 0
    #             new_strides.append(0)
    #         else:
    #             raise ValueError(f"cannot expand shape {self.shape} to {shape}")

    #     return Tensor(
    #         self.storage,
    #         shape=shape,
    #         strides=tuple(new_strides),
    #         offset=self.offset,
    #         requires_grad=self.requires_grad,
    #         _children=(self,),
    #         _op="expand",
    #     )
    
    # def __matmul__(self, other):
    #     if not isinstance(other, Tensor):
    #         raise TypeError("@ only supports Tensor @ Tensor")
    #     if len(self.shape) < 2 or len(other.shape) < 2:
    #         raise ValueError("matmul currently supports only tensors with rank >= 2")

    #     M, K = self.shape[-2:]
    #     K2, N = other.shape[-2:]
    #     if K != K2:
    #         raise ValueError(f"Shapes {self.shape} and {other.shape} not compatible for matmul")

    #     batch_shape = self._broadcast_shape(self.shape[:-2], other.shape[:-2])

    #     a = self._expand_to(batch_shape + (M, K))
    #     b = other._expand_to(batch_shape + (K, N))

    #     out_shape = batch_shape + (M, N)
    #     out = Tensor.zeros(
    #         out_shape,
    #         requires_grad=self.requires_grad or other.requires_grad,
    #     )

    #     for batch_idx in self._iter_indices(batch_shape):
    #         for i in range(M):
    #             for j in range(N):
    #                 acc = 0.0
    #                 for k in range(K):
    #                     acc += a.get(*(batch_idx + (i, k))) * b.get(*(batch_idx + (k, j)))
    #                 out.storage[out._storage_index(batch_idx + (i, j))] = acc

    #     return out
    
    # def softmax(self, axis=-1):
    #     ndim = len(self.shape)
    #     axis = self._canon_dim(axis, ndim)

    #     out = Tensor.zeros(self.shape, requires_grad=self.requires_grad)

    #     outer_shape = self.shape[:axis] + self.shape[axis + 1:]
    #     n = self.shape[axis]

    #     for outer_idx in self._iter_indices(outer_shape):
    #         def full_idx(j):
    #             return outer_idx[:axis] + (j,) + outer_idx[axis:]

    #         m = max(self.get(*full_idx(j)) for j in range(n))
    #         exps = [math.exp(self.get(*full_idx(j)) - m) for j in range(n)]
    #         s = sum(exps)

    #         for j, e in enumerate(exps):
    #             out.storage[out._storage_index(full_idx(j))] = e / s

    #     return out
    
    # def apply_causal_mask(self, mask_value=float("-inf")):
    #     if len(self.shape) < 2:
    #         raise ValueError("causal mask requires rank >= 2")

    #     T1, T2 = self.shape[-2:]
    #     if T1 != T2:
    #         raise ValueError("last two dims must form a square matrix")

    #     out = Tensor.zeros(self.shape, requires_grad=self.requires_grad)

    #     for prefix in self._iter_indices(self.shape[:-2]):
    #         for i in range(T1):
    #             for j in range(T2):
    #                 idx = prefix + (i, j)
    #                 out.storage[out._storage_index(idx)] = (
    #                     self.get(*idx) if j <= i else mask_value
    #                 )

    #     return out
    
    # def relu(self):
    #     out = Tensor.zeros(self.shape, requires_grad=self.requires_grad)
    #     for idx in self._iter_indices(self.shape):
    #         x = self.get(*idx)
    #         out.storage[out._storage_index(idx)] = x if x > 0.0 else 0.0
    #     return out
    
    # def layernorm_last_dim(self, eps=1e-5):
    #     if len(self.shape) == 0:
    #         raise ValueError("layernorm requires rank >= 1")

    #     N = self.shape[-1]
    #     out = Tensor.zeros(self.shape, requires_grad=self.requires_grad)

    #     for outer_idx in self._iter_indices(self.shape[:-1]):
    #         vals = [self.get(*(outer_idx + (j,))) for j in range(N)]
    #         mean = sum(vals) / N
    #         var = sum((x - mean) ** 2 for x in vals) / N
    #         invstd = 1.0 / math.sqrt(var + eps)

    #         for j, x in enumerate(vals):
    #             out.storage[out._storage_index(outer_idx + (j,))] = (x - mean) * invstd

    #     return out
    
    @classmethod
    @contextmanager
    def no_grad(cls):
        prev = cls.grad_enabled
        cls.grad_enabled = False
        try:
            yield
        finally:
            cls.grad_enabled = prev

    @classmethod
    def _should_track(cls, parents):
        return cls.grad_enabled and any(p.requires_grad for p in parents)
    
    @classmethod
    def _make_out(cls, storage, shape, parents, op, strides=None, offset=0):
        track = cls._should_track(parents)
        out = cls(
            storage,
            shape=shape,
            strides=strides,
            offset=offset,
            requires_grad=track,
            _children=parents if track else (),
            _op=op,
        )
        return out, track
    
    def _ensure_grad(self):
        if self.grad is None:
            self.grad = Tensor.zeros(self.shape, requires_grad=False)
        return self.grad
    
    def zero_grad(self):
        if self.grad is not None:
            self.grad = Tensor.zeros(self.shape, requires_grad=False)

    def backward(self):
        if self.shape != ():
            raise ValueError("backward() currently only supports scalar outputs")
        if not self.requires_grad:
            raise ValueError("cannot call backward() on tensor that does not require grad")

        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for p in v._prev:
                    build(p)
                topo.append(v)

        build(self)

        self.grad = Tensor([1.0], shape=(), requires_grad=False)

        for v in reversed(topo):
            v._backward()

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        else:
            shape = tuple(shape)

        if len(shape) < len(self.shape):
            raise ValueError("cannot expand to fewer dimensions")

        pad = len(shape) - len(self.shape)
        old_shape = (1,) * pad + self.shape
        old_strides = (0,) * pad + self.strides

        new_strides = []
        for old_dim, old_stride, new_dim in zip(old_shape, old_strides, shape):
            if new_dim < 0:
                raise ValueError("expand dims must be non-negative")
            if old_dim == new_dim:
                new_strides.append(old_stride)
            elif old_dim == 1:
                new_strides.append(0)
            else:
                raise ValueError(f"cannot expand shape {self.shape} to {shape}")

        out, track = Tensor._make_out(
            self.storage,
            shape=shape,
            parents=(self,),
            op="expand",
            strides=tuple(new_strides),
            offset=self.offset,
        )
        if not track:
            return out

        def _backward():
            sg = self._ensure_grad()
            og = out.grad
            for out_idx in self._iter_indices(out.shape):
                src_idx = self._project_index_to_shape(out_idx, self.shape)
                sg.storage[sg._storage_index(src_idx)] += og.get(*out_idx)

        out._backward = _backward
        return out
    
    def sum(self, axis=None, keepdims=False):
        ndim = len(self.shape)
        axes = self._canon_axes(axis, ndim)

        if keepdims:
            out_shape = tuple(1 if i in axes else self.shape[i] for i in range(ndim))
        else:
            out_shape = tuple(self.shape[i] for i in range(ndim) if i not in axes)

        out_storage = [0.0] * Tensor._prod(out_shape)
        out, track = Tensor._make_out(out_storage, out_shape, (self,), "sum")
        
        for idx in self._iter_indices(self.shape):
            if keepdims:
                out_idx = tuple(0 if i in axes else idx[i] for i in range(ndim))
            else:
                out_idx = tuple(idx[i] for i in range(ndim) if i not in axes)
            out.storage[out._storage_index(out_idx)] += self.get(*idx)

        if not track:
            return out

        def _backward():
            sg = self._ensure_grad()
            og = out.grad
            for idx in self._iter_indices(self.shape):
                if keepdims:
                    out_idx = tuple(0 if i in axes else idx[i] for i in range(ndim))
                else:
                    out_idx = tuple(idx[i] for i in range(ndim) if i not in axes)
                sg.storage[sg._storage_index(idx)] += og.get(*out_idx)

        out._backward = _backward
        return out
    
    def __add__(self, other):
        other = self._coerce(other)
        out_shape = self._broadcast_shape(self.shape, other.shape)

        a = self._expand_to(out_shape)
        b = other._expand_to(out_shape)

        out_storage = [0.0] * Tensor._prod(out_shape)
        out, track = Tensor._make_out(out_storage, out_shape, (a, b), "add")

        for idx in self._iter_indices(out_shape):
            out.storage[out._storage_index(idx)] = a.get(*idx) + b.get(*idx)

        if not track:
            return out

        def _backward():
            og = out.grad
            if a.requires_grad:
                ag = a._ensure_grad()
                for idx in self._iter_indices(out_shape):
                    ag.storage[ag._storage_index(idx)] += og.get(*idx)
            if b.requires_grad:
                bg = b._ensure_grad()
                for idx in self._iter_indices(out_shape):
                    bg.storage[bg._storage_index(idx)] += og.get(*idx)

        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = self._coerce(other)
        out_shape = self._broadcast_shape(self.shape, other.shape)

        a = self._expand_to(out_shape)
        b = other._expand_to(out_shape)

        out_storage = [0.0] * Tensor._prod(out_shape)
        out, track = Tensor._make_out(out_storage, out_shape, (a, b), "mul")

        for idx in self._iter_indices(out_shape):
            out.storage[out._storage_index(idx)] = a.get(*idx) * b.get(*idx)

        if not track:
            return out

        def _backward():
            og = out.grad
            if a.requires_grad:
                ag = a._ensure_grad()
                for idx in self._iter_indices(out_shape):
                    ag.storage[ag._storage_index(idx)] += b.get(*idx) * og.get(*idx)
            if b.requires_grad:
                bg = b._ensure_grad()
                for idx in self._iter_indices(out_shape):
                    bg.storage[bg._storage_index(idx)] += a.get(*idx) * og.get(*idx)

        out._backward = _backward
        return out
    
    __radd__ = __add__
    __rmul__ = __mul__

    def relu(self):
        out_storage = [0.0] * Tensor._prod(self.shape)
        out, track = Tensor._make_out(out_storage, self.shape, (self,), "relu")

        for idx in self._iter_indices(self.shape):
            x = self.get(*idx)
            out.storage[out._storage_index(idx)] = x if x > 0.0 else 0.0

        if not track:
            return out

        def _backward():
            sg = self._ensure_grad()
            og = out.grad
            for idx in self._iter_indices(self.shape):
                if self.get(*idx) > 0.0:
                    sg.storage[sg._storage_index(idx)] += og.get(*idx)

        out._backward = _backward
        return out
    
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        else:
            shape = tuple(shape)

        neg1 = [i for i, d in enumerate(shape) if d == -1]
        if len(neg1) > 1:
            raise ValueError("Only one dimension can be -1")

        if neg1:
            known = 1
            for d in shape:
                if d != -1:
                    if d <= 0:
                        raise ValueError("Dimensions must be positive or -1")
                    known *= d
            total = self.numel()
            if total % known != 0:
                raise ValueError("Total size must be divisible by known dimensions")
            shape = list(shape)
            shape[neg1[0]] = total // known
            shape = tuple(shape)

        if self._prod(shape) != self.numel():
            raise ValueError("Total size of new shape must be unchanged")
        if not self.is_contiguous():
            raise ValueError("Can only reshape contiguous tensors")

        out, track = Tensor._make_out(
            self.storage,
            shape=shape,
            parents=(self,),
            op="reshape",
            strides=self._default_strides(shape),
            offset=self.offset,
        )
        if not track:
            return out

        def _backward():
            sg = self._ensure_grad()
            og = out.grad

            # same flat order, because reshape preserves contiguous layout
            for i in range(self.numel()):
                sg.storage[i] += og.storage[i]

        out._backward = _backward
        return out
    
    def permute(self, *dims):
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        else:
            dims = tuple(dims)

        ndim = len(self.shape)
        dims = tuple(self._canon_dim(d, ndim) for d in dims)

        if sorted(dims) != list(range(ndim)):
            raise ValueError("Invalid permutation of dimensions")

        out, track = Tensor._make_out(
            self.storage,
            shape=tuple(self.shape[d] for d in dims),
            parents=(self,),
            op="permute",
            strides=tuple(self.strides[d] for d in dims),
            offset=self.offset,
        )
        if not track:
            return out

        def _backward():
            sg = self._ensure_grad()
            og = out.grad

            for out_idx in self._iter_indices(out.shape):
                self_idx = [0] * ndim
                for out_axis, in_axis in enumerate(dims):
                    self_idx[in_axis] = out_idx[out_axis]
                self_idx = tuple(self_idx)

                sg.storage[sg._storage_index(self_idx)] += og.get(*out_idx)

        out._backward = _backward
        return out
    
    def narrow(self, dim, start, length):
        dim = self._canon_dim(dim, len(self.shape))
        if start < 0 or length < 0 or start + length > self.shape[dim]:
            raise ValueError("Invalid start or length for narrow")

        new_shape = list(self.shape)
        new_shape[dim] = length
        new_offset = self.offset + start * self.strides[dim]

        out, track = Tensor._make_out(
            self.storage,
            shape=tuple(new_shape),
            parents=(self,),
            op="narrow",
            strides=self.strides,
            offset=new_offset,
        )
        if not track:
            return out

        def _backward():
            sg = self._ensure_grad()
            og = out.grad

            for out_idx in self._iter_indices(out.shape):
                self_idx = list(out_idx)
                self_idx[dim] += start
                self_idx = tuple(self_idx)

                sg.storage[sg._storage_index(self_idx)] += og.get(*out_idx)

        out._backward = _backward
        return out
    
    def select(self, dim, index):
        dim = self._canon_dim(dim, len(self.shape))
        if index < 0:
            index += self.shape[dim]
        if index < 0 or index >= self.shape[dim]:
            raise ValueError("Index out of range for select")

        out_shape = self.shape[:dim] + self.shape[dim + 1:]
        out_strides = self.strides[:dim] + self.strides[dim + 1:]
        out_offset = self.offset + index * self.strides[dim]

        out, track = Tensor._make_out(
            self.storage,
            shape=out_shape,
            parents=(self,),
            op="select",
            strides=out_strides,
            offset=out_offset,
        )
        if not track:
            return out

        def _backward():
            sg = self._ensure_grad()
            og = out.grad

            for out_idx in self._iter_indices(out.shape):
                self_idx = out_idx[:dim] + (index,) + out_idx[dim:]
                sg.storage[sg._storage_index(self_idx)] += og.get(*out_idx)

        out._backward = _backward
        return out
    
    def contiguous(self):
        flat = [self.get(*idx) for idx in self._iter_indices(self.shape)]

        out, track = Tensor._make_out(
            flat,
            shape=self.shape,
            parents=(self,),
            op="contiguous",
            strides=self._default_strides(self.shape),
            offset=0,
        )
        if not track:
            return out

        def _backward():
            sg = self._ensure_grad()
            og = out.grad

            for idx in self._iter_indices(self.shape):
                sg.storage[sg._storage_index(idx)] += og.get(*idx)

        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("@ only supports Tensor @ Tensor")
        if len(self.shape) < 2 or len(other.shape) < 2:
            raise ValueError("matmul currently supports only tensors with rank >= 2")

        M, K = self.shape[-2:]
        K2, N = other.shape[-2:]
        if K != K2:
            raise ValueError(f"Shapes {self.shape} and {other.shape} not compatible for matmul")

        batch_shape = self._broadcast_shape(self.shape[:-2], other.shape[:-2])

        a = self._expand_to(batch_shape + (M, K))
        b = other._expand_to(batch_shape + (K, N))

        out_shape = batch_shape + (M, N)
        out_storage = [0.0] * Tensor._prod(out_shape)
        out, track = Tensor._make_out(out_storage, out_shape, (a, b), "matmul")

        for batch_idx in self._iter_indices(batch_shape):
            for i in range(M):
                for j in range(N):
                    acc = 0.0
                    for k in range(K):
                        acc += a.get(*(batch_idx + (i, k))) * b.get(*(batch_idx + (k, j)))
                    out.storage[out._storage_index(batch_idx + (i, j))] = acc

        if not track:
            return out

        def _backward():
            og = out.grad

            if a.requires_grad:
                ag = a._ensure_grad()
                for batch_idx in self._iter_indices(batch_shape):
                    for i in range(M):
                        for k in range(K):
                            acc = 0.0
                            for j in range(N):
                                acc += og.get(*(batch_idx + (i, j))) * b.get(*(batch_idx + (k, j)))
                            ag.storage[ag._storage_index(batch_idx + (i, k))] += acc

            if b.requires_grad:
                bg = b._ensure_grad()
                for batch_idx in self._iter_indices(batch_shape):
                    for k in range(K):
                        for j in range(N):
                            acc = 0.0
                            for i in range(M):
                                acc += a.get(*(batch_idx + (i, k))) * og.get(*(batch_idx + (i, j)))
                            bg.storage[bg._storage_index(batch_idx + (k, j))] += acc

        out._backward = _backward
        return out
    
    def apply_causal_mask(self, mask_value=float("-inf")):
        if len(self.shape) < 2:
            raise ValueError("causal mask requires rank >= 2")

        T1, T2 = self.shape[-2:]
        if T1 != T2:
            raise ValueError("last two dims must form a square matrix")

        out_storage = [0.0] * Tensor._prod(self.shape)
        out, track = Tensor._make_out(out_storage, self.shape, (self,), "causal_mask")
        for prefix in self._iter_indices(self.shape[:-2]):
            for i in range(T1):
                for j in range(T2):
                    idx = prefix + (i, j)
                    out.storage[out._storage_index(idx)] = (
                        self.get(*idx) if j <= i else mask_value
                    )

        if not track:
            return out

        def _backward():
            sg = self._ensure_grad()
            og = out.grad
            for prefix in self._iter_indices(self.shape[:-2]):
                for i in range(T1):
                    for j in range(T2):
                        if j <= i:
                            idx = prefix + (i, j)
                            sg.storage[sg._storage_index(idx)] += og.get(*idx)

        out._backward = _backward
        return out
    
    def softmax(self, axis=-1):
        ndim = len(self.shape)
        axis = self._canon_dim(axis, ndim)

        out_storage = [0.0] * Tensor._prod(self.shape)
        out, track = Tensor._make_out(out_storage, self.shape, (self,), "softmax")

        outer_shape = self.shape[:axis] + self.shape[axis + 1:]
        n = self.shape[axis]

        for outer_idx in self._iter_indices(outer_shape):
            def full_idx(j):
                return outer_idx[:axis] + (j,) + outer_idx[axis:]

            m = max(self.get(*full_idx(j)) for j in range(n))
            exps = [math.exp(self.get(*full_idx(j)) - m) for j in range(n)]
            s = sum(exps)
            inv_s = 1.0 / s

            for j, e in enumerate(exps):
                out.storage[out._storage_index(full_idx(j))] = e * inv_s

        if not track:
            return out

        def _backward():
            sg = self._ensure_grad()
            og = out.grad

            for outer_idx in self._iter_indices(outer_shape):
                def full_idx(j):
                    return outer_idx[:axis] + (j,) + outer_idx[axis:]

                dot = 0.0
                for j in range(n):
                    idx = full_idx(j)
                    dot += out.get(*idx) * og.get(*idx)

                for j in range(n):
                    idx = full_idx(j)
                    p = out.get(*idx)
                    sg.storage[sg._storage_index(idx)] += p * (og.get(*idx) - dot)

        out._backward = _backward
        return out
    
    def layernorm_last_dim(self, eps=1e-5):
        if len(self.shape) == 0:
            raise ValueError("layernorm requires rank >= 1")

        N = self.shape[-1]
        out_storage = [0.0] * Tensor._prod(self.shape)
        out, track = Tensor._make_out(out_storage, self.shape, (self,), "layernorm_last_dim")

        invstds = {}

        for outer_idx in self._iter_indices(self.shape[:-1]):
            vals = [self.get(*(outer_idx + (j,))) for j in range(N)]
            mean = sum(vals) / N
            var = sum((x - mean) ** 2 for x in vals) / N
            invstd = 1.0 / math.sqrt(var + eps)
            invstds[outer_idx] = invstd

            for j, x in enumerate(vals):
                out.storage[out._storage_index(outer_idx + (j,))] = (x - mean) * invstd

        if not track:
            return out

        def _backward():
            sg = self._ensure_grad()
            og = out.grad

            for outer_idx in self._iter_indices(self.shape[:-1]):
                mean_g = 0.0
                mean_gxhat = 0.0

                for j in range(N):
                    idx = outer_idx + (j,)
                    g = og.get(*idx)
                    xhat = out.get(*idx)
                    mean_g += g
                    mean_gxhat += g * xhat

                mean_g /= N
                mean_gxhat /= N
                invstd = invstds[outer_idx]

                for j in range(N):
                    idx = outer_idx + (j,)
                    g = og.get(*idx)
                    xhat = out.get(*idx)
                    sg.storage[sg._storage_index(idx)] += (
                        g - mean_g - xhat * mean_gxhat
                    ) * invstd

        out._backward = _backward
        return out
    
def _infer_nested_int_shape(x):
    if isinstance(x, int):
        return ()
    if not isinstance(x, list):
        raise ValueError("Expected an int or nested list of ints")
    if len(x) == 0:
        return (0,)
    first = _infer_nested_int_shape(x[0])
    for item in x:
        if _infer_nested_int_shape(item) != first:
            raise ValueError("Nested lists must be rectangular")
    return (len(x),) + first

def _iter_nested_ints(x, prefix=()):
    if isinstance(x, int):
        yield prefix, x
    else:
        for i, item in enumerate(x):
            yield from _iter_nested_ints(item, prefix + (i,))

# def attention(q, k, v, mask=True):
#     Dh = q.shape[-1]
#     scores = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(Dh))

#     if mask:
#         scores = scores.apply_causal_mask()

#     weights = scores.softmax(axis=-1)
#     return weights @ v

# def mha_forward(x, Wq, Wk, Wv, Wo, n_heads):
#     B, T, C = x.shape
#     assert C % n_heads == 0
#     Dh = C // n_heads

#     q = (x @ Wq).reshape(B, T, n_heads, Dh).permute(0, 2, 1, 3)
#     k = (x @ Wk).reshape(B, T, n_heads, Dh).permute(0, 2, 1, 3)
#     v = (x @ Wv).reshape(B, T, n_heads, Dh).permute(0, 2, 1, 3)

#     out = attention(q, k, v, mask=True)                  # (B, H, T, Dh)
#     out = out.permute(0, 2, 1, 3).contiguous()           # (B, T, H, Dh)
#     out = out.reshape(B, T, C)                           # (B, T, C)
#     return out @ Wo

# def transformer_block_forward(x, Wq, Wk, Wv, Wo, W1, W2, n_heads):
#     x = x + mha_forward(x.layernorm_last_dim(), Wq, Wk, Wv, Wo, n_heads)
#     x = x + ((x.layernorm_last_dim() @ W1).relu() @ W2)
#     return x

class Embedding:
    def __init__(self, vocab_size, d_model):
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)

        scale = math.sqrt(6.0 / (self.vocab_size + self.d_model))
        storage = [
            random.uniform(-scale, scale)
            for _ in range(self.vocab_size * self.d_model)
        ]
        self.weight = Tensor(
            storage,
            shape=(self.vocab_size, self.d_model),
            requires_grad=True,
        )

    def __call__(self, token_ids):
        ids_shape = _infer_nested_int_shape(token_ids)
        out_shape = ids_shape + (self.d_model,)
        out_storage = [0.0] * Tensor._prod(out_shape)

        entries = list(_iter_nested_ints(token_ids))

        out, track = Tensor._make_out(
            out_storage,
            out_shape,
            (self.weight,),
            "embedding",
        )

        for prefix, tok in entries:
            if tok < 0 or tok >= self.vocab_size:
                raise ValueError(f"token id {tok} out of range")
            for c in range(self.d_model):
                idx = prefix + (c,)
                out.storage[out._storage_index(idx)] = self.weight.get(tok, c)

        if not track:
            return out

        def _backward():
            wg = self.weight._ensure_grad()
            og = out.grad
            for prefix, tok in entries:
                for c in range(self.d_model):
                    wg.storage[wg._storage_index((tok, c))] += og.get(*(prefix + (c,)))

        out._backward = _backward
        return out

    def parameters(self):
        return [self.weight]
    
class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        scale = math.sqrt(6.0 / (self.in_features + self.out_features))
        w_storage = [
            random.uniform(-scale, scale)
            for _ in range(self.in_features * self.out_features)
        ]

        self.W = Tensor(
            w_storage,
            shape=(self.in_features, self.out_features),
            requires_grad=True,
        )
        self.b = Tensor.zeros((1, self.out_features), requires_grad=True)

    def __call__(self, x):
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]
    
def cross_entropy_loss(logits, target_ids):
    if len(logits.shape) < 1:
        raise ValueError("logits must have rank >= 1")

    V = logits.shape[-1]
    outer_shape = logits.shape[:-1]

    target_shape = _infer_nested_int_shape(target_ids)
    if target_shape != outer_shape:
        raise ValueError(
            f"target shape {target_shape} must match logits.shape[:-1] {outer_shape}"
        )

    entries = list(_iter_nested_ints(target_ids))
    if len(entries) == 0:
        raise ValueError("cross_entropy_loss requires at least one target")

    probs = []
    total = 0.0

    for prefix, tgt in entries:
        if tgt < 0 or tgt >= V:
            raise ValueError(f"target id {tgt} out of range")

        row = [logits.get(*(prefix + (j,))) for j in range(V)]
        m = max(row)
        exps = [math.exp(x - m) for x in row]
        s = sum(exps)

        total += math.log(s) - (row[tgt] - m)
        probs.append(([e / s for e in exps], prefix, tgt))

    loss_value = total / len(entries)

    out, track = Tensor._make_out(
        [loss_value],
        (),
        (logits,),
        "cross_entropy",
    )
    if not track:
        return out

    def _backward():
        lg = logits._ensure_grad()
        scale = out.grad.item() / len(entries)

        for row_probs, prefix, tgt in probs:
            for j in range(V):
                idx = prefix + (j,)
                lg.storage[lg._storage_index(idx)] += row_probs[j] * scale
            tgt_idx = prefix + (tgt,)
            lg.storage[lg._storage_index(tgt_idx)] -= scale

    out._backward = _backward
    return out

def zero_grad_all(params):
    for p in params:
        p.zero_grad()

def sgd_step(params, lr):
    with Tensor.no_grad():
        for p in params:
            if p.grad is None:
                continue
            for i in range(len(p.storage)):
                p.storage[i] -= lr * p.grad.storage[i]

class LayerNorm:
    def __init__(self, d_model):
        self.gamma = Tensor.full((1, d_model), 1.0, requires_grad=True)
        self.beta = Tensor.zeros((1, d_model), requires_grad=True)

    def __call__(self, x):
        return x.layernorm_last_dim() * self.gamma + self.beta

    def parameters(self):
        return [self.gamma, self.beta]
    
class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)

        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)

    def __call__(self, x):
        B, T, C = x.shape
        H = self.n_heads
        Dh = C // H

        q = self.q_proj(x).reshape(B, T, H, Dh).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, H, Dh).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, H, Dh).permute(0, 2, 1, 3)

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(Dh)
        scores = scores.apply_causal_mask()
        weights = scores.softmax(axis=-1)

        out = weights @ v
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, T, C)
        return self.o_proj(out)

    def parameters(self):
        return (
            self.q_proj.parameters()
            + self.k_proj.parameters()
            + self.v_proj.parameters()
            + self.o_proj.parameters()
        )
    
class TransformerBlock:
    def __init__(self, d_model, n_heads, d_ff):
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)

        self.ff1 = Linear(d_model, d_ff)
        self.ff2 = Linear(d_ff, d_model)

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff2(self.ff1(self.ln2(x)).relu())
        return x

    def parameters(self):
        return (
            self.ln1.parameters()
            + self.attn.parameters()
            + self.ln2.parameters()
            + self.ff1.parameters()
            + self.ff2.parameters()
        )
    
class TinyGPT:
    def __init__(self, vocab_size, block_size, d_model, n_heads, d_ff, n_layers):
        self.tok_emb = Embedding(vocab_size, d_model)
        self.pos_emb = Embedding(block_size, d_model)
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.ln_f = LayerNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        self.block_size = int(block_size)

    def __call__(self, token_ids):
        if not isinstance(token_ids, list) or len(token_ids) == 0:
            raise ValueError("token_ids must be a non-empty 2D list")

        if not isinstance(token_ids[0], list) or len(token_ids[0]) == 0:
            raise ValueError("token_ids must contain non-empty sequences")

        T = len(token_ids[0])
        for row in token_ids:
            if not isinstance(row, list):
                raise ValueError("token_ids must be a 2D list")
            if len(row) != T:
                raise ValueError("all sequences in token_ids must have the same length")

        if T > self.block_size:
            raise ValueError("sequence length exceeds block_size")

        x = self.tok_emb(token_ids)
        pos = self.pos_emb(list(range(T)))
        x = x + pos

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.lm_head(x)

    def loss(self, x_ids, y_ids):
        logits = self(x_ids)
        return cross_entropy_loss(logits, y_ids)

    def generate(self, token_ids, max_new_tokens):
        if not isinstance(token_ids, list) or len(token_ids) == 0:
            raise ValueError("token_ids must be a non-empty 2D list")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")

        if not isinstance(token_ids[0], list) or len(token_ids[0]) == 0:
            raise ValueError("token_ids must contain non-empty sequences")

        prompt_len = len(token_ids[0])
        for row in token_ids:
            if not isinstance(row, list):
                raise ValueError("token_ids must be a 2D list")
            if len(row) != prompt_len:
                raise ValueError("all prompts must have the same length for batched generation")

        ids = [row[:] for row in token_ids]

        for _ in range(max_new_tokens):
            ctx = [row[-self.block_size:] for row in ids]

            with Tensor.no_grad():
                logits = self(ctx).tolist()

            for b in range(len(ids)):
                next_logits = logits[b][-1]
                next_id = max(range(len(next_logits)), key=lambda i: next_logits[i])  # greedy
                ids[b].append(next_id)

        return ids

    def parameters(self):
        params = []
        params += self.tok_emb.parameters()
        params += self.pos_emb.parameters()
        for block in self.blocks:
            params += block.parameters()
        params += self.ln_f.parameters()
        params += self.lm_head.parameters()
        return params
    
def get_batch(data, batch_size, block_size):
    if len(data) < block_size + 1:
        raise ValueError("data must contain at least block_size + 1 token ids")
    xs, ys = [], []
    for _ in range(batch_size):
        i = random.randint(0, len(data) - block_size - 1)
        chunk = data[i:i + block_size + 1]
        xs.append(chunk[:-1])
        ys.append(chunk[1:])
    return xs, ys

def estimate_loss(model, data, batch_size, block_size, eval_steps=20):
    losses = []
    with Tensor.no_grad():
        for _ in range(eval_steps):
            xb, yb = get_batch(data, batch_size, block_size)
            losses.append(model.loss(xb, yb).item())
    return sum(losses) / len(losses)

def train_loop(model, train_data, val_data, steps, batch_size, block_size, lr):
    if len(train_data) < block_size + 1:
        raise ValueError("train_data must contain at least block_size + 1 token ids")
    if len(val_data) < block_size + 1:
        raise ValueError("val_data must contain at least block_size + 1 token ids")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")

    params = model.parameters()

    for step in range(steps):
        xb, yb = get_batch(train_data, batch_size, block_size)

        loss = model.loss(xb, yb)
        zero_grad_all(params)
        loss.backward()
        sgd_step(params, lr)

        if step % 100 == 0:
            train_loss = estimate_loss(model, train_data, batch_size, block_size)
            val_loss = estimate_loss(model, val_data, batch_size, block_size)
            print(f"step {step:4d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")



class ByteBPETokenizer:
    def __init__(self):
        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        self.merges = []          # [ ((a, b), new_id), ... ] in merge order
        self.merge_to_id = {}     # (a, b) -> new_id

    @property
    def vocab_size(self):
        return len(self.id_to_bytes)

    @staticmethod
    def _merge_once(tokens, pair, new_id):
        out = []
        i = 0
        a, b = pair

        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(tokens[i])
                i += 1

        return out

    def train(self, text, vocab_size=512, min_pair_count=2, verbose=False):
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256 for byte-level BPE")

        # reset so train() is reusable
        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        self.merges = []
        self.merge_to_id = {}

        tokens = list(text.encode("utf-8"))
        next_id = 256

        while next_id < vocab_size:
            pair_counts = {}

            for a, b in zip(tokens, tokens[1:]):
                pair = (a, b)
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break

            best_pair, best_count = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))

            if best_count < min_pair_count:
                break

            self.merges.append((best_pair, next_id))
            self.merge_to_id[best_pair] = next_id
            self.id_to_bytes[next_id] = (
                self.id_to_bytes[best_pair[0]] + self.id_to_bytes[best_pair[1]]
            )

            tokens = self._merge_once(tokens, best_pair, next_id)

            if verbose and (((next_id - 255) % 100 == 0) or (next_id + 1 == vocab_size)):
                print(f"tokenizer vocab now {next_id + 1}")

            next_id += 1

        return self

    def encode(self, text):
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        tokens = list(text.encode("utf-8"))

        # Apply merges in the exact order they were learned.
        for pair, new_id in self.merges:
            tokens = self._merge_once(tokens, pair, new_id)

        return tokens

    def decode(self, ids):
        out_bytes = b"".join(self.id_to_bytes[int(tok)] for tok in ids)
        return out_bytes.decode("utf-8", errors="replace")

    def save(self, path):
        payload = {
            "merges": [[a, b, new_id] for (a, b), new_id in self.merges]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        tok = cls()
        for a, b, new_id in payload["merges"]:
            a = int(a)
            b = int(b)
            new_id = int(new_id)
            pair = (a, b)

            tok.merges.append((pair, new_id))
            tok.merge_to_id[pair] = new_id
            tok.id_to_bytes[new_id] = tok.id_to_bytes[a] + tok.id_to_bytes[b]

        return tok


def load_text_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def prepare_text_dataset(train_text, val_text, tokenizer, block_size=None):
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    all_ids = train_ids + val_ids

    if len(all_ids) < 2:
        raise ValueError("encoded corpus must contain at least 2 token ids")

    if block_size is not None:
        if len(train_ids) < block_size + 1:
            raise ValueError("train split too short for requested block_size")
        if len(val_ids) < block_size + 1:
            raise ValueError("val split too short for requested block_size")

    return all_ids, train_ids, val_ids


def build_tokenizer_and_dataset(
    text,
    vocab_size=512,
    train_frac=0.9,
    min_pair_count=2,
    block_size=None,
    verbose=True,
):
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if len(text) < 2:
        raise ValueError("text must contain at least 2 characters")

    split = int(len(text) * train_frac)
    split = max(1, min(len(text) - 1, split))

    train_text = text[:split]
    val_text = text[split:]

    tokenizer = ByteBPETokenizer().train(
        train_text,
        vocab_size=vocab_size,
        min_pair_count=min_pair_count,
        verbose=verbose,
    )

    all_ids, train_ids, val_ids = prepare_text_dataset(
        train_text,
        val_text,
        tokenizer,
        block_size=block_size,
    )

    return tokenizer, all_ids, train_ids, val_ids


# text = load_text_file("input.txt")

# tokenizer, all_ids, train_ids, val_ids = build_tokenizer_and_dataset(
#     text,
#     vocab_size=512,
#     train_frac=0.9,
#     min_pair_count=2,
#     block_size=64,
#     verbose=True,
# )

# print("tokenizer vocab size:", tokenizer.vocab_size)
# print("raw characters:", len(text))
# print("total token ids:", len(all_ids))

# model = TinyGPT(
#     vocab_size=tokenizer.vocab_size,
#     block_size=64,
#     d_model=64,
#     n_heads=4,
#     d_ff=256,
#     n_layers=2,
# )

# train_loop(
#     model,
#     train_data=train_ids,
#     val_data=val_ids,
#     steps=3000,
#     batch_size=16,
#     block_size=64,
#     lr=3e-3,
# )

# tokenizer.save("tokenizer.json")


def test_views():
    x = Tensor.from_nested([[1, 2, 3], [4, 5, 6]])

    assert x.shape == (2, 3)
    assert x.strides == (3, 1)
    assert x.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    xt = x.transpose(0, 1)
    assert xt.shape == (3, 2)
    assert xt.tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    assert not xt.is_contiguous()

    xr = x.reshape(3, 2)
    assert xr.tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

    xn = x.narrow(0, 1, 1)
    assert xn.tolist() == [[4.0, 5.0, 6.0]]

    xs = x.select(0, 1)
    assert xs.tolist() == [4.0, 5.0, 6.0]

    yc = xt.contiguous()
    assert yc.is_contiguous()
    assert yc.reshape(6).tolist() == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]

def test_aliasing():
    x = Tensor.from_nested([[1, 2, 3], [4, 5, 6]])
    xt = x.transpose(0, 1)

    x.storage[1] = 99.0
    assert xt.get(1, 0) == 99.0

def test_3d_views():
    x = Tensor.from_nested([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
    ])  # shape (2, 2, 2)

    assert x.permute(1, 0, 2).tolist() == [
        [[1.0, 2.0], [5.0, 6.0]],
        [[3.0, 4.0], [7.0, 8.0]],
    ]
    assert x.narrow(1, 0, 1).tolist() == [
        [[1.0, 2.0]],
        [[5.0, 6.0]],
    ]
    assert x.select(2, 1).tolist() == [
        [2.0, 4.0],
        [6.0, 8.0],
    ]

def test_contiguous_is_copy():
    x = Tensor.from_nested([[1, 2, 3], [4, 5, 6]])
    xt = x.transpose(0, 1)
    y = xt.contiguous()

    x.storage[0] = 99.0
    assert xt.get(0, 0) == 99.0
    assert y.get(0, 0) == 1.0

def test_negative_dims():
    x = Tensor.from_nested([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
    ])

    assert x.transpose(-1, -2).tolist() == [
        [[1.0, 3.0], [2.0, 4.0]],
        [[5.0, 7.0], [6.0, 8.0]],
    ]
    assert x.select(-1, 1).tolist() == [
        [2.0, 4.0],
        [6.0, 8.0],
    ]

def test_negative_permute():
    x = Tensor.from_nested([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
    ])
    assert x.permute(-1, 0, 1).tolist() == [
        [[1.0, 3.0], [5.0, 7.0]],
        [[2.0, 4.0], [6.0, 8.0]],
    ]

def test_add_on_view():
    x = Tensor.from_nested([[1, 2, 3], [4, 5, 6]])
    xt = x.transpose(0, 1)
    assert (xt + 10).tolist() == [[11.0, 14.0], [12.0, 15.0], [13.0, 16.0]]

def test_add_two_views():
    x = Tensor.from_nested([[1, 2, 3], [4, 5, 6]])
    xt = x.transpose(0, 1)
    assert (xt + xt).tolist() == [[2.0, 8.0], [4.0, 10.0], [6.0, 12.0]]

def test_row_broadcast_add():
    x = Tensor.from_nested([[1, 2, 3], [4, 5, 6]])
    row = Tensor.from_nested([[10, 20, 30]])
    assert (x + row).tolist() == [
        [11.0, 22.0, 33.0],
        [14.0, 25.0, 36.0],
    ]

def test_col_broadcast_add():
    x = Tensor.from_nested([[1, 2, 3], [4, 5, 6]])
    col = Tensor.from_nested([[10], [20]])
    assert (x + col).tolist() == [
        [11.0, 12.0, 13.0],
        [24.0, 25.0, 26.0],
    ]

def test_scalar_left_add():
    x = Tensor.from_nested([[1, 2], [3, 4]])
    assert (2 + x).tolist() == [[3.0, 4.0], [5.0, 6.0]]

def test_1d_broadcast_add():
    x = Tensor.from_nested([[1, 2, 3], [4, 5, 6]])
    v = Tensor.from_nested([10, 20, 30])
    assert (x + v).tolist() == [
        [11.0, 22.0, 33.0],
        [14.0, 25.0, 36.0],
    ]

def test_3d_broadcast_add():
    a = Tensor.from_nested([
        [[1, 2, 3]],
        [[4, 5, 6]],
    ])  # shape (2, 1, 3)

    b = Tensor.from_nested([
        [[10], [20], [30], [40]],
    ])  # shape (1, 4, 1)

    assert (a + b).tolist() == [
        [[11.0, 12.0, 13.0], [21.0, 22.0, 23.0], [31.0, 32.0, 33.0], [41.0, 42.0, 43.0]],
        [[14.0, 15.0, 16.0], [24.0, 25.0, 26.0], [34.0, 35.0, 36.0], [44.0, 45.0, 46.0]],
    ]

def test_broadcast_add_on_noncontiguous_view():
    x = Tensor.from_nested([[1, 2, 3], [4, 5, 6]])
    xt = x.transpose(0, 1)            # shape (3, 2), non-contiguous
    v = Tensor.from_nested([10, 20])  # shape (2,)

    assert (xt + v).tolist() == [
        [11.0, 24.0],
        [12.0, 25.0],
        [13.0, 26.0],
    ]

def test_mul_on_view():
    x = Tensor.from_nested([[1, 2, 3], [4, 5, 6]])
    xt = x.transpose(0, 1)
    assert (xt * 10).tolist() == [
        [10.0, 40.0],
        [20.0, 50.0],
        [30.0, 60.0],
    ]

def test_row_broadcast_mul():
    x = Tensor.from_nested([[1, 2, 3], [4, 5, 6]])
    row = Tensor.from_nested([[10, 20, 30]])
    assert (x * row).tolist() == [
        [10.0, 40.0, 90.0],
        [40.0, 100.0, 180.0],
    ]

def test_col_broadcast_mul():
    x = Tensor.from_nested([[1, 2, 3], [4, 5, 6]])
    col = Tensor.from_nested([[10], [20]])
    assert (x * col).tolist() == [
        [10.0, 20.0, 30.0],
        [80.0, 100.0, 120.0],
    ]

def test_3d_broadcast_mul():
    a = Tensor.from_nested([
        [[1, 2, 3]],
        [[4, 5, 6]],
    ])  # (2, 1, 3)

    b = Tensor.from_nested([
        [[10], [20], [30], [40]],
    ])  # (1, 4, 1)

    assert (a * b).tolist() == [
        [[10.0, 20.0, 30.0], [20.0, 40.0, 60.0], [30.0, 60.0, 90.0], [40.0, 80.0, 120.0]],
        [[40.0, 50.0, 60.0], [80.0, 100.0, 120.0], [120.0, 150.0, 180.0], [160.0, 200.0, 240.0]],
    ]

def test_sum_all():
    x = Tensor.from_nested([[1, 2, 3], [4, 5, 6]])
    assert x.sum().shape == ()
    assert x.sum().tolist() == 21.0

def test_sum_axis0():
    x = Tensor.from_nested([[1, 2, 3], [4, 5, 6]])
    assert x.sum(axis=0).tolist() == [5.0, 7.0, 9.0]

def test_sum_axis1_keepdims():
    x = Tensor.from_nested([[1, 2, 3], [4, 5, 6]])
    assert x.sum(axis=1, keepdims=True).tolist() == [[6.0], [15.0]]

def test_sum_on_view():
    x = Tensor.from_nested([[1, 2, 3], [4, 5, 6]])
    xt = x.transpose(0, 1)
    assert xt.sum(axis=0).tolist() == [6.0, 15.0]

def test_sum_multi_axis():
    x = Tensor.from_nested([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
    ])
    assert x.sum(axis=(0, 2)).tolist() == [14.0, 22.0]

def test_matmul_basic():
    a = Tensor.from_nested([[1, 2, 3], [4, 5, 6]])
    b = Tensor.from_nested([[7, 8], [9, 10], [11, 12]])
    assert (a @ b).tolist() == [[58.0, 64.0], [139.0, 154.0]]

def test_matmul_with_view():
    x = Tensor.from_nested([[1, 2, 3], [4, 5, 6]])
    xt = x.transpose(0, 1)   # shape (3, 2)
    y = Tensor.from_nested([[10, 20, 30], [40, 50, 60]])  # shape (2, 3)

    assert (xt @ y).tolist() == [
        [170.0, 220.0, 270.0],
        [220.0, 290.0, 360.0],
        [270.0, 360.0, 450.0],
    ]

def test_matmul_shape_error():
    a = Tensor.from_nested([[1, 2], [3, 4]])
    b = Tensor.from_nested([[1, 2], [3, 4], [5, 6]])
    try:
        a @ b
        assert False, "expected shape mismatch"
    except ValueError:
        pass

def test_mean_axis0():
    x = Tensor.from_nested([[1, 2, 3], [4, 5, 6]])
    assert x.mean(axis=0).tolist() == [2.5, 3.5, 4.5]

def test_sum_keepdims_multi_axis():
    x = Tensor.from_nested([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
    ])
    assert x.sum(axis=(0, 2), keepdims=True).tolist() == [[[14.0], [22.0]]]

def test_expand_row():
    row = Tensor.from_nested([[10, 20, 30]])
    ex = row.expand(2, 3)
    assert ex.shape == (2, 3)
    assert ex.strides == (0, 1)
    assert ex.tolist() == [
        [10.0, 20.0, 30.0],
        [10.0, 20.0, 30.0],
    ]

def test_expand_vector():
    v = Tensor.from_nested([1, 2, 3])
    ex = v.expand(2, 3)
    assert ex.tolist() == [
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ]

def test_expand_aliasing():
    row = Tensor.from_nested([[10, 20, 30]])
    ex = row.expand(2, 3)
    row.storage[1] = 99.0
    assert ex.tolist() == [
        [10.0, 99.0, 30.0],
        [10.0, 99.0, 30.0],
    ]

def test_expand_column():
    col = Tensor.from_nested([[10], [20]])
    ex = col.expand(2, 3)
    assert ex.shape == (2, 3)
    assert ex.strides == (1, 0)
    assert ex.tolist() == [
        [10.0, 10.0, 10.0],
        [20.0, 20.0, 20.0],
    ]

def test_expand_invalid():
    x = Tensor.from_nested([[1, 2, 3], [4, 5, 6]])
    try:
        x.expand(4, 3)
        assert False, "expected invalid expand"
    except ValueError:
        pass

def test_expand_view_addresses():
    row = Tensor.from_nested([[10, 20, 30]])
    ex = row.expand(2, 3)
    assert ex.get(0, 1) == ex.get(1, 1) == 20.0

def test_batched_matmul_basic():
    a = Tensor.from_nested([
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
    ])  # shape (2, 2, 3)

    b = Tensor.from_nested([
        [[1, 2], [3, 4], [5, 6]],
        [[2, 0], [1, 2], [0, 1]],
    ])  # shape (2, 3, 2)

    assert (a @ b).tolist() == [
        [[22.0, 28.0], [49.0, 64.0]],
        [[22.0, 25.0], [31.0, 34.0]],
    ]

def test_batched_matmul_broadcast_rhs():
    a = Tensor.from_nested([
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
    ])  # shape (2, 2, 3)

    b = Tensor.from_nested([
        [1, 0],
        [0, 1],
        [1, 1],
    ])  # shape (3, 2)

    assert (a @ b).tolist() == [
        [[4.0, 5.0], [10.0, 11.0]],
        [[16.0, 17.0], [22.0, 23.0]],
    ]

def test_batched_matmul_with_view():
    x = Tensor.from_nested([
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
    ])  # shape (2, 2, 3)

    xt = x.transpose(-1, -2)  # shape (2, 3, 2), non-contiguous
    y = Tensor.from_nested([
        [1, 2, 3],
        [4, 5, 6],
    ])  # shape (2, 3)

    assert (xt @ y).tolist() == [
        [[17.0, 22.0, 27.0], [22.0, 29.0, 36.0], [27.0, 36.0, 45.0]],
        [[47.0, 64.0, 81.0], [52.0, 71.0, 90.0], [57.0, 78.0, 99.0]],
    ]

def test_batched_matmul_batch_shape_error():
    a = Tensor.from_nested([
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
    ])  # shape (2, 2, 3)

    b = Tensor.from_nested([
        [[1, 2], [3, 4], [5, 6]],
        [[7, 8], [9, 10], [11, 12]],
        [[13, 14], [15, 16], [17, 18]],
    ])  # shape (3, 3, 2)

    try:
        a @ b
        assert False, "expected batch shape mismatch"
    except ValueError:
        pass

def test_softmax_last_dim():
    x = Tensor.from_nested([[1, 2, 3], [1000, 1001, 1002]])
    y = x.softmax(axis=-1)
    rows = y.tolist()

    for row in rows:
        assert abs(sum(row) - 1.0) < 1e-9

    assert rows[0][0] < rows[0][1] < rows[0][2]
    assert rows[1][0] < rows[1][1] < rows[1][2]

def test_softmax_on_view():
    x = Tensor.from_nested([[1, 4], [2, 5], [3, 6]])
    xt = x.transpose(0, 1)   # shape (2, 3), non-contiguous
    y = xt.softmax(axis=-1)

    rows = y.tolist()
    for row in rows:
        assert abs(sum(row) - 1.0) < 1e-9

def test_causal_mask_2d():
    x = Tensor.from_nested([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ])
    y = x.apply_causal_mask().tolist()

    assert y == [
        [1.0, float("-inf"), float("-inf")],
        [4.0, 5.0, float("-inf")],
        [7.0, 8.0, 9.0],
    ]

def test_causal_mask_batched():
    x = Tensor.from_nested([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
    ])  # shape (2, 2, 2)

    y = x.apply_causal_mask().tolist()
    assert y == [
        [[1.0, float("-inf")], [3.0, 4.0]],
        [[5.0, float("-inf")], [7.0, 8.0]],
    ]

# def test_attention_causal_tiny():
#     q = Tensor.from_nested([[[[1.0], [1.0], [1.0]]]])   # shape (1, 1, 3, 1)
#     k = Tensor.from_nested([[[[1.0], [1.0], [1.0]]]])   # shape (1, 1, 3, 1)
#     v = Tensor.from_nested([[[[10.0], [20.0], [30.0]]]])# shape (1, 1, 3, 1)

#     out = attention(q, k, v, mask=True)
#     vals = out.tolist()

#     assert out.shape == (1, 1, 3, 1)
#     assert abs(vals[0][0][0][0] - 10.0) < 1e-9
#     assert abs(vals[0][0][1][0] - 15.0) < 1e-9
#     assert abs(vals[0][0][2][0] - 20.0) < 1e-9

# def test_attention_no_mask_tiny():
#     q = Tensor.from_nested([[[[1.0], [1.0], [1.0]]]])
#     k = Tensor.from_nested([[[[1.0], [1.0], [1.0]]]])
#     v = Tensor.from_nested([[[[10.0], [20.0], [30.0]]]])

#     out = attention(q, k, v, mask=False).tolist()

#     for t in range(3):
#         assert abs(out[0][0][t][0] - 20.0) < 1e-9

# def test_mha_shape():
#     x = Tensor.from_nested([[
#         [1.0, 2.0, 3.0, 4.0],
#         [5.0, 6.0, 7.0, 8.0],
#         [9.0, 10.0, 11.0, 12.0],
#     ]])  # shape (1, 3, 4)

#     I = Tensor.from_nested([
#         [1.0, 0.0, 0.0, 0.0],
#         [0.0, 1.0, 0.0, 0.0],
#         [0.0, 0.0, 1.0, 0.0],
#         [0.0, 0.0, 0.0, 1.0],
#     ])

#     out = mha_forward(x, I, I, I, I, n_heads=2)
#     assert out.shape == (1, 3, 4)

def test_layernorm_last_dim():
    x = Tensor.from_nested([
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
    ])
    y = x.layernorm_last_dim().tolist()

    for batch in y:
        for row in batch:
            mean = sum(row) / len(row)
            var = sum((v - mean) ** 2 for v in row) / len(row)
            assert abs(mean) < 1e-6
            assert abs(var - 1.0) < 1e-4

def test_layernorm_on_view():
    x = Tensor.from_nested([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    xt = x.transpose(0, 1)   # shape (2, 3), non-contiguous
    y = xt.layernorm_last_dim().tolist()

    for row in y:
        mean = sum(row) / len(row)
        var = sum((v - mean) ** 2 for v in row) / len(row)
        assert abs(mean) < 1e-6
        assert abs(var - 1.0) < 1e-4

def test_relu():
    x = Tensor.from_nested([[-1.0, 0.0, 2.0], [3.0, -4.0, 5.0]])
    y = x.relu().tolist()
    assert y == [[0.0, 0.0, 2.0], [3.0, 0.0, 5.0]]

# def test_transformer_block_shape():
#     x = Tensor.from_nested([[
#         [1.0, 2.0, 3.0, 4.0],
#         [5.0, 6.0, 7.0, 8.0],
#         [9.0, 10.0, 11.0, 12.0],
#     ]])  # shape (1, 3, 4)

#     Z44 = Tensor.zeros((4, 4))
#     Z48 = Tensor.zeros((4, 8))
#     Z84 = Tensor.zeros((8, 4))

#     out = transformer_block_forward(x, Z44, Z44, Z44, Z44, Z48, Z84, n_heads=2)
#     assert out.shape == (1, 3, 4)
#     assert out.tolist() == x.tolist()

def test_expand_backward():
    x = Tensor.from_nested([[1.0, 2.0, 3.0]], requires_grad=True)
    y = x.expand(2, 3).sum()
    y.backward()
    assert x.grad.tolist() == [[2.0, 2.0, 2.0]]

def test_add_backward():
    x = Tensor.from_nested([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = (x + x).sum()
    y.backward()
    assert x.grad.tolist() == [[2.0, 2.0], [2.0, 2.0]]

def test_mul_backward_scalar():
    x = Tensor.from_nested([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = (x * 3.0).sum()
    y.backward()
    assert x.grad.tolist() == [[3.0, 3.0], [3.0, 3.0]]

def test_radd_backward():
    x = Tensor.from_nested([[1.0, 2.0]], requires_grad=True)
    y = (2.0 + x).sum()
    y.backward()
    assert x.grad.tolist() == [[1.0, 1.0]]

def test_rmul_backward():
    x = Tensor.from_nested([[1.0, 2.0]], requires_grad=True)
    y = (3.0 * x).sum()
    y.backward()
    assert x.grad.tolist() == [[3.0, 3.0]]

def test_no_grad_expand():
    x = Tensor.from_nested([[1.0, 2.0]], requires_grad=True)
    with Tensor.no_grad():
        y = x.expand(2, 2)
    assert y.requires_grad is False
    assert y._prev == ()

def test_relu_backward():
    x = Tensor.from_nested([[-1.0, 0.0, 2.0], [3.0, -4.0, 5.0]], requires_grad=True)
    y = x.relu().sum()
    y.backward()
    assert x.grad.tolist() == [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]

def test_sum_backward_axis0():
    x = Tensor.from_nested([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = x.sum(axis=0).sum()
    y.backward()
    assert x.grad.tolist() == [[1.0, 1.0], [1.0, 1.0]]

def test_add_backward_row_broadcast():
    x = Tensor.from_nested([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor.from_nested([[10.0, 20.0]], requires_grad=True)
    y = (x + b).sum()
    y.backward()
    assert x.grad.tolist() == [[1.0, 1.0], [1.0, 1.0]]
    assert b.grad.tolist() == [[2.0, 2.0]]

def test_mul_backward_row_broadcast():
    x = Tensor.from_nested([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor.from_nested([[10.0, 20.0]], requires_grad=True)
    y = (x * b).sum()
    y.backward()
    assert x.grad.tolist() == [[10.0, 20.0], [10.0, 20.0]]
    assert b.grad.tolist() == [[4.0, 6.0]]

def test_no_grad_add():
    x = Tensor.from_nested([[1.0, 2.0]], requires_grad=True)
    with Tensor.no_grad():
        y = x + 3.0
    assert y.requires_grad is False
    assert y._prev == ()

def test_reshape_backward_sum():
    x = Tensor.from_nested([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    y = x.reshape(3, 2).sum()
    y.backward()
    assert x.grad.tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]

def test_reshape_backward_layout():
    x = Tensor.from_nested([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    w = Tensor.from_nested([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = (x.reshape(3, 2) * w).sum()
    y.backward()
    assert x.grad.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

def test_no_grad_reshape():
    x = Tensor.from_nested([[1.0, 2.0, 3.0]], requires_grad=True)
    with Tensor.no_grad():
        y = x.reshape(3, 1)
    assert y.requires_grad is False
    assert y._prev == ()

def test_permute_backward_sum():
    x = Tensor.from_nested([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
    ], requires_grad=True)

    y = x.permute(1, 0, 2).sum()
    y.backward()

    assert x.grad.tolist() == [
        [[1.0, 1.0], [1.0, 1.0]],
        [[1.0, 1.0], [1.0, 1.0]],
    ]

def test_permute_backward_layout():
    x = Tensor.from_nested([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
    ], requires_grad=True)

    w = Tensor.from_nested([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
    ])

    y = (x.permute(1, 0, 2) * w).sum()
    y.backward()

    assert x.grad.tolist() == [
        [[1.0, 2.0], [5.0, 6.0]],
        [[3.0, 4.0], [7.0, 8.0]],
    ]

def test_no_grad_permute():
    x = Tensor.from_nested([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    with Tensor.no_grad():
        y = x.permute(1, 0)

    assert y.requires_grad is False
    assert y._prev == ()

def test_narrow_backward_sum():
    x = Tensor.from_nested([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    y = x.narrow(1, 1, 2).sum()
    y.backward()
    assert x.grad.tolist() == [[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]

def test_narrow_backward_layout():
    x = Tensor.from_nested([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    w = Tensor.from_nested([[10.0, 20.0], [30.0, 40.0]])
    y = (x.narrow(1, 1, 2) * w).sum()
    y.backward()
    assert x.grad.tolist() == [[0.0, 10.0, 20.0], [0.0, 30.0, 40.0]]

def test_no_grad_narrow():
    x = Tensor.from_nested([[1.0, 2.0, 3.0]], requires_grad=True)
    with Tensor.no_grad():
        y = x.narrow(1, 1, 2)
    assert y.requires_grad is False
    assert y._prev == ()

def test_select_backward_sum():
    x = Tensor.from_nested([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    y = x.select(1, 1).sum()
    y.backward()
    assert x.grad.tolist() == [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]

def test_select_backward_layout():
    x = Tensor.from_nested([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    w = Tensor.from_nested([10.0, 20.0])
    y = (x.select(1, 1) * w).sum()
    y.backward()
    assert x.grad.tolist() == [[0.0, 10.0, 0.0], [0.0, 20.0, 0.0]]

def test_select_backward_negative_index():
    x = Tensor.from_nested([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    y = x.select(-1, -1).sum()
    y.backward()
    assert x.grad.tolist() == [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]

def test_no_grad_select():
    x = Tensor.from_nested([[1.0, 2.0, 3.0]], requires_grad=True)
    with Tensor.no_grad():
        y = x.select(1, 1)
    assert y.requires_grad is False
    assert y._prev == ()

def test_contiguous_backward_sum():
    x = Tensor.from_nested([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    y = x.transpose(0, 1).contiguous().sum()
    y.backward()
    assert x.grad.tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]

def test_contiguous_backward_layout():
    x = Tensor.from_nested([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    xt = x.transpose(0, 1)
    w = Tensor.from_nested([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    y = (xt.contiguous() * w).sum()
    y.backward()

    assert x.grad.tolist() == [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]

def test_no_grad_contiguous():
    x = Tensor.from_nested([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    with Tensor.no_grad():
        y = x.transpose(0, 1).contiguous()
    assert y.requires_grad is False
    assert y._prev == ()

def test_matmul_backward_basic():
    x = Tensor.from_nested([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    w = Tensor.from_nested([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], requires_grad=True)

    y = (x @ w).sum()
    y.backward()

    assert x.grad.tolist() == [
        [15.0, 19.0, 23.0],
        [15.0, 19.0, 23.0],
    ]
    assert w.grad.tolist() == [
        [5.0, 5.0],
        [7.0, 7.0],
        [9.0, 9.0],
    ]

def test_matmul_backward_with_view():
    x = Tensor.from_nested([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    y = Tensor.from_nested([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])

    loss = (x.transpose(0, 1) @ y).sum()
    loss.backward()

    assert x.grad.tolist() == [
        [60.0, 60.0, 60.0],
        [150.0, 150.0, 150.0],
    ]

def test_matmul_backward_broadcast_rhs_input_grad():
    a = Tensor.from_nested([
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
    ], requires_grad=True)

    b = Tensor.from_nested([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])

    y = (a @ b).sum()
    y.backward()

    assert a.grad.tolist() == [
        [[1.0, 1.0, 2.0], [1.0, 1.0, 2.0]],
        [[1.0, 1.0, 2.0], [1.0, 1.0, 2.0]],
    ]

def test_matmul_backward_broadcast_rhs_weight_grad():
    a = Tensor.from_nested([
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
    ])

    b = Tensor.from_nested([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ], requires_grad=True)

    y = (a @ b).sum()
    y.backward()

    assert b.grad.tolist() == [
        [22.0, 22.0],
        [26.0, 26.0],
        [30.0, 30.0],
    ]

def test_no_grad_matmul():
    x = Tensor.from_nested([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    w = Tensor.from_nested([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

    with Tensor.no_grad():
        y = x @ w

    assert y.requires_grad is False
    assert y._prev == ()

def test_causal_mask_backward():
    x = Tensor.from_nested([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ], requires_grad=True)

    y = x.apply_causal_mask().sum()
    y.backward()

    assert x.grad.tolist() == [
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ]

def test_no_grad_causal_mask():
    x = Tensor.from_nested([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    with Tensor.no_grad():
        y = x.apply_causal_mask()
    assert y.requires_grad is False
    assert y._prev == ()

def test_softmax_backward_sum_zero():
    x = Tensor.from_nested([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    y = x.softmax(axis=-1).sum()
    y.backward()

    for row in x.grad.tolist():
        for g in row:
            assert abs(g) < 1e-9

def test_softmax_backward_weighted():
    x = Tensor.from_nested([[0.0, 0.0]], requires_grad=True)
    w = Tensor.from_nested([[1.0, 2.0]])

    y = (x.softmax(axis=-1) * w).sum()
    y.backward()

    got = x.grad.tolist()[0]
    assert abs(got[0] + 0.25) < 1e-9
    assert abs(got[1] - 0.25) < 1e-9

def test_no_grad_softmax():
    x = Tensor.from_nested([[1.0, 2.0, 3.0]], requires_grad=True)
    with Tensor.no_grad():
        y = x.softmax(axis=-1)
    assert y.requires_grad is False
    assert y._prev == ()

def test_layernorm_backward_sum_zero():
    x = Tensor.from_nested([
        [1.0, 2.0, 4.0],
        [0.5, -1.0, 3.0],
    ], requires_grad=True)

    y = x.layernorm_last_dim().sum()
    y.backward()

    for row in x.grad.tolist():
        for g in row:
            assert abs(g) < 1e-6

def test_layernorm_backward_numerical():
    vals = [1.0, 2.0, 4.0]
    w = [1.0, 0.0, 0.0]

    x = Tensor.from_nested([vals], requires_grad=True)
    wt = Tensor.from_nested([w])

    y = (x.layernorm_last_dim() * wt).sum()
    y.backward()
    got = x.grad.tolist()[0]

    def ln_row(vs, eps=1e-5):
        mean = sum(vs) / len(vs)
        var = sum((v - mean) ** 2 for v in vs) / len(vs)
        invstd = 1.0 / math.sqrt(var + eps)
        return [(v - mean) * invstd for v in vs]

    h = 1e-5
    expected = []
    for i in range(len(vals)):
        vp = vals[:]
        vm = vals[:]
        vp[i] += h
        vm[i] -= h
        fp = sum(a * b for a, b in zip(ln_row(vp), w))
        fm = sum(a * b for a, b in zip(ln_row(vm), w))
        expected.append((fp - fm) / (2.0 * h))

    for g, e in zip(got, expected):
        assert abs(g - e) < 1e-4

def test_no_grad_layernorm():
    x = Tensor.from_nested([[1.0, 2.0, 3.0]], requires_grad=True)
    with Tensor.no_grad():
        y = x.layernorm_last_dim()
    assert y.requires_grad is False
    assert y._prev == ()

# def test_attention_backward_smoke():
#     q = Tensor.from_nested([[[[1.0], [2.0], [3.0]]]], requires_grad=True)
#     k = Tensor.from_nested([[[[1.0], [0.0], [1.0]]]], requires_grad=True)
#     v = Tensor.from_nested([[[[10.0], [20.0], [30.0]]]], requires_grad=True)

#     y = attention(q, k, v, mask=True).sum()
#     y.backward()

#     assert q.grad is not None and q.grad.shape == q.shape
#     assert k.grad is not None and k.grad.shape == k.shape
#     assert v.grad is not None and v.grad.shape == v.shape

# def test_transformer_block_backward_smoke():
#     x = Tensor.from_nested([[
#         [1.0, 2.0, 3.0, 4.0],
#         [5.0, 6.0, 7.0, 8.0],
#         [9.0, 10.0, 11.0, 12.0],
#     ]], requires_grad=True)

#     Wq = Tensor.zeros((4, 4), requires_grad=True)
#     Wk = Tensor.zeros((4, 4), requires_grad=True)
#     Wv = Tensor.zeros((4, 4), requires_grad=True)
#     Wo = Tensor.zeros((4, 4), requires_grad=True)
#     W1 = Tensor.zeros((4, 8), requires_grad=True)
#     W2 = Tensor.zeros((8, 4), requires_grad=True)

#     y = transformer_block_forward(x, Wq, Wk, Wv, Wo, W1, W2, n_heads=2).sum()
#     y.backward()

#     assert x.grad is not None and x.grad.shape == x.shape
#     assert Wq.grad is not None and Wq.grad.shape == Wq.shape
#     assert Wk.grad is not None and Wk.grad.shape == Wk.shape
#     assert Wv.grad is not None and Wv.grad.shape == Wv.shape
#     assert Wo.grad is not None and Wo.grad.shape == Wo.shape
#     assert W1.grad is not None and W1.grad.shape == W1.shape
#     assert W2.grad is not None and W2.grad.shape == W2.shape

def test_item_neg_sub_div():
    x = Tensor.from_nested([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = ((-x) + 10.0 - 1.0) / 2.0
    assert y.tolist() == [[4.0, 3.5], [3.0, 2.5]]

    loss = y.sum()
    loss.backward()
    assert x.grad.tolist() == [[-0.5, -0.5], [-0.5, -0.5]]

def test_embedding_shape():
    emb = Embedding(vocab_size=5, d_model=3)
    x = emb([1, 3, 4])
    assert x.shape == (3, 3)

def test_embedding_backward():
    emb = Embedding(vocab_size=5, d_model=2)
    y = emb([2, 1, 2]).sum()
    y.backward()

    g = emb.weight.grad.tolist()
    assert g[0] == [0.0, 0.0]
    assert g[1] == [1.0, 1.0]
    assert g[2] == [2.0, 2.0]
    assert g[3] == [0.0, 0.0]
    assert g[4] == [0.0, 0.0]

def test_embedding_batched_shape():
    emb = Embedding(vocab_size=10, d_model=4)
    x = emb([[1, 2, 3], [4, 5, 6]])
    assert x.shape == (2, 3, 4)

def test_linear_shape():
    lin = Linear(4, 6)
    x = Tensor.from_nested([
        [[1.0, 2.0, 3.0, 4.0],
         [5.0, 6.0, 7.0, 8.0]]
    ])
    y = lin(x)
    assert y.shape == (1, 2, 6)

def test_linear_backward_smoke():
    lin = Linear(3, 2)
    x = Tensor.from_nested([[1.0, 2.0, 3.0]], requires_grad=True)
    y = lin(x).sum()
    y.backward()

    assert x.grad is not None and x.grad.shape == x.shape
    assert lin.W.grad is not None and lin.W.grad.shape == lin.W.shape
    assert lin.b.grad is not None and lin.b.grad.shape == lin.b.shape

def test_cross_entropy_loss_value():
    logits = Tensor.from_nested([[0.0, 0.0]])
    loss = cross_entropy_loss(logits, [0])
    assert abs(loss.item() - math.log(2.0)) < 1e-9

def test_cross_entropy_backward():
    logits = Tensor.from_nested([[0.0, 0.0]], requires_grad=True)
    loss = cross_entropy_loss(logits, [0])
    loss.backward()

    got = logits.grad.tolist()[0]
    assert abs(got[0] + 0.5) < 1e-9
    assert abs(got[1] - 0.5) < 1e-9

def test_cross_entropy_batched_shape():
    logits = Tensor.from_nested([
        [[2.0, 1.0, 0.0], [0.5, 0.0, -1.0]],
        [[1.0, 3.0, 2.0], [2.0, 2.0, 2.0]],
    ], requires_grad=True)

    loss = cross_entropy_loss(logits, [[0, 1], [1, 2]])
    loss.backward()

    assert loss.shape == ()
    assert logits.grad is not None and logits.grad.shape == logits.shape

def test_no_grad_embedding():
    emb = Embedding(vocab_size=5, d_model=3)
    with Tensor.no_grad():
        y = emb([1, 2, 3])
    assert y.requires_grad is False
    assert y._prev == ()

def test_no_grad_cross_entropy():
    logits = Tensor.from_nested([[1.0, 2.0, 3.0]], requires_grad=True)
    with Tensor.no_grad():
        loss = cross_entropy_loss(logits, [2])
    assert loss.requires_grad is False
    assert loss._prev == ()

def test_sgd_step():
    x = Tensor.from_nested([[1.0, 2.0]], requires_grad=True)
    y = (x * 3.0).sum()
    y.backward()

    before = x.tolist()
    sgd_step([x], lr=0.1)
    after = x.tolist()

    assert before != after
    assert after == [[1.0 - 0.3, 2.0 - 0.3]]

def test_multihead_attention_module_smoke():
    mha = MultiHeadAttention(d_model=4, n_heads=2)
    x = Tensor.from_nested([[
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
    ]], requires_grad=True)

    y = mha(x)
    assert y.shape == (1, 3, 4)

    loss = y.sum()
    loss.backward()

    assert x.grad is not None and x.grad.shape == x.shape
    for p in mha.parameters():
        assert p.grad is not None and p.grad.shape == p.shape


def test_transformer_block_module_smoke():
    block = TransformerBlock(d_model=4, n_heads=2, d_ff=8)
    x = Tensor.from_nested([[
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
    ]], requires_grad=True)

    y = block(x)
    assert y.shape == (1, 3, 4)

    loss = y.sum()
    loss.backward()

    assert x.grad is not None and x.grad.shape == x.shape
    for p in block.parameters():
        assert p.grad is not None and p.grad.shape == p.shape


def test_tinygpt_forward_loss_generate():
    model = TinyGPT(
        vocab_size=11,
        block_size=4,
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=2,
    )

    xb = [[1, 2, 3, 4], [4, 3, 2, 1]]
    yb = [[2, 3, 4, 5], [3, 2, 1, 0]]

    logits = model(xb)
    assert logits.shape == (2, 4, 11)

    loss = model.loss(xb, yb)
    assert loss.shape == ()

    loss.backward()
    for p in model.parameters():
        assert p.grad is not None and p.grad.shape == p.shape

    out = model.generate([[1, 2, 3]], max_new_tokens=2)
    assert len(out) == 1
    assert len(out[0]) == 5

def test_get_batch_shapes():
    data = list(range(30))
    xb, yb = get_batch(data, batch_size=4, block_size=5)

    assert len(xb) == 4
    assert len(yb) == 4

    for xrow, yrow in zip(xb, yb):
        assert len(xrow) == 5
        assert len(yrow) == 5
        assert yrow[:-1] == xrow[1:]

def test_overfit_one_batch_smoke():
    random.seed(0)

    data = [0, 1, 2, 3, 4] * 40

    model = TinyGPT(
        vocab_size=5,
        block_size=4,
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=1,
    )

    xb, yb = get_batch(data, batch_size=4, block_size=4)
    params = model.parameters()

    start = model.loss(xb, yb).item()

    for _ in range(80):
        loss = model.loss(xb, yb)
        zero_grad_all(params)
        loss.backward()
        sgd_step(params, lr=0.05)

    end = model.loss(xb, yb).item()
    assert end < start

def test_bpe_roundtrip_ascii():
    text = "hello hello world\n"
    tok = ByteBPETokenizer().train(text, vocab_size=300)
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_bpe_roundtrip_unicode():
    text = "héllo 🌍\nhello again\n"
    tok = ByteBPETokenizer().train(text, vocab_size=320)
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_bpe_dataset_prep():
    text = "to be or not to be, that is the question"
    tok, all_ids, train_ids, val_ids = build_tokenizer_and_dataset(
        text,
        vocab_size=280,
        train_frac=0.8,
        min_pair_count=2,
        verbose=False,
    )

    assert len(all_ids) == len(train_ids) + len(val_ids)
    assert len(train_ids) > 0
    assert len(val_ids) > 0


def test_bpe_save_load():
    text = "abc abc abc"
    tok = ByteBPETokenizer().train(text, vocab_size=270)
    tok.save("tmp_tokenizer.json")

    tok2 = ByteBPETokenizer.load("tmp_tokenizer.json")
    assert tok2.decode(tok2.encode(text)) == text
    assert tok2.vocab_size == tok.vocab_size


def test_tinygpt_with_tokenizer_shapes():
    text = "hello world hello world"
    tok, _, train_ids, val_ids = build_tokenizer_and_dataset(
        text,
        vocab_size=256,   # no merges; pure byte-level tokenizer
        train_frac=0.9,
        verbose=False,
    )

    model = TinyGPT(
        vocab_size=tok.vocab_size,
        block_size=4,
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=1,
    )

    xb, yb = get_batch(train_ids, batch_size=2, block_size=4)
    logits = model(xb)
    assert logits.shape == (2, 4, tok.vocab_size)
    loss = model.loss(xb, yb)
    assert loss.shape == ()

test_views()
test_aliasing()
test_3d_views()
test_contiguous_is_copy()
test_negative_dims()
test_negative_permute()
test_add_on_view()
test_add_two_views()
test_row_broadcast_add()
test_col_broadcast_add()
test_scalar_left_add()
test_1d_broadcast_add()
test_3d_broadcast_add()
test_broadcast_add_on_noncontiguous_view()
test_mul_on_view()
test_row_broadcast_mul()
test_col_broadcast_mul()
test_3d_broadcast_mul()
test_sum_all()
test_sum_axis0()
test_sum_axis1_keepdims()
test_sum_on_view()
test_sum_multi_axis()
test_matmul_basic()
test_matmul_with_view()
test_matmul_shape_error()
test_mean_axis0()
test_sum_keepdims_multi_axis()
test_expand_row()
test_expand_vector()
test_expand_aliasing()
test_expand_column()
test_expand_invalid()
test_expand_view_addresses()
test_batched_matmul_basic()
test_batched_matmul_broadcast_rhs()
test_batched_matmul_with_view()
test_batched_matmul_batch_shape_error()
test_softmax_last_dim()
test_softmax_on_view()
test_causal_mask_2d()
test_causal_mask_batched()
# test_attention_causal_tiny()
# test_attention_no_mask_tiny()
# test_mha_shape()
test_layernorm_last_dim()
test_layernorm_on_view()
test_relu()
# test_transformer_block_shape()
test_expand_backward()
test_add_backward()
test_mul_backward_scalar()
test_radd_backward()
test_rmul_backward()
test_no_grad_expand()
test_relu_backward()
test_sum_backward_axis0()
test_add_backward_row_broadcast()
test_mul_backward_row_broadcast()
test_no_grad_add()
test_reshape_backward_sum()
test_reshape_backward_layout()
test_no_grad_reshape()
test_permute_backward_sum()
test_permute_backward_layout()
test_no_grad_permute()
test_narrow_backward_sum()
test_narrow_backward_layout()
test_no_grad_narrow()
test_select_backward_sum()
test_select_backward_layout()
test_select_backward_negative_index()
test_no_grad_select()
test_contiguous_backward_sum()
test_contiguous_backward_layout()
test_no_grad_contiguous()
test_matmul_backward_basic()
test_matmul_backward_with_view()
test_matmul_backward_broadcast_rhs_input_grad()
test_matmul_backward_broadcast_rhs_weight_grad()
test_no_grad_matmul()
test_causal_mask_backward()
test_no_grad_causal_mask()
test_softmax_backward_sum_zero()
test_softmax_backward_weighted()
test_no_grad_softmax()
test_layernorm_backward_sum_zero()
test_layernorm_backward_numerical()
test_no_grad_layernorm()
# test_attention_backward_smoke()
# test_transformer_block_backward_smoke()
test_item_neg_sub_div()
test_embedding_shape()
test_embedding_backward()
test_embedding_batched_shape()
test_linear_shape()
test_linear_backward_smoke()
test_cross_entropy_loss_value()
test_cross_entropy_backward()
test_cross_entropy_batched_shape()
test_no_grad_embedding()
test_no_grad_cross_entropy()
test_sgd_step()
test_multihead_attention_module_smoke()
test_transformer_block_module_smoke()
test_tinygpt_forward_loss_generate()
test_get_batch_shapes()
test_overfit_one_batch_smoke()
test_bpe_roundtrip_ascii()
test_bpe_roundtrip_unicode()
test_bpe_dataset_prep()
test_bpe_save_load()
test_tinygpt_with_tokenizer_shapes()
print("All tests passed!")
