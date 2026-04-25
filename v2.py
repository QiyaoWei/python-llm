import math

class Tensor:
    __slots__ = ("storage", "shape", "strides", "offset",
        "requires_grad", "grad", "_prev", "_op", "_backward",)
    
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
    
    def contiguous(self):
        flat = []

        def collect(prefix, dim):
            if dim == len(self.shape):
                flat.append(self.get(*prefix))
                return
            for i in range(self.shape[dim]):
                collect(prefix + (i,), dim + 1)

        collect((), 0)
        return Tensor(flat, shape=self.shape, requires_grad=self.requires_grad)
    
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
        return Tensor(
            self.storage,
            shape=shape,
            strides=self._default_strides(shape),
            offset=self.offset,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="reshape",
        )
    
    def permute(self, *dims):
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        else:
            dims = tuple(dims)

        ndim = len(self.shape)
        dims = tuple(self._canon_dim(d, ndim) for d in dims)

        if sorted(dims) != list(range(len(self.shape))):
            raise ValueError("Invalid permutation of dimensions")
        
        return Tensor(
            self.storage,
            shape=tuple(self.shape[d] for d in dims),
            strides=tuple(self.strides[d] for d in dims),
            offset=self.offset,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="permute",
        )
    
    def transpose(self, dim0, dim1):
        ndim = len(self.shape)
        dim0 = self._canon_dim(dim0, ndim)
        dim1 = self._canon_dim(dim1, ndim)
        dims = list(range(ndim))
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        return self.permute(dims)
    
    def narrow(self, dim, start, length):
        dim = self._canon_dim(dim, len(self.shape))
        if start < 0 or length < 0 or start + length > self.shape[dim]:
            raise ValueError("Invalid start or length for narrow")
        
        new_shape = list(self.shape)
        new_shape[dim] = length
        new_offset = self.offset + start * self.strides[dim]
        return Tensor(
            self.storage,
            shape=tuple(new_shape),
            strides=self.strides,
            offset=new_offset,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="narrow",
        )
    
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

    # @staticmethod
    # def _broadcast_index(out_idx, in_shape):
    #     pad = len(out_idx) - len(in_shape)
    #     idx = []

    #     for i, dim in enumerate(in_shape):
    #         oi = out_idx[pad + i]
    #         idx.append(0 if dim == 1 else oi)

    #     return tuple(idx)
    
    def _expand_to(self, shape):
        shape = tuple(shape)
        return self if self.shape == shape else self.expand(shape)

    def __add__(self, other):
        other = self._coerce(other)
        out_shape = self._broadcast_shape(self.shape, other.shape)

        a = self._expand_to(out_shape)
        b = other._expand_to(out_shape)

        out = Tensor.zeros(
            out_shape,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        for idx in self._iter_indices(out_shape):
            out.storage[out._storage_index(idx)] = a.get(*idx) + b.get(*idx)

        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = self._coerce(other)
        out_shape = self._broadcast_shape(self.shape, other.shape)

        a = self._expand_to(out_shape)
        b = other._expand_to(out_shape)

        out = Tensor.zeros(
            out_shape,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        for idx in self._iter_indices(out_shape):
            out.storage[out._storage_index(idx)] = a.get(*idx) * b.get(*idx)

        return out

    __rmul__ = __mul__

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
    
    def sum(self, axis=None, keepdims=False):
        ndim = len(self.shape)
        axes = self._canon_axes(axis, ndim)

        if keepdims:
            out_shape = tuple(1 if i in axes else self.shape[i] for i in range(ndim))
        else:
            out_shape = tuple(self.shape[i] for i in range(ndim) if i not in axes)

        out = Tensor.zeros(out_shape, requires_grad=self.requires_grad)

        for idx in self._iter_indices(self.shape):
            if keepdims:
                out_idx = tuple(0 if i in axes else idx[i] for i in range(ndim))
            else:
                out_idx = tuple(idx[i] for i in range(ndim) if i not in axes)

            out.storage[out._storage_index(out_idx)] += self.get(*idx)

        return out
    
    def mean(self, axis=None, keepdims=False):
        ndim = len(self.shape)
        axes = self._canon_axes(axis, ndim)

        denom = 1
        for a in axes:
            denom *= self.shape[a]

        return self.sum(axis=axes, keepdims=keepdims) * (1.0 / denom)
    
    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        else:
            shape = tuple(shape)

        if len(shape) < len(self.shape):
            raise ValueError("cannot expand to fewer dimensions")

        # Right-align the old shape with the new one.
        pad = len(shape) - len(self.shape)
        old_shape = (1,) * pad + self.shape
        old_strides = (0,) * pad + self.strides

        new_strides = []
        for old_dim, old_stride, new_dim in zip(old_shape, old_strides, shape):
            if new_dim < 0:
                raise ValueError("expand dims must be non-negative")

            if old_dim == new_dim:
                # same size: keep the old stride
                new_strides.append(old_stride)
            elif old_dim == 1:
                # expanded singleton dimension: stride 0
                new_strides.append(0)
            else:
                raise ValueError(f"cannot expand shape {self.shape} to {shape}")

        return Tensor(
            self.storage,
            shape=shape,
            strides=tuple(new_strides),
            offset=self.offset,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="expand",
        )
    
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
        out = Tensor.zeros(
            out_shape,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        for batch_idx in self._iter_indices(batch_shape):
            for i in range(M):
                for j in range(N):
                    acc = 0.0
                    for k in range(K):
                        acc += a.get(*(batch_idx + (i, k))) * b.get(*(batch_idx + (k, j)))
                    out.storage[out._storage_index(batch_idx + (i, j))] = acc

        return out
    
    def softmax(self, axis=-1):
        ndim = len(self.shape)
        axis = self._canon_dim(axis, ndim)

        out = Tensor.zeros(self.shape, requires_grad=self.requires_grad)

        outer_shape = self.shape[:axis] + self.shape[axis + 1:]
        n = self.shape[axis]

        for outer_idx in self._iter_indices(outer_shape):
            def full_idx(j):
                return outer_idx[:axis] + (j,) + outer_idx[axis:]

            m = max(self.get(*full_idx(j)) for j in range(n))
            exps = [math.exp(self.get(*full_idx(j)) - m) for j in range(n)]
            s = sum(exps)

            for j, e in enumerate(exps):
                out.storage[out._storage_index(full_idx(j))] = e / s

        return out
    
    def apply_causal_mask(self, mask_value=float("-inf")):
        if len(self.shape) < 2:
            raise ValueError("causal mask requires rank >= 2")

        T1, T2 = self.shape[-2:]
        if T1 != T2:
            raise ValueError("last two dims must form a square matrix")

        out = Tensor.zeros(self.shape, requires_grad=self.requires_grad)

        for prefix in self._iter_indices(self.shape[:-2]):
            for i in range(T1):
                for j in range(T2):
                    idx = prefix + (i, j)
                    out.storage[out._storage_index(idx)] = (
                        self.get(*idx) if j <= i else mask_value
                    )

        return out
    
    def relu(self):
        out = Tensor.zeros(self.shape, requires_grad=self.requires_grad)
        for idx in self._iter_indices(self.shape):
            x = self.get(*idx)
            out.storage[out._storage_index(idx)] = x if x > 0.0 else 0.0
        return out
    
    def layernorm_last_dim(self, eps=1e-5):
        if len(self.shape) == 0:
            raise ValueError("layernorm requires rank >= 1")

        N = self.shape[-1]
        out = Tensor.zeros(self.shape, requires_grad=self.requires_grad)

        for outer_idx in self._iter_indices(self.shape[:-1]):
            vals = [self.get(*(outer_idx + (j,))) for j in range(N)]
            mean = sum(vals) / N
            var = sum((x - mean) ** 2 for x in vals) / N
            invstd = 1.0 / math.sqrt(var + eps)

            for j, x in enumerate(vals):
                out.storage[out._storage_index(outer_idx + (j,))] = (x - mean) * invstd

        return out
    
def attention(q, k, v, mask=True):
    Dh = q.shape[-1]
    scores = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(Dh))

    if mask:
        scores = scores.apply_causal_mask()

    weights = scores.softmax(axis=-1)
    return weights @ v

def mha_forward(x, Wq, Wk, Wv, Wo, n_heads):
    B, T, C = x.shape
    assert C % n_heads == 0
    Dh = C // n_heads

    q = (x @ Wq).reshape(B, T, n_heads, Dh).permute(0, 2, 1, 3)
    k = (x @ Wk).reshape(B, T, n_heads, Dh).permute(0, 2, 1, 3)
    v = (x @ Wv).reshape(B, T, n_heads, Dh).permute(0, 2, 1, 3)

    out = attention(q, k, v, mask=True)                  # (B, H, T, Dh)
    out = out.permute(0, 2, 1, 3).contiguous()           # (B, T, H, Dh)
    out = out.reshape(B, T, C)                           # (B, T, C)
    return out @ Wo

def transformer_block_forward(x, Wq, Wk, Wv, Wo, W1, W2, n_heads):
    x = x + mha_forward(x.layernorm_last_dim(), Wq, Wk, Wv, Wo, n_heads)
    x = x + ((x.layernorm_last_dim() @ W1).relu() @ W2)
    return x
    
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

def test_attention_causal_tiny():
    q = Tensor.from_nested([[[[1.0], [1.0], [1.0]]]])   # shape (1, 1, 3, 1)
    k = Tensor.from_nested([[[[1.0], [1.0], [1.0]]]])   # shape (1, 1, 3, 1)
    v = Tensor.from_nested([[[[10.0], [20.0], [30.0]]]])# shape (1, 1, 3, 1)

    out = attention(q, k, v, mask=True)
    vals = out.tolist()

    assert out.shape == (1, 1, 3, 1)
    assert abs(vals[0][0][0][0] - 10.0) < 1e-9
    assert abs(vals[0][0][1][0] - 15.0) < 1e-9
    assert abs(vals[0][0][2][0] - 20.0) < 1e-9

def test_attention_no_mask_tiny():
    q = Tensor.from_nested([[[[1.0], [1.0], [1.0]]]])
    k = Tensor.from_nested([[[[1.0], [1.0], [1.0]]]])
    v = Tensor.from_nested([[[[10.0], [20.0], [30.0]]]])

    out = attention(q, k, v, mask=False).tolist()

    for t in range(3):
        assert abs(out[0][0][t][0] - 20.0) < 1e-9

def test_mha_shape():
    x = Tensor.from_nested([[
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
    ]])  # shape (1, 3, 4)

    I = Tensor.from_nested([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    out = mha_forward(x, I, I, I, I, n_heads=2)
    assert out.shape == (1, 3, 4)

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

def test_transformer_block_shape():
    x = Tensor.from_nested([[
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
    ]])  # shape (1, 3, 4)

    Z44 = Tensor.zeros((4, 4))
    Z48 = Tensor.zeros((4, 8))
    Z84 = Tensor.zeros((8, 4))

    out = transformer_block_forward(x, Z44, Z44, Z44, Z44, Z48, Z84, n_heads=2)
    assert out.shape == (1, 3, 4)
    assert out.tolist() == x.tolist()

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
test_attention_causal_tiny()
test_attention_no_mask_tiny()
test_mha_shape()
test_layernorm_last_dim()
test_layernorm_on_view()
test_relu()
test_transformer_block_shape()
print("All tests passed!")
