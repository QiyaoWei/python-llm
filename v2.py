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
        if any(s <= 0 for s in self.strides):
            raise ValueError("strides must be positive")
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

test_views()
print("All tests passed!")
