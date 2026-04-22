from contextlib import contextmanager
import random
import math

class Tensor2D:

    __slots__ = ("data", "shape", "requires_grad", "grad", "_prev", "_op", "_backward")
    grad_enabled = True

    def __init__(self, data, shape=None, _children=(), _op="", requires_grad=False):
        if shape is None:
            M = len(data)
            N = len(data[0]) if M > 0 else 0
            for row in data:
                if len(row) != N:
                    raise ValueError("All rows must have the same number of columns")
                
            flat = []
            for row in data:
                flat.extend(float(x) for x in row)
            shape = (M, N)
        else:
            M, N = shape
            flat = [float(x) for x in data]
            if len(flat) != M * N:
                raise ValueError("Data length does not match shape")
            
        self.data = flat
        self.shape = shape
        self.requires_grad = requires_grad
        self.grad = None
        self._prev = tuple(_children)
        self._op = _op
        self._backward = lambda: None

    @classmethod
    def zeros(cls, shape, requires_grad=False):
        M, N = shape
        return cls([0.0] * (M * N), shape=shape, requires_grad=requires_grad)
    
    @classmethod
    def full(cls, shape, fill_value, requires_grad=False):
        M, N = shape
        return cls([float(fill_value)] * (M * N), shape=shape, requires_grad=requires_grad)
    
    @classmethod
    def random_uniform(cls, shape, low=0.0, high=1.0, requires_grad=False):
        M, N = shape
        data = [random.uniform(low, high) for _ in range(M * N)]
        return cls(data, shape=shape, requires_grad=requires_grad)
    
    @staticmethod
    def _coerce(value):
        if isinstance(value, Tensor2D):
            return value
        elif isinstance(value, (int, float)):
            return Tensor2D([float(value)], shape=(1, 1), requires_grad=False)
        raise TypeError(f"Cannot coerce {value} to Tensor2D")
    
    @classmethod
    @contextmanager
    def no_grad(cls):
        old_state = cls.grad_enabled
        cls.grad_enabled = False
        try:
            yield
        finally:
            cls.grad_enabled = old_state

    @classmethod
    def _should_track(cls, parents):
        return cls.grad_enabled and any(p.requires_grad for p in parents)
    
    @classmethod
    def _make_out(cls, data, shape, parents, op):
        track = cls._should_track(parents)
        return cls(
            data,
            shape=shape,
            _children=parents if track else (),
            _op=op,
            requires_grad=track,
        ), track
    
    def _ensure_grad(self):
        if self.grad is None:
            M, N = self.shape
            self.grad = [0.0] * (M * N)
        return self.grad
    
    def item(self):
        if self.shape != (1, 1):
            raise ValueError("item() can only be called on a tensor with shape (1, 1)")
        return self.data[0]
    
    def row(self, i):
        M, N = self.shape
        if i < 0:
            i += M
        if i < 0 or i >= M:
            raise IndexError("Row index out of range")
        start = i * N
        end = start + N
        return self.data[start:end]
    
    def rows(self):
        M, N = self.shape
        return [self.data[i * N:(i + 1) * N] for i in range(M)]
    
    def detach(self):
        return Tensor2D(self.data[:], shape=self.shape, requires_grad=False)
    
    def zero_grad(self):
        if self.grad is not None:
            for i in range(len(self.grad)):
                self.grad[i] = 0.0

    def backward(self):

        if self.shape != (1, 1):
            raise ValueError("backward() can only be called on a tensor with shape (1, 1)")
        if not self.requires_grad:
            raise ValueError("Cannot call backward() on a tensor that does not require grad")
        
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = [1.0]
        for v in reversed(topo):
            v._backward()

    @staticmethod
    def _add_case(a_shape, b_shape):
        M, N = a_shape
        if a_shape == b_shape:
            return "same", a_shape
        if b_shape == (1, 1):
            return "other_scalar", a_shape
        if a_shape == (1, 1):
            return "self_scalar", b_shape
        if b_shape == (1, N):
            return "other_row", a_shape
        if a_shape[0] == 1 and a_shape[1] == b_shape[1]:
            return "self_row", b_shape
        raise ValueError(f"Shapes {a_shape} and {b_shape} are not compatible for addition")
    
    def __add__(self, other):
        other = Tensor2D._coerce(other)
        case, out_shape = Tensor2D._add_case(self.shape, other.shape)
        M, N = out_shape
        a = self.data
        b = other.data
        out_data = [0.0] * (M * N)

        if case == "same":
            for i in range(M * N):
                out_data[i] = a[i] + b[i]
        elif case == "other_scalar":
            scalar = b[0]
            for i in range(M * N):
                out_data[i] = a[i] + scalar
        elif case == "self_scalar":
            scalar = a[0]
            for i in range(M * N):
                out_data[i] = scalar + b[i]
        elif case == "other_row":
            for i in range(M):
                row_offset = i * N
                for j in range(N):
                    out_data[row_offset + j] = a[row_offset + j] + b[j]
        elif case == "self_row":
            for i in range(M):
                row_offset = i * N
                for j in range(N):
                    out_data[row_offset + j] = a[j] + b[row_offset + j]
        
        out, requires_grad = Tensor2D._make_out(out_data, out_shape, (self, other), "add")
        if not requires_grad:
            return out
        
        def _backward():
            og = out.grad
            if self.requires_grad:
                sg = self._ensure_grad()
                if case in ("same", "other_scalar", "other_row"):
                    for i in range(M * N):
                        sg[i] += og[i]
                elif case == "self_scalar":
                    scalar_grad = sum(og)
                    sg[0] += scalar_grad
                elif case == "self_row":
                    for i in range(M):
                        row_offset = i * N
                        for j in range(N):
                            sg[j] += og[row_offset + j]
            if other.requires_grad:
                pg = other._ensure_grad()
                if case in ("same", "self_scalar", "self_row"):
                    for i in range(M * N):
                        pg[i] += og[i]
                elif case == "other_scalar":
                    scalar_grad = sum(og)
                    pg[0] += scalar_grad
                elif case == "other_row":
                    for i in range(M):
                        row_offset = i * N
                        for j in range(N):
                            pg[j] += og[row_offset + j]
        out._backward = _backward
        return out
    
    __radd__ = __add__

    @staticmethod
    def _mul_case(a_shape, b_shape):
        M, N = a_shape
        if a_shape == b_shape:
            return "same", a_shape
        if b_shape == (1, 1):
            return "other_scalar", a_shape
        if a_shape == (1, 1):
            return "self_scalar", b_shape
        raise ValueError(f"Shapes {a_shape} and {b_shape} are not compatible for multiplication")
    
    def __mul__(self, other):
        other = Tensor2D._coerce(other)
        case, out_shape = Tensor2D._mul_case(self.shape, other.shape)
        M, N = out_shape
        a = self.data
        b = other.data
        out_data = [0.0] * (M * N)

        if case == "same":
            for i in range(M * N):
                out_data[i] = a[i] * b[i]
        elif case == "other_scalar":
            scalar = b[0]
            for i in range(M * N):
                out_data[i] = a[i] * scalar
        elif case == "self_scalar":
            scalar = a[0]
            for i in range(M * N):
                out_data[i] = scalar * b[i]

        out, requires_grad = Tensor2D._make_out(out_data, out_shape, (self, other), "mul")
        if not requires_grad:
            return out
        
        def _backward():
            og = out.grad
            if self.requires_grad:
                sg = self._ensure_grad()
                if case == "same":
                    for i in range(M * N):
                        sg[i] += b[i] * og[i]
                elif case == "other_scalar":
                    scalar = b[0]
                    for i in range(M * N):
                        sg[i] += scalar * og[i]
                elif case == "self_scalar":
                    acc = 0.0
                    for i in range(M * N):
                        acc += b[i] * og[i]
                    sg[0] += acc
            if other.requires_grad:
                pg = other._ensure_grad()
                if case == "same":
                    for i in range(M * N):
                        pg[i] += a[i] * og[i]
                elif case == "other_scalar":
                    acc = 0.0
                    for i in range(M * N):
                        acc += a[i] * og[i]
                    pg[0] += acc
                elif case == "self_scalar":
                    scalar = a[0]
                    for i in range(M * N):
                        pg[i] += scalar * og[i]
        out._backward = _backward
        return out
    
    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-1.0) * other

    def __rsub__(self, other):
        return Tensor2D._coerce(other) + (-1.0) * self
    
    def __matmul__(self, other):

        if not isinstance(other, Tensor2D):
            raise TypeError("Matrix multiplication is only supported between Tensor2D instances")
        M, K = self.shape
        K2, N = other.shape
        if K != K2:
            raise TypeError(f"Shapes {self.shape} and {other.shape} are not compatible for matrix multiplication")
        
        a = self.data
        b = other.data
        out_data = [0.0] * (M * N)

        for i in range(M):
            for j in range(N):
                acc = 0.0
                for k in range(K):
                    acc += a[i * K + k] * b[k * N + j]
                out_data[i * N + j] = acc

        out, requires_grad = Tensor2D._make_out(out_data, (M, N), (self, other), "matmul")
        if not requires_grad:
            return out

        def _backward():
            og = out.grad
            if self.requires_grad:
                sg = self._ensure_grad()
                for i in range(M):
                    for j in range(K):
                        acc = 0.0
                        for n in range(N):
                            acc += og[i * N + n] * b[j * N + n]
                        sg[i * K + j] += acc
            if other.requires_grad:
                pg = other._ensure_grad()
                for j in range(N):
                    for k in range(K):
                        acc = 0.0
                        for m in range(M):
                            acc += og[m * N + j] * a[m * K + k]
                        pg[k * N + j] += acc
        out._backward = _backward
        return out
    
    def T(self):
        M, N = self.shape
        out_data = [0.0] * (M * N)
        for i in range(M):
            for j in range(N):
                out_data[j * M + i] = self.data[i * N + j]
        out, requires_grad = Tensor2D._make_out(out_data, (N, M), (self,), "transpose")
        if not requires_grad:
            return out
        
        def _backward():
            og = out.grad
            if self.requires_grad:
                sg = self._ensure_grad()
                for i in range(M):
                    for j in range(N):
                        sg[i * N + j] += og[j * M + i]
        out._backward = _backward
        return out
    
    def relu(self):
        M, N = self.shape
        out_data = [0.0] * (M * N)
        for i in range(M * N):
            out_data[i] = max(0.0, self.data[i])
        out, requires_grad = Tensor2D._make_out(out_data, self.shape, (self,), "relu")
        if not requires_grad:
            return out
        
        def _backward():
            og = out.grad
            if self.requires_grad:
                sg = self._ensure_grad()
                for i in range(M * N):
                    if self.data[i] > 0.0:
                        sg[i] += og[i]
        out._backward = _backward
        return out
    
    def softmax_row(self):
        M, N = self.shape
        out_data = [0.0] * (M * N)
        for i in range(M):
            base = i * N
            row = self.data[base:base + N]
            max_val = max(row)
            exps = [math.exp(x - max_val) for x in row]
            sum_exps = sum(exps)
            inv = 1.0 / sum_exps
            for j in range(N):
                out_data[base + j] = exps[j] * inv
        out, requires_grad = Tensor2D._make_out(out_data, (M, N), (self,), "softmax_row")
        if not requires_grad:
            return out
        
        def _backward():
            sg = self._ensure_grad()
            og = out.grad
            od = out.data

            for i in range(M):
                base = i * N
                dot = 0.0
                for j in range(N):
                    dot += od[base + j] * og[base + j]
                for j in range(N):
                    p = od[base + j]
                    sg[base + j] += p * (og[base + j] - dot)

        out._backward = _backward
        return out
    
    def layernorm(self, gamma=None, beta=None, eps=1e-5):
        M, N = self.shape

        if gamma is not None and gamma.shape != (1, N):
            raise ValueError(f"gamma must have shape (1, {N})")
        if beta is not None and beta.shape != (1, N):
            raise ValueError(f"beta must have shape (1, {N})")

        gamma_data = gamma.data if gamma is not None else [1.0] * N
        beta_data = beta.data if beta is not None else [0.0] * N

        xhat_flat = [0.0] * (M * N)
        invstds = [0.0] * M
        out_flat = [0.0] * (M * N)

        for i in range(M):
            base = i * N
            row = self.data[base:base + N]
            mean = sum(row) / N
            var = sum((x - mean) ** 2 for x in row) / N
            invstd = 1.0 / math.sqrt(var + eps)
            invstds[i] = invstd

            for j in range(N):
                xhat = (row[j] - mean) * invstd
                xhat_flat[base + j] = xhat
                out_flat[base + j] = xhat * gamma_data[j] + beta_data[j]

        parents = [self]
        if gamma is not None:
            parents.append(gamma)
        if beta is not None:
            parents.append(beta)

        out, track = Tensor2D._make_out(out_flat, (M, N), tuple(parents), 'layernorm')
        if not track:
            return out

        def _backward():
            og = out.grad
            sg = self._ensure_grad() if self.requires_grad else None
            gg = gamma._ensure_grad() if gamma is not None and gamma.requires_grad else None
            bg = beta._ensure_grad() if beta is not None and beta.requires_grad else None

            for i in range(M):
                base = i * N
                dxhat = [0.0] * N
                mean_dxhat = 0.0
                mean_dxhat_xhat = 0.0

                for j in range(N):
                    g = og[base + j]
                    xhat = xhat_flat[base + j]

                    if gg is not None:
                        gg[j] += g * xhat
                    if bg is not None:
                        bg[j] += g

                    dx = g * gamma_data[j]
                    dxhat[j] = dx
                    mean_dxhat += dx
                    mean_dxhat_xhat += dx * xhat

                mean_dxhat /= N
                mean_dxhat_xhat /= N
                invstd = invstds[i]

                if sg is not None:
                    for j in range(N):
                        sg[base + j] += (
                            dxhat[j] - mean_dxhat - xhat_flat[base + j] * mean_dxhat_xhat
                        ) * invstd

        out._backward = _backward
        return out
    
    def apply_causal_mask(self):
        M, N = self.shape
        if M != N:
            raise ValueError("causal mask expects a square (T, T) tensor")

        out_flat = self.data[:]
        neg_inf = float('-inf')

        for i in range(M):
            base = i * N
            for j in range(i + 1, N):
                out_flat[base + j] = neg_inf

        out, track = Tensor2D._make_out(out_flat, (M, N), (self,), 'causal_mask')
        if not track:
            return out

        def _backward():
            sg = self._ensure_grad()
            og = out.grad
            for i in range(M):
                base = i * N
                for j in range(i + 1):
                    sg[base + j] += og[base + j]

        out._backward = _backward
        return out
    
    def slice_cols(self, start, end):
        M, N = self.shape
        if not (0 <= start <= end <= N):
            raise IndexError(f"Column slice [{start}:{end}] out of range for shape {self.shape}")

        width = end - start
        out_data = [0.0] * (M * width)

        for i in range(M):
            src = i * N + start
            dst = i * width
            out_data[dst:dst + width] = self.data[src:src + width]

        out, track = Tensor2D._make_out(
            out_data,
            shape=(M, width),
            parents=(self,),
            op=f"slice_cols[{start}:{end}]",
        )
        if not track:
            return out

        def _backward():
            og = out.grad
            sg = self._ensure_grad()

            for i in range(M):
                src = i * N + start
                dst = i * width
                for j in range(width):
                    sg[src + j] += og[dst + j]

        out._backward = _backward
        return out
    
class Embedding:
    def __init__(self, vocab_size, d_model):
        scale = (6.0 / (vocab_size + d_model)) ** 0.5
        self.W = Tensor2D.random_uniform((vocab_size, d_model), low=-scale, high=scale, requires_grad=True)

    def __call__(self, token_ids):
        vocab_size, d_model = self.W.shape
        T = len(token_ids)
        out_data = [0.0] * (T * d_model)

        for t, token_id in enumerate(token_ids):
            if token_id < 0 or token_id >= vocab_size:
                raise ValueError(f"Token ID {token_id} is out of bounds for vocab size {vocab_size}")
            start = token_id * d_model
            end = t * d_model
            out_data[end:end + d_model] = self.W.data[start:start + d_model]

        out, requires_grad = Tensor2D._make_out(out_data, (T, d_model), (self.W,), "embedding_lookup")
        if not requires_grad:
            return out
        
        def _backward():
            og = out.grad
            Wg = self.W._ensure_grad()
            for t, token_id in enumerate(token_ids):
                start = token_id * d_model
                end = t * d_model
                for i in range(d_model):
                    Wg[start + i] += og[end + i]
        out._backward = _backward
        return out
    
    def parameters(self):
        return [self.W]
    
def cross_entropy_loss(logits, target_ids):
    T, V = logits.shape
    if len(target_ids) != T:
        raise ValueError(f"Length of target_ids {len(target_ids)} does not match number of rows in logits {T}")

    probs = [] if logits.requires_grad and Tensor2D.grad_enabled else None
    total = 0.0

    for i, tgt in enumerate(target_ids):
        if not (0 <= tgt < V):
            raise IndexError(f"target id {tgt} out of range [0, {V})")
        base = i * V
        row = logits.data[base:base + V]
        max_val = max(row)
        exps = [math.exp(x - max_val) for x in row]
        sum_exps = sum(exps)
        total += math.log(sum_exps) - (row[tgt] - max_val)
        if probs is not None:
            inv_sum = 1.0 / sum_exps
            probs.append([e * inv_sum for e in exps])

    out, requires_grad = Tensor2D._make_out([total / T], shape=(1, 1), parents=(logits,), op="cross_entropy_loss")
    if not requires_grad:
        return out
    
    def _backward():
        og = out.grad[0] / T
        sg = logits._ensure_grad()
        for i, tgt in enumerate(target_ids):
            base = i * V
            for j in range(V):
                sg[base + j] += probs[i][j] * og
            sg[base + tgt] -= og
    out._backward = _backward
    return out

class Linear:
    def __init__(self, in_features, out_features):
        scale = (6.0 / (in_features + out_features)) ** 0.5
        self.W = Tensor2D.random_uniform((in_features, out_features), low=-scale, high=scale, requires_grad=True)
        self.b = Tensor2D.zeros((1, out_features), requires_grad=True)

    def __call__(self, x):
        return x @ self.W + self.b
    
    def parameters(self):
        return [self.W, self.b]
    
class FeedForward:
    def __init__(self, d_model, d_ff):
        self.lin1 = Linear(d_model, d_ff)
        self.lin2 = Linear(d_ff, d_model)

    def __call__(self, x):
        return self.lin2(self.lin1(x).relu())
    
    def parameters(self):
        return self.lin1.parameters() + self.lin2.parameters()
    
class MultiHeadAttention:
    def __init__(self, d_model, n_heads, head_size):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_size = head_size
        total = n_heads * head_size

        # one big Q, K, V projection each
        self.q_proj = Linear(d_model, total)
        self.k_proj = Linear(d_model, total)
        self.v_proj = Linear(d_model, total)

        # keep output projection as per-head row blocks for now
        scale = (6.0 / (head_size + d_model)) ** 0.5
        self.W_out_blocks = [
            Tensor2D.random_uniform(
                (head_size, d_model),
                low=-scale,
                high=scale,
                requires_grad=True,
            )
            for _ in range(n_heads)
        ]
        self.b_out = Tensor2D.zeros((1, d_model), requires_grad=True)

    def __call__(self, x, mask=True):
        # x: (T, d_model)
        Q = self.q_proj(x)   # (T, n_heads * head_size)
        K = self.k_proj(x)   # (T, n_heads * head_size)
        V = self.v_proj(x)   # (T, n_heads * head_size)

        scale = 1.0 / math.sqrt(self.head_size)
        out = None

        for h in range(self.n_heads):
            start = h * self.head_size
            end = start + self.head_size

            qh = Q.slice_cols(start, end)   # (T, head_size)
            kh = K.slice_cols(start, end)   # (T, head_size)
            vh = V.slice_cols(start, end)   # (T, head_size)

            scores = (qh @ kh.T()) * scale  # (T, T)
            if mask:
                scores = scores.apply_causal_mask()

            weights = scores.softmax_row()  # (T, T)
            head_out = weights @ vh         # (T, head_size)

            proj = head_out @ self.W_out_blocks[h]   # (T, d_model)
            out = proj if out is None else out + proj

        return out + self.b_out

    def parameters(self):
        params = []
        params.extend(self.q_proj.parameters())
        params.extend(self.k_proj.parameters())
        params.extend(self.v_proj.parameters())
        params.extend(self.W_out_blocks)
        params.append(self.b_out)
        return params
    
# class AttentionHead:
#     def __init__(self, d_model, d_head):
#         self.Wq = Linear(d_model, d_head)
#         self.Wk = Linear(d_model, d_head)
#         self.Wv = Linear(d_model, d_head)

#     def __call__(self, x, mask_causal=False):
#         q = self.Wq(x)
#         k = self.Wk(x)
#         v = self.Wv(x)

#         scores = q @ k.T() * (q.shape[1] ** -0.5)
#         if mask_causal:
#             scores = scores.apply_causal_mask()
#         weights = scores.softmax_row()
#         out = weights @ v
#         return out
    
#     def parameters(self):
#         return self.Wq.parameters() + self.Wk.parameters() + self.Wv.parameters()
    
# class MultiHeadAttention:
#     def __init__(self, d_model, n_heads):
#         assert d_model % n_heads == 0
#         head_size = d_model // n_heads
#         self.heads = [AttentionHead(d_model, head_size) for _ in range(n_heads)]
#         scale = (6.0 / (head_size + d_model)) ** 0.5
#         self.W_out_blocks = [
#             Tensor2D.random_uniform(
#                 (head_size, d_model),
#                 low=-scale,
#                 high=scale,
#                 requires_grad=True,
#             )
#             for _ in range(n_heads)
#         ]
#         self.b_out = Tensor2D.zeros((1, d_model), requires_grad=True)

#     def __call__(self, x, mask_causal=False):
#         out = None
#         for head, W_out in zip(self.heads, self.W_out_blocks):
#             head_out = head(x, mask_causal=mask_causal) @ W_out
#             out = head_out if out is None else out + head_out
#         out = out + self.b_out
#         return out
    
#     def parameters(self):
#         params = self.b_out.parameters()
#         for head, W_out in zip(self.heads, self.W_out_blocks):
#             params += head.parameters()
#             params += W_out.parameters()
#         return params

class TransformerBlock:
    def __init__(self, d_model, n_heads, head_size, hidden_dim):
        self.ln1_gamma = Tensor2D.full((1, d_model), 1.0, requires_grad=True)
        self.ln1_beta  = Tensor2D.zeros((1, d_model), requires_grad=True)
        self.ln2_gamma = Tensor2D.full((1, d_model), 1.0, requires_grad=True)
        self.ln2_beta  = Tensor2D.zeros((1, d_model), requires_grad=True)

        self.attn = MultiHeadAttention(d_model, n_heads, head_size)
        self.ff = FeedForward(d_model, hidden_dim)

    def __call__(self, x):
        x = x + self.attn(x.layernorm(self.ln1_gamma, self.ln1_beta), mask=True)
        x = x + self.ff(x.layernorm(self.ln2_gamma, self.ln2_beta))
        return x

    def parameters(self):
        return (
            [self.ln1_gamma, self.ln1_beta, self.ln2_gamma, self.ln2_beta]
            + self.attn.parameters()
            + self.ff.parameters()
        )


class TransformerLM:
    def __init__(self, vocab_size, d_model, n_heads, head_size, hidden_dim, n_layers, max_len):
        self.tok_emb = Embedding(vocab_size, d_model)
        self.pos_emb = Embedding(max_len, d_model)

        self.blocks = [
            TransformerBlock(d_model, n_heads, head_size, hidden_dim)
            for _ in range(n_layers)
        ]

        self.ln_f_gamma = Tensor2D.full((1, d_model), 1.0, requires_grad=True)
        self.ln_f_beta  = Tensor2D.zeros((1, d_model), requires_grad=True)

        scale = (6.0 / (d_model + vocab_size)) ** 0.5
        self.W_vocab = Tensor2D.random_uniform(
            (d_model, vocab_size),
            low=-scale,
            high=scale,
            requires_grad=True,
        )
        self.b_vocab = Tensor2D.zeros((1, vocab_size), requires_grad=True)

    def __call__(self, token_ids):
        T = len(token_ids)
        x = self.tok_emb(token_ids) + self.pos_emb(list(range(T)))

        for block in self.blocks:
            x = block(x)

        x = x.layernorm(self.ln_f_gamma, self.ln_f_beta)
        logits = (x @ self.W_vocab) + self.b_vocab
        return logits

    def parameters(self):
        params = []
        params.extend(self.tok_emb.parameters())
        params.extend(self.pos_emb.parameters())
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend([self.ln_f_gamma, self.ln_f_beta, self.W_vocab, self.b_vocab])
        return params
    
# # x = Tensor2D([[1.0, -2.0], [3.0, 4.0]], requires_grad=True)

# # with Tensor2D.no_grad():
# #     y = x.relu()

# # print("requires_grad:", y.requires_grad)   # expect False
# # print("_prev:", y._prev)                   # expect ()

# # assert y.requires_grad is False
# # assert y._prev == ()
# # print("no_grad smoke test passed")

# def zero_grad_params(params):
#     for p in params:
#         p.zero_grad()

# def sgd_step(params, lr):
#     for p in params:
#         if p.grad is None:
#             continue
#         for k in range(len(p.data)):
#             p.data[k] -= lr * p.grad[k]

# def decode(ids, itos):
#     return ''.join(itos[i] for i in ids)

# def greedy_pred_ids(logits):
#     T, V = logits.shape
#     preds = []
#     for i in range(T):
#         row = logits.row(i)
#         preds.append(max(range(V), key=lambda j: row[j]))
#     return preds

# def get_example(data_ids, block_size):
#     start = random.randint(0, len(data_ids) - block_size - 1)
#     chunk = data_ids[start:start + block_size + 1]
#     return chunk[:-1], chunk[1:]

# def eval_loss(model, data_ids, block_size, n_batches=10):
#     total = 0.0
#     with Tensor2D.no_grad():
#         for _ in range(n_batches):
#             x_ids, y_ids = get_example(data_ids, block_size)
#             loss = cross_entropy_loss(model(x_ids), y_ids)
#             total += loss.item()
#     return total / n_batches

# def sample_from_logits(logit_row, temperature=1.0):
#     scaled = [x / temperature for x in logit_row]
#     m = max(scaled)
#     exps = [math.exp(x - m) for x in scaled]
#     s = sum(exps)
#     probs = [e / s for e in exps]
#     return random.choices(range(len(probs)), weights=probs, k=1)[0]

# def generate(model, seed_ids, max_new_tokens, block_size, temperature=1.0):
#     ids = seed_ids[:]
#     with Tensor2D.no_grad():
#         for _ in range(max_new_tokens):
#             ctx = ids[-block_size:]
#             logits = model(ctx)
#             next_id = sample_from_logits(logits.row(-1), temperature=temperature)
#             ids.append(next_id)
#     return ids

# random.seed(42)

# tiny_text = "hello world\n"
# chars = sorted(set(tiny_text))
# stoi = {ch: i for i, ch in enumerate(chars)}
# itos = {i: ch for ch, i in stoi.items()}
# vocab_size = len(chars)

# x_text = tiny_text[:-1]
# y_text = tiny_text[1:]

# x_ids = [stoi[ch] for ch in x_text]
# y_ids = [stoi[ch] for ch in y_text]

# T = len(x_ids)

# model = TransformerLM(
#     vocab_size=vocab_size,
#     d_model=16,
#     n_heads=2,
#     head_size=8,
#     hidden_dim=32,
#     n_layers=1,
#     max_len=T,
# )

# params = model.parameters()
# lr = 0.03

# for step in range(301):
#     zero_grad_params(params)

#     logits = model(x_ids)
#     loss = cross_entropy_loss(logits, y_ids)

#     loss.backward()
#     sgd_step(params, lr)

#     if step % 25 == 0:
#         pred_ids = greedy_pred_ids(logits)
#         matches = sum(int(p == t) for p, t in zip(pred_ids, y_ids))
#         acc = matches / len(y_ids)

#         print(f"step {step:3d} | loss {loss.item():.4f} | token-acc {acc:.1%}")
#         print("x:   ", repr(x_text))
#         print("y:   ", repr(y_text))
#         print("pred:", repr(decode(pred_ids, itos)))
#         print()

#         if loss.item() < 0.05:
#             break


import urllib.request


# -----------------------------
# helpers
# -----------------------------

def zero_grad_params(params):
    for p in params:
        p.zero_grad()


def clip_grad_norm(params, max_norm=1.0):
    total_sq = 0.0
    for p in params:
        if p.grad is None:
            continue
        for g in p.grad:
            total_sq += g * g

    total_norm = math.sqrt(total_sq)
    if total_norm <= max_norm:
        return total_norm

    scale = max_norm / (total_norm + 1e-6)
    for p in params:
        if p.grad is None:
            continue
        for i in range(len(p.grad)):
            p.grad[i] *= scale
    return total_norm


class Adam:
    def __init__(self, params, lr=3e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0

        self.m = [[0.0] * len(p.data) for p in params]
        self.v = [[0.0] * len(p.data) for p in params]

    def step(self):
        self.t += 1
        b1_corr = 1.0 - (self.b1 ** self.t)
        b2_corr = 1.0 - (self.b2 ** self.t)

        for idx, p in enumerate(self.params):
            if p.grad is None:
                continue

            m = self.m[idx]
            v = self.v[idx]

            for i in range(len(p.data)):
                g = p.grad[i]

                m[i] = self.b1 * m[i] + (1.0 - self.b1) * g
                v[i] = self.b2 * v[i] + (1.0 - self.b2) * (g * g)

                mhat = m[i] / b1_corr
                vhat = v[i] / b2_corr

                p.data[i] -= self.lr * mhat / (math.sqrt(vhat) + self.eps)


def decode(ids, itos):
    return ''.join(itos[i] for i in ids)


def greedy_pred_ids(logits):
    T, V = logits.shape
    pred = []
    for i in range(T):
        row = logits.row(i)
        pred.append(max(range(V), key=lambda j: row[j]))
    return pred


def get_example(data_ids, block_size):
    start = random.randint(0, len(data_ids) - block_size - 1)
    chunk = data_ids[start:start + block_size + 1]
    return chunk[:-1], chunk[1:]


def eval_loss(model, data_ids, block_size, n_batches=3):
    total = 0.0
    with Tensor2D.no_grad():
        for _ in range(n_batches):
            x_ids, y_ids = get_example(data_ids, block_size)
            logits = model(x_ids)
            loss = cross_entropy_loss(logits, y_ids)
            total += loss.item()
    return total / n_batches


def sample_from_logits(logit_row, temperature=1.0):
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")

    scaled = [x / temperature for x in logit_row]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    s = sum(exps)
    probs = [e / s for e in exps]
    return random.choices(range(len(probs)), weights=probs, k=1)[0]


def generate(model, seed_ids, max_new_tokens, block_size, temperature=0.8):
    ids = seed_ids[:]
    with Tensor2D.no_grad():
        for _ in range(max_new_tokens):
            ctx = ids[-block_size:]
            logits = model(ctx)
            next_id = sample_from_logits(logits.row(-1), temperature=temperature)
            ids.append(next_id)
    return ids


# -----------------------------
# load Tiny Shakespeare
# -----------------------------

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
with urllib.request.urlopen(url) as f:
    text = f.read().decode("utf-8")

chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars)

data = [stoi[ch] for ch in text]
split = int(0.9 * len(data))
train_ids = data[:split]
val_ids = data[split:]

# # -----------------------------
# # stage 1: fixed-window Tiny Shakespeare overfit
# # -----------------------------

# random.seed(42)

# block_size = 32
# chunk = train_ids[:block_size + 1]
# x_ids = chunk[:-1]
# y_ids = chunk[1:]

# print("x_text:", repr(decode(x_ids, itos)))
# print("y_text:", repr(decode(y_ids, itos)))
# print()

# model = TransformerLM(
#     vocab_size=vocab_size,
#     d_model=32,
#     n_heads=4,
#     head_size=8,
#     hidden_dim=64,
#     n_layers=1,
#     max_len=block_size,
# )

# params = model.parameters()
# opt = Adam(params, lr=3e-3)

# for step in range(401):
#     zero_grad_params(params)

#     logits = model(x_ids)
#     loss = cross_entropy_loss(logits, y_ids)

#     loss.backward()
#     clip_grad_norm(params, 1.0)
#     opt.step()

#     if step % 25 == 0:
#         pred_ids = greedy_pred_ids(logits)
#         matches = sum(int(p == t) for p, t in zip(pred_ids, y_ids))
#         acc = matches / len(y_ids)

#         print(f"step {step:3d} | loss {loss.item():.4f} | token-acc {acc:.1%}")
#         print("pred:", repr(decode(pred_ids, itos)))
#         print()

#         if loss.item() < 0.05:
#             break

# -----------------------------
# stage 2: small random-window Tiny Shakespeare training
# -----------------------------

random.seed(42)

block_size = 32
model = TransformerLM(
    vocab_size=vocab_size,
    d_model=32,
    n_heads=4,
    head_size=8,
    hidden_dim=64,
    n_layers=1,
    max_len=block_size,
)

params = model.parameters()
opt = Adam(params, lr=3e-3)

micro_batches = 4

for step in range(20001):
    zero_grad_params(params)
    train_loss_sum = 0.0

    for _ in range(micro_batches):
        x_ids, y_ids = get_example(train_ids, block_size)
        logits = model(x_ids)
        loss = cross_entropy_loss(logits, y_ids)

        train_loss_sum += loss.item()
        (loss * (1.0 / micro_batches)).backward()

    clip_grad_norm(params, 1.0)
    opt.step()

    if step % 100 == 0:
        train_loss = train_loss_sum / micro_batches
        val = eval_loss(model, val_ids, block_size, n_batches=5)
        print(f"step {step:4d} | train loss {train_loss:.4f} | val loss {val:.4f}")

    if step % 200 == 0:
        seed_text = "First "
        seed_ids = [stoi[ch] for ch in seed_text if ch in stoi]
        out_ids = generate(model, seed_ids, max_new_tokens=60, block_size=block_size, temperature=0.8)
        print(repr(decode(out_ids, itos)))
        print()
