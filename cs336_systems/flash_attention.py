"""
FlashAttention-2: PyTorch (tiled) forward and Triton forward + PyTorch backward.
"""
from __future__ import annotations

import math
import torch

# Triton is optional for CPU-only tests
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

# Default tile sizes (at least 16x16 per assignment)
Q_TILE_SIZE = 64
K_TILE_SIZE = 64


def _flash_forward_pytorch_tiled(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False,
    Bq: int = 16, Bk: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    FlashAttention-2 forward using tiled PyTorch (Algorithm 1).
    Q, K, V: (batch, seq, d). Returns (O, L) with O (batch, n_queries, d), L (batch, n_queries).
    """
    batch, n_queries, d = Q.shape
    n_keys = K.shape[1]
    scale = 1.0 / math.sqrt(d)
    device = Q.device
    dtype = Q.dtype
    # Accumulate in float32 for stability
    dtype_acc = torch.float32

    O = torch.zeros(batch, n_queries, d, device=device, dtype=dtype_acc)
    L = torch.zeros(batch, n_queries, device=device, dtype=dtype_acc)

    Tq = (n_queries + Bq - 1) // Bq
    Tk = (n_keys + Bk - 1) // Bk

    for i in range(Tq):
        q_start = i * Bq
        q_end = min(q_start + Bq, n_queries)
        q_actual = q_end - q_start
        Qi = Q[:, q_start:q_end, :].to(dtype_acc)  # (batch, Bq, d)

        O_i = torch.zeros(batch, q_actual, d, device=device, dtype=dtype_acc)
        l_i = torch.zeros(batch, q_actual, device=device, dtype=dtype_acc)
        m_i = torch.full((batch, q_actual), float("-inf"), device=device, dtype=dtype_acc)

        for j in range(Tk):
            k_start = j * Bk
            k_end = min(k_start + Bk, n_keys)
            k_actual = k_end - k_start
            Kj = K[:, k_start:k_end, :].to(dtype_acc)   # (batch, Bk, d)
            Vj = V[:, k_start:k_end, :].to(dtype_acc)   # (batch, Bk, d)

            # S_ij = Q_i K_j^T / sqrt(d)  -> (batch, Bq, Bk)
            S_ij = torch.matmul(Qi, Kj.transpose(-2, -1)) * scale

            if is_causal:
                q_idx = torch.arange(q_start, q_end, device=device).view(1, -1, 1)
                k_idx = torch.arange(k_start, k_end, device=device).view(1, 1, -1)
                causal_mask = q_idx >= k_idx
                S_ij = torch.where(causal_mask, S_ij, torch.tensor(-1e6, device=device, dtype=S_ij.dtype))

            m_old = m_i
            m_new = torch.maximum(m_i, S_ij.max(dim=-1, keepdim=False).values)
            m_i = m_new

            P_tilde = torch.exp(S_ij - m_i.unsqueeze(-1))
            l_i = torch.exp(m_old - m_i) * l_i + P_tilde.sum(dim=-1)
            O_i = torch.exp(m_old - m_i).unsqueeze(-1) * O_i + torch.matmul(P_tilde, Vj)

        O_i = O_i / l_i.unsqueeze(-1).clamp(min=1e-10)
        L_i = m_i + torch.log(l_i.clamp(min=1e-10))
        O[:, q_start:q_end, :] = O_i.to(dtype)
        L[:, q_start:q_end] = L_i

    return O, L


class FlashAttention2PyTorch(torch.autograd.Function):
    """FlashAttention-2 forward in pure PyTorch (tiled). Backward can be NotImplemented or implemented."""

    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False):
        # Q, K, V: (batch, seq, d)
        O, L = _flash_forward_pytorch_tiled(Q, K, V, is_causal=is_causal, Bq=Q_TILE_SIZE, Bk=K_TILE_SIZE)
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        ctx.scale = 1.0 / math.sqrt(Q.shape[-1])
        return O

    @staticmethod
    def backward(ctx, dO: torch.Tensor):
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = ctx.is_causal
        scale = ctx.scale
        d = Q.shape[-1]
        # D = rowsum(O * dO), shape (batch, n_queries)
        D = (O * dO).sum(dim=-1)
        # Eqs 13-19: S = QK^T/sqrt(d), P = exp(S - L), dV = P^T dO, dP = dO V^T,
        # dS = P * (dP - D), dQ = dS K/sqrt(d), dK = dS^T Q/sqrt(d)
        S = torch.matmul(Q, K.transpose(-2, -1)) * scale
        if is_causal:
            n_queries, n_keys = Q.shape[1], K.shape[1]
            q_idx = torch.arange(n_queries, device=Q.device).view(1, -1, 1)
            k_idx = torch.arange(n_keys, device=Q.device).view(1, 1, -1)
            causal_mask = q_idx >= k_idx
            S = torch.where(causal_mask, S, torch.tensor(-1e6, device=S.device, dtype=S.dtype))
        P = torch.exp(S - L.unsqueeze(-1))
        dV = torch.matmul(P.transpose(-2, -1), dO)
        dP = torch.matmul(dO, V.transpose(-2, -1))
        dS = P * (dP - D.unsqueeze(-1))
        dQ = torch.matmul(dS, K) * scale
        dK = torch.matmul(dS.transpose(-2, -1), Q) * scale
        return dQ, dK, dV, None


if TRITON_AVAILABLE:

    @triton.jit
    def _flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            O_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,),
            strides=(stride_lq,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )

        # Load Q tile
        Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        # Accumulators in float32
        O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
        l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
        m_i = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)

        n_k_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
        for j in range(n_k_tiles):
            # Load K as a transposed tile (D, K_TILE_SIZE), so tl.dot(Q, Kt) computes QK^T.
            K_block_ptr = tl.make_block_ptr(
                K_ptr + batch_index * stride_kb,
                shape=(D, N_KEYS),
                strides=(stride_kd, stride_kk),
                offsets=(0, j * K_TILE_SIZE),
                block_shape=(D, K_TILE_SIZE),
                order=(1, 0),
            )
            V_block_ptr = tl.make_block_ptr(
                V_ptr + batch_index * stride_vb,
                shape=(N_KEYS, D),
                strides=(stride_vk, stride_vd),
                offsets=(j * K_TILE_SIZE, 0),
                block_shape=(K_TILE_SIZE, D),
                order=(1, 0),
            )
            Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
            Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

            # S_ij = Q_i @ K_j^T * scale  (Q_TILE_SIZE, K_TILE_SIZE), float32 for stability
            S_ij = tl.dot(Qi.to(tl.float32), Kj.to(tl.float32)) * scale

            if IS_CAUSAL:
                q_offset = query_tile_index * Q_TILE_SIZE
                k_offset = j * K_TILE_SIZE
                q_idx = tl.arange(0, Q_TILE_SIZE)
                k_idx = tl.arange(0, K_TILE_SIZE)
                q_idx = q_offset + q_idx
                k_idx = k_offset + k_idx
                # mask where q < k
                mask = q_idx[:, None] >= k_idx[None, :]
                S_ij = tl.where(mask, S_ij, -1e6)

            m_old = m_i
            m_new = tl.maximum(m_i, tl.max(S_ij, axis=1))
            m_i = m_new

            P_tilde = tl.exp(S_ij - m_i[:, None])
            P_tilde = P_tilde.to(Vj.dtype)
            l_i = tl.exp(m_old - m_i) * l_i + tl.sum(P_tilde, axis=1)
            # Accumulate in float32 for stability (O_i is float32)
            O_i = tl.exp(m_old - m_i)[:, None].to(O_i.dtype) * O_i + tl.dot(P_tilde.to(tl.float32), Vj.to(tl.float32))

        O_i = O_i / l_i[:, None]
        L_i = m_i + tl.log(l_i)
        O_i = O_i.to(O_block_ptr.type.element_ty)
        tl.store(O_block_ptr, O_i, boundary_check=(0, 1))
        tl.store(L_block_ptr, L_i.to(L_block_ptr.type.element_ty), boundary_check=(0,))


def _flash_forward_triton(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False,
    Bq: int = 64, Bk: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FlashAttention-2 forward using Triton kernel. Q, K, V on CUDA."""
    batch, n_queries, d = Q.shape
    n_keys = K.shape[1]
    scale = 1.0 / math.sqrt(d)
    assert Q.is_cuda and K.is_cuda and V.is_cuda

    O = torch.empty_like(Q)
    L = torch.empty(batch, n_queries, device=Q.device, dtype=torch.float32)

    Tq = triton.cdiv(n_queries, Bq)
    grid = (Tq, batch)

    _flash_fwd_kernel[grid](
        Q, K, V, O, L,
        stride_qb=Q.stride(0), stride_qq=Q.stride(1), stride_qd=Q.stride(2),
        stride_kb=K.stride(0), stride_kk=K.stride(1), stride_kd=K.stride(2),
        stride_vb=V.stride(0), stride_vk=V.stride(1), stride_vd=V.stride(2),
        stride_ob=O.stride(0), stride_oq=O.stride(1), stride_od=O.stride(2),
        stride_lb=L.stride(0), stride_lq=L.stride(1),
        N_QUERIES=n_queries, N_KEYS=n_keys,
        scale=scale, D=d,
        Q_TILE_SIZE=Bq, K_TILE_SIZE=Bk,
        IS_CAUSAL=is_causal,
    )
    return O, L


@torch.compile
def _flash_backward_pytorch(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    O: torch.Tensor, dO: torch.Tensor, L: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass for FlashAttention-2 using Eqs 13-19 (PyTorch, compiled)."""
    scale = 1.0 / math.sqrt(Q.shape[-1])
    D = (O * dO).sum(dim=-1)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    if is_causal:
        n_queries, n_keys = Q.shape[1], K.shape[1]
        q_idx = torch.arange(n_queries, device=Q.device).view(1, -1, 1)
        k_idx = torch.arange(n_keys, device=Q.device).view(1, 1, -1)
        causal_mask = q_idx >= k_idx
        S = torch.where(causal_mask, S, torch.tensor(-1e6, device=S.device, dtype=S.dtype))
    P = torch.exp(S - L.unsqueeze(-1))
    dV = torch.matmul(P.transpose(-2, -1), dO)
    dP = torch.matmul(dO, V.transpose(-2, -1))
    dS = P * (dP - D.unsqueeze(-1))
    dQ = torch.matmul(dS, K) * scale
    dK = torch.matmul(dS.transpose(-2, -1), Q) * scale
    return dQ, dK, dV


class FlashAttention2Triton(torch.autograd.Function):
    """FlashAttention-2 forward with Triton kernel, backward with PyTorch (torch.compile)."""

    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False):
        O, L = _flash_forward_triton(Q, K, V, is_causal=is_causal, Bq=Q_TILE_SIZE, Bk=K_TILE_SIZE)
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO: torch.Tensor):
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = ctx.is_causal
        dQ, dK, dV = _flash_backward_pytorch(Q, K, V, O, dO, L, is_causal)
        return dQ, dK, dV, None
