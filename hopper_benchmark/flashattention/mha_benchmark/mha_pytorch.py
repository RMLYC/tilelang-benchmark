#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
torch_bench_mha.py

使用 PyTorch 原生 SDPA 实现的 Multi-Head Attention 基准程序。
目标是在支持的 GPU 上调用最高效的内核后端，并提供与类 FA3 相似的
forward、backward 与 profile 接口，便于集成与横向对比。

风格: Google Python Style
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from contextlib import nullcontext

from mha_config import MHAConfig


# ================================ Utilities ================================ #

def _require_cuda() -> None:
    """确保当前环境可用 CUDA 且设备就绪。"""
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到可用的 CUDA 设备。请在支持 CUDA 的环境下运行。")


def _sdpa_fastest_ctx():
    """返回一个上下文，优先选择最快的 SDPA 后端。
    优先 FLASH_ATTENTION，其次 EFFICIENT_ATTENTION（PyTorch 2.5+）。
    若新 API 不可用，则回退到旧 API；再不行就空上下文。
    """
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        # 新 API：按优先级指定后端
        return sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])
    except Exception:
        # 老版本回退（在新版本上会给出弃用告警，但可用）
        try:
            return torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=True
            )
        except Exception:
            return nullcontext()



def attn_flops_forward(cfg: MHAConfig) -> float:
    """计算前向 FLOPs 的近似值，仅统计 QK^T 与 P·V 主项。

    公式:
      FLOPs ≈ 4 * B * H * S^2 * Hd
    """
    B, H, S, Hd = cfg.batch, cfg.heads, cfg.seq_len, cfg.dim
    return 4.0 * B * H * (S ** 2) * Hd


def attn_flops_forward_backward(cfg: MHAConfig) -> float:
    """估算前向加反向的总 FLOPs。

    经验规律:
      反向大约是前向的两倍，因此 FWD_BWD ≈ 3.5 × FWD。
    """
    return 3.5 * attn_flops_forward(cfg)


def _make_inputs(cfg: MHAConfig, device: torch.device, seed: int = 17
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """根据配置生成随机 Q, K, V。

    形状:
      Q, K, V: [B, S, H, Hd]
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    q = torch.randn(cfg.batch, cfg.seq_len, cfg.heads, cfg.dim,
                    device=device, dtype=cfg.dtype, generator=g)
    k = torch.randn_like(q, memory_format=torch.contiguous_format)
    v = torch.randn_like(q, memory_format=torch.contiguous_format)
    return q, k, v


# ================================ Kernel Class ============================= #

class TorchMHAKernel:
    """基于 PyTorch SDPA 的 MHA 封装类。"""

    def __init__(self,
                 batch: int,
                 seq_len: int,
                 heads: int,
                 dim: int,
                 causal: bool,
                 dtype: torch.dtype = torch.float16,
                 device: str = "cuda:0") -> None:
        """初始化内核配置与运行环境。

        Args:
          batch: 批大小 B
          seq_len: 序列长度 S
          heads: 注意力头数 H
          dim: 每头维度 Hd
          causal: 是否使用因果掩码
          dtype: torch.float16 或 torch.bfloat16
          device: 设备字符串, 例如 "cuda:0"
        """
        _require_cuda()

        self.cfg = MHAConfig(
            batch=batch,
            seq_len=seq_len,
            heads=heads,
            dim=dim,
            causal=causal,
            dtype=dtype,
            dropout_p=0.0,
        )
        self.device = torch.device(device)

    # ----------------------------- Core APIs ----------------------------- #

    @torch.inference_mode(False)
    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                softmax_scale: Optional[float] = None) -> torch.Tensor:
        """执行前向计算。

        说明:
          输入须为 [B, S, H, Hd]。本函数不做缓存，完全由调用者管理输入输出。

        Args:
          q: Query 张量 [B, S, H, Hd]
          k: Key 张量 [B, S, H, Hd]
          v: Value 张量 [B, S, H, Hd]
          softmax_scale: softmax 缩放因子。不传则使用 1/sqrt(Hd)

        Returns:
          输出张量 [B, S, H, Hd]
        """
        cfg = self.cfg

        q = q.to(device=self.device, dtype=cfg.dtype)
        k = k.to(device=self.device, dtype=cfg.dtype)
        v = v.to(device=self.device, dtype=cfg.dtype)

        # PyTorch SDPA 支持 [B, H, S, D] 或 [B, S, H, D]
        # 这里直接使用 [B, H, S, D] 格式以匹配后端要求
        q_bhsd = q.transpose(1, 2)   # [B, H, S, D]
        k_bhsd = k.transpose(1, 2)
        v_bhsd = v.transpose(1, 2)

        # is_causal 控制下三角掩码。scale 可覆盖默认缩放
        with _sdpa_fastest_ctx():
            out_bhsd = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=cfg.causal,
                scale=softmax_scale
            )
        out = out_bhsd.transpose(1, 2).contiguous()  # 回到 [B, S, H, D]
        return out

    @torch.inference_mode(False)
    def backward(self,
                 q: torch.Tensor,
                 k: torch.Tensor,
                 v: torch.Tensor,
                 dout: Optional[torch.Tensor] = None,
                 softmax_scale: Optional[float] = None
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """对给定输入执行一次前向和反向，返回 dQ, dK, dV。

        说明:
          若不提供 dout，则使用均方损失构造标量 loss。

        Args:
          q: Query 张量 [B, S, H, Hd]
          k: Key 张量 [B, S, H, Hd]
          v: Value 张量 [B, S, H, Hd]
          dout: 上游梯度 [B, S, H, Hd]。不提供则内部构造标量损失
          softmax_scale: softmax 缩放因子

        Returns:
          三元组 (dQ, dK, dV)，形状均为 [B, S, H, Hd]
        """
        cfg = self.cfg

        q = q.to(device=self.device, dtype=cfg.dtype).detach().requires_grad_(True)
        k = k.to(device=self.device, dtype=cfg.dtype).detach().requires_grad_(True)
        v = v.to(device=self.device, dtype=cfg.dtype).detach().requires_grad_(True)

        q_bhsd = q.transpose(1, 2)
        k_bhsd = k.transpose(1, 2)
        v_bhsd = v.transpose(1, 2)

        with _sdpa_fastest_ctx():
            out_bhsd = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=cfg.causal,
                scale=softmax_scale
            )
        out = out_bhsd.transpose(1, 2).contiguous()

        if dout is None:
            loss = out.float().pow(2).mean()
        else:
            if dout.shape != out.shape:
                raise ValueError(f"dout 形状 {dout.shape} 与 out 形状 {out.shape} 不一致")
            loss = (out * dout.to(out.dtype)).float().mean()

        loss.backward()

        d_q = q.grad.detach().clone()
        d_k = k.grad.detach().clone()
        d_v = v.grad.detach().clone()

        # 清理梯度，避免外部重复调用时累积
        for t in (q, k, v):
            if t.grad is not None:
                t.grad.zero_()

        return d_q, d_k, d_v

    # ----------------------------- Profiling ----------------------------- #

    @torch.inference_mode(False)
    def profile(self,
                q: Optional[torch.Tensor] = None,
                k: Optional[torch.Tensor] = None,
                v: Optional[torch.Tensor] = None,
                dout: Optional[torch.Tensor] = None,
                warmup: int = 50,
                iters: int = 300,
                do_backward: bool = False,
                softmax_scale: Optional[float] = None
                ) -> Dict[str, Optional[float]]:
        """基准测试接口，返回延迟与 TFLOPs。

        说明:
          若未提供 q, k, v，将根据 cfg 生成固定随机张量。
          度量时将使用 CUDA events 并同步，确保时间准确。

        Args:
          q: Query [B, S, H, Hd]
          k: Key [B, S, H, Hd]
          v: Value [B, S, H, Hd]
          dout: 上游梯度 [B, S, H, Hd]。do_backward 为 True 时可提供
          warmup: 预热轮次
          iters: 计时轮次
          do_backward: 是否计量反向
          softmax_scale: softmax 缩放因子

        Returns:
          包含以下键的字典
            - fwd_latency_ms
            - fwd_tflops
            - bwd_latency_ms
            - bwd_tflops
        """
        cfg = self.cfg
        device = self.device

        if q is None or k is None or v is None:
            q, k, v = _make_inputs(cfg, device)
        else:
            q = q.to(device=device, dtype=cfg.dtype)
            k = k.to(device=device, dtype=cfg.dtype)
            v = v.to(device=device, dtype=cfg.dtype)

        if dout is None:
            dout = torch.randn_like(q)

        # 为了公平计时，固定输入
        q_f = q.detach()
        k_f = k.detach()
        v_f = v.detach()
        dout_f = dout.detach()

        def _fwd_once(q_, k_, v_) -> torch.Tensor:
            # 转为 [B, H, S, D]
            with _sdpa_fastest_ctx():
                out = F.scaled_dot_product_attention(
                    q_.transpose(1, 2), k_.transpose(1, 2), v_.transpose(1, 2),
                    attn_mask=None, dropout_p=0.0,
                    is_causal=cfg.causal, scale=softmax_scale
                )
            return out.transpose(1, 2).contiguous()

        # 预热前向
        for _ in range(max(1, warmup)):
            out = _fwd_once(q_f, k_f, v_f)
            torch.cuda.synchronize()

        # 前向计时
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        total_fwd_ms = 0.0
        for _ in range(iters):
            starter.record()
            out = _fwd_once(q_f, k_f, v_f)
            ender.record()
            torch.cuda.synchronize()
            total_fwd_ms += starter.elapsed_time(ender)
        avg_fwd_ms = total_fwd_ms / iters

        # 计算前向 TFLOPs
        fwd_flops = attn_flops_forward(cfg)
        fwd_tflops = fwd_flops / (avg_fwd_ms * 1e-3) / 1e12

        # 可选反向
        avg_bwd_ms: Optional[float] = None
        bwd_tflops: Optional[float] = None
        if do_backward:
            # 预热前向加反向
            for _ in range(max(1, warmup)):
                q_b = q_f.clone().detach().requires_grad_(True)
                k_b = k_f.clone().detach().requires_grad_(True)
                v_b = v_f.clone().detach().requires_grad_(True)
                out = _fwd_once(q_b, k_b, v_b)
                loss = (out * dout_f).float().mean()
                loss.backward()
                torch.cuda.synchronize()

            # 计时前向加反向
            total_fb_ms = 0.0
            for _ in range(iters):
                q_b = q_f.clone().detach().requires_grad_(True)
                k_b = k_f.clone().detach().requires_grad_(True)
                v_b = v_f.clone().detach().requires_grad_(True)

                starter.record()
                out = _fwd_once(q_b, k_b, v_b)
                loss = (out * dout_f).float().mean()
                loss.backward()
                ender.record()
                torch.cuda.synchronize()
                total_fb_ms += starter.elapsed_time(ender)

            avg_fb_ms = total_fb_ms / iters
            # 反向时间用总时间减去仅前向时间
            avg_bwd_ms = max(0.0, avg_fb_ms - avg_fwd_ms)

            # 反向 FLOPs 近似值
            fwd_bwd_flops = attn_flops_forward_backward(cfg)
            bwd_flops = max(0.0, fwd_bwd_flops - fwd_flops)
            bwd_tflops = bwd_flops / (avg_bwd_ms * 1e-3) / 1e12 if avg_bwd_ms > 0 else None

        return {
            "fwd_latency_ms": float(avg_fwd_ms),
            "fwd_tflops": float(fwd_tflops),
            "bwd_latency_ms": None if avg_bwd_ms is None else float(avg_bwd_ms),
            "bwd_tflops": None if bwd_tflops is None else float(bwd_tflops),
        }


# =================================== CLI =================================== #

def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch SDPA MHA 基准测试。")
    parser.add_argument("--batch", type=int, required=True, help="批大小 B")
    parser.add_argument("--seq_len", type=int, required=True, help="序列长度 S")
    parser.add_argument("--heads", type=int, required=True, help="注意力头数 H")
    parser.add_argument("--dim", type=int, required=True, help="每头维度 Hd")
    parser.add_argument("--causal", action="store_true", help="是否使用因果掩码")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"], help="计算精度")
    parser.add_argument("--warmup", type=int, default=50, help="预热轮次")
    parser.add_argument("--iters", type=int, default=300, help="计时轮次")
    parser.add_argument("--bwd", action="store_true", help="是否计量反向")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    kernel = TorchMHAKernel(
        batch=args.batch,
        seq_len=args.seq_len,
        heads=args.heads,
        dim=args.dim,
        causal=bool(args.causal),
        dtype=dtype,
        device="cuda:0",
    )

    # 如未提供外部输入，这里生成固定随机输入
    q, k, v = _make_inputs(kernel.cfg, torch.device("cuda:0"))
    metrics = kernel.profile(
        q=q, k=k, v=v,
        dout=None,
        warmup=args.warmup,
        iters=args.iters,
        do_backward=bool(args.bwd),
    )

    d_model = args.heads * args.dim
    print(f"[Config] B={args.batch} S={args.seq_len} H={args.heads} Hd={args.dim} D={d_model} "
          f"Causal={bool(args.causal)} DType={dtype}")
    print(f"[Device] {torch.cuda.get_device_name(0)} | CC {torch.cuda.get_device_capability()}")

    print(f"[Result-FWD] avg_latency = {metrics['fwd_latency_ms']:.3f} ms | "
          f"throughput = {metrics['fwd_tflops']:.2f} TFLOPs")
    if args.bwd:
        bms = metrics["bwd_latency_ms"]
        bt = metrics["bwd_tflops"]
        bms_str = f"{bms:.3f} ms" if bms is not None else "N/A"
        bt_str = f"{bt:.2f} TFLOPs" if bt is not None else "N/A"
        print(f"[Result-BWD] avg_latency = {bms_str} | throughput = {bt_str}")

    print(f"[Info] softmax_scale 默认值 = 1/sqrt(Hd) = {1.0 / math.sqrt(args.dim):.6f}")


if __name__ == "__main__":
    main()
