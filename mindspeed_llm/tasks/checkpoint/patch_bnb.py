import ctypes as ct
import os
import torch
from typing import Optional

import bitsandbytes as bnb
from bitsandbytes.utils import QuantState
from bitsandbytes.cextension import lib
from bitsandbytes.functional import get_4bit_type, get_ptr


def _patched_quantize_4bit(
    self,
    A: torch.Tensor,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: Optional[int] = None,
    compress_statistics=False,
    quant_type="nf4",
    quant_storage=torch.uint8,
):
    if quant_type not in ["nf4"]:
        raise NotImplementedError(f"4-bit quantization data type {quant_type} is not implemented.")
    if blocksize is None:
        blocksize = 64
    total_blocks = A.numel() // blocksize
    chunks = max(128, total_blocks // 2048)         # number of chunks is 128 instead of 8
    chunksize = (total_blocks + chunks - 1) // chunks
    prev_device = torch.npu.current_device()
    torch.npu.set_device(A.device)
    if A.dtype in [torch.float32, torch.float16, torch.bfloat16]:
        data = [
            -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
            0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
            0.7229568362236023, 1.0,
        ]
        data = torch.tensor(data, device="npu", dtype=torch.float32).view(1, -1)
        chunks_absmax = []
        chunks_out = []
        for i in range(chunks):
            start = i * chunksize * blocksize
            end = min((i + 1) * chunksize * blocksize, A.numel())
            chunk_data = A.view(-1)[start:end].view(-1, blocksize)
            absmax = chunk_data.abs().max(dim=1, keepdim=True).values
            chunks_absmax.append(absmax)
            a = chunk_data / absmax.float()
            diff = torch.abs(a.unsqueeze(-1) - data)
            out = (torch.argmin(diff, dim=-1) + 8) % 16
            out = out.reshape(-1, 2)
            out = (out[:, 0] + out[:, 1] * 16).to(torch.uint8)
            chunks_out.append(out)
            del diff, a, chunk_data, absmax
        absmax = torch.cat(chunks_absmax, dim=0)
        out = torch.cat(chunks_out, dim=0)
    else:
        raise ValueError(f"Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}")
    torch.npu.set_device(prev_device)
    code = get_4bit_type(quant_type, device=A.device)
    state = QuantState(
        absmax=absmax,
        shape=A.shape,
        dtype=A.dtype,
        blocksize=blocksize,
        code=code,
        quant_type=quant_type,
    )
    return out, state


_QUANT_THREADS_CONFIGURED = False


def _configure_quant_threads():
    """Raise ATen intra-op threads (the env often pins OMP_NUM_THREADS=1)."""
    global _QUANT_THREADS_CONFIGURED
    if _QUANT_THREADS_CONFIGURED:
        return
    _QUANT_THREADS_CONFIGURED = True
    env_threads = os.environ.get("QLORA_QUANT_CPU_THREADS")
    try:
        target = int(env_threads) if env_threads else min(96, os.cpu_count() or 1)
        if torch.get_num_threads() < target:
            torch.set_num_threads(target)
    except Exception:
        pass


def quantize_nf4_cpu(A: torch.Tensor, blocksize: int = 64):
    """CPU replica of the patched NPUBackend.quantize_4bit above (nf4 only).

    Runs the exact same pure-PyTorch math without any torch.npu calls, so the
    output is bit-identical to the NPU path: per-block absmax kept in the input
    dtype as a [blocks, 1] tensor, nearest-neighbor NF4 lookup with the
    (argmin + 8) % 16 nibble rotation expected by the AscendC dequant kernel in
    libbitsandbytes_npu.so, packed uint8 pairs as a flat [n/2] tensor.

    Chunk sizes only bound peak memory (blocks are independent), so they are
    sized for parallel efficiency instead of matching the NPU chunking. Thread
    count can be tuned via QLORA_QUANT_CPU_THREADS (default: min(96, ncpu)).
    """
    _configure_quant_threads()
    if A.device.type != "cpu":
        A = A.to("cpu")
    A = A.contiguous()
    target_chunk_elements = 4 * 1024 * 1024
    chunks = max(1, -(-A.numel() // target_chunk_elements))
    chunksize = (-(-A.numel() // chunks) + blocksize - 1) // blocksize
    if A.dtype in [torch.float32, torch.float16, torch.bfloat16]:
        data = [
            -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
            0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
            0.7229568362236023, 1.0,
        ]
        data = torch.tensor(data, dtype=torch.float32).view(1, -1)
        chunks_absmax = []
        chunks_out = []
        for i in range(chunks):
            start = i * chunksize * blocksize
            end = min((i + 1) * chunksize * blocksize, A.numel())
            chunk_data = A.view(-1)[start:end].view(-1, blocksize)
            absmax = chunk_data.abs().max(dim=1, keepdim=True).values
            chunks_absmax.append(absmax)
            a = chunk_data / absmax.float()
            diff = torch.abs(a.unsqueeze(-1) - data)
            out = (torch.argmin(diff, dim=-1) + 8) % 16
            out = out.reshape(-1, 2)
            out = (out[:, 0] + out[:, 1] * 16).to(torch.uint8)
            chunks_out.append(out)
            del diff, a, chunk_data, absmax
        absmax = torch.cat(chunks_absmax, dim=0)
        out = torch.cat(chunks_out, dim=0)
    else:
        raise ValueError(f"Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}")
    code = get_4bit_type("nf4", device=A.device)
    state = QuantState(
        absmax=absmax,
        shape=A.shape,
        dtype=A.dtype,
        blocksize=blocksize,
        code=code,
        quant_type="nf4",
    )
    return out, state


def _patched_dequantize_4bit(
    self,
    A: torch.Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: Optional[int] = None,
    quant_type="nf4",
):
    """Chunked copy of NPUBackend.dequantize_4bit (bitsandbytes npu_beta 0.45.3).

    The stock bf16 path allocates a full-size fp32 copy of the output plus a
    full-size bf16 temporary on every call, peaking at ~3x the output size in
    extra NPU memory. Here the kernel writes into a small reusable fp32 staging
    buffer chunk by chunk and copy_ casts back to bf16 without extra temporaries.

    Chunk sizes must keep every kernel call at >= 1561 64-element packages:
    the kernel is launched with a fixed blockDim=40 and its unguarded blocks
    (blockIdx >= coreNum, possible only when coreNum < 40) write out of bounds.
    """
    if blocksize is None:
        blocksize = 64
    supported_blocksizes = [2048, 4096, 1024, 512, 256, 128, 64]
    if blocksize not in supported_blocksizes:
        raise ValueError(
            f"The blockwise of {blocksize} is not supported. Supported values: {supported_blocksizes}"
        )

    if quant_state is None:
        assert absmax is not None and out is not None
        quant_state = QuantState(
            absmax=absmax, shape=out.shape, dtype=out.dtype, blocksize=blocksize, quant_type=quant_type
        )
    else:
        absmax = quant_state.absmax

    if out is None:
        out = torch.empty(quant_state.shape, dtype=quant_state.dtype, device=A.device)

    n = out.numel()

    prev_device = torch.npu.current_device()
    torch.npu.set_device(A.device)

    if quant_state.quant_type not in ["nf4"]:
        raise NotImplementedError(
            f"4-bit quantization data type {quant_state.quant_type} is not implemented."
        )

    if out.dtype == torch.float32:
        lib.cdequantize_blockwise_fp32_nf4(
            get_ptr(A),
            get_ptr(absmax),
            get_ptr(out),
            ct.c_int(quant_state.blocksize),
            ct.c_int(n),
            torch.npu.current_stream(),
        )
    elif out.dtype == torch.float16:
        lib.cdequantize_blockwise_fp16_nf4(
            get_ptr(A),
            get_ptr(absmax),
            get_ptr(out),
            ct.c_int(quant_state.blocksize),
            ct.c_int(n),
            torch.npu.current_stream(),
        )
    elif out.dtype == torch.bfloat16:
        blocksize_ = quant_state.blocksize
        absmax_flat = absmax.to(torch.float32).reshape(-1)
        A_flat = A.reshape(-1)
        out_flat = out.view(-1)
        # The prebuilt AscendC kernel (libbitsandbytes_npu.so) is launched with a
        # fixed blockDim=40 and does not guard blocks with blockIdx >= coreNum
        # (unsigned `coreNum - blockIdx` underflows, so ghost blocks run the full
        # singleCoreNumel path and write past the output buffer). coreNum == 40 is
        # guaranteed only when the package count P = numel / blocksize >= 1561, so
        # every kernel call must cover at least 1561 * 64 = 99904 elements. Use
        # MIN_CHUNK = 102400 (1600 packages, multiple of 2560) which also keeps the
        # A/absmax slice pointers 32-byte aligned, and merge a would-be small tail
        # chunk into the previous one. The number of staging pieces targets a fixed
        # staging memory budget (~64 MB fp32) instead of a fixed piece count, so
        # smaller dequantize calls (e.g. per-expert chunks) do not pay a full
        # 128-piece launch sequence.
        MIN_CHUNK = 102400
        STAGING_BUDGET = 16 * 1024 * 1024
        if n <= MIN_CHUNK:
            chunks, chunk_numel = 1, n
        else:
            chunks = max(1, min(128, (n + STAGING_BUDGET - 1) // STAGING_BUDGET))
            chunk_numel = ((n + chunks - 1) // chunks + 2559) // 2560 * 2560
            while chunks > 1 and n - chunk_numel * (chunks - 1) < MIN_CHUNK:
                chunks -= 1
                chunk_numel = ((n + chunks - 1) // chunks + 2559) // 2560 * 2560
            chunk_numel = min(chunk_numel, n)
        staging = torch.empty(chunk_numel, dtype=torch.float32, device=out.device)
        stream = torch.npu.current_stream()
        for start in range(0, n, chunk_numel):
            end = min(start + chunk_numel, n)
            cn = end - start
            n_blocks = (cn + blocksize_ - 1) // blocksize_
            lib.cdequantize_blockwise_fp32_nf4(
                get_ptr(A_flat[start // 2 : (end + 1) // 2]),
                get_ptr(absmax_flat[start // blocksize_ : start // blocksize_ + n_blocks]),
                get_ptr(staging[:cn]),
                ct.c_int(blocksize_),
                ct.c_int(cn),
                stream,
            )
            # copy_ casts fp32 -> bf16 without a full-size temporary
            out_flat[start:end].copy_(staging[:cn])
    else:
        raise ValueError(f"Blockwise quantization only supports 16/32-bit floats, but got {out.dtype}")
    torch.npu.set_device(prev_device)
    is_transposed = True if A.shape[0] == 1 else False

    if is_transposed:
        return out.t()
    else:
        return out


def apply_bnb_npu_patches():
    """Patch bitsandbytes NPU kernels to reduce peak NPU memory usage."""
    from bitsandbytes.backends.npu import NPUBackend

    NPUBackend.quantize_4bit = _patched_quantize_4bit
    NPUBackend.dequantize_4bit = _patched_dequantize_4bit
