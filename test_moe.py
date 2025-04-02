from pathlib import Path
from PIL import Image

import pytest
import torch
from gguf import GGMLQuantizationType, GGUFReader, ReaderTensor, dequantize
import numpy as np
from huggingface_hub import snapshot_download

import vllm._custom_ops as ops
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.quantization.gguf import _fused_moe_gguf, _fused_moe_gguf_new
from vllm.platforms import current_platform
from torch.utils.cpp_extension import load

GGUF_SAMPLE_MOE = snapshot_download("SzymonOzog/test-gguf-moe-sample")

sources = ["./csrc/my_bindings.cpp", "./csrc/quantization/gguf/gguf_kernel.cu"]
my_extension = load(
    name="my_extension",
    sources=sources,
    extra_cuda_cflags=["-arch=sm_80", "-lineinfo"],  # for CUDA 8.0 arch
    extra_include_paths=["./csrc"],
    verbose=True,
)




def get_gguf_MoE_tensors(
        hidden_size: int,
        quant_type: GGMLQuantizationType) -> list[ReaderTensor]:
    sample_dir = GGUF_SAMPLE_MOE
    filename = f"Quant_{quant_type.name}_{hidden_size}.gguf"
    sample_file = Path(sample_dir) / filename
    return GGUFReader(sample_file).tensors


@torch.inference_mode()
def test_moe(num_tokens: int, hidden_size: int, dtype: torch.dtype,
             quant_type: GGMLQuantizationType, top_k: int):
    current_platform.seed_everything(0)
    H, E = 1024, 256

    x = torch.rand((num_tokens, H), dtype=dtype, device="cuda")

    topk_weights = torch.rand(num_tokens, top_k, device="cuda", dtype=dtype)
    topk_ids = torch.randint(0, E, (num_tokens, top_k), device="cuda")

    tensors = get_gguf_MoE_tensors(hidden_size, quant_type)

    w13 = tensors[0]
    w2 = tensors[1]

    w13_dequant = torch.tensor(dequantize(w13.data, quant_type),
                               device="cuda").to(dtype)

    w2_dequant = torch.tensor(dequantize(w2.data, quant_type),
                              device="cuda").to(dtype)
    act = SiluAndMul()
    w = torch.tensor(w13.data, device="cuda")
    print(w.shape)
    # w = w[..., :w.shape[-1]//2]
    print(w.shape)

    output = _fused_moe_gguf(x, w,
                             torch.tensor(w2.data,
                                          device="cuda"), topk_weights,
                             topk_ids, quant_type, quant_type, act)

    output_fast = _fused_moe_gguf_new(x, w,
                             torch.tensor(w2.data,
                                          device="cuda"), topk_weights,
                             topk_ids, quant_type, quant_type, act, my_extension)
    torch.cuda.synchronize()

    print(output.shape)

    # ref_output = fused_experts(x, w13_dequant, w2_dequant, topk_weights,
    #                            topk_ids).reshape(output.shape)

    # torch.testing.assert_close(output, ref_output, atol=1, rtol=1e-1)
    print(output)
    print(output_fast)

    # print(output[7, 12:42])
    # print(output_fast[7, 12:42])
    # print(ref_output[7, 12:42])
    img = torch.isclose(output, output_fast, atol=1e-1, rtol=1e-1).cpu().numpy().astype(np.uint8) * 255
    # Convert to PIL image
    image = Image.fromarray(img, mode="L")  # "L" mode for grayscale

    image.save("boolean_tensor.png")

    torch.testing.assert_close(output, output_fast, atol=1e-2, rtol=1e-1)
test_moe(1, 512, torch.bfloat16, GGMLQuantizationType.Q4_K, 8)
