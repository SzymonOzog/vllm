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
from torch.profiler import profile, record_function, ProfilerActivity
import os
torch.utils.cpp_extension.COMMON_NVCC_FLAGS = [
    # '-D__CUDA_NO_HALF_OPERATORS__',
    # '-D__CUDA_NO_HALF_CONVERSIONS__',
    # '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
    # '-D__CUDA_NO_HALF2_OPERATORS__',
    '--expt-relaxed-constexpr'
]

GGUF_SAMPLE_MOE = snapshot_download("SzymonOzog/test-gguf-moe-sample")
GGUF_SAMPLE = snapshot_download("Isotr0py/test-gguf-sample")

sources = ["./csrc/my_bindings.cpp", "./csrc/quantization/gguf/gguf_kernel.cu"]
my_extension = load(
    name="my_extension",
    sources=sources,
    extra_cuda_cflags=["-arch=sm_80", "-lineinfo"],  # for CUDA 8.0 arch
    extra_include_paths=["./csrc"],
    verbose=True,
)

def get_gguf_sample_tensors(
        hidden_size: int,
        quant_type: GGMLQuantizationType) -> list[ReaderTensor]:
    sample_dir = GGUF_SAMPLE
    filename = f"Quant_{quant_type.name}_{hidden_size}.gguf"
    sample_file = Path(sample_dir) / filename
    return GGUFReader(sample_file).tensors

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
    # x = torch.ones((num_tokens, H), dtype=dtype, device="cuda")

    topk_weights = torch.rand(num_tokens, top_k, device="cuda", dtype=dtype)
    topk_ids = torch.randint(0, E, (num_tokens, top_k), device="cuda", dtype=torch.int32)

    tensors = get_gguf_MoE_tensors(hidden_size, quant_type)

    w13 = tensors[0]
    w2 = tensors[1]

    w13_dequant = torch.tensor(dequantize(w13.data, quant_type),
                               device="cuda").to(dtype)

    w2_dequant = torch.tensor(dequantize(w2.data, quant_type),
                              device="cuda").to(dtype)
    act = SiluAndMul()
    w = torch.tensor(w13.data, device="cuda")
    w22 = torch.tensor(w2.data, device="cuda")


    output = _fused_moe_gguf(x, w,
                             torch.tensor(w2.data,
                                          device="cuda"), topk_weights,
                             topk_ids, quant_type, quant_type, act, my_extension)

    output_fast = _fused_moe_gguf_new(x, w, w22, topk_weights,
                             topk_ids, quant_type, quant_type, act, my_extension)

    torch.cuda.synchronize()

    print(output.shape)
    print(output)
    print(output_fast)

    ref_output = fused_experts(x, w13_dequant, w2_dequant, topk_weights,
                               topk_ids).reshape(output.shape)
    #
    torch.testing.assert_close(output, ref_output, atol=1, rtol=1e-1)
    torch.testing.assert_close(output_fast, output, atol=1, rtol=1e-1)
    print(ref_output)
    # print(output_fast2)

    # if(num_tokens >= 8):
    #     print(output[56, :22])
    #     print(output_fast[56, :22])
    # print(ref_output[7, 12:42])
    # img = torch.isclose(output, output_fast, atol=1e-2, rtol=1e-1).cpu().numpy().astype(np.uint8) * 255
    # # Convert to PIL image
    # image = Image.fromarray(img, mode="L")  # "L" mode for grayscale
    #
    # image.save("boolean_tensor.png")
#
    # torch.testing.assert_close(output, output_fast, atol=1e-2, rtol=1e-1)

@torch.inference_mode()
def test_mmq(num_tokens: int, hidden_size: int, dtype: torch.dtype,
             quant_type: GGMLQuantizationType):
    current_platform.seed_everything(0)

    tensors = get_gguf_sample_tensors(hidden_size, quant_type)
    x = torch.rand((num_tokens, hidden_size), dtype=dtype, device="cuda")
    for tensor in tensors:

        qweight = torch.tensor(tensor.data, device="cuda")
        output = my_extension.ggml_mul_mat_a8(qweight, x, quant_type, qweight.shape[0])
        ref_output = my_extension.ggml_mul_mat_vec_a8(qweight, x, quant_type, qweight.shape[0])
        atols = {torch.half: 1, torch.bfloat16: 1.5, torch.float: 1.2}
        # test matrix has inputs centered around 0 and lower precision from
        # bfloat16 tends to accumulate and can greatly inflate rtol
        # since outputs are also very close to 0
        rtols = {torch.half: 1e-1, torch.bfloat16: 1e4, torch.float: 2e1}
        print(output.shape, ref_output.shape)
        # torch.testing.assert_close(output,
        #                            ref_output,
        #                            atol=atols[dtype],
        #                            rtol=rtols[dtype])
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
#     for n_tokens in range(1,16):
#         with record_function(f"n_tokens {n_tokens}"):
#             test_mmq(n_tokens, 1024, torch.bfloat16, GGMLQuantizationType.Q4_K)
# prof.export_chrome_trace("trace.json")

test_moe(1, 512, torch.bfloat16, GGMLQuantizationType.Q4_K, 8)
test_moe(8, 512, torch.bfloat16, GGMLQuantizationType.Q4_K, 8)
# test_moe(2048, 512, torch.bfloat16, GGMLQuantizationType.Q4_K, 8)
# # test_moe(8192, 512, torch.bfloat16, GGMLQuantizationType.Q4_K, 8)
# test_moe(16*8192, 512, torch.bfloat16, GGMLQuantizationType.Q4_K, 8)
