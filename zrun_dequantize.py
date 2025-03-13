# SPDX-License-Identifier: Apache-2.0
import torch
from torch.utils.cpp_extension import load
from gguf import dequantize, GGMLQuantizationType, GGUFReader, ReaderTensor
from huggingface_hub import snapshot_download
from vllm.scalar_type import scalar_types

from pathlib import Path
from typing import List


def get_gguf_sample_tensors(
        hidden_size: int,
        quant_type: GGMLQuantizationType) -> List[ReaderTensor]:
    sample_dir = GGUF_SAMPLE
    filename = f"Quant_{quant_type.name}_{hidden_size}.gguf"
    sample_file = Path(sample_dir) / filename
    return GGUFReader(sample_file).tensors


quant_type = GGMLQuantizationType.Q4_K
GGUF_SAMPLE = snapshot_download("Isotr0py/test-gguf-sample")
tensors = get_gguf_sample_tensors(256, quant_type)
weight = tensors[0]

sources = ["./csrc/my_bindins_deq.cpp", "./csrc/quantization/gguf/gguf_kernel.cu"]
my_extension = load(
    name="my_extension",
    sources=sources,
    extra_cuda_cflags=["-arch=sm_80"],  # for CUDA 8.0 arch
    extra_include_paths=["./csrc"],
    verbose=True,
)

dtype = torch.float32
shape_str = weight.name.split("_")[-1]
shape = map(int, shape_str.split("x"))

x = torch.ones(1, device="cuda", dtype=dtype)

res = my_extension.ggml_deq(torch.tensor(weight.data, device="cuda", dtype=torch.uint8), quant_type, *list(shape), dtype)
ref_output = torch.tensor(dequantize(weight.data, quant_type), device="cuda").to(dtype)
print(quant_type)
print(ref_output[-1])
print(res[-1])
print(torch.allclose(res, ref_output, atol=1e-2, rtol=4e-2))
