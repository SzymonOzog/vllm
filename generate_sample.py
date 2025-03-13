from __future__ import annotations

import argparse
from math import prod
from pathlib import Path
import ctypes
import logging
import numpy as np
import torch
import safetensors
from tqdm import tqdm
import gguf
from gguf.constants import GGMLQuantizationType


logger = logging.getLogger(__name__)


c_float_p = ctypes.POINTER(ctypes.c_float)


class ggml_init_params(ctypes.Structure):
    _fields_ = [
        ("mem_size", ctypes.c_size_t),
        ("mem_buffer", ctypes.c_void_p),
        ("no_alloc", ctypes.c_bool),
    ]


class GGMLQuants:
    libggml: ctypes.CDLL

    def __init__(self, libggml: Path):
        self.libggml = ctypes.CDLL(str(libggml), winmode=0)
        # self.libggml = ctypes.WinDLL(str(libggml), winmode=0)
        self.libggml.ggml_quantize_chunk.restype = ctypes.c_size_t
        # enum ggml_type   type,
        #    const float * src,
        #           void * dst,
        #        int64_t   start,
        #        int64_t   nrows,
        #        int64_t   n_per_row,
        #    const float * imatrix) {
        self.libggml.ggml_quantize_chunk.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
        )

        self.libggml.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
        self.libggml.ggml_quantize_requires_imatrix.argtypes = (ctypes.c_int,)

        for t in (
            "q4_0", "q4_1", "q5_0", "q5_1", "q8_0",
            "q2_K", "q3_K", "q4_K", "q5_K", "q6_K",
            "tq1_0", "tq2_0",
            "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s", "iq1_s", "iq1_m",
            "iq4_nl", "iq4_xs",
        ):
            dequant_func: ctypes._NamedFuncPointer = getattr(self.libggml, "dequantize_row_" + t)
            dequant_func.restype = None
            dequant_func.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64)

        self.libggml.ggml_fp16_to_fp32_row.restype = None
        self.libggml.ggml_fp16_to_fp32_row.argtypes = (ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_float), ctypes.c_int64)
        self.libggml.ggml_bf16_to_fp32_row.restype = None
        self.libggml.ggml_bf16_to_fp32_row.argtypes = (ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_float), ctypes.c_int64)

        self.libggml.ggml_init.argtypes = (ggml_init_params,)

        self.libggml.ggml_init(ggml_init_params(1 * 1024 * 1024, 0, False))

    def dequantize(self, tensor: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray:
        result = np.zeros(gguf.quant_shape_from_byte_shape(tensor.shape, qtype), dtype=np.float32, order="C")
        if qtype == GGMLQuantizationType.F32:
            # no-op
            result = tensor.view(np.float32)
        elif qtype == GGMLQuantizationType.F16:
            self.libggml.ggml_fp16_to_fp32_row(tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), result.ctypes.data_as(c_float_p), result.size)
        elif qtype == GGMLQuantizationType.BF16:
            self.libggml.ggml_bf16_to_fp32_row(tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), result.ctypes.data_as(c_float_p), result.size)
        else:
            lw_qname = qtype.name.lower()
            if lw_qname[-1] == "k":
                lw_qname = lw_qname[:-1] + "K"
            dequant_func: ctypes._NamedFuncPointer = getattr(self.libggml, "dequantize_row_" + lw_qname)
            dequant_func(tensor.ctypes.data_as(ctypes.c_void_p), result.ctypes.data_as(c_float_p), result.size)
        return result

    def quantize(self, data: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray:
        result = np.zeros(gguf.quant_shape_to_byte_shape(data.shape, qtype), dtype=np.uint8, order="C")
        if self.libggml.ggml_quantize_requires_imatrix(qtype.value):
            # TODO: is a column-wise sum of squares appropriate?
            qw = np.sum((data * data).reshape((-1, data.shape[-1])), axis=0).ctypes.data_as(c_float_p)
        else:
            qw = ctypes.cast(0, c_float_p)
        result_size = self.libggml.ggml_quantize_chunk(qtype.value, data.ctypes.data_as(c_float_p), result.ctypes.data_as(ctypes.c_void_p), 0, prod(data.shape[:-1]), data.shape[-1], qw)
        assert result.size == result_size
        return result

def merge_and_upcast(key, tensors):
    output_tensors = []
    for value in range(256):
        formatted_key = key.format(value)
        t = tensors[formatted_key].to(torch.float32)
        upcast_key = formatted_key + "_scale_inv"    
        expanded = tensors[upcast_key].repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)

        h, w = t.shape
        pad_h = (128 - h % 128) % 128
        pad_w = (128 - w % 128) % 128
        padded_t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h))

        multiplied = padded_t * expanded
        unpadded = multiplied[:h, :w]

        output_tensors.append(unpadded)

    return torch.stack(output_tensors)


def upcast(key, tensors):
    formatted_key = key
    t = tensors[formatted_key].to(torch.float32)
    upcast_key = formatted_key + "_scale_inv"
    expanded = tensors[upcast_key].repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)

    h, w = t.shape
    pad_h = (128 - h % 128) % 128
    pad_w = (128 - w % 128) % 128
    padded_t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h))

    multiplied = padded_t * expanded
    unpadded = multiplied[:h, :w]

    return unpadded

def create_sample(ggml_quants: GGMLQuants, qtype: GGMLQuantizationType):
    gguf_writer = gguf.GGUFWriter(f"/nfs/scratch_2/szymon_ozog/DeepSeek_MTP/DeepSeek-R1-MTP-{qtype.name}.gguf", "llama")
    
    tensors = {}
    for i in ["160", "161", "162", "163"]:
        with safetensors.safe_open(f"/nfs/scratch_2/szymon_ozog/DeepSeekR1_full/model-00{i}-of-000163.safetensors", framework="pt", device='cpu') as f:
            for k in f.keys():
                if 'model.layers.61' in k:
                    tensors[k] = f.get_tensor(k)


    # merging tensors :rocket:
    export_tensors = {}

    key = "model.layers.61.mlp.experts.{}.up_proj.weight"
    export_tensors['blk.61.ffn_up_exps.weight'] = merge_and_upcast(key, tensors)

    key = "model.layers.61.mlp.experts.{}.down_proj.weight"
    export_tensors['blk.61.ffn_down_exps.weight'] = merge_and_upcast(key, tensors)

    key = "model.layers.61.mlp.experts.{}.gate_proj.weight"
    export_tensors['blk.61.ffn_gate_exps.weight'] = merge_and_upcast(key, tensors)

    key = "model.layers.61.mlp.shared_experts.down_proj.weight"
    export_tensors['blk.61.ffn_down_shexps.weight'] = upcast(key, tensors)

    key = "model.layers.61.mlp.shared_experts.up_proj.weight"
    export_tensors['blk.61.ffn_up_shexps.weight'] = upcast(key, tensors)

    key = "model.layers.61.mlp.shared_experts.gate_proj.weight"
    export_tensors['blk.61.ffn_gate_shexps.weight'] = upcast(key, tensors)

    export_tensors['blk.61.ffn_gate_inp.weight'] = tensors['model.layers.61.mlp.gate.weight'].to(torch.float32)
    export_tensors['blk.61.exp_probs_b.bias'] = tensors['model.layers.61.mlp.gate.e_score_correction_bias'].to(torch.float32)

    export_tensors['blk.61.attn_kv_a_mqa.weight'] = upcast('model.layers.61.self_attn.kv_a_proj_with_mqa.weight', tensors)
    export_tensors['blk.61.attn_kv_b.weight'] = upcast('model.layers.61.self_attn.kv_b_proj.weight', tensors)

    export_tensors['blk.61.attn_q_a.weight'] = upcast('model.layers.61.self_attn.q_a_proj.weight', tensors)
    export_tensors['blk.61.attn_q_b.weight'] = upcast('model.layers.61.self_attn.q_b_proj.weight', tensors)

    export_tensors['blk.61.attn_output.weight'] = upcast('model.layers.61.self_attn.o_proj.weight', tensors)

    export_tensors['blk.61.attn_kv_a_norm.weight'] = tensors['model.layers.61.self_attn.kv_a_layernorm.weight'].to(torch.float32)
    export_tensors['blk.61.attn_q_a_norm.weight'] = tensors['model.layers.61.self_attn.q_a_layernorm.weight'].to(torch.float32)

    export_tensors['blk.61.attn_norm.weight'] = tensors['model.layers.61.input_layernorm.weight'].to(torch.float32)
    export_tensors['blk.61.ffn_norm.weight'] = tensors['model.layers.61.post_attention_layernorm.weight'].to(torch.float32)

    # new keys?
    export_tensors['token_embd.weight'] = tensors['model.layers.61.embed_tokens.weight']
    export_tensors['blk.61.shared_head.norm.weight'] = tensors['model.layers.61.shared_head.norm.weight'].to(torch.float32)
    export_tensors['blk.61.shared_head.head.weight'] = tensors['model.layers.61.shared_head.head.weight'].to(torch.float32)
    export_tensors['blk.61.eh_proj.weight'] = tensors['model.layers.61.eh_proj.weight'].to(torch.float32)
    export_tensors['blk.61.enorm.weight'] = tensors['model.layers.61.enorm.weight'].to(torch.float32)
    export_tensors['blk.61.hnorm.weight'] = tensors['model.layers.61.hnorm.weight'].to(torch.float32)

    print(export_tensors.keys())
  # 'blk.58.attn_q_a.weight': 'model.layers.58.self_attn.q_a_proj.weight',
  # 'blk.58.attn_q_a_norm.weight': 'model.layers.58.self_attn.q_a_layernorm.weight',
  # 'blk.58.attn_q_b.weight': 'model.layers.58.self_attn.q_b_proj.weight',
  # 'blk.58.attn_kv_a_mqa.weight': 'model.layers.58.self_attn.kv_a_proj_with_mqa.weight',
  # 'blk.58.attn_kv_a_norm.weight': 'model.layers.58.self_attn.kv_a_layernorm.weight',
  # 'blk.58.attn_kv_b.weight': 'model.layers.58.self_attn.kv_b_proj.weight',
  # 'blk.58.attn_output.weight': 'model.layers.58.self_attn.o_proj.weight',
  # 'blk.58.ffn_gate_inp.weight': 'model.layers.58.mlp.gate.weight',
  # 'blk.58.ffn_gate_inp.e_score_correction_bias': 'model.layers.58.mlp.gate.e_score_correction_bias',
  # 'blk.58.ffn_gate_shexp.weight': 'model.layers.58.mlp.shared_experts.gate_proj.weight',
  # 'blk.58.ffn_up_shexp.weight': 'model.layers.58.mlp.shared_experts.up_proj.weight',
  # 'blk.58.ffn_down_shexp.weight': 'model.layers.58.mlp.shared_experts.down_proj.weight',
  # 'blk.58.attn_norm.weight': 'model.layers.58.input_layernorm.weight',
  # 'blk.58.ffn_norm.weight': 'model.layers.58.post_attention_layernorm.weight'



    #gguf_writer.add_tensor(f"blk.61.", ggml_quants.quantize(tensors[k], qtype), raw_dtype=qtype)
    for k, v in tqdm(export_tensors.items()):
        if "down_" in k:
            qtype = GGMLQuantizationType.Q6_K
        elif "norm" in k or "_gate_inp" in k or "exp_probs" in k or "eh_proj" in k:
            qtype = GGMLQuantizationType.F32
        else:
            qtype = GGMLQuantizationType.Q4_K
        print(k, "got dtype", qtype)
        gguf_writer.add_tensor(k, ggml_quants.quantize(v.float().numpy(), qtype), raw_dtype=qtype)

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()

    gguf_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Python (de)quantization against the reference C implementation")
    parser.add_argument("--libggml", type=Path, default="libggml.so", help="The path to libggml.so")
    np.random.seed(0)

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    ggml_quants = GGMLQuants(args.libggml)

    qtypes = [
        GGMLQuantizationType.Q4_K,
    ]

    for qtype in qtypes:
        create_sample(ggml_quants, qtype)

