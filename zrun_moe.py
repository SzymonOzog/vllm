# SPDX-License-Identifier: Apache-2.0
import torch
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.cpp_extension import load

from vllm import _custom_ops as ops
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size
from vllm.model_executor.layers.quantization.gguf import _fuse_mul_mat

sources = ["./csrc/my_bindins.cpp", "./csrc/quantization/gguf/gguf_kernel.cu"]
my_extension = load(
    name="my_extension",
    sources=sources,
    extra_cuda_cflags=["-arch=sm_80", "-lineinfo"],  # for CUDA 8.0 arch
    extra_include_paths=["./csrc"],
    verbose=True,
)

def reload_ext():
    sources = ["./csrc/my_bindins.cpp", "./csrc/quantization/gguf/gguf_kernel.cu"]
    return load(
        name="my_extension",
        sources=sources,
        extra_cuda_cflags=["-arch=sm_80", "-lineinfo"],  # for CUDA 8.0 arch
        extra_include_paths=["./csrc"],
        verbose=True,
    )

def _fused_moe_gguf(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    qweight_type: int,
    qweight_type2: int,
    act,
) -> torch.Tensor:

    num_tokens, _ = x.shape
    E, N, _ = w1.shape
    top_k = topk_ids.shape[1]
    out_hidden_states = torch.empty_like(x)
    # TODO get real block size
    BLOCK_SIZE = 4

    sorted_token_ids, expert_ids, _ = moe_align_block_size(
        topk_ids, BLOCK_SIZE, E)
    out = my_extension.ggmp_moe_a8(x, w1, sorted_token_ids, expert_ids,
                                   qweight_type, N, top_k, num_tokens)
    out = act(out)
    out = my_extension.ggmp_moe_a8(out, w2, sorted_token_ids, expert_ids,
                                   qweight_type2, w2.shape[1], 1,
                                   num_tokens * top_k)
    out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(topk_weights.view(num_tokens, top_k, 1))
    ops.moe_sum(out, out_hidden_states)
    return out_hidden_states


num_tokens = 8
# y = torch.arange(7168, device="cuda", dtype=torch.float16) * 0.01
# print(y)
# x = torch.vstack([y for i in range(num_tokens)])
# print(x, x.shape)
# y = torch.arange(num_tokens, device="cuda", dtype=float16) * 0.01 x = torch.ones(num_tokens, 7168, device="cuda", dtype=torch.float16)
act = SiluAndMul()
state = torch.load("curr_state.pt")

w13_qweight = state["w13_qweight"].to("cuda")
w2_qweight = state["w2_qweight"].to("cuda")

w13_qweight_type = 10
w2_qweight_type = 11
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    for num_tokens in range(1, 2049):
    # for num_tokens in [1]:
        x = torch.randn(num_tokens, 7168, device="cuda", dtype=torch.float16)
        topk_weights = torch.randn(num_tokens,
                                   8,
                                   device="cuda",
                                   dtype=torch.float16)
        topk_ids = torch.randint(0, 256, (num_tokens, 8), device="cuda")
        # topk_ids = torch.ones(num_tokens, 8, device="cuda", dtype=torch.int64) * 255
        # print(topk_ids)

        final_hidden_states = torch.empty_like(x)
        with record_function(f"Custom moe {num_tokens}"):
            final_hidden_states_kern = _fused_moe_gguf(x, w13_qweight,
                                                       w2_qweight,
                                                       topk_weights, topk_ids,
                                                       w13_qweight_type,
                                                       w2_qweight_type, act)
            torch.cuda.synchronize()
        up_proj = torch.empty_like(final_hidden_states_kern)
        out_toks = []
        with record_function(f"naive moe {num_tokens}"):
            for tok, (w, idx) in enumerate(zip(topk_weights, topk_ids)):
                inp = x[tok].reshape((1, ) + x.shape[1:])
                current_hidden_state = None
                i = 0
                for ww, ii in zip(w, idx):
                    expert_up = w13_qweight[ii]

                    out = _fuse_mul_mat(inp, expert_up, w13_qweight_type)
                    # if tok == 0:
                    #     print(out[0, :3], out.shape)
                    out = act(out)

                    expert_down = w2_qweight[ii]
                    current_state = _fuse_mul_mat(out, expert_down,
                                                  w2_qweight_type).mul_(ww)
                    # up_proj[tok * 8 + i] = current_state
                    if current_hidden_state is None:
                        current_hidden_state = current_state
                    else:
                        current_hidden_state.add_(current_state)
                    i += 1
                final_hidden_states[tok] = current_hidden_state
            torch.cuda.synchronize()
        atol = 1e-2
        test = final_hidden_states
        # print(test[..., -10:])
        # print(final_hidden_states_kern[..., -10:])
        print(test[..., -5:] - final_hidden_states_kern[..., -5:])
        print(f"""teasting moe full num tokens {num_tokens}
            allclose = 
              {torch.allclose(test, final_hidden_states_kern, atol=atol),}
mean abs difference{torch.abs(test - final_hidden_states_kern).mean()},
max abs difference {torch.abs(test - final_hidden_states_kern).max()}""")
        # print(f"""teasting moe experts up projection num tokens {num_tokens}
        #     allclose = 
        #     {torch.allclose(up_proj, final_hidden_states_kern, atol=atol)}""")
prof.export_chrome_trace("trace.json")
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
# mean abs difference{torch.abs(up_proj - final_hidden_states_kern).mean()},
# max abs difference {torch.abs(up_proj - final_hidden_states_kern).max()}""")
