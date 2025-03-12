# SPDX-License-Identifier: Apache-2.0
from torch import nn

from vllm.config import LoadFormat, VllmConfig
from vllm.model_executor.model_loader.loader import (BaseModelLoader,
                                                     get_model_loader)
from vllm.model_executor.model_loader.utils import (
    get_architecture_class_name, get_model_architecture)


def get_model(*, vllm_config: VllmConfig) -> nn.Module:
    if "gguf" not in vllm_config.model_config.model:
        load_config = vllm_config.load_config
        load_config.load_format = LoadFormat("safetensors")
        loader = get_model_loader(load_config)
    else:
        loader = get_model_loader(vllm_config.load_config)
    return loader.load_model(vllm_config=vllm_config)


__all__ = [
    "get_model", "get_model_loader", "BaseModelLoader",
    "get_architecture_class_name", "get_model_architecture"
]
