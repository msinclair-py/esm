import inspect
from typing import Callable

import torch
import torch.nn as nn
from accelerate import init_empty_weights

from esm.models.esm3 import ESM3
from esm.models.esmc import ESMC
from esm.models.function_decoder import FunctionTokenDecoder
from esm.models.vqvae import StructureTokenDecoder, StructureTokenEncoder
from esm.tokenization import get_esm3_model_tokenizers, get_esmc_model_tokenizers
from esm.utils.constants.esm3 import data_root
from esm.utils.constants.models import (
    ESM3_FUNCTION_DECODER_V0,
    ESM3_OPEN_SMALL,
    ESM3_STRUCTURE_DECODER_V0,
    ESM3_STRUCTURE_ENCODER_V0,
    ESMC_6B,
    ESMC_300M,
    ESMC_600M,
)

ModelBuilder = Callable[[torch.device | str], nn.Module]


def ESM3_structure_encoder_v0(device: torch.device | str = "cpu"):
    with init_empty_weights():
        model = StructureTokenEncoder(
            d_model=1024, n_heads=1, v_heads=128, n_layers=2, d_out=128, n_codes=4096
        ).eval()
    state_dict = torch.load(
        data_root("esm3") / "data/weights/esm3_structure_encoder_v0.pth",
        map_location=device,
    )
    model.load_state_dict(state_dict, assign=True)
    model = model.to(device).to(torch.float32)
    return model


def ESM3_structure_decoder_v0(device: torch.device | str = "cpu"):
    with init_empty_weights():
        model = StructureTokenDecoder(d_model=1280, n_heads=20, n_layers=30).eval()
    state_dict = torch.load(
        data_root("esm3") / "data/weights/esm3_structure_decoder_v0.pth",
        map_location=device,
    )
    model.load_state_dict(state_dict, assign=True)
    model = model.to(device)
    return model


def ESM3_function_decoder_v0(device: torch.device | str = "cpu"):
    with init_empty_weights():
        model = FunctionTokenDecoder().eval()
    state_dict = torch.load(
        data_root("esm3") / "data/weights/esm3_function_decoder_v0.pth",
        map_location=device,
    )
    model.load_state_dict(state_dict, assign=True)
    model = model.to(device)
    return model


def _load_esmc_state_dict(model_key: str, device: torch.device | str):
    """Return the esm-native ``ESMC`` state dict from an ESMC HF snapshot.

    Replaces ``huggingface_hub.load_torch_model`` (broken on hub >= 0.36, which defaults
    to safetensors + a root-level filename pattern). The ESMC repos use two different
    layouts, so both are handled here:

    * **Sharded/single safetensors at the snapshot root** (e.g. ESMC-6B):
      ``model[-NNNNN-of-NNNNN].safetensors`` (+ ``model.safetensors.index.json``). These
      are packaged for the transformers ``ESMCForMaskedLM`` wrapper, so keys are remapped
      to the esm-native ``ESMC`` module: strip the ``esmc.`` submodule prefix and rename
      the output head ``lm_head.`` -> ``sequence_head.``.
    * **Single pickled ``.pth`` under ``data/weights/``** (e.g. ESMC-300M/600M): the same
      layout the ESM3 builders above load explicitly.

    Keys are returned so the caller's ``load_state_dict(..., assign=True)`` (strict)
    surfaces any mismatch — unlike the old ``strict=False`` path, which would silently
    drop the prefixed keys and leave the model on random weights.
    """
    root = data_root(model_key)

    index = root / "model.safetensors.index.json"
    safetensors_files = sorted(root.glob("*.safetensors"))
    if index.exists() or safetensors_files:
        import json

        from safetensors.torch import load_file

        if index.exists():
            shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
            safetensors_files = [root / s for s in shards]
        state_dict: dict = {}
        for shard in safetensors_files:
            state_dict.update(load_file(str(shard), device="cpu"))

        def _to_native(key: str) -> str:
            # ESMCForMaskedLM module prefix -> esm-native ESMC, and head rename.
            key = key.removeprefix("esmc.")
            if key.startswith("lm_head."):
                key = "sequence_head." + key[len("lm_head.") :]
            return key

        return {_to_native(k): v for k, v in state_dict.items()}

    weights_dir = root / "data" / "weights"
    pths = sorted(weights_dir.glob("*.pth"))
    if len(pths) != 1:
        raise FileNotFoundError(
            f"no safetensors at {root} and expected exactly one .pth under "
            f"{weights_dir}, found {[p.name for p in pths]}"
        )
    return torch.load(pths[0], map_location=device)


def ESMC_300M_202412(device: torch.device | str = "cpu", use_flash_attn: bool = True):
    with init_empty_weights():
        model = ESMC(
            d_model=960,
            n_heads=15,
            n_layers=30,
            tokenizer=get_esmc_model_tokenizers(),
            use_flash_attn=use_flash_attn,
        ).eval()
    model.load_state_dict(_load_esmc_state_dict("esmc-300", device), assign=True)
    model = model.to(device)
    return model


def ESMC_600M_202412(device: torch.device | str = "cpu", use_flash_attn: bool = True):
    with init_empty_weights():
        model = ESMC(
            d_model=1152,
            n_heads=18,
            n_layers=36,
            tokenizer=get_esmc_model_tokenizers(),
            use_flash_attn=use_flash_attn,
        ).eval()
    model.load_state_dict(_load_esmc_state_dict("esmc-600", device), assign=True)
    model = model.to(device)
    return model


def ESMC_6B_202412(device: torch.device | str = "cpu", use_flash_attn: bool = True):
    with init_empty_weights():
        model = ESMC(
            d_model=2560,
            n_heads=40,
            n_layers=80,
            tokenizer=get_esmc_model_tokenizers(),
            use_flash_attn=use_flash_attn,
        ).eval()
    model.load_state_dict(_load_esmc_state_dict("esmc-6b", device), assign=True)
    model = model.to(device)
    return model


def ESM3_sm_open_v0(device: torch.device | str = "cpu"):
    with init_empty_weights():
        model = ESM3(
            d_model=1536,
            n_heads=24,
            v_heads=256,
            n_layers=48,
            structure_encoder_fn=ESM3_structure_encoder_v0,
            structure_decoder_fn=ESM3_structure_decoder_v0,
            function_decoder_fn=ESM3_function_decoder_v0,
            tokenizers=get_esm3_model_tokenizers(ESM3_OPEN_SMALL),
        ).eval()
    state_dict = torch.load(
        data_root("esm3") / "data/weights/esm3_sm_open_v1.pth", map_location=device
    )
    model.load_state_dict(state_dict, assign=True)
    model = model.to(device)
    return model


LOCAL_MODEL_REGISTRY: dict[str, ModelBuilder] = {
    ESM3_OPEN_SMALL: ESM3_sm_open_v0,
    ESM3_STRUCTURE_ENCODER_V0: ESM3_structure_encoder_v0,
    ESM3_STRUCTURE_DECODER_V0: ESM3_structure_decoder_v0,
    ESM3_FUNCTION_DECODER_V0: ESM3_function_decoder_v0,
    ESMC_600M: ESMC_600M_202412,
    ESMC_300M: ESMC_300M_202412,
    ESMC_6B: ESMC_6B_202412,
}


def load_local_model(
    model_name: str,
    device: torch.device = torch.device("cpu"),
    use_flash_attn: bool = True,
) -> nn.Module:
    if model_name not in LOCAL_MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in local model registry.")
    builder = LOCAL_MODEL_REGISTRY[model_name]
    kwargs = {}
    if "use_flash_attn" in inspect.signature(builder).parameters:
        kwargs["use_flash_attn"] = use_flash_attn
    return builder(device, **kwargs)


# Register custom versions of ESM3 for use with the local inference API
def register_local_model(model_name: str, model_builder: ModelBuilder) -> None:
    LOCAL_MODEL_REGISTRY[model_name] = model_builder
