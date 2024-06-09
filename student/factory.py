import json
import logging
import os
import pathlib
import re
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from student.models import CLIP, get_cast_dtype, convert_weights_to_lp


_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_config/"]
print(_MODEL_CONFIG_PATHS)
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        with open(cf, 'r') as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}


_rescan_model_configs()  # initial populate of model config registry


def list_models():
    """ enumerate available model architectures based on config files """
    return list(_MODEL_CONFIGS.keys())


def add_model_config(path):
    """ add model config path or file and update registry """
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()


def get_model_config(model_name):
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None


def create_model(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_custom_text: bool = False,
        pretrained_image: bool = False,
        cache_dir: Optional[str] = None,
):
    model_name = model_name.replace('/', '-')  # for callers using old naming with / in ViT names
    if isinstance(device, str):
        device = torch.device(device)


    model_cfg = get_model_config(model_name)
    if model_cfg is not None:
        logging.info(f'Loaded {model_name} model config.')
    else:
        raise RuntimeError(f'Model config for {model_name} not found.')


    cast_dtype = get_cast_dtype(precision)
    custom_text = model_cfg.pop('custom_text', False) or force_custom_text or ('hf_model_name' in model_cfg['text_cfg'])

    model = CLIP(**model_cfg, cast_dtype=cast_dtype)


    model.to(device=device)
    if precision in ("fp16", "bf16"):
        convert_weights_to_lp(model, dtype=torch.bfloat16 if precision == 'bf16' else torch.float16)

    # set image / mean metadata from pretrained_cfg if available, or use default
    model.visual.image_mean = OPENAI_DATASET_MEAN
    model.visual.image_std = OPENAI_DATASET_STD

    if jit:
        model = torch.jit.script(model)

    return model


def create_kd_model_and_transforms(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_custom_text: bool = False,
        cache_dir: Optional[str] = None,
):
    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_custom_text=force_custom_text,
        cache_dir=cache_dir,
    )


    return model