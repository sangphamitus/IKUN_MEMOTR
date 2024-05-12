# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
import torch

from utils.utils import distributed_rank
from .memotr import build as build_memotr
from .ikun import build as build_memotr_ikun
from .utils import load_pretrained_model


def build_model(config: dict):
    model = build_memotr(config=config)
    if config["AVAILABLE_GPUS"] is not None and config["DEVICE"] == "cuda":
        model.to(device=torch.device(config["DEVICE"], distributed_rank()))
    else:
        model.to(device=torch.device(config["DEVICE"]))
    if config["PRETRAINED_MODEL"] is not None:
        model = load_pretrained_model(model, config["PRETRAINED_MODEL"], show_details=False)

    if config["FREEZE_MEMOTR"]:
        for  param in model.parameters():
            param.requires_grad = False
    
    my_model = build_memotr_ikun(memotr_model=model, config=config)
    return my_model
