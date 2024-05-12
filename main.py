# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : Main Function.
import os
import argparse
import torch.distributed
import torch.backends.cuda
import torch.backends.cudnn

from utils.utils import distributed_rank
from utils.utils import yaml_to_dict
from configs.utils import update_config


def parse_option():
    parser = argparse.ArgumentParser("Network training and evaluation script.", add_help=True)

    parser.add_argument("--git-version", type=str)

    # About system, Like GPUs:
    parser.add_argument("--available-gpus", type=str, help="Available GPUs, like '0,1,2,3'.")
    parser.add_argument("--use-distributed", action="store_true", help="Use distributed training.")
    parser.add_argument("--use-checkpoint", action="store_true", help="Use gradient checkpoint to save GPU memory.")
    parser.add_argument("--checkpoint-level", type=int)

    # Running mode, Training? Evaluation? or ?
    parser.add_argument("--mode", type=str, help="Running mode.")

    # Only For **Result Submit Process**:
    parser.add_argument("--submit-dir", type=str)
    parser.add_argument("--submit-model", type=str)
    parser.add_argument("--submit-data-split", type=str)

    # Only For **Model Eval Process**:
    parser.add_argument("--eval-dir", type=str)
    parser.add_argument("--eval-mode", type=str)
    parser.add_argument("--eval-model", type=str)
    parser.add_argument("--eval-threads", type=int)
    parser.add_argument("--eval-port", type=int)
    parser.add_argument("--eval-data-split", type=str)
    # basic parameters
    parser.add_argument('--img_hw', nargs='+', type=tuple,
                                default=[(224, 224), (448, 448), (672, 672)])
    parser.add_argument('--norm_mean', type=list, default=[0.48145466, 0.4578275, 0.40821073])
    parser.add_argument('--norm_std', type=list, default=[0.26862954, 0.26130258, 0.27577711])
    parser.add_argument('--random_crop_ratio', nargs='+', type=float, default=[0.8, 1.0])

    # Pretrained Model Load:
    parser.add_argument("--pretrained-model", type=str, help="Pretrained model path.")
    # Resume
    parser.add_argument("--resume", type=str, help="Resume checkpoint path.")
    parser.add_argument("--resume-scheduler", type=str, help="Whether resume the training scheduler.")

    # About Paths:
    # Config file:
    parser.add_argument("--config-path", type=str, help="Config file path.",
                        default="./configs/train_mot17_coco.yaml")
    # Data Path:
    parser.add_argument("--data-root", type=str, help="Dataset root dir.",default="./DATA_DIR")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--data-path", type=str)
    # Log outputs:
    parser.add_argument("--outputs-dir", type=str, help="Outputs dir path.",default="./outputs/memotr_mot17_coco/")

    # Data:
    parser.add_argument("--accumulation-steps", type=int, help="Gradient accumulation steps.")
    parser.add_argument("--batch-size", type=int, help="Batch size for training.",default=1)
    parser.add_argument("--coco-size", type=str)
    parser.add_argument("--overflow-bbox", type=str)
    parser.add_argument("--reverse-clip", type=float)
    parser.add_argument("--use-motsynth", type=str)
    parser.add_argument("--use-crowdhuman", type=str)
    parser.add_argument("--motsynth-rate", type=float)
    parser.add_argument("--sample-steps", type=int, nargs="*")
    parser.add_argument("--sample-lengths", type=int, nargs="*")

    # Training setting：
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lr-points", type=float)
    parser.add_argument("--lr-backbone", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr-drop-milestones", type=int, nargs="*")

    # Submit setting：
    parser.add_argument("--miss-tolerance", type=float)

    # Model setting：
    parser.add_argument("--num-det-queries", type=int)
    parser.add_argument("--merge-det-track-layer", type=int)

    # Training augmentation:
    parser.add_argument("--tp-drop-rate", type=float)
    parser.add_argument("--train-coco", type=str)
    parser.add_argument("--fp-insert-rate", type=float)
    parser.add_argument("--f", type=str)
    return parser.parse_args()


def main(config: dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = config["AVAILABLE_GPUS"]
    config["CONFIG_PATH"] = os.path.join(config["PARENT_ROOT"], config["CONFIG_PATH"])
    config["OUTPUTS_DIR"] = os.path.join(config["PARENT_ROOT"], config["OUTPUTS_DIR"])
    config["PRETRAINED_MODEL"] = os.path.join(config["PARENT_ROOT"], config["PRETRAINED_MODEL"])
    config["TRAIN_COCO"] = os.path.join(config["PARENT_ROOT"], config["TRAIN_COCO"])
    config["CLIP_MODEL"] = os.path.join(config["PARENT_ROOT"],"CLIP", config["CLIP_MODEL"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if config["USE_DISTRIBUTED"]:
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(distributed_rank())

    from train_engine import train
    from submit_engine import submit
    from eval_engine import evaluate
    if config["MODE"] == "train":
        train(config=config)
    elif config["MODE"] == "submit":
        submit(config=config)
    elif config["MODE"] == "eval":
        evaluate(config=config)
    else:
        raise ValueError(f"Unsupported mode '{config['MODE']}'")
    return


if __name__ == '__main__':
    opt = parse_option()                  # runtime options
    cfg = yaml_to_dict(opt.config_path)   # configs

    # Merge parser option and .yaml config, then run main function.
    merged_config = update_config(config=cfg, option=opt)
    merged_config["CONFIG_PATH"] = opt.config_path
    main(config=merged_config)
