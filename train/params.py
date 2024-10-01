import argparse

from utils.utils import get_datetime
import utils.pairsampler as pair

LookupChoices = type('', (argparse.Action, ), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))

def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4}
    else:
        return {"lr": 5.0e-4}


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--t_model",
        type=str,
        default="RN50",
        help="Name of the teacher vision backbone to use.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RN50",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--t_model_checkpoint",
        type=str,
        default="RN50",
        help="teacher checkpoint path",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="O",
        help="benchmark"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="resume"
    )
    parser.add_argument(
        "--op_dir",
        type=str,
        default=None,
        help="path to ckpt folder"
    )
    parser.add_argument(
        "--report_logger_path",
        type=str,
        default=None,
        help="path to log"
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action='store_true',
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )
    parser.add_argument(
        "--force-custom-text",
        default=False,
        action='store_true',
        help="Force use of CustomTextCLIP model (separate text-tower).",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="dataset root",
    )
    parser.add_argument(
        "--dataroot",
        type=str,
        default=None,
        help="dataset root",
    )
    parser.add_argument(
        "--pretrained",
        default='',
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--return_prob",
        default=False,
        help="return probability and label in csv"
    )
    parser.add_argument(
        "--swin",
        default=False,
        help="use swin transformer instead of vit"
    )
    parser.add_argument(
        "--vis",
        type=bool,
        default=False,
        help="attention map visualizing"
    )

    
    parser.add_argument("--iterations", type=int, default=4000, help="Number of iterations to train for.")
    parser.add_argument("--epochs", type=int, default=32, help="Number of epochs to train for.")
    parser.add_argument("--run", type=int, default=5, help="run")

    parser.add_argument("--scheduler", type=str, default="", help="set scheduler")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=None, help="Weight Decay")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU.")
    parser.add_argument("--total_batch_size", type=int, default=64, help="Batch size per GPU.")
    parser.add_argument("--t_batch_size", type=int, default=64, help="Batch size per GPU.") 

    parser.add_argument("--alpha_cls_loss", type=float, default=1., help="CLS loss weight")
    parser.add_argument("--alpha_sim_loss", type=float, default=1., help="sim loss weight")
    parser.add_argument("--alpha_l2_loss", type=float, default=1., help="l2 loss weight")

    parser.add_argument("--alpha_ckd_loss", type=float, default=0., help="CRD loss weight")
    parser.add_argument("--alpha_fd_loss", type=float, default=0., help="FD loss weight")
    parser.add_argument("--alpha_affinity_loss", type=float, default=0., help="affinity loss weight")
    parser.add_argument("--alpha_gd_loss", type=float, default=0., help="gradient loss weight")
    parser.add_argument("--attn_ratio", type=float, default=0., help="attention loss weight")

    parser.add_argument('--triplet_ratio', default=0, type=float)
    parser.add_argument('--dist_ratio', default=0, type=float)
    parser.add_argument('--angle_ratio', default=0, type=float)

    parser.add_argument('--dark_ratio', default=0, type=float)
    parser.add_argument('--dark_alpha', default=2, type=float)
    parser.add_argument('--dark_beta', default=3, type=float)

    parser.add_argument('--at_ratio', default=0, type=float)

    parser.add_argument('--triplet_sample',
                        choices=dict(random=pair.RandomNegative,
                                    hard=pair.HardNegative,
                                    all=pair.AllPairs,
                                    semihard=pair.SemiHardNegative,
                                    distance=pair.DistanceWeighted),
                        default=pair.DistanceWeighted,
                        action=LookupChoices)

    parser.add_argument('--triplet_margin', type=float, default=0.2)

    parser.add_argument('--distkd_ratio', type=bool, default=False)

    parser.add_argument("--current_time", type=float, default=get_datetime(), help="datetime in Seoul")

    # wandb
    parser.add_argument("--set_wandb", type=bool, default=True, help="wandb")
    parser.add_argument("--user", type=str, default="", help="user name")

    args = parser.parse_args(args)

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args
