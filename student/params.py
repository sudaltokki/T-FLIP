import argparse


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
        "--t-model-checkpoint",
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

    
    parser.add_argument("--epochs", type=int, default=32, help="Number of epochs to train for.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per GPU.")
    parser.add_argument("--t_batch_size", type=int, default=64, help="Batch size per GPU.")   
    parser.add_argument("--alpha_ckd_loss", type=float, default=0., help="CRD loss weight")
    parser.add_argument("--alpha_icl_loss", type=float, default=0., help="ICL_loss weight")
    parser.add_argument("--alpha_fd_loss", type=float, default=0., help="FD_loss weight")
    parser.add_argument("--alpha_affinity_loss", type=float, default=0., help="affinity loss weight")

    args = parser.parse_args(args)

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args
