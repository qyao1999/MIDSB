import argparse
import os
from random import random

import torch

from MIDSB.model import DiffusionBridge
from utils.config import read_config_from_yaml

os.environ['OMP_NUM_THREADS'] = '4'
def parse_args():
    default_config = read_config_from_yaml('config/default.yml')

    parser = argparse.ArgumentParser(description="Argument parser for model training and evaluation.")
    parser.add_argument('--config_path', type=str, default='config/default.yml',
                        help="Path to the configuration file")
    parser.add_argument('--run_name', type=str, default=default_config.run_name,
                        help="Name for the current run (used for logging and checkpointing)")

    parser.add_argument('--dataset', type=str, default=default_config.dataset,
                        help="Dataset to use for training and evaluation")
    parser.add_argument('--scorer_backbone', type=str, default=default_config.scorer_backbone,
                        help="Backbone architecture for the scorer")
    parser.add_argument('--denoiser_backbone', type=str, default=default_config.denoiser_backbone,
                        help="Backbone architecture for the denoiser")

    parser.add_argument('--beta', type=float, default=default_config.beta,
                        help="Value for the noise schedule 'Beta'")
    parser.add_argument('--bridge_type', type=str, default=default_config.bridge_type,
                        help="Training strategy of denoiser")
    parser.add_argument('--condition', type=str, default=default_config.condition,
                        help="Type of conditioning")
    parser.add_argument('--no_mean_inverting', action='store_true',
                        help="Whether not to use Mean Inversion")
    parser.add_argument('--ot_ode', action='store_true',
                        help="Whether to use optimal transport ODE")

    parser.add_argument('--denoiser_training_strategy', type=str, default=default_config.denoiser_training_strategy,
                        help="Training strategy of denoiser")
    parser.add_argument('--denoiser_loss_weight', type=float, default=default_config.denoiser_loss_weight,
                        help="Weight for the denoiser loss term")
    parser.add_argument('--loss_weight_type', type=str, default=default_config.loss_weight_type,
                        help="Type of loss weight calculation")

    parser.add_argument('--learning_rate', type=float, default=default_config.learning_rate,
                        help="Learning rate for the optimizer")
    parser.add_argument('--batch_size', type=int, default=default_config.batch_size,
                        help="Batch size for training")
    parser.add_argument('--evaluate_batch_size', type=int, default=default_config.evaluate_batch_size,
                        help="Batch size for evaluation")
    parser.add_argument('--patience', type=int, default=default_config.patience,
                        help="Patience for early stopping mechanism")
    parser.add_argument('--seed', type=int, default=default_config.seed,
                        help="Random seed for reproducibility")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Steps for gradient accumulation")

    parser.add_argument('--gpus', type=str, default='0',
                        help="Comma-separated list of GPU indices to use")
    parser.add_argument('--amp', action='store_true',
                        help="Enable Automatic Mixed Precision training")
    parser.add_argument('--float32_precision_high', action='store_true',
                        help="Enable high precision mode for float32 operations")
    parser.add_argument('--dummy', action='store_true',
                        help="Dummy flag for testing purposes")
    parser.add_argument('--resume', action='store_true',
                        help="Resume from a previous checkpoint")
    parser.add_argument('--wandb_log', action='store_true',
                        help="wandb log")
    args = parser.parse_args()

    assert args.condition in ['none', 'y', 'mean', 'both'], 'unsupported condition strategy'
    assert args.bridge_type in ['VP', 'VE'], 'unsupported bridge type'
    assert args.denoiser_training_strategy in ['frozen_denoiser', 'pretrained_denoiser', 'joint_training'], 'unsupported denoiser training strategy'
    assert args.loss_weight_type == 'constant' or args.loss_weight_type.startswith('min_snr'), 'unsupported loss weight type'
    return args


def configure_gpus(args):
    """根据提供的GPU参数配置GPU使用。"""
    gpu_list = [int(gpu.strip()) for gpu in args.gpus.split(',')]
    gpu_num = len(gpu_list)

    args.gpu_num = gpu_num
    if gpu_num > 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpu_list)
        torch.distributed.init_process_group('nccl')
    local_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    return local_rank


def main():
    args = parse_args()
    local_rank = configure_gpus(args)

    if args.float32_precision_high:
        torch.set_float32_matmul_precision('high')

    config = read_config_from_yaml(args.config_path)

    config_updates = {
        'wandb_log': args.wandb_log,
        'resume': args.resume,
        'run_name': args.run_name if args.resume else (
            f'MIDSB_{args.bridge_type}_{args.loss_weight_type}{"_condition_" + args.condition if args.condition != "none" else ""}'
            if args.run_name == 'MIDSB' else args.run_name),
        'dataset': args.dataset if not args.resume else None,
        'scorer_backbone': args.scorer_backbone if not args.resume else None,
        'denoiser_backbone': args.denoiser_backbone if not args.resume else None,
        'batch_size': args.batch_size if not args.resume else None,
        'evaluate_batch_size': args.evaluate_batch_size if not args.resume else None,
        'learning_rate': args.learning_rate if not args.resume else None,
        'beta': args.beta if not args.resume else None,
        'denoiser_loss_weight': args.denoiser_loss_weight if not args.resume else None,
        'mean_inverting': not args.no_mean_inverting if not args.resume else None,
        'loss_weight_type': args.loss_weight_type if not args.resume else None,
        'bridge_type': args.bridge_type if not args.resume else None,
        'denoiser_training_strategy': args.denoiser_training_strategy if not args.resume else None,
        'ot_ode': args.ot_ode if not args.resume else None,
        'condition': args.condition if not args.resume else None,
        'amp': args.amp if not args.resume else None,
        'dummy': args.dummy if not args.resume else None,
        'seed': args.seed if args.seed >= 0 else random.randint(0, 10000),
        'patience': args.patience,
        'gpu_num': args.gpu_num,
        'device': torch.device('cuda', local_rank),
        'batch_size_per_gpu': args.batch_size // args.gpu_num // args.gradient_accumulation_steps if not args.resume else None,
        'gradient_accumulation_steps': args.gradient_accumulation_steps
    }

    # 移除值为None的键值对
    config_updates = {k: v for k, v in config_updates.items() if v is not None}
    # 使用update方法更新config
    config.update(config_updates)
    config.print()
    bridge = DiffusionBridge(config, local_rank)
    bridge.train()


if __name__ == '__main__':
    main()
