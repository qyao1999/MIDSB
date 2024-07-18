# Generative Speech Enhancement using Mean-Inverting Diffusion Schrödinger Bridge

Official Pytorch Implementation of [Generative Speech Enhancement using Mean-Inverting Diffusion Schrödinger Bridge](). 

<p align="center">
  <img src="https://qyao1999.github.io/assets/bridge-_Oti9JSU.png" width="100%"/>
</p>

# Installation


# Datasets


# Pre-trained models


# Training

To train, run
```bash
torchrun --nproc_per_node=4 train_discriminator.py --dataset <target_dataset> --gpus 0,1,2,3 
```

To train, run
```bash
torchrun --nproc_per_node=4 train.py --dataset <target_dataset> --gpus 0,1,2,3 
```
Here are some key arguments you can modify:
- `--config_path`: Path to the YAML configuration file. The default is `config/default.yml`.
- `--dataset`: Dataset to use for training and evaluation.
- `--scorer_backbone`: Backbone architecture for the scorer.
- `--denoiser_backbone`: Backbone architecture for the denoiser.
- `--beta`: Value for the noise schedule 'Beta'.
- `--learning_rate`: Learning rate for the optimizer.
- `--batch_size`: Batch size for training.
- `--evaluate_batch_size`: Batch size for evaluation.
- `--patience`: Patience for early stopping mechanism.
- `--gpus`: Comma-separated list of GPU indices to use. The default is `0`.
- `--resume`: Resume from a previous checkpoint.
- `--wandb_log`: Whether to log to Weights & Biases.

For a full list of arguments, run `python train.py --help`.

# Inference

For inference, run
```bash
python enhancement.py --run_name <your_run_name> --dataset <target_dataset> --sampling_method hybrid --skip_type time_uniform --sample_step 3 --calc_metrics --max_workers 8
```

# Evaluating Metrics

# Citation