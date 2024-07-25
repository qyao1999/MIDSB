# Generative Speech Enhancement using Mean-Inverting Diffusion Schrödinger Bridge 

Official Pytorch Implementation of [Generative Speech Enhancement using Mean-Inverting Diffusion Schrödinger Bridge](). 

Audio samples are available at [Demo Page](https://qyao1999.github.io/MIDSB/) .

<p align="center">
  <img src="https://qyao1999.github.io/assets/bridge-_Oti9JSU.png" width="100%"/>
</p>



# Installation


# Datasets

We use `Voicebank+DEMAND` and `TIMIT+WHAM!` for training and testing. 

**(IMPORTANT)** Please set the path of these two dataset directories by the `datasets` attribute in `config/default_dataset.yml`. You can also add the paths to your custom dataset directories. 


# Training

To pre-train the discriminator, run
```bash
torchrun --nproc_per_node=4 train_discriminator.py --discriminator_backbone <choosed_backbone> --dataset <target_dataset> --gpus 0,1,2,3 
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
python enhancement.py --run_name <your_run_name> --dataset <target_dataset> --sampling_method hybrid --skip_type time_uniform --NFE 3 --calc_metrics --max_workers 8
```

Here are some key arguments you can modify:
- `--dataset`: Dataset to use for inference.
- `--run_name`: Name of the run to evaluate.
- `--sampling_method`: Sampling method for inference. Available options are `hybrid`, `ddim`.
- `--skip_type`: Type of skip sampling for inference. Available options are `time_uniform`, `logSNR`, `time_quadratic`.
- `--NFE`: The number of function evaluations for inference.
- `--calc_metrics`: Whether to calculate metrics.

For a full list of arguments, run `python enhancement.py --help`.

# Evaluating Metrics
You can also evaluate the metrics by running

```bash
python calculate_metrics.py --test_dir <path_to_your_testset> --enhanced_dir <path_to_your_enhanced_audios> --suffix .wav
```
Here are some key arguments you can modify:
- `--test_dir`: Path to the directory containing the testset.
- `--enhanced_dir`: Path to the directory containing the enhanced audios.
- `--suffix`: Suffix of the audio files.
- 
For a full list of arguments, run `python calculate_metrics.py --help`.
