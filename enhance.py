import argparse
import os

import torch
import torchaudio
from tqdm import tqdm

from MIDSB.dataset import AudioFolder
from MIDSB.model import DiffusionBridge
from utils.config import read_config_from_yaml
from utils.log import Logger


def main(config, noisy_path, test_path=None):
    noisy_audios = AudioFolder(noisy_path, return_path=True)
    dataloader = noisy_audios.get_data_loader(reverse=config.reverse)

    bridge = DiffusionBridge(config, 0)

    os.makedirs(config.output_path, exist_ok=True)
    config.save(save_path=config.output_path)

    if config.ot_ode:
        config.eval_ot_ode = True
        Logger.warning(f'Set config.eval_ot_ode to True, because config.ot_ode is True')
    with torch.no_grad():
        iter = tqdm(dataloader, desc=f'Enhancing Audios')
        for _x, _audio_name in iter:
            x, audio_name = _x.squeeze(0), _audio_name[0]
            audio_path = os.path.join(config.output_path, f'{audio_name}.wav')
            if not os.path.exists(audio_path) or config.overwrite:
                x, xs, pred_x0s = bridge.enhancement(x, num_step=config.NFE, sampling_method=config.sampling_method, skip_type=config.skip_type, ot_ode=config.eval_ot_ode)
                torchaudio.save(audio_path, x.type(torch.float32).cpu().squeeze().unsqueeze(0), config.data.get('sample_rate', 16000))
    Logger.info('Enhancement process has successfully completed.')

    if config.calc_metrics:
        if test_path is not None:
            from calc_metircs import evaluate_metrics
            evaluate_metrics(test_path, config.output_path, sample_rate = config.data.get('sample_rate', 16000), max_workers=config.max_workers, overwrite=config.overwrite)
        else:
            raise ValueError('test_path is not provided')


if __name__ == '__main__':
    default_config = read_config_from_yaml('config/default_run.yml')
    parser = argparse.ArgumentParser()

    # Run Management
    parser.add_argument("--run_name", type=str, required=True, help="Identifier for run management.")

    # Dataset Configuration
    parser.add_argument("--dataset", type=str, default="voicebank", help="Dataset name, inferred if not provided.")
    parser.add_argument("--subset", type=str, default="test", help="Data subset: train, validation, or test.")
    parser.add_argument("--noisy_path", type=str, default="", help="Path to noisy audio files (optional).")
    parser.add_argument("--test_path", type=str, default="", help="Directory for dataset with 'clean' and 'noisy' subdirs (optional).")

    # Sampling Parameters
    parser.add_argument("--sampling_method", type=str, default="hybrid", help="Sampling method: ddim, hybrid, etc.")
    parser.add_argument("--skip_type", type=str, default="time_uniform", help="How steps are skipped during sampling.")
    parser.add_argument("--NFE", type=int, default=3, help="The number of function evaluations.")

    # Evaluation Flags
    parser.add_argument("--calc_metrics", action="store_true", help="Whether to calculate metrics.")
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite existing results.")
    parser.add_argument("--reverse", action="store_true", help="Whether to reverse processing order.")
    parser.add_argument("--eval_ot_ode", action="store_true", help="Whether to enable OT-ODE evaluation.")
    parser.add_argument("--max_workers", type=int, default=8, help='Max number of workers for calculating the metrics.')

    parser.add_argument('--gpu', type=int, default=0,help="GPU index for inference")

    args = parser.parse_args()


    if args.dataset in default_config.dataset.keys():
        noisy_path = os.path.join(default_config.dataset[args.dataset], 'test', 'noisy')
        test_path = os.path.join(default_config.dataset[args.dataset], 'test')
    elif args.noisy_path != '':
        noisy_path = args.noisy_path
        test_path = args.test_path if args.test_path != '' else None
    else:
        raise ValueError("Please specify a dataset or noisy path")

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    eval_config = {
        "run_path": os.path.join(default_config.run_dir, args.run_name),
        "checkpoint_name": "best.pt",
        "NFE": args.NFE,
        "sampling_method": args.sampling_method,
        "skip_type": args.skip_type,
        "overwrite": args.overwrite,
        "subset": args.subset,
        "eval_ot_ode": args.eval_ot_ode,
        "device": torch.device('cuda:0'),
        "reverse": args.reverse,
        "calc_metrics": args.calc_metrics,
        "sample_rate": default_config.data.get('sample_rate', 16000),
        "output_dir": os.path.join(default_config.output_dir, args.run_name),
        "max_workers": args.max_workers
    }
    config = read_config_from_yaml(config_path=os.path.join(eval_config['run_path'], 'config.yml'))
    config.update(eval_config)

    config.name = f'eval_step={config.NFE}_method={config.sampling_method}_{config.skip_type}'
    config.output_path = os.path.join(config.output_dir, os.path.basename(config.run_path), config.name)
    config.train = False
    config.resume = True
    config.amp = False
    main(config, noisy_path=noisy_path, test_path=test_path)
