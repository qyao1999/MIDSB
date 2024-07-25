import concurrent
import json
import os
from argparse import ArgumentParser
from glob import glob
from os.path import join

import numpy as np
import pandas as pd
from soundfile import read
from tabulate import tabulate
from tqdm import tqdm

from evaluate import MetricRegister
from utils.log import Logger


def mean_std(data):
    data = data[~np.isnan(data)]
    mean = np.mean(data)
    std = np.std(data)
    return mean, std


def evaluate(metrics:dict, clean_dir:str, noisy_dir:str, enhanced_dir:str, filename:str, suffix, sample_rate:int=16000):
    x, _ = read(join(clean_dir, filename))
    y, _ = read(join(noisy_dir, filename))

    x_hat_path = join(enhanced_dir, filename.replace('.wav', suffix))
    x_hat, _ = read(x_hat_path)

    len = min(x.size, y.size, x_hat.size)
    x, y = x[:len], y[:len]
    x_hat = x_hat[:len]
    n = y - x

    result = {"filename": filename}
    for metric in metrics.values():
        metric_res = metric.compute(ref_wav=x, deg_wav=x_hat, noise_wav=n, sample_rate=sample_rate, wav_path=x_hat_path)
        result.update(metric_res)
    return result


def evaluate_metrics(test_dir:str, enhanced_dir:str, suffix:str='.wav', sample_rate:int=16000, max_workers:int=8, overwrite:bool=True):
    Logger.info(f"Enhanced Dictionary: \033[95m{enhanced_dir}\033[0m.")
    Logger.info(f"Test Dictionary: \033[95m{test_dir}\033[0m.")

    output_json_path = join(enhanced_dir, "metrics.json")
    results = None
    if os.path.exists(output_json_path) and not overwrite:
        # Load and print the existing JSON data
        with open(output_json_path, "r") as json_file:
            results = json.load(json_file)
        Logger.info(f"Results are loaded from existing JSON file: \033[95m`{output_json_path}`\033[0m.")
    else:
        clean_dir = join(test_dir, "clean/")
        noisy_dir = join(test_dir, "noisy/")

        data = {"filename": []}

        # Evaluate standard metrics
        noisy_files = sorted(glob('{}/*.wav'.format(noisy_dir)))

        file_names = []
        for noisy_file in tqdm(noisy_files, desc=f'Recognizing'):
            filename = noisy_file.split('/')[-1]
            if not os.path.exists(join(enhanced_dir, filename.replace('.wav', suffix))):
                raise ValueError(f'The suffix `{suffix}` may be wrong')
            file_names.append(filename)
        Logger.info(f"Number of files: {len(file_names)}")

        metric_fns = MetricRegister.fetch(['pesq', 'estoi', 'composite', 'energy_ratios', 'dnsmos'])
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_name = {executor.submit(evaluate, metric_fns, clean_dir, noisy_dir, enhanced_dir, filename, suffix, sample_rate): filename for filename in file_names}
            for future in tqdm(concurrent.futures.as_completed(future_to_name), desc=f'Calculating', total=len(file_names)):
                name = future_to_name[future]
                try:
                    result = future.result()
                except Exception as exc:
                    Logger.error(f"{name} generated an exception: {exc}")
                else:
                    for item in result.keys():
                        if item not in data.keys():
                            data[item] = []
                        data[item].append(result[item])

        # Save results as DataFrame
        df = pd.DataFrame(data)
        df.to_csv(join(enhanced_dir, "_results.csv"), index=False)

        results = {
            "Metric": [],
            "Mean": [],
            "Std": []
        }

        for metric in [k for k in data.keys() if k != 'filename']:
            mean, std = mean_std(df[metric].to_numpy(dtype=np.float64))  # Explicitly convert to float64
            results["Metric"].append(metric.upper())
            results["Mean"].append(np.float64(mean))
            results["Std"].append(np.float64(std))

        output_json_path = join(enhanced_dir, "metrics.json")
        with open(output_json_path, "w") as json_file:
            json.dump(results, json_file, indent=4)
        Logger.info(f"Results have been saved to {output_json_path}")

    headers = ["Metric", "Mean", "Std"]
    Logger.info("\nSummary of Results:")
    print(tabulate(results, headers=headers, floatfmt=".4f", numalign="right"))




if __name__ == '__main__':
    parser = ArgumentParser(description='Script to evaluate audio quality metrics on a set of enhanced audio files.')
    parser.add_argument(
        "--test_dir", type=str, default='/data/SpeechEnhancement/VoiceBank+DEMAND/data/test/',
        help='Directory path where the original test data is stored. This directory must contain subdirectories named "clean/" and "noisy/" for accessing clean and noisy versions of the audio files.'
    )
    parser.add_argument(
        "--enhanced_dir", type=str, required=True,
        help='Directory path where the enhanced audio files are located. These are the files that will be evaluated against the original clean audio files.'
    )
    parser.add_argument(
        "--suffix", type=str, default='.wav',
        help='Suffix used to identify enhanced audio files within the `enhanced_dir`. This should match the naming convention of your enhanced files.'
    )
    parser.add_argument(
        "--sample_rate", type=int, default=16000,
        help='Sample rate of the audio files. This script assumes all audio files have the same sample rate.'
    )
    parser.add_argument(
        "--max_workers", type=int, default=8,
        help='Max number of workers.'
    )
    args = parser.parse_args()


    evaluate_metrics(
        test_dir=args.test_dir,
        enhanced_dir=args.enhanced_dir,
        suffix=args.suffix,
        sample_rate=args.sample_rate,
        max_workers=args.max_workers
    )
