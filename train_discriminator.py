import argparse
import os

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm
from yotool.util.logger import Logger

from MIDSB.dataset import ComplexSpec, STFTUtil
from backbone import BackboneRegister
from evaluate import MetricRegister
from utils.config import Config, read_config_from_yaml


def main(config, local_rank=0):
    device = torch.device(config.device)
    os.makedirs(os.path.join(config.output_path, config.dataset), exist_ok=True)
    discriminator = BackboneRegister.fetch(config.discriminator_backbone)(input_channels=2, discriminative=True)

    discriminator.to(device)
    if torch.distributed.is_initialized():
        discriminator = DDP(discriminator, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    print(f"Trainable parameters {sum(p.numel() for p in discriminator.parameters() if p.requires_grad) / 1e6 : .2f} M")

    if config.ema:
        ema = ExponentialMovingAverage(discriminator.parameters(), decay=config.ema_rate)
        ema.to(device)

    optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.learning_rate)

    train_dataset = ComplexSpec(dataset=config.dataset, subset='train', shuffle_spec=True, return_spec=True)

    if torch.distributed.is_initialized():
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            sampler=train_sampler)
    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True)

    valid_dataset = ComplexSpec(dataset=config.dataset, subset='valid', shuffle_spec=False, return_spec=True)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True)

    _reduce_op = lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    earlystop_counter = 0
    num_step = 0
    best_valid_loss = 1e7
    for epoch in range(config.num_epoch):
        if torch.distributed.is_initialized():
            train_sampler.set_epoch(epoch)

        if config.ema and ema.collected_params is not None:
            ema.restore(discriminator.parameters())
        discriminator.train()

        iter = tqdm(train_dataloader, desc=f'Epoch {epoch}', leave=False, ncols=200) if local_rank == 0 else train_dataloader
        for clean_b, noisy_b in iter:
            x, y = clean_b.to(device), noisy_b.to(device)

            x_hat = discriminator(y)

            complex_loss = torch.square(torch.abs(x_hat - x))
            complex_loss = torch.mean(_reduce_op(complex_loss.reshape(complex_loss.shape[0], -1), dim=-1))

            train_loss = complex_loss

            train_loss.backward()

            optimizer.step()
            if config.ema:
                ema.update(discriminator.parameters())
            optimizer.zero_grad()

            if local_rank == 0 and num_step % 10 == 0:
                iter.set_postfix_str({'step': num_step, 'loss': train_loss.item()})
            num_step += 1

        if local_rank == 0:
            discriminator.eval()

            if config.ema:
                ema.store(discriminator.parameters())  # store current params in EMA
                ema.copy_to(discriminator.parameters())  # copy EMA parameters over current params for evaluation
            valid_total_loss = 0
            with torch.no_grad():
                valid_iter = tqdm(valid_dataloader, desc='Valid', leave=False, ncols=200)
                for clean_b, noisy_b in valid_iter:
                    x, y = clean_b.to(device), noisy_b.to(device)
                    x_hat = discriminator(y)
                    complex_loss = torch.square(torch.abs(x_hat - x))
                    complex_loss = torch.mean(_reduce_op(complex_loss.reshape(complex_loss.shape[0], -1), dim=-1))
                    valid_loss = complex_loss
                    valid_total_loss += valid_loss.item()

            valid_loss = valid_total_loss / len(valid_dataloader)
            dataset = ComplexSpec(subset='valid', return_raw=True)

            m_pesq, m_estoi, m_si_sdr = 0.0, 0.0, 0.0
            n_samples = 20

            pesq_fn, estoi_fn, si_sdr_fn = MetricRegister.fetch('pesq'), MetricRegister.fetch('estoi'), MetricRegister.fetch('si_sdr')
            with torch.no_grad():
                for i in tqdm(range(0, n_samples), desc=f'Evaluate Metrics', leave=False, ncols=200):
                    x, y = dataset[i * (len(dataset) // (n_samples - 1))]
                    y, invert = STFTUtil.to_stft(y, device)
                    x_hat = discriminator(y)
                    x_hat = invert(x_hat)
                    clean_sig, enhanced_sig = x.detach().squeeze().numpy(), x_hat.type(torch.float32).detach().squeeze().numpy()

                    m_pesq += pesq_fn(ref_wav = clean_sig, deg_wav = enhanced_sig, sample_rate = 16000)
                    m_estoi += estoi_fn(ef_wav = clean_sig, deg_wav = enhanced_sig, sample_rate = 16000)
                    m_si_sdr += si_sdr_fn(ef_wav = clean_sig, deg_wav = enhanced_sig, sample_rate = 16000)
                m_pesq, m_estoi, m_si_sdr = m_pesq / n_samples, m_estoi / n_samples, m_si_sdr / n_samples

            result = {'pesq': m_pesq, 'estoi': m_estoi, 'si_sdr': m_si_sdr}
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                Logger.info(f'Epoch {epoch} - Best Valid Loss {best_valid_loss}. Save checkpoint to \033[95m{os.path.abspath(config.output_path)}\033[0m')
                earlystop_counter = 0
                Logger.info(result)
                if torch.distributed.is_initialized():
                    torch.save(discriminator.module.state_dict(), os.path.join(config.output_path, config.dataset, f'discriminator_{config.discriminator_backbone}1.pt'))
                else:
                    torch.save(discriminator.state_dict(), os.path.join(config.output_path, config.dataset, f'discriminator_{config.discriminator_backbone}1.pt'))
            else:
                earlystop_counter += 1
                Logger.info(f'Epoch {epoch} - Valid Loss {valid_loss} Best Valid Loss {best_valid_loss}')
                print(result)
                if earlystop_counter >= config.patience:
                    Logger.info(f"Early stop after {epoch} epoch(es)")
                    break


if __name__ == '__main__':
    default_config = read_config_from_yaml('config/default_run.yml')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=default_config.dataset, help="Dataset to use for training and evaluation")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument('--discriminator_backbone', type=str, default=default_config.discriminator_backbone, help="Backbone architecture for the discriminator")
    parser.add_argument('--gpus', type=str, default='0',
                        help="Comma-separated list of GPU indices to use")
    args = parser.parse_args()

    gpu_list = [int(gpu.strip()) for gpu in args.gpus.split(',')]
    gpu_num = len(gpu_list)

    if gpu_num > 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpu_list)
        torch.distributed.init_process_group('nccl')
    local_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    config = Config({
        'device': torch.device('cuda', local_rank),
        'discriminator_backbone': args.discriminator_backbone,
        'batch_size': args.batch_size // gpu_num,
        'dataset': args.dataset,
        'num_workers': 4,
        'num_epoch': 200,
        'learning_rate': args.learning_rate,
        'ema_rate': 0.999,
        'ema': True,
        'patience': 10,
        'output_path': 'pretrained_discriminator/',
    })
    config.print()
    main(config, local_rank)
