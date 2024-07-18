import os
import time
from datetime import datetime

import torch
import wandb
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from backbone.registry import BackboneRegister
from evaluate.registry import MetricRegister
from utils.common import seed_everything, scaler_format
from utils.config import Config, read_config_from_yaml
from utils.log import progress_visualize, ProcessBar, Logger
from .dataset import ComplexSpec, STFTUtil
from .sdes import VP_DiffusionBridgeSDE, VE_DiffusionBridgeSDE


class DiffusionBridge():
    def __init__(self, con=None, local_rank=0):
        self.local_rank = local_rank

        # Configuration
        assert con is not None, 'Configuration must not be None'
        if isinstance(con, dict):
            self.config = Config(con)
        elif isinstance(con, str):
            if os.path.exists(con) and con.endswith(('.yml', '.yaml')):
                self.config = read_config_from_yaml(config_path=con)
            else:
                raise NotImplementedError(f'Configuration file from \'{con}\' doesn\'t exist or is not a yaml file.')
        elif isinstance(con, Config):
            self.config = con
        else:
            raise NotImplementedError(f'The type of configuration  is not supported.')

        # SEED
        self.config.seed = seed_everything(self.config.seed + local_rank)
        self.device = self.config.device
        self.config.device = None

        self.is_resume = self.config.get('resume', False)
        self.is_train = self.config.get('train', True)

        if self.is_train:
            # wandb
            if not self.is_resume:
                self.config.run_name = self.config.run_name + '_' + datetime.now().strftime('%m%d%H%M')
                self.config.run_path = os.path.join(self.config.run_dir, self.config.run_name)
                os.makedirs(self.config.run_path, exist_ok=True)
                self.config.wandb_id = wandb.util.generate_id()

            if self.local_rank == 0:
                if self.config.wandb_log:
                    self.wandb_run = wandb.init(id=self.config.wandb_id, project="MIDSB", name=self.config.run_name, config=self.config.dict(), resume='allow')
                    bridge_artifact = wandb.Artifact(name='bridge', type='code')
                    bridge_artifact.add_file(os.path.join(os.path.dirname(__file__), 'model.py'))
                    wandb.log_artifact(bridge_artifact)
                self.config.save()

        if self.local_rank == 0:
            self.config.print()

        # Model
        # generator
        if self.config.condition in ['none']:
            input_channels = 2
        elif self.config.condition in ['y', 'mean']:
            input_channels = 4
        elif self.config.condition in ['both']:
            input_channels = 6
        else:
            raise NotImplementedError(f'Condition {self.config.condition} is not supported yet.')

        self.generator = BackboneRegister.fetch(self.config.generator_backbone)(input_channels=input_channels)
        self.generator.to(self.device)

        # discriminator
        if self.config.discriminator_training_strategy != 'none':
            self.discriminator = BackboneRegister.fetch(self.config.discriminator_backbone)(input_channels=2, discriminative=True)
            self.discriminator.to(self.device)
            if self.is_train and not self.is_resume and self.config.discriminator_training_strategy in ['frozen_discriminator', 'pretrained_discriminator']:
                discriminator_checkpoint_path = os.path.join('pretrained_discriminator', self.config.dataset, f'discriminator_{self.config.discriminator_backbone}.pt')
                if not os.path.exists(discriminator_checkpoint_path):
                    raise RuntimeError(f"The discriminator checkpoint at path '{discriminator_checkpoint_path}' does not exist. Please ensure the path is correct or the discriminator has been pre-trained.")
                checkpoint = torch.load(discriminator_checkpoint_path, map_location='cpu')
                self.discriminator.load_state_dict(checkpoint)

        if local_rank == 0:
            from torchinfo import summary
            summary(self.generator, col_names=("num_params", "params_percent", "trainable"))

        if self.is_train:
            self.ema = ExponentialMovingAverage(self.parameters(), decay=self.config.ema_rate)
            self.ema.to(self.device)

            if self.config.optimizer == 'Adam':
                self.optimizer = optim.Adam(self.parameters(), lr=self.config.learning_rate)
            elif self.config.optimizer == 'AdamW':
                self.optimizer = optim.AdamW(self.parameters(), lr=self.config.learning_rate)
            else:
                raise NotImplementedError(f'Optimizer {self.config.optimizer} not supported yet!')

            self.amp = self.config.get('amp', False)
            self.autocast = torch.cuda.amp.autocast(enabled=self.amp)
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

            if not self.is_resume:
                self.config.current_epoch = -1
                self.config.best_valid_loss = 1e8
                self.config.best_epoch = -1
                self.config.num_step = 0

        # Output directory
        self.checkpoint_dir = os.path.join(self.config.run_path, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if self.is_resume:
            self.restore()

        sde_cls = None
        if self.config.bridge_type == 'VP':
            sde_cls = VP_DiffusionBridgeSDE
        elif self.config.bridge_type == 'VE':
            sde_cls = VE_DiffusionBridgeSDE
        self.sde = sde_cls(beta=self.config.beta, t_max=self.config.t_max, loss_weight_type=self.config.loss_weight_type, device=self.device)

        self.t_max, self.t_min = self.config.t_max, self.config.t_min

        if self.is_train and torch.distributed.is_initialized():
            self.generator = DDP(self.generator, device_ids=[local_rank], output_device=local_rank)
            if self.config.discriminator != 'none':
                self.discriminator = DDP(self.discriminator, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        self._reduce_op = lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

        # Function
        self.generator_loss_fn = self._generator_loss_fn()
        self.step_fn = self._step_fn()

    def parameters(self):
        if self.config.discriminator_training_strategy not in ['none', 'frozen_discriminator']:
            def merge_parameters(*iterables):
                for it in iterables:
                    yield from it

            return merge_parameters(self.generator.parameters(), self.discriminator.parameters())
        return self.generator.parameters()

    def state(self):
        if not self.is_train:
            return {'generator': self.generator.state_dict()}

        generator_state_dict = self.generator.module.state_dict() if isinstance(self.generator, DDP) else self.generator.state_dict()
        discriminator_state_dict = self.discriminator.module.state_dict() if isinstance(self.discriminator, DDP) else self.discriminator.state_dict()
        state = {'ema': self.ema.state_dict(), 'generator': generator_state_dict, 'optimizer': self.optimizer.state_dict()}
        if self.config.discriminator_training_strategy != 'none':
            state['discriminator'] = discriminator_state_dict
        return state

    def save(self):
        assert self.is_train
        torch.save(self.state(), os.path.join(self.config.run_path, f'best.pt'))
        self.config.save()

    def restore(self):
        checkpoint = torch.load(os.path.join(self.config.run_path, f'best.pt'), map_location="cpu")
        self.generator.load_state_dict(checkpoint['model'])

        if self.is_train:
            if self.config.discriminator != 'none':
                self.discriminator.load_state_dict(checkpoint['discriminator'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            if self.config.condition in ['mean', 'both']:
                self.discriminator.load_state_dict(checkpoint['discriminator'])

    def generator_fn(self, x, t, cond=None):
        input = torch.cat([x] + cond, dim=1) if cond is not None else x
        return self.generator(input, t)

    def discriminator_fn(self, x):
        assert self.config.discriminator_training_strategy != 'none'
        x_bar = self.discriminator(x)
        return x_bar

    def _generator_loss_fn(self):
        def _generator_loss_fn(x, y, x_bar=None):
            timestep = torch.rand(x.shape[0], device=x.device) * (self.t_max - self.t_min) + self.t_min

            if self.config.discriminator_training_strategy != 'none' and self.config.mean_inverting:
                xt, x0_hat = self.sde.q_sample(t=timestep, x0=x, x1=y, ot_ode=self.config.ot_ode, x0_bar=x_bar)
                label = self.sde.compute_label(t=timestep, x0=x, xt=xt, x0_hat=x0_hat)
            else:
                xt = self.sde.q_sample(t=timestep, x0=x, x1=y, ot_ode=self.config.ot_ode)
                label = self.sde.compute_label(t=timestep, x0=x, xt=xt)

            condition_map = {
                'none': [],
                'y': [x],
                'mean': [x_bar],
                'both': [x_bar, x]
            }
            cond = condition_map.get(self.config.condition, [])

            score = self.generator_fn(xt, timestep, cond=cond)

            complex_batch_loss = torch.square(torch.abs(score - label))
            complex_loss = torch.mean(self._reduce_op(complex_batch_loss.reshape(complex_batch_loss.shape[0], -1), dim=-1))
            generator_loss = complex_loss

            return generator_loss

        return _generator_loss_fn

    def _step_fn(self):
        def _step_fn(x, y):
            with self.autocast:
                if self.config.discriminator_training_strategy != 'none':
                    with torch.set_grad_enabled(self.config.discriminator_training_strategy != 'frozen_discriminator'):
                        x_bar = self.discriminator_fn(y)

                    if self.config.discriminator_training_strategy != 'frozen_discriminator':
                        batch_loss = torch.square(torch.abs(x - x_bar))
                        discriminator_loss = torch.mean(self._reduce_op(batch_loss.reshape(batch_loss.shape[0], -1), dim=-1))

                    generator_loss = self.generator_loss_fn(x, y=y, x_bar=x_bar)
                else:
                    generator_loss = self.generator_loss_fn(x, y=y)

                if self.config.discriminator_training_strategy not in ['none', 'frozen_discriminator']:
                    return (1 - self.config.discriminator_loss_weight) * generator_loss + self.config.discriminator_loss_weight * discriminator_loss, {"train/generator_loss": generator_loss.item(), "train/discriminator_loss": discriminator_loss.item()}
                else:
                    return generator_loss, {"train/generator_loss": generator_loss.item()}

        return _step_fn

    def train_dataloader(self, ):
        train_dataset = ComplexSpec(dataset=self.config.dataset, subset='train', shuffle_spec=True, return_spec=True, dummy=self.config.dummy)

        if torch.distributed.is_initialized():
            self.train_sampler = DistributedSampler(train_dataset, shuffle=True)
            return torch.utils.data.DataLoader(train_dataset, batch_size=self.config.batch_size_per_gpu, shuffle=False, num_workers=self.config.num_workers, pin_memory=True, sampler=self.train_sampler)
        else:
            return torch.utils.data.DataLoader(train_dataset, batch_size=self.config.batch_size_per_gpu, shuffle=True, num_workers=self.config.num_workers, pin_memory=True)

    def valid_dataloader(self, ):
        valid_dataset = ComplexSpec(dataset=self.config.dataset, subset='valid', shuffle_spec=True, return_spec=True, dummy=self.config.dummy)
        return torch.utils.data.DataLoader(valid_dataset, batch_size=self.config.evaluate_batch_size, shuffle=False, num_workers=self.config.num_workers, pin_memory=True, )

    def prepair_data(self, clean_b, noisy_b):
        clean_b, noisy_b = clean_b.to(self.device), noisy_b.to(self.device)
        return clean_b, noisy_b

    def train(self):
        train_dataloader, valid_dataloader = self.train_dataloader(), (self.valid_dataloader() if self.local_rank == 0 else None)
        num_step = int(self.config.num_step)
        if self.local_rank == 0:
            earlystop_counter = 0
            best_valid_loss = self.config.best_valid_loss
            best_epoch = self.config.best_epoch
            valid_loss_history = []

        for epoch in range(self.config.current_epoch + 1, self.config.num_epoch):
            train_loss_history = []
            if torch.distributed.is_initialized():
                self.train_sampler.set_epoch(epoch)
            # Train
            if self.ema.collected_params is not None:
                self.ema.restore(self.parameters())
            self.generator.train()

            if self.local_rank == 0:
                train_bar = ProcessBar(train_dataloader, len(train_dataloader) // self.config.logging_per_step) if self.local_rank == 0 else train_dataloader
                iter = train_bar.get_iter()
                train_bar.update(epoch, train_loss_history, {'step': num_step})
            else:
                iter = train_dataloader

            accumulation_loss = 0
            train_loss_per_epoch = 0
            for clean_b, noisy_b in iter:
                x, y = self.prepair_data(clean_b, noisy_b)
                loss, all_loss = self.step_fn(x, y)

                if self.amp:
                    loss = self.scaler.scale(loss)

                loss = loss / self.config.gradient_accumulation_steps
                accumulation_loss += loss.item()
                train_loss_per_epoch += loss.item()
                loss.backward()

                all_loss = {k: v / self.config.gradient_accumulation_steps for k, v in all_loss.items()}

                if (num_step + 1) % self.config.gradient_accumulation_steps == 0 or (num_step + 1) % len(train_dataloader) == 0:
                    if self.amp:
                        self.scaler.unscale_(self.optimizer)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.ema.update(self.parameters())
                    self.optimizer.zero_grad()

                    global_step = int((num_step + 1) / self.config.gradient_accumulation_steps)
                    if self.local_rank == 0 and global_step % self.config.logging_per_step == 0:
                        train_loss_history.append(accumulation_loss)
                        train_bar.update(epoch, train_loss_history, {'step': global_step, 'loss': scaler_format(accumulation_loss, 4)})
                        if self.config.wandb_log:
                            wandb.log({'step': global_step, 'train/loss_per_step': accumulation_loss, **all_loss})
                    accumulation_loss = 0
                num_step += 1


            if self.local_rank == 0:
                train_loss_per_epoch = train_loss_per_epoch / len(train_dataloader)
                if self.config.wandb_log:
                    wandb.log({'epoch': epoch, 'train/loss_per_epoch': train_loss_per_epoch})
                # valid
                self.ema.store(self.parameters())  # store current params in EMA
                self.ema.copy_to(self.parameters())  # copy EMA parameters over current params for evaluation
                self.generator.eval()
                valid_loss = 0.0
                valid_step = 0
                with torch.no_grad():
                    valid_iter = tqdm(valid_dataloader, desc=f'Epoch {epoch} Validing', leave=False, ncols=200)

                    for clean_b, noisy_b in valid_iter:
                        x, y = self.prepair_data(clean_b, noisy_b)
                        loss, _ = self.step_fn(x, y)
                        valid_loss += loss.item()
                        if valid_step % self.config.logging_per_step == 0:
                            valid_iter.set_postfix_str({'valid_loss': scaler_format(loss.item(), 4)})
                        valid_step += 1

                valid_loss = valid_loss / len(valid_dataloader)
                if self.config.wandb_log:
                    wandb.log({'epoch': epoch, 'valid/loss': valid_loss})

                self.config.current_epoch = epoch
                self.config.num_step = num_step
                valid_loss_history.append(valid_loss)

                valid_metrics = self.evaluate_metrics(subset='valid', epoch=epoch, n_samples=10)
                test_metrics = self.evaluate_metrics(subset='test', epoch=epoch, n_samples=10)
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_epoch = epoch
                    earlystop_counter = 0
                    self.config.best_epoch = best_epoch
                    self.config.best_valid_loss = best_valid_loss
                    self.save()  # self.save(epoch)
                    Logger.info(
                        f'「{time.strftime("%m-%d %H:%M:%S", time.localtime())}」Epoch {epoch} - History |{progress_visualize(valid_loss_history, sample_evenly=False)}| Train Loss = {scaler_format(train_loss_per_epoch, 4)}, Valid Loss = {scaler_format(valid_loss, 4)} (best: {scaler_format(best_valid_loss, 4)} in Epoch {best_epoch}), '
                        f'PESQ = {scaler_format(valid_metrics["pesq"], 2)}(val)/{scaler_format(test_metrics["pesq"], 2)}(test), SI-SDR = {scaler_format(valid_metrics["si_sdr"], 2)}(val)/{scaler_format(test_metrics["si_sdr"], 2)}(test)')
                    Logger.info(f'Save checkpoint to \033[95m{os.path.abspath(self.config.run_path)}\033[0m.')
                else:
                    earlystop_counter += 1
                    Logger.info(
                        f'「{time.strftime("%m-%d %H:%M:%S", time.localtime())}」Epoch {epoch} - History |{progress_visualize(valid_loss_history, sample_evenly=False)}| Train Loss = {scaler_format(train_loss_per_epoch, 4)}, Valid Loss = {scaler_format(valid_loss, 4)} (best: {scaler_format(best_valid_loss, 4)} in Epoch {best_epoch}), '
                        f'PESQ = {scaler_format(valid_metrics["pesq"], 2)}(val)/{scaler_format(test_metrics["pesq"], 2)}(test), SI-SDR = {scaler_format(valid_metrics["si_sdr"], 2)}(val)/{scaler_format(test_metrics["si_sdr"], 2)}(test) | '
                        f'Early Stopping: {earlystop_counter}/{self.config.patience}')
                    if earlystop_counter >= self.config.patience:
                        Logger.info(f"Early stop after {epoch} epoch(es)")
                        break

        self.config.best_valid_loss = best_valid_loss
        self.config.best_epoch = best_epoch

        if self.local_rank == 0 and self.config.wandb_log:
            model_artifact = wandb.Artifact(name='model', type='checkpoint')
            model_artifact.add_file(os.path.join(self.config.run_path, f'best.pt'))
            wandb.log_artifact(model_artifact)

    def evaluate_metrics(self, subset='valid', epoch=None, n_samples=0, sampling_method='ddim', num_step=5):
        dataset = ComplexSpec(dataset=self.config.dataset, subset=subset, return_raw=True)

        if n_samples < 1:
            n_samples = len(dataset)
        else:
            n_samples = min(n_samples, len(dataset))

        metrics = MetricRegister.fetch(['pesq', 'estoi', 'si_sdr'])

        result = {}
        for i in tqdm(range(0, n_samples), desc=f'Evaluate Metrics {n_samples}/{len(dataset)}', leave=False, ncols=200):
            x, y = dataset[i * (len(dataset) // (n_samples - 1))]

            x_bar, _, _ = self.enhancement(y, num_step=num_step, sampling_method=sampling_method, ot_ode=self.config.ot_ode)
            clean_sig, enhanced_sig = x.cpu().squeeze().numpy(), x_bar.type(torch.float32).cpu().squeeze().numpy()

            for metric in metrics.values():
                metric_res = metric.compute(ref_wav=clean_sig, deg_wav=enhanced_sig, sample_rate=self.config.sample_rate)
                for item in metric_res.keys():
                    if item not in result.keys():
                        result[item] = 0.0
                    result[item] += metric_res[item]


        result = { k: v/n_samples for k,v in result.items()}


        if self.config.wandb_log:
            log_dict = {f'{subset}/{name}': metric for name, metric in result.items()}
            if epoch is not None:
                log_dict['epoch'] = epoch
            wandb.log(log_dict)

        return result

    def enhancement(self, audio, num_step=5, sampling_method='ddim', skip_type='time_uniform', ot_ode=None):
        if ot_ode is None:
            ot_ode = self.config.ot_ode

        x, invert_fn = STFTUtil.to_stft(audio, device=self.device)

        condition_map = {
            'none': [],
            'y': [x],
            'mean': [self.discriminator_fn(x)],
            'both': [self.discriminator_fn(x), x]
        }
        cond = condition_map.get(self.config.condition, [])

        global NFE
        NFE = 0

        @torch.no_grad()
        def pred_x0_fn(xt, timestep):
            global NFE
            timestep = torch.full((xt.shape[0],), timestep, device=self.device, dtype=torch.float32)
            out = self.generator_fn(xt, timestep, cond=cond)
            NFE = NFE + 1
            return self.sde.compute_pred_x0(t=timestep, xt=xt, net_out=out)

        if sampling_method == 'ddim':
            sampler = self.sde.get_ddim_solver(model_fn=pred_x0_fn, num_step=num_step, ot_ode=ot_ode)
        elif sampling_method == 'hybrid':
            sampler = self.sde.get_hybrid_solver(model_fn=pred_x0_fn, num_step=num_step, skip_type=skip_type, ot_ode=ot_ode)
        else:
            raise NotImplementedError
        xs, pred_x0s = sampler.sampling(x=x)

        audio = invert_fn(xs[:, 0, :, :])
        assert NFE == num_step, "The number of function evaluations should match the number of steps."

        return audio, xs, pred_x0s
