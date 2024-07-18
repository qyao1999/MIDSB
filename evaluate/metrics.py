import os
from typing import Dict

import numpy as np
from pesq import pesq
from pystoi import stoi
from scipy.linalg import toeplitz
from .DNSMOS.dnsmos_local import ComputeScore
from .composite_metric import eval_composite
from .registry import MetricRegister

class AudioMetric:
    def __init__(self, **kwargs):
        pass

    def calculate(self, ref_wav, deg_wav, noise_wav, wav_path, sample_rate=16000, **kwargs) -> dict:
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def compute(cls, *args, **kwargs) -> dict:
        instance = cls()
        return instance.calculate(*args, **kwargs)

@MetricRegister.register('pesq')
class PESQMetric(AudioMetric):
    def calculate(self, ref_wav, deg_wav, sample_rate=16000, **kwargs):
        psq_mode = "wb" if sample_rate == 16000 else "nb"
        return {'PESQ': pesq(sample_rate, ref_wav, deg_wav, psq_mode)}

@MetricRegister.register('estoi')
class ESTOIMetric(AudioMetric):
    def calculate(self, ref_wav, deg_wav, sample_rate=16000, **kwargs):
        return {"ESTOI": stoi(ref_wav, deg_wav, sample_rate, extended=True)}

@MetricRegister.register('composite')
class CompositeMetric(AudioMetric):
    def calculate(self, ref_wav, deg_wav, sample_rate=16000, **kwargs):
        result = eval_composite(ref_wav, deg_wav, sample_rate)
        return {"CSIG": result['csig'], "CBAK": result['cbak'], "COVL": result['covl']}

@MetricRegister.register("si_sdr")
class SiSdrMetric(AudioMetric):
    def calculate(self, ref_wav, deg_wav, **kwargs):
        alpha = np.dot(deg_wav, ref_wav) / np.linalg.norm(ref_wav) ** 2
        sdr = 10 * np.log10(np.linalg.norm(alpha * ref_wav) ** 2 / np.linalg.norm(alpha * ref_wav - deg_wav) ** 2)
        return {'SI_SDR': sdr}

@MetricRegister.register("energy_ratios")
class EnergyRatiosMetric(AudioMetric):
    def calculate(self, ref_wav, deg_wav, noise_wav, **kwargs):
        sdr, sir, sar = self.energy_ratios(deg_wav, ref_wav, noise_wav)
        return {'SI_SDR': sdr, 'SI_SIR': sir, 'SI_SAR': sar}

    def energy_ratios(self, s_hat, s, n, eps=1e-10):
        s_target, e_noise, e_art = self.si_sdr_components(s_hat, s, n)
        si_sdr = 10 * np.log10(eps + np.linalg.norm(s_target) ** 2 / (eps + np.linalg.norm(e_noise + e_art) ** 2))
        si_sir = 10 * np.log10(eps + np.linalg.norm(s_target) ** 2 / (eps + np.linalg.norm(e_noise) ** 2))
        si_sar = 10 * np.log10(eps + np.linalg.norm(s_target) ** 2 / (eps + np.linalg.norm(e_art) ** 2))
        return si_sdr, si_sir, si_sar

    def si_sdr_components(self, s_hat, s, n, eps=1e-10):
        alpha_s = np.dot(s_hat, s) / (eps + np.linalg.norm(s) ** 2)
        s_target = alpha_s * s
        alpha_n = np.dot(s_hat, n) / (eps + np.linalg.norm(n) ** 2)
        e_noise = alpha_n * n
        e_art = s_hat - s_target - e_noise
        return s_target, e_noise, e_art

@MetricRegister.register("dnsmos")
class DNSMOSMetric(AudioMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        p808_model_path = os.path.join('evaluate', 'DNSMOS', 'DNSMOS', 'model_v8.onnx')
        primary_model_path = os.path.join('evaluate', 'DNSMOS', 'DNSMOS', 'sig_bak_ovr.onnx')
        self.compute = ComputeScore(primary_model_path, p808_model_path)

    def calculate(self, wav_path, sample_rate=16000, **kwargs):
        result = self.compute(wav_path, sampling_rate=sample_rate, is_personalized_MOS=False)
        return {
            'DNSMOS_SIG': result['SIG'],
            'DNSMOS_BAK': result['BAK'],
            'DNSMOS_OVRL': result['OVRL'],
            'DNSMOS_P808': result['P808_MOS']
        }