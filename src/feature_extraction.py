import numpy as np
from scipy import signal
import pandas as pd


class FeatureExtractor:
    """
    用于从时域和频域提取特征的类
    """

    def __init__(self, sampling_rate):
        """
        初始化特征提取器

        参数:
        sampling_rate (float): 信号采样率，单位Hz
        """
        self.sampling_rate = sampling_rate

    def extract_all_features(self, signal_data):
        """
        提取所有六个特征

        参数:
        signal_data (numpy.ndarray): 原始信号数据

        返回:
        dict: 包含所有特征的字典
        """
        # 提取时域特征
        kurtosis = self.calculate_kurtosis(signal_data)
        rms = self.calculate_rms(signal_data)
        crest_factor = self.calculate_crest_factor(signal_data)

        # 计算频谱
        frequencies, spectrum = self.calculate_spectrum(signal_data)

        # 提取频域特征
        intensity = self.calculate_intensity(spectrum)
        contrast = self.calculate_contrast(spectrum)
        freq_band_energy_ratio = self.calculate_frequency_band_energy_ratio(frequencies, spectrum)

        # 汇总所有特征
        features = {
            'kurtosis': kurtosis,
            'rms': rms,
            'crest_factor': crest_factor,
            'intensity': intensity,
            'contrast': contrast,
            'freq_band_energy_ratio': freq_band_energy_ratio
        }

        return features

    def calculate_kurtosis(self, signal_data):
        """
        计算信号的峭度

        参数:
        signal_data (numpy.ndarray): 原始信号数据

        返回:
        float: 峭度值
        """
        mean = np.mean(signal_data)
        std = np.std(signal_data)
        n = len(signal_data)
        kurtosis = np.sum(((signal_data - mean) / std) ** 4) / n
        return kurtosis

    def calculate_rms(self, signal_data):
        """
        计算信号的有效值(RMS)

        参数:
        signal_data (numpy.ndarray): 原始信号数据

        返回:
        float: RMS值
        """
        return np.sqrt(np.mean(np.square(signal_data)))

    def calculate_crest_factor(self, signal_data):
        """
        计算信号的峰值因子

        参数:
        signal_data (numpy.ndarray): 原始信号数据

        返回:
        float: 峰值因子
        """
        peak = np.max(np.abs(signal_data))
        rms = self.calculate_rms(signal_data)
        return peak / rms if rms != 0 else 0

    def calculate_spectrum(self, signal_data):
        """
        计算信号的频谱

        参数:
        signal_data (numpy.ndarray): 原始信号数据

        返回:
        tuple: (frequencies, spectrum)
        """
        n = len(signal_data)
        frequencies = np.fft.rfftfreq(n, 1 / self.sampling_rate)
        spectrum = np.abs(np.fft.rfft(signal_data)) / n
        return frequencies, spectrum

    def calculate_intensity(self, spectrum):
        """
        计算频谱的烈度

        参数:
        spectrum (numpy.ndarray): 频谱数据

        返回:
        float: 烈度值
        """
        return np.sum(spectrum ** 2)

    def calculate_contrast(self, spectrum):
        """
        计算频谱的对比度

        参数:
        spectrum (numpy.ndarray): 频谱数据

        返回:
        float: 对比度值
        """
        if len(spectrum) <= 1:
            return 0
        normalized_spectrum = spectrum / np.max(spectrum) if np.max(spectrum) > 0 else spectrum
        contrast = np.mean(np.abs(np.diff(normalized_spectrum)))
        return contrast

    def calculate_frequency_band_energy_ratio(self, frequencies, spectrum,
                                              band_start=None, band_end=None):
        """
        计算特定频带能量与总能量的比值

        参数:
        frequencies (numpy.ndarray): 频率数组
        spectrum (numpy.ndarray): 频谱数据
        band_start (float, optional): 频带开始频率
        band_end (float, optional): 频带结束频率

        返回:
        float: 频带能量比
        """
        # 如果未指定频带，使用频谱的1/4到1/2范围作为默认值
        if band_start is None:
            band_start = frequencies[len(frequencies) // 4]
        if band_end is None:
            band_end = frequencies[len(frequencies) // 2]

        # 计算总能量
        total_energy = np.sum(spectrum ** 2)

        # 找出频带范围内的索引
        band_indices = np.where((frequencies >= band_start) & (frequencies <= band_end))

        # 计算频带内能量
        band_energy = np.sum(spectrum[band_indices] ** 2)

        # 计算比值
        ratio = band_energy / total_energy if total_energy > 0 else 0

        return ratio

    def extract_features_batch(self, signal_data_batch):
        """
        批量提取特征

        参数:
        signal_data_batch (list): 信号数据列表

        返回:
        pandas.DataFrame: 特征数据框
        """
        features_list = []

        for signal_data in signal_data_batch:
            features = self.extract_all_features(signal_data)
            features_list.append(features)

        return pd.DataFrame(features_list)