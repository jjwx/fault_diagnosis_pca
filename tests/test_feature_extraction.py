import unittest
import numpy as np
from src.feature_extraction import FeatureExtractor


class TestFeatureExtraction(unittest.TestCase):

    def setUp(self):
        self.sampling_rate = 1000
        self.extractor = FeatureExtractor(self.sampling_rate)

        # 创建测试信号
        t = np.arange(0, 1, 1 / self.sampling_rate)
        self.sine_signal = np.sin(2 * np.pi * 50 * t)  # 50Hz正弦波
        self.impulse_signal = np.zeros_like(t)
        self.impulse_signal[100:110] = 2.0  # 添加脉冲
        self.test_signal = self.sine_signal + self.impulse_signal + np.random.normal(0, 0.1, len(t))

    def test_kurtosis(self):
        # 测试峭度计算
        kurtosis = self.extractor.calculate_kurtosis(self.test_signal)

        # 检查峭度是否为正数(通常脉冲信号峭度较高)
        self.assertGreater(kurtosis, 0)

        # 对于纯正弦波，理论峭度值为-1.5(减去了正态分布峭度3)
        pure_sine_kurtosis = self.extractor.calculate_kurtosis(self.sine_signal)
        self.assertAlmostEqual(pure_sine_kurtosis, -1.5, delta=0.2)

    def test_rms(self):
        # 测试有效值计算
        rms = self.extractor.calculate_rms(self.sine_signal)

        # 对于幅值为1的正弦波，理论RMS值为1/sqrt(2)≈0.7071
        self.assertAlmostEqual(rms, 1 / np.sqrt(2), delta=0.01)

    def test_crest_factor(self):
        # 测试峰值因子计算
        crest_factor = self.extractor.calculate_crest_factor(self.sine_signal)

        # 对于幅值为1的正弦波，理论峰值因子为sqrt(2)≈1.4142
        self.assertAlmostEqual(crest_factor, np.sqrt(2), delta=0.01)

        # 对于有脉冲的信号，峰值因子应该更高
        impulse_crest_factor = self.extractor.calculate_crest_factor(self.test_signal)
        self.assertGreater(impulse_crest_factor, crest_factor)

    def test_spectrum_calculation(self):
        # 测试频谱计算
        frequencies, spectrum = self.extractor.calculate_spectrum(self.sine_signal)

        # 检查频率范围
        self.assertEqual(len(frequencies), len(self.sine_signal) // 2 + 1)
        self.assertLessEqual(max(frequencies), self.sampling_rate / 2)

        # 对于50Hz正弦波，主要频率分量应该在50Hz附近
        max_freq_idx = np.argmax(spectrum)
        self.assertAlmostEqual(frequencies[max_freq_idx], 50, delta=1)

    def test_intensity(self):
        # 测试烈度计算
        _, spectrum = self.extractor.calculate_spectrum(self.sine_signal)
        intensity = self.extractor.calculate_intensity(spectrum)

        # 烈度应该为正数
        self.assertGreater(intensity, 0)

    def test_contrast(self):
        # 测试对比度计算
        _, spectrum = self.extractor.calculate_spectrum(self.sine_signal)
        contrast = self.extractor.calculate_contrast(spectrum)

        # 对比度应该为正数
        self.assertGreater(contrast, 0)

        # 对于有脉冲的信号，频谱应该更复杂，对比度应该更高
        _, impulse_spectrum = self.extractor.calculate_spectrum(self.test_signal)
        impulse_contrast = self.extractor.calculate_contrast(impulse_spectrum)
        self.assertGreaterEqual(impulse_contrast, contrast)

    def test_frequency_band_energy_ratio(self):
        # 测试频带能量比计算
        frequencies, spectrum = self.extractor.calculate_spectrum(self.test_signal)

        # 不指定频带，应该使用默认频带
        ratio = self.extractor.calculate_frequency_band_energy_ratio(frequencies, spectrum)

        # 比率应该在0到1之间
        self.assertGreaterEqual(ratio, 0)
        self.assertLessEqual(ratio, 1)

        # 指定包含主要频率(50Hz)的频带
        specific_ratio = self.extractor.calculate_frequency_band_energy_ratio(
            frequencies, spectrum, 40, 60)

        # 此频带中的能量比例应该相对较高
        self.assertGreater(specific_ratio, 0.5)

    def test_extract_all_features(self):
        # 测试完整的特征提取
        features = self.extractor.extract_all_features(self.test_signal)

        # 检查所有特征是否存在
        self.assertIn('kurtosis', features)
        self.assertIn('rms', features)
        self.assertIn('crest_factor', features)
        self.assertIn('intensity', features)
        self.assertIn('contrast', features)
        self.assertIn('freq_band_energy_ratio', features)

        # 检查特征值类型
        for key, value in features.items():
            self.assertIsInstance(value, (int, float))


if __name__ == '__main__':
    unittest.main()