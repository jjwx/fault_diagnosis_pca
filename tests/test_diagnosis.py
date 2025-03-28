import unittest
import numpy as np
import pandas as pd
import os
import tempfile
from src.feature_extraction import FeatureExtractor
from src.preprocessing import Preprocessor
from src.pca_model import PCADiagnosisModel
from src.diagnosis import FaultDiagnoser


class TestDiagnosis(unittest.TestCase):

    def setUp(self):
        # 创建组件
        self.sampling_rate = 1000
        self.feature_extractor = FeatureExtractor(self.sampling_rate)
        self.preprocessor = Preprocessor()
        self.pca_model = PCADiagnosisModel(n_components=2)
        self.diagnoser = FaultDiagnoser(self.feature_extractor, self.preprocessor, self.pca_model)

        # 创建测试信号
        self.signal_length = 1000
        t = np.arange(0, 1, 1 / self.sampling_rate)

        # 生成正常信号
        self.normal_signals = []
        for i in range(50):
            signal = np.sin(2 * np.pi * 50 * t) + np.random.normal(0, 0.05, self.signal_length)
            self.normal_signals.append(signal)

        # 生成故障信号 - 添加脉冲
        self.fault_signals = []
        for i in range(20):
            signal = np.sin(2 * np.pi * 50 * t)
            impulse_positions = np.random.choice(self.signal_length, size=5, replace=False)
            impulse = np.zeros_like(t)
            impulse[impulse_positions] = 2.0
            signal = signal + impulse + np.random.normal(0, 0.1, self.signal_length)
            self.fault_signals.append(signal)

        # 训练集和测试集
        self.train_signals = self.normal_signals[:40]
        self.test_normal = self.normal_signals[40:]
        self.test_fault = self.fault_signals
        self.test_signals = self.test_normal + self.test_fault
        self.test_labels = [0] * len(self.test_normal) + [1] * len(self.test_fault)

        # 训练模型
        self.diagnoser.train_model(self.train_signals)

    def test_diagnose_normal(self):
        # 测试对正常信号的诊断
        normal_result = self.diagnoser.diagnose(self.test_normal[0])

        # 应该诊断为正常(无故障)
        self.assertFalse(normal_result['fault_detected'])

        # T2和SPE值应在阈值以下
        self.assertLessEqual(normal_result['t2_value'], normal_result['t2_threshold'])
        self.assertLessEqual(normal_result['spe_value'], normal_result['spe_threshold'])

    def test_diagnose_fault(self):
        # 测试对故障信号的诊断
        fault_result = self.diagnoser.diagnose(self.test_fault[0])

        # 应该诊断为故障
        self.assertTrue(fault_result['fault_detected'])

        # T2或SPE值应超过阈值
        self.assertTrue(
            fault_result['t2_value'] > fault_result['t2_threshold'] or
            fault_result['spe_value'] > fault_result['spe_threshold']
        )

        # 检查贡献率是否存在
        if 'max_t2_contribution_feature' in fault_result:
            self.assertIn(fault_result['max_t2_contribution_feature'],
                          ['kurtosis', 'rms', 'crest_factor', 'intensity', 'contrast', 'freq_band_energy_ratio'])

    def test_batch_diagnosis(self):
        # 测试批量诊断
        results = self.diagnoser.diagnose_batch(self.test_signals)

        # 检查结果数据框大小
        self.assertEqual(len(results), len(self.test_signals))

        # 检查列是否存在
        self.assertIn('Fault_Detected', results.columns)
        self.assertIn('T2', results.columns)
        self.assertIn('SPE', results.columns)

    def test_performance_evaluation(self):
        # 测试性能评估
        performance = self.diagnoser.evaluate_performance(self.test_signals, self.test_labels)

        # 检查性能指标是否存在
        self.assertIn('accuracy', performance)
        self.assertIn('precision', performance)
        self.assertIn('recall', performance)
        self.assertIn('f1_score', performance)

        # 检查指标范围
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            self.assertGreaterEqual(performance[metric], 0)
            self.assertLessEqual(performance[metric], 1)

    def test_model_save_load(self):
        # 测试模型保存和加载
        with tempfile.TemporaryDirectory() as temp_dir:
            # 保存模型
            self.diagnoser.save_models(temp_dir)

            # 创建新的诊断器
            new_diagnoser = FaultDiagnoser(self.feature_extractor, Preprocessor(), PCADiagnosisModel())

            # 加载模型
            new_diagnoser.load_models(temp_dir)

            # 测试新诊断器对故障信号的诊断
            fault_result = new_diagnoser.diagnose(self.test_fault[0])

            # 应该诊断为故障
            self.assertTrue(fault_result['fault_detected'])


if __name__ == '__main__':
    unittest.main()