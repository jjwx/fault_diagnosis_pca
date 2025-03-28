import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class FaultDiagnoser:
    """
    故障诊断系统类，集成特征提取、预处理和PCA模型
    """

    def __init__(self, feature_extractor, preprocessor, pca_model):
        """
        初始化故障诊断器

        参数:
        feature_extractor: 特征提取器实例
        preprocessor: 预处理器实例
        pca_model: PCA模型实例
        """
        self.feature_extractor = feature_extractor
        self.preprocessor = preprocessor
        self.pca_model = pca_model

    def process_signal(self, signal_data, filter_signal=True, remove_noise=False):
        """
        处理原始信号并提取特征

        参数:
        signal_data (numpy.ndarray): 原始信号数据
        filter_signal (bool): 是否进行滤波
        remove_noise (bool): 是否去噪

        返回:
        pandas.DataFrame: 特征数据框
        """
        # 信号预处理
        processed_signal = signal_data.copy()

        if filter_signal:
            processed_signal = self.preprocessor.filter_signal(processed_signal)

        if remove_noise:
            processed_signal = self.preprocessor.remove_noise(processed_signal)

        # 提取特征
        features = self.feature_extractor.extract_all_features(processed_signal)
        features_df = pd.DataFrame([features])

        return features_df

    def process_signal_batch(self, signal_data_batch, filter_signal=True, remove_noise=False):
        """
        批量处理信号并提取特征

        参数:
        signal_data_batch (list): 信号数据列表
        filter_signal (bool): 是否进行滤波
        remove_noise (bool): 是否去噪

        返回:
        pandas.DataFrame: 特征数据框
        """
        processed_signals = []

        for signal_data in signal_data_batch:
            # 信号预处理
            processed_signal = signal_data.copy()

            if filter_signal:
                processed_signal = self.preprocessor.filter_signal(processed_signal)

            if remove_noise:
                processed_signal = self.preprocessor.remove_noise(processed_signal)

            processed_signals.append(processed_signal)

        # 批量提取特征
        features_df = self.feature_extractor.extract_features_batch(processed_signals)

        return features_df

    def train_model(self, normal_signals, filter_signal=True, remove_noise=False, n_components=2):
        """
        使用正常信号训练模型

        参数:
        normal_signals (list): 正常运行状态下的信号列表
        filter_signal (bool): 是否进行滤波
        remove_noise (bool): 是否去噪
        n_components (int): PCA保留的主成分数量

        返回:
        self: 训练好的诊断器实例
        """
        # 处理信号并提取特征
        features_df = self.process_signal_batch(normal_signals, filter_signal, remove_noise)

        # 标准化特征
        normalized_features = self.preprocessor.normalize_features(features_df, fit=True)

        # 设置PCA组件数量
        self.pca_model.n_components = n_components

        # 训练PCA模型
        self.pca_model.fit(normalized_features)

        return self

    def diagnose(self, signal_data, filter_signal=True, remove_noise=False):
        """
        对新信号进行故障诊断

        参数:
        signal_data (numpy.ndarray): 原始信号数据
        filter_signal (bool): 是否进行滤波
        remove_noise (bool): 是否去噪

        返回:
        dict: 诊断结果
        """
        # 处理信号并提取特征
        features_df = self.process_signal(signal_data, filter_signal, remove_noise)

        # 标准化特征
        normalized_features = self.preprocessor.normalize_features(features_df)

        # 使用PCA模型进行诊断
        diagnosis_result = self.pca_model.diagnose(normalized_features)

        # 提取关键诊断信息
        result = {
            'fault_detected': diagnosis_result['Fault_Detected'].iloc[0],
            't2_value': diagnosis_result['T2'].iloc[0],
            't2_threshold': diagnosis_result['T2_Threshold'].iloc[0],
            't2_violation': diagnosis_result['T2_Violation'].iloc[0],
            'spe_value': diagnosis_result['SPE'].iloc[0],
            'spe_threshold': diagnosis_result['SPE_Threshold'].iloc[0],
            'spe_violation': diagnosis_result['SPE_Violation'].iloc[0],
            'fault_score': diagnosis_result['Fault_Score'].iloc[0]
        }

        # 如果检测到故障，计算贡献率
        if result['fault_detected']:
            t2_contrib, spe_contrib = self.pca_model.calculate_contribution(normalized_features)

            # 查找贡献最大的特征
            max_t2_feature = t2_contrib.iloc[0].idxmax()
            max_spe_feature = spe_contrib.iloc[0].idxmax()

            result['max_t2_contribution_feature'] = max_t2_feature
            result['max_spe_contribution_feature'] = max_spe_feature
            result['t2_contributions'] = t2_contrib.iloc[0].to_dict()
            result['spe_contributions'] = spe_contrib.iloc[0].to_dict()

        return result

    def diagnose_batch(self, signal_data_batch, filter_signal=True, remove_noise=False):
        """
        批量诊断信号

        参数:
        signal_data_batch (list): 信号数据列表
        filter_signal (bool): 是否进行滤波
        remove_noise (bool): 是否去噪

        返回:
        pandas.DataFrame: 诊断结果数据框
        """
        # 处理信号并提取特征
        features_df = self.process_signal_batch(signal_data_batch, filter_signal, remove_noise)

        # 标准化特征
        normalized_features = self.preprocessor.normalize_features(features_df)

        # 使用PCA模型进行诊断
        diagnosis_results = self.pca_model.diagnose(normalized_features)

        return diagnosis_results

    def evaluate_performance(self, signal_data_batch, true_labels, filter_signal=True, remove_noise=False):
        """
        评估诊断性能

        参数:
        signal_data_batch (list): 信号数据列表
        true_labels (list): 真实标签列表 (0: 正常, 1: 故障)
        filter_signal (bool): 是否进行滤波
        remove_noise (bool): 是否去噪

        返回:
        dict: 性能指标
        """
        # 批量诊断
        diagnosis_results = self.diagnose_batch(signal_data_batch, filter_signal, remove_noise)

        # 提取预测标签
        predicted_labels = diagnosis_results['Fault_Detected'].astype(int)

        # 计算性能指标
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)

        # 如果有连续的故障得分，还可以计算AUC
        if 'Fault_Score' in diagnosis_results.columns:
            auc = roc_auc_score(true_labels, diagnosis_results['Fault_Score'])
        else:
            auc = None

        performance = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }

        return performance

    def save_models(self, base_path):
        """
        保存所有模型组件

        参数:
        base_path (str): 基础保存路径
        """
        import os

        # 创建目录
        os.makedirs(base_path, exist_ok=True)

        # 保存标准化器
        self.preprocessor.save_scaler(os.path.join(base_path, 'scaler.pkl'))

        # 保存PCA模型
        self.pca_model.save_model(os.path.join(base_path, 'pca_model.pkl'))

    def load_models(self, base_path):
        """
        加载所有模型组件

        参数:
        base_path (str): 基础加载路径
        """
        import os

        # 加载标准化器
        self.preprocessor.load_scaler(os.path.join(base_path, 'scaler.pkl'))

        # 加载PCA模型
        self.pca_model.load_model(os.path.join(base_path, 'pca_model.pkl'))