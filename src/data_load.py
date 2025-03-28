import os
import numpy as np
import pandas as pd


class DataLoader:
    """
    加载和处理真实振动数据
    """

    def __init__(self, data_dir=None):
        """
        初始化数据加载器

        参数:
            data_dir: 数据目录路径
        """
        self.data_dir = data_dir

    def load_csv_data(self, file_path, signal_column='signal', label_column=None):
        """
        从CSV文件加载振动数据

        参数:
            file_path: CSV文件路径
            signal_column: 包含信号数据的列名
            label_column: 包含标签的列名（可选）

        返回:
            signal_data: 信号数据
            labels: 标签（如果有）
        """
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到文件: {file_path}")

        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 提取信号数据
        if signal_column in df.columns:
            signal_data = df[signal_column].values
        else:
            raise ValueError(f"CSV文件中没有名为'{signal_column}'的列")

        # 提取标签（如果有）
        labels = None
        if label_column is not None and label_column in df.columns:
            labels = df[label_column].values

        return signal_data, labels

    def load_numpy_data(self, file_path):
        """
        从NumPy文件加载振动数据

        参数:
            file_path: NumPy (.npy或.npz)文件路径

        返回:
            signal_data: 信号数据
            labels: 标签（如果是.npz文件且包含'labels'）
        """
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到文件: {file_path}")

        # 加载NumPy文件
        if file_path.endswith('.npy'):
            # 如果是.npy文件，只包含信号数据
            signal_data = np.load(file_path)
            labels = None
        elif file_path.endswith('.npz'):
            # 如果是.npz文件，可能包含信号和标签
            data = np.load(file_path)
            if 'signals' in data:
                signal_data = data['signals']
            else:
                # 假设第一个数组是信号
                signal_data = data[list(data.keys())[0]]

            if 'labels' in data:
                labels = data['labels']
            else:
                labels = None
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")

        return signal_data, labels

    def segment_signals(self, signals, segment_length, overlap=0.5):
        """
        将长信号分割成固定长度的片段

        参数:
            signals: 原始信号数组 (n_samples, signal_length)
            segment_length: 每个片段的长度
            overlap: 相邻片段之间的重叠比例 (0到1之间)

        返回:
            segments: 分割后的片段 (n_segments, segment_length)
        """
        if len(signals.shape) == 1:
            # 单个信号
            signal_length = signals.shape[0]
            step = int(segment_length * (1 - overlap))
            n_segments = max(1, (signal_length - segment_length) // step + 1)
            segments = np.zeros((n_segments, segment_length))

            for i in range(n_segments):
                start = i * step
                end = start + segment_length
                if end <= signal_length:
                    segments[i] = signals[start:end]
                else:
                    # 最后一个片段可能超出范围，取最后segment_length个点
                    segments[i] = signals[-segment_length:]

            return segments
        else:
            # 多个信号
            n_samples = signals.shape[0]
            signal_length = signals.shape[1]
            step = int(segment_length * (1 - overlap))
            n_segments_per_sample = max(1, (signal_length - segment_length) // step + 1)
            total_segments = n_samples * n_segments_per_sample
            segments = np.zeros((total_segments, segment_length))

            segment_idx = 0
            for i in range(n_samples):
                for j in range(n_segments_per_sample):
                    start = j * step
                    end = start + segment_length
                    if end <= signal_length:
                        segments[segment_idx] = signals[i, start:end]
                    else:
                        # 最后一个片段可能超出范围
                        segments[segment_idx] = signals[i, -segment_length:]
                    segment_idx += 1

            return segments

    def normalize_signals(self, signals, method='standard'):
        """
        归一化信号数据

        参数:
            signals: 原始信号数组
            method: 归一化方法，'standard'(标准化)或'minmax'(最小-最大缩放)

        返回:
            normalized_signals: 归一化后的信号
        """
        if method == 'standard':
            # Z-score标准化
            mean = np.mean(signals, axis=1, keepdims=True)
            std = np.std(signals, axis=1, keepdims=True)
            std[std == 0] = 1.0  # 防止除零
            return (signals - mean) / std

        elif method == 'minmax':
            # 最小-最大缩放
            min_vals = np.min(signals, axis=1, keepdims=True)
            max_vals = np.max(signals, axis=1, keepdims=True)
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1.0  # 防止除零
            return (signals - min_vals) / range_vals

        else:
            raise ValueError(f"不支持的归一化方法: {method}")