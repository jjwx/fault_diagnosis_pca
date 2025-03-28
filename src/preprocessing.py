# import numpy as np
# from scipy import signal
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
#
#
# class Preprocessor:
#     """
#     数据预处理类，用于信号处理和特征标准化
#     """
#
#     def __init__(self):
#         """
#         初始化预处理器
#         """
#         self.scaler = StandardScaler()
#         self.is_fitted = False
#
#     def filter_signal(self, signal_data, lowcut=10, highcut=1000, fs=2048, order=5):
#         """
#         使用带通滤波器过滤信号
#
#         参数:
#         signal_data (numpy.ndarray): 原始信号数据
#         lowcut (float): 低截止频率
#         highcut (float): 高截止频率
#         fs (float): 采样率
#         order (int): 滤波器阶数
#
#         返回:
#         numpy.ndarray: 滤波后的信号
#         """
#         nyq = 0.5 * fs
#         low = lowcut / nyq
#         high = highcut / nyq
#
#         b, a = signal.butter(order, [low, high], btype='band')
#         filtered_signal = signal.filtfilt(b, a, signal_data)
#
#         return filtered_signal
#
#     def remove_noise(self, signal_data, window_size=5):
#         """
#         使用中值滤波去除噪声
#
#         参数:
#         signal_data (numpy.ndarray): 原始信号数据
#         window_size (int): 滤波窗口大小
#
#         返回:
#         numpy.ndarray: 去噪后的信号
#         """
#         return signal.medfilt(signal_data, window_size)
#
#     def normalize_features(self, features_df, fit=False):
#         """
#         标准化特征
#
#         参数:
#         features_df (pandas.DataFrame): 包含特征的数据框
#         fit (bool): 是否训练标准化器
#
#         返回:
#         pandas.DataFrame: 标准化后的特征
#         """
#         if fit or not self.is_fitted:
#             self.scaler.fit(features_df)
#             self.is_fitted = True
#
#         scaled_features = self.scaler.transform(features_df)
#         return pd.DataFrame(scaled_features, columns=features_df.columns)
#
#     def segment_signal(self, signal_data, segment_length, overlap=0):
#         """
#         将长信号分割成多个片段
#
#         参数:
#         signal_data (numpy.ndarray): 原始信号数据
#         segment_length (int): 每个片段的长度
#         overlap (float): 重叠比例，范围[0,1)
#
#         返回:
#         list: 信号片段列表
#         """
#         step = int(segment_length * (1 - overlap))
#         segments = []
#
#         for i in range(0, len(signal_data) - segment_length + 1, step):
#             segment = signal_data[i:i + segment_length]
#             segments.append(segment)
#
#         return segments
#
#     def save_scaler(self, filepath):
#         """
#         保存标准化器
#
#         参数:
#         filepath (str): 保存路径
#         """
#         if self.is_fitted:
#             import joblib
#             joblib.dump(self.scaler, filepath)
#         else:
#             raise ValueError("Scaler has not been fitted yet.")
#
#     def load_scaler(self, filepath):
#         """
#         加载标准化器
#
#         参数:
#         filepath (str): 加载路径
#         """
#         import joblib
#         self.scaler = joblib.load(filepath)
#         self.is_fitted = True


import numpy as np
from scipy import signal
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    """
    数据预处理类，用于信号处理和特征标准化
    """

    def __init__(self, filter_enabled=True, filter_lowcut=10, filter_highcut=1000,
                 filter_order=5, sampling_rate=2048, noise_removal_enabled=False,
                 noise_window_size=5, segment_overlap=0.5):
        """
        初始化预处理器

        参数:
        filter_enabled (bool): 是否启用带通滤波
        filter_lowcut (float): 低截止频率
        filter_highcut (float): 高截止频率
        filter_order (int): 滤波器阶数
        sampling_rate (float): 采样率
        noise_removal_enabled (bool): 是否启用噪声去除
        noise_window_size (int): 噪声去除窗口大小
        segment_overlap (float): 分段重叠比例
        """
        self.filter_enabled = filter_enabled
        self.filter_lowcut = filter_lowcut
        self.filter_highcut = filter_highcut
        self.filter_order = filter_order
        self.sampling_rate = sampling_rate
        self.noise_removal_enabled = noise_removal_enabled
        self.noise_window_size = noise_window_size
        self.segment_overlap = segment_overlap

        self.scaler = StandardScaler()
        self.is_fitted = False

    def process_signal(self, signal_data):
        """
        处理信号：应用滤波和噪声去除

        参数:
        signal_data (numpy.ndarray): 原始信号数据

        返回:
        numpy.ndarray: 处理后的信号
        """
        processed_signal = signal_data

        # 应用带通滤波
        if self.filter_enabled:
            processed_signal = self.filter_signal(
                processed_signal,
                self.filter_lowcut,
                self.filter_highcut,
                self.sampling_rate,
                self.filter_order
            )

        # 应用噪声去除
        if self.noise_removal_enabled:
            processed_signal = self.remove_noise(
                processed_signal,
                self.noise_window_size
            )

        return processed_signal

    def process_signal_batch(self, signal_batch):
        """
        批量处理信号

        参数:
        signal_batch (list): 信号数据列表

        返回:
        list: 处理后的信号列表
        """
        return [self.process_signal(signal) for signal in signal_batch]

    def filter_signal(self, signal_data, lowcut=None, highcut=None, fs=None, order=None):
        """
        使用带通滤波器过滤信号

        参数:
        signal_data (numpy.ndarray): 原始信号数据
        lowcut (float): 低截止频率，如果None则使用实例变量
        highcut (float): 高截止频率，如果None则使用实例变量
        fs (float): 采样率，如果None则使用实例变量
        order (int): 滤波器阶数，如果None则使用实例变量

        返回:
        numpy.ndarray: 滤波后的信号
        """
        # 使用传入的参数或默认值
        lowcut = lowcut if lowcut is not None else self.filter_lowcut
        highcut = highcut if highcut is not None else self.filter_highcut
        fs = fs if fs is not None else self.sampling_rate
        order = order if order is not None else self.filter_order

        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq

        b, a = signal.butter(order, [low, high], btype='band')
        filtered_signal = signal.filtfilt(b, a, signal_data)

        return filtered_signal

    def remove_noise(self, signal_data, window_size=None):
        """
        使用中值滤波去除噪声

        参数:
        signal_data (numpy.ndarray): 原始信号数据
        window_size (int): 滤波窗口大小，如果None则使用实例变量

        返回:
        numpy.ndarray: 去噪后的信号
        """
        window_size = window_size if window_size is not None else self.noise_window_size
        return signal.medfilt(signal_data, window_size)

    def normalize_features(self, features_df, fit=False):
        """
        标准化特征

        参数:
        features_df (pandas.DataFrame): 包含特征的数据框
        fit (bool): 是否训练标准化器

        返回:
        pandas.DataFrame: 标准化后的特征
        """
        if fit or not self.is_fitted:
            self.scaler.fit(features_df)
            self.is_fitted = True

        scaled_features = self.scaler.transform(features_df)
        return pd.DataFrame(scaled_features, columns=features_df.columns)

    def segment_signal(self, signal_data, segment_length, overlap=None):
        """
        将长信号分割成多个片段

        参数:
        signal_data (numpy.ndarray): 原始信号数据
        segment_length (int): 每个片段的长度
        overlap (float): 重叠比例，范围[0,1)，如果None则使用实例变量

        返回:
        list: 信号片段列表
        """
        overlap = overlap if overlap is not None else self.segment_overlap
        step = int(segment_length * (1 - overlap))
        segments = []

        for i in range(0, len(signal_data) - segment_length + 1, step):
            segment = signal_data[i:i + segment_length]
            segments.append(segment)

        return segments

    def save_scaler(self, filepath):
        """
        保存标准化器

        参数:
        filepath (str): 保存路径
        """
        if self.is_fitted:
            import joblib
            joblib.dump(self.scaler, filepath)
        else:
            raise ValueError("Scaler has not been fitted yet.")

    def load_scaler(self, filepath):
        """
        加载标准化器

        参数:
        filepath (str): 加载路径
        """
        import joblib
        self.scaler = joblib.load(filepath)
        self.is_fitted = True