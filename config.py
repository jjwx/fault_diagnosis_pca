"""
故障诊断系统配置文件
"""

# 数据采集和预处理配置
DATA_CONFIG = {
    'sampling_rate': 2048,       # 采样率 (Hz)
    'signal_length': 1024,       # 每个信号片段的长度
    'filter_enabled': True,      # 是否启用滤波
    'filter_lowcut': 10,         # 带通滤波器低截止频率 (Hz)
    'filter_highcut': 1000,      # 带通滤波器高截止频率 (Hz)
    'filter_order': 5,           # 滤波器阶数
    'noise_removal_enabled': False,  # 是否启用噪声去除
    'noise_window_size': 5,      # 中值滤波窗口大小
    'segment_overlap': 0.5,      # 信号分段重叠比例
}

# 特征提取配置
FEATURE_CONFIG = {
    'time_domain_features': ['kurtosis', 'rms', 'crest_factor'],
    'freq_domain_features': ['intensity', 'contrast', 'freq_band_energy_ratio'],
    'freq_band_start': None,     # 特定频带的起始频率 (Hz)，None表示自动选择
    'freq_band_end': None,       # 特定频带的结束频率 (Hz)，None表示自动选择
}

# PCA模型配置
PCA_CONFIG = {
    'n_components': 3,           # 保留的主成分数量
    'significance_level': 0.01,  # 显著性水平
}

# 路径配置
PATH_CONFIG = {
    'data_dir': 'data',          # 数据目录
    'raw_data_dir': 'data/raw',  # 原始数据目录
    'processed_data_dir': 'data/processed',  # 处理后的数据目录
    'model_dir': 'models',       # 模型保存目录
    'result_dir': 'results',     # 结果保存目录
}

# 可视化配置
VIZ_CONFIG = {
    'figsize': (12, 8),          # 图形默认大小
    'dpi': 100,                  # 图形DPI
    'save_format': 'png',        # 图形保存格式
}