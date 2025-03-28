import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import matplotlib.gridspec as gridspec


class Visualizer:
    """
    数据和结果可视化类
    """

    def __init__(self, figsize=(12, 8)):
        """
        初始化可视化器

        参数:
        figsize (tuple): 图形默认大小
        """
        self.figsize = figsize
        plt.style.use('seaborn-whitegrid')

    def plot_signal(self, signal_data, sampling_rate=1, title='Signal Waveform'):
        """
        绘制时域信号波形

        参数:
        signal_data (numpy.ndarray): 信号数据
        sampling_rate (float): 采样率
        title (str): 图形标题

        返回:
        matplotlib.figure.Figure: 图形对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        time = np.arange(len(signal_data)) / sampling_rate
        ax.plot(time, signal_data)

        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True)

        return fig

    def plot_spectrum(self, signal_data, sampling_rate=1, title='Signal Spectrum'):
        """
        绘制频谱图

        参数:
        signal_data (numpy.ndarray): 信号数据
        sampling_rate (float): 采样率
        title (str): 图形标题

        返回:
        matplotlib.figure.Figure: 图形对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        frequencies, spectrum = self._calculate_spectrum(signal_data, sampling_rate)

        ax.plot(frequencies, spectrum)
        ax.set_title(title)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude')
        ax.grid(True)

        # 对数刻度更适合查看频谱细节
        ax.set_yscale('log')

        return fig

    def _calculate_spectrum(self, signal_data, sampling_rate):
        """
        计算信号频谱

        参数:
        signal_data (numpy.ndarray): 信号数据
        sampling_rate (float): 采样率

        返回:
        tuple: (frequencies, spectrum)
        """
        n = len(signal_data)
        frequencies = np.fft.rfftfreq(n, 1 / sampling_rate)
        spectrum = np.abs(np.fft.rfft(signal_data)) / n
        return frequencies, spectrum

    def plot_time_frequency(self, signal_data, sampling_rate=1, title='Time-Frequency Analysis'):
        """
        绘制时频分析图（短时傅里叶变换）

        参数:
        signal_data (numpy.ndarray): 信号数据
        sampling_rate (float): 采样率
        title (str): 图形标题

        返回:
        matplotlib.figure.Figure: 图形对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        f, t, Sxx = signal.spectrogram(signal_data, fs=sampling_rate, window='hann',
                                       nperseg=min(256, len(signal_data) // 10),
                                       noverlap=min(128, len(signal_data) // 20))

        pcm = ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
        fig.colorbar(pcm, ax=ax, label='功率/频率 (dB/Hz)')

        ax.set_title(title)
        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('频率 (Hz)')

        return fig

    def plot_features(self, features_df, title='特征分布'):
        """
        绘制特征分布图

        参数:
        features_df (pandas.DataFrame): 特征数据框
        title (str): 图形标题

        返回:
        matplotlib.figure.Figure: 图形对象
        """
        # 计算需要的子图数量
        n_features = features_df.shape[1]
        n_cols = min(3, n_features)
        n_rows = int(np.ceil(n_features / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        fig.suptitle(title, fontsize=16)

        # 扁平化axes数组以便于索引
        if n_rows > 1 or n_cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]

        # 绘制每个特征的分布
        for i, feature in enumerate(features_df.columns):
            if i < len(axes):
                sns.histplot(features_df[feature], kde=True, ax=axes[i])
                axes[i].set_title(feature)
                axes[i].set_xlabel('值')
                axes[i].set_ylabel('频数')

        # 隐藏多余的子图
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        return fig

    def plot_feature_correlation(self, features_df, title='特征相关性'):
        """
        绘制特征相关性热图

        参数:
        features_df (pandas.DataFrame): 特征数据框
        title (str): 图形标题

        返回:
        matplotlib.figure.Figure: 图形对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # 计算相关系数矩阵
        corr_matrix = features_df.corr()

        # 绘制热图
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)

        ax.set_title(title)

        return fig

    def plot_pca_explained_variance(self, pca_model, title='PCA解释方差'):
        """
        绘制PCA解释方差比例图

        参数:
        pca_model: PCA模型实例
        title (str): 图形标题

        返回:
        matplotlib.figure.Figure: 图形对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # 获取解释方差比例
        explained_variance_ratio = pca_model.pca.explained_variance_ratio_

        # 绘制条形图
        ax.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio,
               alpha=0.7, color='skyblue')

        # 绘制累积解释方差
        cumulative = np.cumsum(explained_variance_ratio)
        ax.step(range(1, len(explained_variance_ratio) + 1), cumulative, where='mid',
                label='累积解释方差', color='red', linewidth=2)

        ax.set_title(title)
        ax.set_xlabel('主成分')
        ax.set_ylabel('解释方差比例')
        ax.set_xticks(range(1, len(explained_variance_ratio) + 1))
        ax.set_ylim([0, 1.05])
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

        return fig

    def plot_diagnosis_summary(self, diagnosis_result):
        """
        绘制诊断结果摘要图

        参数:
        diagnosis_result (dict): 诊断结果字典

        返回:
        matplotlib.figure.Figure: 图形对象
        """
        fig = plt.figure(figsize=self.figsize)
        gs = gridspec.GridSpec(2, 2, figure=fig)

        # T2统计量子图
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.bar(['T² 统计量'], [diagnosis_result['t2_value']], color='skyblue')
        ax1.axhline(y=diagnosis_result['t2_threshold'], color='red', linestyle='--', label='阈值')
        ax1.set_title('T² 统计量')
        ax1.legend()

        # SPE统计量子图
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.bar(['SPE 统计量'], [diagnosis_result['spe_value']], color='lightgreen')
        ax2.axhline(y=diagnosis_result['spe_threshold'], color='red', linestyle='--', label='阈值')
        ax2.set_title('SPE 统计量')
        ax2.legend()

        # 故障状态子图
        ax3 = fig.add_subplot(gs[1, 0])
        status = '故障' if diagnosis_result['fault_detected'] else '正常'
        color = 'red' if diagnosis_result['fault_detected'] else 'green'
        ax3.text(0.5, 0.5, f'设备状态: {status}', horizontalalignment='center',
                 verticalalignment='center', fontsize=18, color=color)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_title('诊断结果')

        # 故障得分子图
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.bar(['故障得分'], [diagnosis_result['fault_score']], color='orange')
        ax4.set_title('故障得分')

        # 如果检测到故障，显示贡献最大的特征
        if diagnosis_result['fault_detected'] and 'max_t2_contribution_feature' in diagnosis_result:
            ax3.text(0.5, 0.3,
                     f"主要影响特征:\n{diagnosis_result['max_t2_contribution_feature']} (T²)\n{diagnosis_result['max_spe_contribution_feature']} (SPE)",
                     horizontalalignment='center', verticalalignment='center', fontsize=12)

        plt.tight_layout()

        return fig

    def plot_feature_contributions(self, diagnosis_result):
        """
        绘制特征贡献率图

        参数:
        diagnosis_result (dict): 包含特征贡献的诊断结果

        返回:
        matplotlib.figure.Figure: 图形对象
        """
        if 't2_contributions' not in diagnosis_result or 'spe_contributions' not in diagnosis_result:
            return None

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)

        # 将字典转换为Series并排序
        t2_contrib = pd.Series(diagnosis_result['t2_contributions']).sort_values()
        spe_contrib = pd.Series(diagnosis_result['spe_contributions']).sort_values()

        # T2贡献率条形图
        t2_contrib.plot(kind='barh', ax=ax1, color='skyblue')
        ax1.set_title('T² 贡献率')
        ax1.set_xlabel('贡献值')
        ax1.grid(True, linestyle='--', alpha=0.7)

        # SPE贡献率条形图
        spe_contrib.plot(kind='barh', ax=ax2, color='lightgreen')
        ax2.set_title('SPE 贡献率')
        ax2.set_xlabel('贡献值')
        ax2.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()

        return fig