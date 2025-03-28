import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class PCADiagnosisModel:
    """
    基于PCA的故障诊断模型
    """

    def __init__(self, n_components=2):
        """
        初始化PCA模型

        参数:
        n_components (int): 保留的主成分数量
        """
        self.pca = PCA(n_components=n_components)
        self.n_components = n_components
        self.is_fitted = False
        self.normal_t2_threshold = None
        self.normal_spe_threshold = None
        self.feature_names = None

    def fit(self, features_df, significance_level=0.05):
        """
        使用正常设备数据训练PCA模型

        参数:
        features_df (pandas.DataFrame): 正常运行状态下的特征数据
        significance_level (float): 显著性水平，用于设置阈值

        返回:
        self: 训练好的模型实例
        """
        self.feature_names = features_df.columns

        # 训练PCA模型
        self.pca.fit(features_df)

        # 计算T2和SPE(Q)统计量
        principal_components = self.pca.transform(features_df)
        reconstructed = self.pca.inverse_transform(principal_components)

        # 计算T2统计量
        t2_values = self._calculate_t2(principal_components)

        # 计算SPE统计量 (Q统计量)
        spe_values = self._calculate_spe(features_df, reconstructed)

        # 设置阈值
        from scipy import stats
        # T2阈值基于F分布
        # 使用F分布的临界值，其中自由度为(主成分数, 样本数-主成分数)
        n_samples = features_df.shape[0]
        f_critical = stats.f.ppf(1 - significance_level, self.n_components, n_samples - self.n_components)
        self.normal_t2_threshold = (self.n_components * (n_samples - 1) * f_critical) / (n_samples - self.n_components)

        # SPE阈值，通常使用经验方法设置为正常数据SPE的95%或99%分位数
        self.normal_spe_threshold = np.percentile(spe_values, (1 - significance_level) * 100)

        self.is_fitted = True
        return self

    def _calculate_t2(self, principal_components):
        """
        计算T2统计量

        参数:
        principal_components (numpy.ndarray): 主成分得分

        返回:
        numpy.ndarray: T2统计量
        """
        # 计算主成分得分的协方差矩阵的逆
        cov_inv = np.diag(1.0 / self.pca.explained_variance_[:self.n_components])

        # 计算T2统计量
        t2_values = []
        for pc in principal_components:
            t2 = pc.dot(cov_inv).dot(pc.T)
            t2_values.append(t2)

        return np.array(t2_values)

    def _calculate_spe(self, original_data, reconstructed_data):
        """
        计算SPE (Q)统计量

        参数:
        original_data (numpy.ndarray): 原始数据
        reconstructed_data (numpy.ndarray): 重构数据

        返回:
        numpy.ndarray: SPE统计量
        """
        # 计算残差
        residuals = original_data - reconstructed_data

        # 计算SPE (平方预测误差)
        spe_values = np.sum(residuals ** 2, axis=1)

        return spe_values

    def diagnose(self, features_df):
        """
        对新数据进行故障诊断

        参数:
        features_df (pandas.DataFrame): 新的特征数据

        返回:
        pandas.DataFrame: 包含T2、SPE和故障判断的结果
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")

        # 确保输入数据列顺序与训练数据一致
        if self.feature_names is not None:
            features_df = features_df[self.feature_names]

        # 投影到主成分空间
        principal_components = self.pca.transform(features_df)
        reconstructed = self.pca.inverse_transform(principal_components)

        # 计算T2和SPE统计量
        t2_values = self._calculate_t2(principal_components)
        spe_values = self._calculate_spe(features_df, reconstructed)

        # 判断是否异常
        t2_violations = t2_values > self.normal_t2_threshold
        spe_violations = spe_values > self.normal_spe_threshold

        # 计算故障得分
        fault_scores = t2_values / self.normal_t2_threshold + spe_values / self.normal_spe_threshold

        # 最终故障判断
        fault_detected = np.logical_or(t2_violations, spe_violations)

        # 创建诊断结果数据框
        results = pd.DataFrame({
            'T2': t2_values,
            'SPE': spe_values,
            'T2_Threshold': self.normal_t2_threshold,
            'SPE_Threshold': self.normal_spe_threshold,
            'T2_Violation': t2_violations,
            'SPE_Violation': spe_violations,
            'Fault_Score': fault_scores,
            'Fault_Detected': fault_detected
        })

        return results

    def calculate_contribution(self, features_df):
        """
        计算各特征对T2和SPE的贡献

        参数:
        features_df (pandas.DataFrame): 特征数据

        返回:
        tuple: (t2_contribution_df, spe_contribution_df) T2和SPE贡献率数据框
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")

        # 确保输入数据列顺序与训练数据一致
        if self.feature_names is not None:
            features_df = features_df[self.feature_names]

        # 获取加载矩阵和特征数量
        loadings = self.pca.components_
        n_features = features_df.shape[1]

        # 标准化数据
        features_std = (features_df - np.mean(features_df, axis=0)) / np.std(features_df, axis=0)

        # 计算T2贡献
        principal_components = self.pca.transform(features_df)
        t2_contributions = np.zeros((features_df.shape[0], n_features))

        for i in range(features_df.shape[0]):
            for j in range(n_features):
                contribution = 0
                for k in range(self.n_components):
                    contribution += (principal_components[i, k] * loadings[k, j] /
                                     np.sqrt(self.pca.explained_variance_[k]))
                t2_contributions[i, j] = contribution ** 2

        # 计算SPE贡献
        reconstructed = self.pca.inverse_transform(principal_components)
        residuals = features_df.values - reconstructed
        spe_contributions = residuals ** 2

        # 创建贡献率数据框
        t2_contribution_df = pd.DataFrame(t2_contributions, columns=self.feature_names)
        spe_contribution_df = pd.DataFrame(spe_contributions, columns=self.feature_names)

        return t2_contribution_df, spe_contribution_df

    def save_model(self, filepath):
        """
        保存模型

        参数:
        filepath (str): 保存路径
        """
        if self.is_fitted:
            model_data = {
                'pca': self.pca,
                'n_components': self.n_components,
                'normal_t2_threshold': self.normal_t2_threshold,
                'normal_spe_threshold': self.normal_spe_threshold,
                'feature_names': self.feature_names
            }
            joblib.dump(model_data, filepath)
        else:
            raise ValueError("Model has not been fitted yet.")

    def load_model(self, filepath):
        """
        加载模型

        参数:
        filepath (str): 加载路径
        """
        model_data = joblib.load(filepath)
        self.pca = model_data['pca']
        self.n_components = model_data['n_components']
        self.normal_t2_threshold = model_data['normal_t2_threshold']
        self.normal_spe_threshold = model_data['normal_spe_threshold']
        self.feature_names = model_data['feature_names']
        self.is_fitted = True

    def plot_scores(self, features_df, true_labels=None):
        """
        绘制T2和SPE得分图

        参数:
        features_df (pandas.DataFrame): 特征数据
        true_labels (numpy.ndarray, optional): 真实标签

        返回:
        matplotlib.figure.Figure: 图形对象
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")

        # 诊断结果
        results = self.diagnose(features_df)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # T2统计量图
        if true_labels is not None:
            scatter = ax1.scatter(range(len(results)), results['T2'], c=true_labels, cmap='coolwarm', alpha=0.7)
            legend1 = ax1.legend(*scatter.legend_elements(), title="故障类型")
            ax1.add_artist(legend1)
        else:
            ax1.scatter(range(len(results)), results['T2'], c=results['Fault_Detected'], cmap='coolwarm', alpha=0.7)

        ax1.axhline(y=self.normal_t2_threshold, color='r', linestyle='--', label='阈值')
        ax1.set_title('T²统计量')
        ax1.set_xlabel('样本索引')
        ax1.set_ylabel('T²值')
        ax1.legend()

        # SPE统计量图
        if true_labels is not None:
            scatter = ax2.scatter(range(len(results)), results['SPE'], c=true_labels, cmap='coolwarm', alpha=0.7)
        else:
            ax2.scatter(range(len(results)), results['SPE'], c=results['Fault_Detected'], cmap='coolwarm', alpha=0.7)

        ax2.axhline(y=self.normal_spe_threshold, color='r', linestyle='--', label='阈值')
        ax2.set_title('SPE统计量')
        ax2.set_xlabel('样本索引')
        ax2.set_ylabel('SPE值')
        ax2.legend()

        plt.tight_layout()
        return fig

    def plot_pca_space(self, features_df, true_labels=None):
        """
        在PCA空间中绘制样本

        参数:
        features_df (pandas.DataFrame): 特征数据
        true_labels (numpy.ndarray, optional): 真实标签

        返回:
        matplotlib.figure.Figure: 图形对象
        """
        if not self.is_fitted or self.n_components < 2:
            raise ValueError("Model has not been fitted or n_components < 2.")

        # 投影到PCA空间
        principal_components = self.pca.transform(features_df)

        fig, ax = plt.subplots(figsize=(10, 8))

        # 绘制散点图
        if true_labels is not None:
            scatter = ax.scatter(principal_components[:, 0], principal_components[:, 1],
                                 c=true_labels, cmap='coolwarm', alpha=0.7)
            legend1 = ax.legend(*scatter.legend_elements(), title="故障类型")
            ax.add_artist(legend1)
        else:
            # 使用诊断结果作为颜色
            results = self.diagnose(features_df)
            scatter = ax.scatter(principal_components[:, 0], principal_components[:, 1],
                                 c=results['Fault_Detected'], cmap='coolwarm', alpha=0.7)
            legend1 = ax.legend(*scatter.legend_elements(), title="故障检测")
            ax.add_artist(legend1)

        ax.set_title('PCA空间中的样本分布')
        ax.set_xlabel(f'主成分1 ({self.pca.explained_variance_ratio_[0]:.2%} 方差)')
        ax.set_ylabel(f'主成分2 ({self.pca.explained_variance_ratio_[1]:.2%} 方差)')
        ax.grid(True, linestyle='--', alpha=0.7)

        return fig

    def plot_contribution(self, features_df, sample_idx=0):
        """
        绘制特征贡献率图

        参数:
        features_df (pandas.DataFrame): 特征数据
        sample_idx (int): 样本索引

        返回:
        matplotlib.figure.Figure: 图形对象
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")

        # 计算贡献率
        t2_contrib, spe_contrib = self.calculate_contribution(features_df)

        # 获取指定样本的贡献率
        t2_sample_contrib = t2_contrib.iloc[sample_idx]
        spe_sample_contrib = spe_contrib.iloc[sample_idx]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # T2贡献率条形图
        t2_sample_contrib.sort_values().plot(kind='barh', ax=ax1, color='skyblue')
        ax1.set_title(f'样本 {sample_idx} 的 T² 贡献率')
        ax1.set_xlabel('贡献值')
        ax1.grid(True, linestyle='--', alpha=0.7)

        # SPE贡献率条形图
        spe_sample_contrib.sort_values().plot(kind='barh', ax=ax2, color='lightgreen')
        ax2.set_title(f'样本 {sample_idx} 的 SPE 贡献率')
        ax2.set_xlabel('贡献值')
        ax2.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        return fig

    def plot_confusion_matrix(self, features_df, true_labels):
        """
        绘制混淆矩阵

        参数:
        features_df (pandas.DataFrame): 特征数据
        true_labels (numpy.ndarray): 真实标签 (0: 正常, 1: 故障)

        返回:
        matplotlib.figure.Figure: 图形对象
        """
        # 诊断结果
        results = self.diagnose(features_df)
        predicted_labels = results['Fault_Detected'].astype(int)

        # 计算混淆矩阵
        cm = confusion_matrix(true_labels, predicted_labels)

        # 绘制混淆矩阵
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)

        ax.set_title('故障诊断混淆矩阵')
        ax.set_xlabel('预测标签')
        ax.set_ylabel('真实标签')
        ax.set_xticklabels(['正常', '故障'])
        ax.set_yticklabels(['正常', '故障'])

        return fig