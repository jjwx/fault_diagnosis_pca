import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from src.feature_extraction import FeatureExtractor
from src.preprocessing import Preprocessor
from src.pca_model import PCADiagnosisModel
from src.diagnosis import FaultDiagnoser
from src.visualization import Visualizer

# 导入配置文件
from config import DATA_CONFIG, FEATURE_CONFIG, PCA_CONFIG, PATH_CONFIG, VIZ_CONFIG


def load_example_data(normal_count=100, fault_count=50, signal_length=None,
                      sampling_rate=None, seed=42):
    """
    生成示例数据用于演示

    参数:
    normal_count (int): 正常信号的数量
    fault_count (int): 故障信号的数量
    signal_length (int): 信号长度
    sampling_rate (int): 采样率
    seed (int): 随机种子

    返回:
    tuple: (normal_signals, fault_signals, test_signals, test_labels)
    """
    # 使用配置文件中的默认值，如果未提供特定值
    if signal_length is None:
        signal_length = DATA_CONFIG['signal_length']
    if sampling_rate is None:
        sampling_rate = DATA_CONFIG['sampling_rate']

    np.random.seed(seed)

    # 生成时间轴
    t = np.arange(signal_length) / sampling_rate

    # 正常信号：基本正弦波加少量噪声
    normal_signals = []
    for i in range(normal_count):
        freq = 50 + np.random.normal(0, 2)  # 频率略有变化
        amp = 1.0 + np.random.normal(0, 0.1)  # 幅度略有变化
        base_signal = amp * np.sin(2 * np.pi * freq * t)
        noise = np.random.normal(0, 0.05, signal_length)  # 小噪声
        signal = base_signal + noise
        normal_signals.append(signal)

    # 故障信号类型1：增加冲击分量
    fault_signals_1 = []
    for i in range(fault_count // 2):
        freq = 50 + np.random.normal(0, 2)
        amp = 1.0 + np.random.normal(0, 0.1)
        base_signal = amp * np.sin(2 * np.pi * freq * t)

        # 添加脉冲噪声（模拟轴承故障）
        impulse_positions = np.random.choice(signal_length, size=5, replace=False)
        impulse = np.zeros(signal_length)
        impulse[impulse_positions] = 2.0  # 脉冲幅度

        noise = np.random.normal(0, 0.1, signal_length)
        signal = base_signal + impulse + noise
        fault_signals_1.append(signal)

    # 故障信号类型2：增加调制分量（模拟齿轮故障）
    fault_signals_2 = []
    for i in range(fault_count - len(fault_signals_1)):
        freq = 50 + np.random.normal(0, 2)
        amp = 1.0 + np.random.normal(0, 0.1)
        mod_freq = 10  # 调制频率
        base_signal = amp * np.sin(2 * np.pi * freq * t)
        modulation = 0.5 * np.sin(2 * np.pi * mod_freq * t)

        # 调幅信号
        signal = (1 + modulation) * base_signal + np.random.normal(0, 0.1, signal_length)
        fault_signals_2.append(signal)

    # 合并故障信号
    fault_signals = fault_signals_1 + fault_signals_2

    # 创建测试集：取一部分正常和故障信号混合
    test_normal = normal_signals[int(0.8 * normal_count):]
    test_fault = fault_signals[int(0.8 * len(fault_signals)):]

    test_signals = test_normal + test_fault
    test_labels = [0] * len(test_normal) + [1] * len(test_fault)

    # 训练集：剩余的正常信号
    train_normal = normal_signals[:int(0.8 * normal_count)]

    return train_normal, fault_signals, test_signals, test_labels


def load_real_data(data_path, file_format='csv', normal_label=0, train_ratio=0.8,
                   signal_length=None):
    """
    从真实数据文件加载振动信号数据

    参数:
    data_path (str): 数据文件或目录的路径
    file_format (str): 文件格式，支持'csv'、'excel'、'mat'等
    normal_label (int/str): 标识正常样本的标签值
    train_ratio (float): 用于训练的正常数据比例
    signal_length (int): 信号长度，用于验证数据是否符合预期

    返回:
    tuple: (train_normal_signals, fault_signals, test_signals, test_labels)
    """
    # 使用配置中的信号长度，如果未提供
    if signal_length is None:
        signal_length = DATA_CONFIG['signal_length']

    try:
        print(f"正在从 {data_path} 加载真实数据...")

        # 根据文件格式加载数据
        if file_format.lower() == 'csv':
            # 假设CSV文件包含信号数据和标签
            # 格式: 每行一个样本，最后一列是标签
            df = pd.read_csv(data_path)
        elif file_format.lower() in ['xls', 'xlsx', 'excel']:
            df = pd.read_excel(data_path)
        elif file_format.lower() == 'mat':
            from scipy.io import loadmat
            mat_data = loadmat(data_path)
            # 根据mat文件的具体结构进行处理
            # 这里需要根据实际情况修改
            signals = mat_data.get('signals', [])
            labels = mat_data.get('labels', [])
            df = pd.DataFrame({
                'signal': signals,
                'label': labels
            })
        else:
            # 处理其他格式，或自定义加载函数
            raise ValueError(f"不支持的文件格式: {file_format}")

        # 假设数据结构：最后一列是标签，其余列是信号或已提取的特征
        # 如果数据结构不同，需要相应调整

        # 区分标签列和信号数据
        if 'label' in df.columns:
            labels = df['label'].values
            signals_df = df.drop('label', axis=1)
        else:
            # 假设最后一列是标签
            labels = df.iloc[:, -1].values
            signals_df = df.iloc[:, :-1]

        signals = []

        # 检查数据格式：如果每行是一个信号（而不是已提取的特征），则转换为信号列表
        if signals_df.shape[1] > 10:  # 假设如果列数很多，则为原始信号数据
            for i in range(len(signals_df)):
                signals.append(signals_df.iloc[i].values)
        else:
            # 如果是已提取的特征，可能需要另一种处理方式
            print("警告：输入数据似乎是特征而非原始信号，请确保数据格式正确")
            for i in range(len(signals_df)):
                signals.append(signals_df.iloc[i].values)

        # 区分正常和故障信号
        normal_signals = [signals[i] for i in range(len(signals)) if labels[i] == normal_label]
        fault_signals = [signals[i] for i in range(len(signals)) if labels[i] != normal_label]

        # 分割正常信号为训练集和测试集
        train_size = int(len(normal_signals) * train_ratio)
        train_normal = normal_signals[:train_size]
        test_normal = normal_signals[train_size:]

        # 创建测试集：混合部分正常信号和部分故障信号
        test_fault = fault_signals[:]  # 使用所有故障信号进行测试

        # 如果故障信号太多，可以限制数量
        if len(test_fault) > len(test_normal) * 2:
            import random
            random.seed(42)  # 固定随机种子以保证可重复性
            test_fault = random.sample(test_fault, len(test_normal) * 2)

        test_signals = test_normal + test_fault
        test_labels = [normal_label] * len(test_normal) + [1] * len(test_fault)  # 故障标签统一为1

        print(f"已加载数据: {len(train_normal)} 训练正常信号, {len(test_normal)} 测试正常信号, {len(test_fault)} 测试故障信号")

        return train_normal, fault_signals, test_signals, test_labels

    except Exception as e:
        print(f"加载数据时出错: {e}")
        # 发生错误时返回示例数据
        print("使用示例数据替代...")
        return load_example_data()


# def main():
#     """主函数，程序入口点"""
#     parser = argparse.ArgumentParser(description='故障诊断系统')
#     parser.add_argument('--mode', choices=['train', 'test', 'demo'], default='train',
#                         help='程序运行模式：训练、测试或演示')
#     parser.add_argument('--save_path', default=PATH_CONFIG['model_dir'],
#                         help='模型保存路径')
#     parser.add_argument('--load_path', default=PATH_CONFIG['model_dir'],
#                         help='模型加载路径')
#     parser.add_argument('--visualize', action='store_true',
#                         help='是否可视化结果')
#     # 添加数据加载相关参数
#     parser.add_argument('--data_path', default=None,
#                         help='真实数据文件路径')
#     parser.add_argument('--data_format', default='csv',
#                         help='数据文件格式: csv, excel, mat等')
#     parser.add_argument('--use_example_data', action='store_true',
#                         help='使用示例数据而非真实数据')
#     parser.add_argument('--normal_label', default=0, type=int,
#                         help='正常样本的标签值')
#     parser.add_argument('--train_ratio', default=0.8, type=float,
#                         help='用于训练的正常数据比例')
#     parser.add_argument('--sampling_rate', default=DATA_CONFIG['sampling_rate'], type=int,
#                         help='信号采样率 (Hz)')
#     parser.add_argument('--signal_length', default=DATA_CONFIG['signal_length'], type=int,
#                         help='信号长度')
#     parser.add_argument('--result_dir', default=PATH_CONFIG['result_dir'],
#                         help='结果保存目录')
#
#     args = parser.parse_args()
#
#
#     # 创建必要的目录
#     os.makedirs(args.save_path, exist_ok=True)
#     os.makedirs(PATH_CONFIG['result_dir'], exist_ok=True)
#     os.makedirs(PATH_CONFIG['processed_data_dir'], exist_ok=True)
#
#     # 初始化组件，只使用当前类支持的参数
#     sampling_rate = DATA_CONFIG['sampling_rate']
#     feature_extractor = FeatureExtractor(sampling_rate)
#
#     # 初始化预处理器（如果Preprocessor支持这些参数则使用，否则简化初始化）
#     try:
#         preprocessor = Preprocessor(
#             filter_enabled=DATA_CONFIG['filter_enabled'],
#             filter_lowcut=DATA_CONFIG['filter_lowcut'],
#             filter_highcut=DATA_CONFIG['filter_highcut'],
#             filter_order=DATA_CONFIG['filter_order']
#         )
#     except TypeError:
#         # 如果Preprocessor不支持这些参数，使用默认初始化
#         preprocessor = Preprocessor()
#
#     # 初始化PCA模型
#     pca_model = PCADiagnosisModel(n_components=PCA_CONFIG['n_components'])
#
#     # 尝试传递significance_level参数（如果支持）
#     try:
#         pca_model.significance_level = PCA_CONFIG['significance_level']
#     except:
#         pass
#
#     diagnoser = FaultDiagnoser(feature_extractor, preprocessor, pca_model)
#     visualizer = Visualizer(figsize=VIZ_CONFIG['figsize'])
#
#     # 加载数据
#     if args.use_example_data or args.data_path is None:
#         print("加载示例数据...")
#         normal_signals, fault_signals, test_signals, test_labels = load_example_data(
#             signal_length=args.signal_length,
#             sampling_rate=sampling_rate
#         )
#     else:
#         # 加载真实数据
#         normal_signals, fault_signals, test_signals, test_labels = load_real_data(
#             data_path=args.data_path,
#             file_format=args.data_format,
#             normal_label=args.normal_label,
#             train_ratio=args.train_ratio,
#             signal_length=args.signal_length
#         )
#
#     if args.mode == 'train' or args.mode == 'demo':
#         print("训练诊断模型...")
#         diagnoser.train_model(normal_signals)
#
#         # 保存模型
#         print(f"保存模型到 {args.save_path}...")
#         diagnoser.save_models(args.save_path)
#
#     if args.mode == 'test' or args.mode == 'demo':
#         if args.mode == 'test':
#             # 加载模型
#             print(f"从 {args.load_path} 加载模型...")
#             diagnoser.load_models(args.load_path)
#
#         # 评估模型性能
#         print("评估模型性能...")
#         performance = diagnoser.evaluate_performance(test_signals, test_labels)
#
#         print("\n======== 性能评估 ========")
#         print(f"准确率: {performance['accuracy']:.4f}")
#         print(f"精确率: {performance['precision']:.4f}")
#         print(f"召回率: {performance['recall']:.4f}")
#         print(f"F1分数: {performance['f1_score']:.4f}")
#         if performance['auc'] is not None:
#             print(f"AUC: {performance['auc']:.4f}")
#
#         # 诊断示例
#         print("\n======== 诊断示例 ========")
#
#         # 诊断一个正常信号
#         normal_idx = test_labels.index(0)
#         normal_result = diagnoser.diagnose(test_signals[normal_idx])
#
#         print("\n正常信号诊断结果:")
#         print(f"检测到故障: {'是' if normal_result['fault_detected'] else '否'}")
#         print(f"T² 值: {normal_result['t2_value']:.4f} (阈值: {normal_result['t2_threshold']:.4f})")
#         print(f"SPE 值: {normal_result['spe_value']:.4f} (阈值: {normal_result['spe_threshold']:.4f})")
#         print(f"故障得分: {normal_result['fault_score']:.4f}")
#
#         # 诊断一个故障信号
#         fault_idx = test_labels.index(1)
#         fault_result = diagnoser.diagnose(test_signals[fault_idx])
#
#         print("\n故障信号诊断结果:")
#         print(f"检测到故障: {'是' if fault_result['fault_detected'] else '否'}")
#         print(f"T² 值: {fault_result['t2_value']:.4f} (阈值: {fault_result['t2_threshold']:.4f})")
#         print(f"SPE 值: {fault_result['spe_value']:.4f} (阈值: {fault_result['spe_threshold']:.4f})")
#         print(f"故障得分: {fault_result['fault_score']:.4f}")
#
#         if fault_result['fault_detected'] and 'max_t2_contribution_feature' in fault_result:
#             print(f"T² 贡献最大的特征: {fault_result['max_t2_contribution_feature']}")
#             print(f"SPE 贡献最大的特征: {fault_result['max_spe_contribution_feature']}")
#
#         # 可视化结果
#         if args.visualize:
#             print("\n生成可视化结果...")
#
#             # 对所有测试数据进行批量诊断
#             features_df = diagnoser.process_signal_batch(test_signals)
#             normalized_features = preprocessor.normalize_features(features_df)
#
#             # 设置保存路径为结果目录
#             result_save_path = args.result_dir
#
#             # 设置图形DPI
#             plt.rcParams['figure.dpi'] = VIZ_CONFIG['dpi']
#
#             # 在PCA空间中绘制样本分布
#             fig1 = pca_model.plot_pca_space(normalized_features, test_labels)
#             plt.savefig(os.path.join(result_save_path, f'pca_space.{VIZ_CONFIG["save_format"]}'))
#
#             # 绘制T2和SPE得分
#             fig2 = pca_model.plot_scores(normalized_features, test_labels)
#             plt.savefig(os.path.join(result_save_path, f'scores.{VIZ_CONFIG["save_format"]}'))
#
#             # 混淆矩阵
#             fig3 = pca_model.plot_confusion_matrix(normalized_features, test_labels)
#             plt.savefig(os.path.join(result_save_path, f'confusion_matrix.{VIZ_CONFIG["save_format"]}'))
#
#             # 绘制故障信号的诊断结果摘要
#             fig4 = visualizer.plot_diagnosis_summary(fault_result)
#             plt.savefig(os.path.join(result_save_path, f'diagnosis_summary.{VIZ_CONFIG["save_format"]}'))
#
#             # 如果检测到故障，绘制特征贡献图
#             if fault_result['fault_detected'] and 't2_contributions' in fault_result:
#                 fig5 = visualizer.plot_feature_contributions(fault_result)
#                 plt.savefig(os.path.join(result_save_path, f'feature_contributions.{VIZ_CONFIG["save_format"]}'))
#
#             # 绘制特征相关性热图
#             fig6 = visualizer.plot_feature_correlation(features_df)
#             plt.savefig(os.path.join(result_save_path, f'feature_correlation.{VIZ_CONFIG["save_format"]}'))
#
#             # 绘制PCA解释方差
#             fig7 = visualizer.plot_pca_explained_variance(pca_model)
#             plt.savefig(os.path.join(result_save_path, f'pca_explained_variance.{VIZ_CONFIG["save_format"]}'))
#
#             print(f"可视化结果已保存到 {result_save_path}")
#
#             # 显示图形
#             plt.show()
#
#
# if __name__ == '__main__':
#     main()

# def main():
#     """主函数，程序入口点"""
#     parser = argparse.ArgumentParser(description='故障诊断系统')
#     parser.add_argument('--mode', choices=['train', 'test', 'demo'], default='demo',
#                         help='程序运行模式：训练、测试或演示')
#     parser.add_argument('--save_path', default=PATH_CONFIG['model_dir'],
#                         help='模型保存路径')
#     parser.add_argument('--load_path', default=PATH_CONFIG['model_dir'],
#                         help='模型加载路径')
#     parser.add_argument('--visualize', action='store_true',
#                         help='是否可视化结果')
#     parser.add_argument('--data_path', default=None,
#                         help='真实数据文件路径')
#     parser.add_argument('--data_format', default='csv',
#                         help='数据文件格式: csv, excel, mat等')
#     parser.add_argument('--use_example_data', action='store_true',
#                         help='使用示例数据而非真实数据')
#     parser.add_argument('--normal_label', default=0, type=int,
#                         help='正常样本的标签值')
#     parser.add_argument('--train_ratio', default=0.8, type=float,
#                         help='用于训练的正常数据比例')
#     parser.add_argument('--config', default='config.py',
#                         help='配置文件路径')
#
#     args = parser.parse_args()
#
#     # 创建必要的目录
#     os.makedirs(args.save_path, exist_ok=True)
#     os.makedirs(PATH_CONFIG['result_dir'], exist_ok=True)
#     os.makedirs(PATH_CONFIG['processed_data_dir'], exist_ok=True)
#
#     # 初始化组件，使用配置文件参数
#     sampling_rate = DATA_CONFIG['sampling_rate']
#
#     # 初始化特征提取器
#     feature_extractor = FeatureExtractor(sampling_rate)
#
#     # 使用配置文件参数初始化预处理器
#     preprocessor = Preprocessor(
#         filter_enabled=DATA_CONFIG['filter_enabled'],
#         filter_lowcut=DATA_CONFIG['filter_lowcut'],
#         filter_highcut=DATA_CONFIG['filter_highcut'],
#         filter_order=DATA_CONFIG['filter_order'],
#         sampling_rate=DATA_CONFIG['sampling_rate'],
#         noise_removal_enabled=DATA_CONFIG['noise_removal_enabled'],
#         noise_window_size=DATA_CONFIG['noise_window_size'],
#         segment_overlap=DATA_CONFIG['segment_overlap']
#     )
#
#     # 初始化PCA模型
#     # 初始化PCA模型，只传递n_components参数
#     pca_model = PCADiagnosisModel(n_components=PCA_CONFIG['n_components'])
#
#     # 初始化诊断器和可视化器
#     diagnoser = FaultDiagnoser(feature_extractor, preprocessor, pca_model)
#     visualizer = Visualizer()
#
#     # 设置可视化参数
#     if hasattr(visualizer, 'set_params'):
#         visualizer.set_params(
#             figsize=VIZ_CONFIG['figsize'],
#             dpi=VIZ_CONFIG['dpi']
#         )
#
#     # 设置matplotlib参数
#     plt.rcParams['figure.dpi'] = VIZ_CONFIG.get('dpi', 100)
#     plt.rcParams['figure.figsize'] = VIZ_CONFIG.get('figsize', (12, 8))
#
#     # 加载数据
#     if args.use_example_data or args.data_path is None:
#         print("加载示例数据...")
#         normal_signals, fault_signals, test_signals, test_labels = load_example_data(
#             signal_length=DATA_CONFIG['signal_length'],
#             sampling_rate=DATA_CONFIG['sampling_rate']
#         )
#     else:
#         # 加载真实数据
#         normal_signals, fault_signals, test_signals, test_labels = load_real_data(
#             args.data_path, args.data_format, args.normal_label, args.train_ratio
#         )
#
#     # 预处理信号
#     print("预处理信号...")
#     normal_signals = preprocessor.process_signal_batch(normal_signals)
#     test_signals = preprocessor.process_signal_batch(test_signals)
#
#     # 训练模式
#     if args.mode == 'train' or args.mode == 'demo':
#         print("训练诊断模型...")
#         diagnoser.train_model(normal_signals)
#
#         # 保存模型
#         print(f"保存模型到 {args.save_path}...")
#         diagnoser.save_models(args.save_path)
#
#     # 测试模式
#     if args.mode == 'test' or args.mode == 'demo':
#         if args.mode == 'test':
#             # 加载模型
#             print(f"从 {args.load_path} 加载模型...")
#             diagnoser.load_models(args.load_path)
#
#         # 评估模型性能
#         print("评估模型性能...")
#         performance = diagnoser.evaluate_performance(test_signals, test_labels)
#
#         print("\n======== 性能评估 ========")
#         print(f"准确率: {performance['accuracy']:.4f}")
#         print(f"精确率: {performance['precision']:.4f}")
#         print(f"召回率: {performance['recall']:.4f}")
#         print(f"F1分数: {performance['f1_score']:.4f}")
#         if performance.get('auc') is not None:
#             print(f"AUC: {performance['auc']:.4f}")
#
#         # 诊断示例
#         print("\n======== 诊断示例 ========")
#
#         # 诊断一个正常信号
#         normal_idx = test_labels.index(0)
#         normal_result = diagnoser.diagnose(test_signals[normal_idx])
#
#         print("\n正常信号诊断结果:")
#         print(f"检测到故障: {'是' if normal_result['fault_detected'] else '否'}")
#         print(f"T² 值: {normal_result['t2_value']:.4f} (阈值: {normal_result['t2_threshold']:.4f})")
#         print(f"SPE 值: {normal_result['spe_value']:.4f} (阈值: {normal_result['spe_threshold']:.4f})")
#         print(f"故障得分: {normal_result['fault_score']:.4f}")
#
#         # 诊断一个故障信号
#         fault_idx = test_labels.index(1)
#         fault_result = diagnoser.diagnose(test_signals[fault_idx])
#
#         print("\n故障信号诊断结果:")
#         print(f"检测到故障: {'是' if fault_result['fault_detected'] else '否'}")
#         print(f"T² 值: {fault_result['t2_value']:.4f} (阈值: {fault_result['t2_threshold']:.4f})")
#         print(f"SPE 值: {fault_result['spe_value']:.4f} (阈值: {fault_result['spe_threshold']:.4f})")
#         print(f"故障得分: {fault_result['fault_score']:.4f}")
#
#         if fault_result['fault_detected'] and 'max_t2_contribution_feature' in fault_result:
#             print(f"T² 贡献最大的特征: {fault_result['max_t2_contribution_feature']}")
#             print(f"SPE 贡献最大的特征: {fault_result['max_spe_contribution_feature']}")
#
#         # 可视化结果
#         if args.visualize:
#             print("\n生成可视化结果...")
#             save_format = VIZ_CONFIG.get('save_format', 'png')
#             result_dir = PATH_CONFIG['result_dir']
#
#             # 对所有测试数据进行批量诊断
#             features_df = diagnoser.process_signal_batch(test_signals)
#             normalized_features = preprocessor.normalize_features(features_df)
#
#             # 在PCA空间中绘制样本分布
#             fig1 = pca_model.plot_pca_space(normalized_features, test_labels)
#             plt.savefig(os.path.join(result_dir, f'pca_space.{save_format}'))
#
#             # 绘制T2和SPE得分
#             fig2 = pca_model.plot_scores(normalized_features, test_labels)
#             plt.savefig(os.path.join(result_dir, f'scores.{save_format}'))
#
#             # 混淆矩阵
#             fig3 = pca_model.plot_confusion_matrix(normalized_features, test_labels)
#             plt.savefig(os.path.join(result_dir, f'confusion_matrix.{save_format}'))
#
#             # 绘制故障信号的诊断结果摘要
#             fig4 = visualizer.plot_diagnosis_summary(fault_result)
#             plt.savefig(os.path.join(result_dir, f'diagnosis_summary.{save_format}'))
#
#             # 如果检测到故障，绘制特征贡献图
#             if fault_result['fault_detected'] and 't2_contributions' in fault_result:
#                 fig5 = visualizer.plot_feature_contributions(fault_result)
#                 plt.savefig(os.path.join(result_dir, f'feature_contributions.{save_format}'))
#
#             # 绘制特征相关性热图
#             fig6 = visualizer.plot_feature_correlation(features_df)
#             plt.savefig(os.path.join(result_dir, f'feature_correlation.{save_format}'))
#
#             # 绘制PCA解释方差
#             fig7 = visualizer.plot_pca_explained_variance(pca_model)
#             plt.savefig(os.path.join(result_dir, f'pca_explained_variance.{save_format}'))
#
#             print(f"可视化结果已保存到 {result_dir}")
#
#             # 显示图形
#             plt.show()
#
# if __name__ == '__main__':
#     main()


def main():
    """主函数，程序入口点"""
    parser = argparse.ArgumentParser(description='故障诊断系统')
    parser.add_argument('--mode', choices=['train', 'test', 'demo'], default='demo'
                                                                             '',
                        help='程序运行模式：训练、测试或演示')
    parser.add_argument('--save_path', default=PATH_CONFIG['model_dir'],
                        help='模型保存路径')

    parser.add_argument('--load_path', default=PATH_CONFIG['model_dir'],
                        help='模型加载路径')
    parser.add_argument('--visualize', action='store_true',
                        help='是否可视化结果')
    parser.add_argument('--data_path', default=None,
                        help='真实数据文件路径')
    parser.add_argument('--data_format', default='csv',
                        help='数据文件格式: csv, excel, mat等')
    parser.add_argument('--use_example_data', action='store_true',
                        help='使用示例数据而非真实数据')
    parser.add_argument('--normal_label', default=0, type=int,
                        help='正常样本的标签值')
    parser.add_argument('--train_ratio', default=0.8, type=float,
                        help='用于训练的正常数据比例')
    parser.add_argument('--config', default='config.py',
                        help='配置文件路径')

    args = parser.parse_args()

    # 创建必要的目录
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(PATH_CONFIG['result_dir'], exist_ok=True)
    os.makedirs(PATH_CONFIG['processed_data_dir'], exist_ok=True)

    # 初始化组件，使用配置文件参数
    sampling_rate = DATA_CONFIG['sampling_rate']

    # 初始化特征提取器
    feature_extractor = FeatureExtractor(sampling_rate)

    # 使用配置文件参数初始化预处理器
    preprocessor = Preprocessor(
        filter_enabled=DATA_CONFIG['filter_enabled'],
        filter_lowcut=DATA_CONFIG['filter_lowcut'],
        filter_highcut=DATA_CONFIG['filter_highcut'],
        filter_order=DATA_CONFIG['filter_order'],
        sampling_rate=DATA_CONFIG['sampling_rate'],
        noise_removal_enabled=DATA_CONFIG['noise_removal_enabled'],
        noise_window_size=DATA_CONFIG['noise_window_size'],
        segment_overlap=DATA_CONFIG['segment_overlap']
    )

    # 初始化PCA模型
    pca_model = PCADiagnosisModel(n_components=PCA_CONFIG['n_components'])

    # 初始化可视化器
    visualizer = Visualizer(figsize=VIZ_CONFIG['figsize'])

    # 设置matplotlib参数
    plt.rcParams['figure.dpi'] = VIZ_CONFIG.get('dpi', 100)

    # 加载数据
    if args.use_example_data or args.data_path is None:
        print("加载示例数据...")
        normal_signals, fault_signals, test_signals, test_labels = load_example_data(
            signal_length=DATA_CONFIG['signal_length'],
            sampling_rate=DATA_CONFIG['sampling_rate']
        )
    else:
        # 加载真实数据
        normal_signals, fault_signals, test_signals, test_labels = load_real_data(
            args.data_path, args.data_format, args.normal_label, args.train_ratio
        )

    # 预处理信号```````````````````````
    print("预处理信号...")
    normal_signals = preprocessor.process_signal_batch(normal_signals)
    test_signals = preprocessor.process_signal_batch(test_signals)

    # 训练模式
    if args.mode == 'train' or args.mode == 'demo':
        print("训练诊断模型...")

        # 提取正常信号特征
        normal_features_list = []
        for signal in normal_signals:
            features = feature_extractor.extract_all_features(signal)
            normal_features_list.append(features)

        normal_features_df = pd.DataFrame(normal_features_list)

        # 标准化特征
        normalized_normal_features = preprocessor.normalize_features(normal_features_df, fit=True)

        # 使用significance_level从配置文件中获取
        significance_level = PCA_CONFIG.get('significance_level', 0.01)

        # 训练PCA模型
        pca_model.fit(normalized_normal_features, significance_level=significance_level)

        # 如果使用可视化，绘制PCA解释方差
        if args.visualize:
            fig = visualizer.plot_pca_explained_variance(pca_model)
            plt.savefig(os.path.join(PATH_CONFIG['result_dir'], f'pca_explained_variance.{VIZ_CONFIG["save_format"]}'))

        # 保存模型
        print(f"保存模型到 {args.save_path}...")
        model_path = os.path.join(args.save_path, 'pca_model.pkl')
        pca_model.save_model(model_path)

        # 保存预处理器
        preprocessor.save_scaler(os.path.join(args.save_path, 'scaler.pkl'))

    # 测试模式
    if args.mode == 'test' or args.mode == 'demo':
        if args.mode == 'test':
            # 加载模型
            print(f"从 {args.load_path} 加载模型...")
            model_path = os.path.join(args.load_path, 'pca_model.pkl')
            pca_model.load_model(model_path)

            # 加载预处理器
            try:
                preprocessor.load_scaler(os.path.join(args.load_path, 'scaler.pkl'))
            except:
                print("警告: 无法加载标准化器，将使用原始特征")

        # 提取测试信号特征
        test_features_list = []
        for signal in test_signals:
            features = feature_extractor.extract_all_features(signal)
            test_features_list.append(features)

        test_features_df = pd.DataFrame(test_features_list)

        # 标准化特征
        if preprocessor.is_fitted:
            normalized_test_features = preprocessor.normalize_features(test_features_df)
        else:
            normalized_test_features = test_features_df

        # 批量诊断
        diagnosis_results = pca_model.diagnose(normalized_test_features)

        # 计算性能指标
        predicted_labels = diagnosis_results['Fault_Detected'].astype(int)
        test_labels_np = np.array(test_labels)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        performance = {
            'accuracy': accuracy_score(test_labels_np, predicted_labels),
            'precision': precision_score(test_labels_np, predicted_labels),
            'recall': recall_score(test_labels_np, predicted_labels),
            'f1_score': f1_score(test_labels_np, predicted_labels),
        }

        # 计算AUC（如果有多于一个类）
        if len(np.unique(test_labels_np)) > 1:
            performance['auc'] = roc_auc_score(test_labels_np, diagnosis_results['Fault_Score'])

        print("\n======== 性能评估 ========")
        print(f"准确率: {performance['accuracy']:.4f}")
        print(f"精确率: {performance['precision']:.4f}")
        print(f"召回率: {performance['recall']:.4f}")
        print(f"F1分数: {performance['f1_score']:.4f}")
        if 'auc' in performance:
            print(f"AUC: {performance['auc']:.4f}")

        # 诊断示例
        print("\n======== 诊断示例 ========")

        # 诊断一个正常信号
        normal_idx = test_labels.index(0)
        normal_signal = test_signals[normal_idx]

        # 绘制正常信号波形
        if args.visualize:
            fig = visualizer.plot_signal(normal_signal, sampling_rate, title='正常信号波形')
            plt.show()
            plt.savefig(os.path.join(PATH_CONFIG['result_dir'], f'normal_signal.{VIZ_CONFIG["save_format"]}'))

            fig = visualizer.plot_spectrum(normal_signal, sampling_rate, title='正常信号频谱')
            plt.show()
            plt.savefig(os.path.join(PATH_CONFIG['result_dir'], f'normal_spectrum.{VIZ_CONFIG["save_format"]}'))

        # 提取特征并诊断
        normal_features = feature_extractor.extract_all_features(normal_signal)
        normal_features_df = pd.DataFrame([normal_features])

        if preprocessor.is_fitted:
            normal_features_df = preprocessor.normalize_features(normal_features_df)

        # 诊断
        normal_diagnosis = pca_model.diagnose(normal_features_df)
        normal_result = normal_diagnosis.iloc[0].to_dict()  # 转换为字典

        # 转换为plot_diagnosis_summary所需的格式
        normal_plot_result = {
            't2_value': normal_result['T2'],
            't2_threshold': normal_result['T2_Threshold'],
            'spe_value': normal_result['SPE'],
            'spe_threshold': normal_result['SPE_Threshold'],
            'fault_score': normal_result['Fault_Score'],
            'fault_detected': normal_result['Fault_Detected']
        }

        print("\n正常信号诊断结果:")
        print(f"检测到故障: {'是' if normal_result['Fault_Detected'] else '否'}")
        print(f"T² 值: {normal_result['T2']:.4f} (阈值: {normal_result['T2_Threshold']:.4f})")
        print(f"SPE 值: {normal_result['SPE']:.4f} (阈值: {normal_result['SPE_Threshold']:.4f})")
        print(f"故障得分: {normal_result['Fault_Score']:.4f}")

        # 诊断一个故障信号
        fault_idx = test_labels.index(1)
        fault_signal = test_signals[fault_idx]

        # 绘制故障信号波形
        if args.visualize:
            fig = visualizer.plot_signal(fault_signal, sampling_rate, title='故障信号波形')
            plt.savefig(os.path.join(PATH_CONFIG['result_dir'], f'fault_signal.{VIZ_CONFIG["save_format"]}'))

            fig = visualizer.plot_spectrum(fault_signal, sampling_rate, title='故障信号频谱')
            plt.savefig(os.path.join(PATH_CONFIG['result_dir'], f'fault_spectrum.{VIZ_CONFIG["save_format"]}'))

            fig = visualizer.plot_time_frequency(fault_signal, sampling_rate, title='故障信号时频分析')
            plt.savefig(os.path.join(PATH_CONFIG['result_dir'], f'fault_timefreq.{VIZ_CONFIG["save_format"]}'))

        # 提取特征并诊断
        fault_features = feature_extractor.extract_all_features(fault_signal)
        fault_features_df = pd.DataFrame([fault_features])

        if preprocessor.is_fitted:
            fault_features_df = preprocessor.normalize_features(fault_features_df)

        # 诊断
        fault_diagnosis = pca_model.diagnose(fault_features_df)
        fault_result = fault_diagnosis.iloc[0].to_dict()  # 转换为字典

        # 如果检测到故障，计算特征贡献
        if fault_result['Fault_Detected']:
            t2_contrib, spe_contrib = pca_model.calculate_contribution(fault_features_df)

            # 找出贡献最大的特征
            max_t2_feature = t2_contrib.iloc[0].idxmax()
            max_spe_feature = spe_contrib.iloc[0].idxmax()

            # 添加到结果中
            fault_plot_result = {
                't2_value': fault_result['T2'],
                't2_threshold': fault_result['T2_Threshold'],
                'spe_value': fault_result['SPE'],
                'spe_threshold': fault_result['SPE_Threshold'],
                'fault_score': fault_result['Fault_Score'],
                'fault_detected': fault_result['Fault_Detected'],
                'max_t2_contribution_feature': max_t2_feature,
                'max_spe_contribution_feature': max_spe_feature,
                't2_contributions': t2_contrib.iloc[0].to_dict(),
                'spe_contributions': spe_contrib.iloc[0].to_dict()
            }
        else:
            fault_plot_result = {
                't2_value': fault_result['T2'],
                't2_threshold': fault_result['T2_Threshold'],
                'spe_value': fault_result['SPE'],
                'spe_threshold': fault_result['SPE_Threshold'],
                'fault_score': fault_result['Fault_Score'],
                'fault_detected': fault_result['Fault_Detected']
            }

        print("\n故障信号诊断结果:")
        print(f"检测到故障: {'是' if fault_result['Fault_Detected'] else '否'}")
        print(f"T² 值: {fault_result['T2']:.4f} (阈值: {fault_result['T2_Threshold']:.4f})")
        print(f"SPE 值: {fault_result['SPE']:.4f} (阈值: {fault_result['SPE_Threshold']:.4f})")
        print(f"故障得分: {fault_result['Fault_Score']:.4f}")

        if fault_result['Fault_Detected'] and 'max_t2_contribution_feature' in fault_plot_result:
            print(f"T² 贡献最大的特征: {fault_plot_result['max_t2_contribution_feature']}")
            print(f"SPE 贡献最大的特征: {fault_plot_result['max_spe_contribution_feature']}")

        # 可视化结果
        if args.visualize:
            print("\n生成可视化结果...")
            save_format = VIZ_CONFIG.get('save_format', 'png')
            result_dir = PATH_CONFIG['result_dir']

            # 绘制特征分布
            fig = visualizer.plot_features(test_features_df, title='测试信号特征分布')
            plt.savefig(os.path.join(result_dir, f'feature_distribution.{save_format}'))

            # 绘制特征相关性
            fig = visualizer.plot_feature_correlation(test_features_df)
            plt.savefig(os.path.join(result_dir, f'feature_correlation.{save_format}'))

            # 使用PCA model的方法绘制PCA空间
            fig = pca_model.plot_pca_space(normalized_test_features, test_labels)
            plt.savefig(os.path.join(result_dir, f'pca_space.{save_format}'))

            # 使用PCA model的方法绘制T2和SPE得分
            fig = pca_model.plot_scores(normalized_test_features, test_labels)
            plt.savefig(os.path.join(result_dir, f'scores.{save_format}'))

            # 使用PCA model的方法绘制混淆矩阵
            fig = pca_model.plot_confusion_matrix(normalized_test_features, test_labels)
            plt.savefig(os.path.join(result_dir, f'confusion_matrix.{save_format}'))

            # 绘制诊断结果摘要
            fig = visualizer.plot_diagnosis_summary(fault_plot_result)
            plt.savefig(os.path.join(result_dir, f'diagnosis_summary.{save_format}'))

            # 如果检测到故障，绘制特征贡献图
            if fault_result['Fault_Detected']:
                fig = visualizer.plot_feature_contributions(fault_plot_result)
                plt.savefig(os.path.join(result_dir, f'feature_contributions.{save_format}'))

                # 或者使用PCA model的方法绘制特征贡献
                fig = pca_model.plot_contribution(fault_features_df)
                plt.savefig(os.path.join(result_dir, f'pca_contribution.{save_format}'))

            print(f"可视化结果已保存到 {result_dir}")

            # 显示图形
            plt.show()

if __name__ == '__main__':
    # import sys
    # if '--visualize' not in sys.argv:
    #     sys.argv.append('--visualize')

    main()