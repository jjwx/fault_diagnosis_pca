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
from src.data_load import DataLoader
import random
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

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


def load_real_data(data_dir, normal_pattern='*normal*.csv', fault_pattern='*fault*.csv',
                   signal_column='signal', train_ratio=0.8, segment_length=1024, overlap=0.5, mode='train'):
    """
    从目录中加载真实振动信号数据，并进行分段处理

    参数:
    data_dir (str): 数据文件目录的路径
    normal_pattern (str): 匹配正常数据文件的模式
    fault_pattern (str): 匹配故障数据文件的模式
    signal_column (str): CSV文件中信号数据的列名
    train_ratio (float): 用于训练的正常数据比例 (仅训练模式使用)
    segment_length (int): 信号分段长度
    overlap (float): 分段重叠比例(0到1之间)
    mode (str): 'train'或'test'，指示当前模式

    返回:
    tuple: (train_normal_signals, fault_signals, test_signals, test_labels)
    """
    import fnmatch

    print(f"\n加载数据从目录: {data_dir}...")
    data_loader = DataLoader(data_dir)

    # 查找并加载所有正常数据文件
    normal_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if fnmatch.fnmatch(file, normal_pattern):
                normal_files.append(os.path.join(root, file))

    # 查找并加载所有故障数据文件
    fault_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if fnmatch.fnmatch(file, fault_pattern):
                fault_files.append(os.path.join(root, file))

    print(f"找到 {len(normal_files)} 个正常数据文件和 {len(fault_files)} 个故障数据文件")

    # 训练模式
    if mode == 'train':
        # 训练模式下检查是否有足够的正常数据
        if not normal_files:
            print("错误: 训练模式下没有找到正常数据文件，无法训练模型")
            return load_example_data()  # 如果没有找到正常文件，返回示例数据

        # 加载正常数据
        normal_signals = []
        for file in normal_files:
            try:
                signal_data, _ = data_loader.load_csv_data(file, signal_column=signal_column)
                normal_signals.append(signal_data)
            except Exception as e:
                print(f"加载文件 {file} 时出错: {e}")

        # 加载故障数据(如果有，用于测试)
        fault_signals_raw = []
        for file in fault_files:
            try:
                signal_data, _ = data_loader.load_csv_data(file, signal_column=signal_column)
                fault_signals_raw.append(signal_data)
            except Exception as e:
                print(f"加载文件 {file} 时出错: {e}")

        # 分段处理正常信号
        print("\n对信号进行分段处理...")
        segmented_normal_signals = []
        for signal in normal_signals:
            # 将信号转换为numpy数组（如果还不是）
            signal_array = np.array(signal)
            segments = data_loader.segment_signals(signal_array, segment_length, overlap)
            segmented_normal_signals.extend(segments)

        # 分段处理故障信号（如果有）
        segmented_fault_signals = []
        if fault_signals_raw and len(fault_signals_raw) > 0:
            for signal in fault_signals_raw:
                # 将信号转换为numpy数组（如果还不是）
                signal_array = np.array(signal)
                segments = data_loader.segment_signals(signal_array, segment_length, overlap)
                segmented_fault_signals.extend(segments)

        print(
            f"分段处理完成，共有 {len(segmented_normal_signals)} 个正常信号片段和 {len(segmented_fault_signals) if segmented_fault_signals else 0} 个故障信号片段")

        # 分割正常信号为训练集和测试集
        train_size = int(len(segmented_normal_signals) * train_ratio)
        train_normal = segmented_normal_signals[:train_size]
        test_normal = segmented_normal_signals[train_size:]

        # 创建测试集：混合部分正常信号和部分故障信号
        test_fault = segmented_fault_signals

        # 如果故障信号太多，可以限制数量
        if test_fault and len(test_fault) > len(test_normal) * 2:
            import random
            random.seed(42)  # 固定随机种子以保证可重复性
            test_fault = random.sample(test_fault, len(test_normal) * 2)

        # 组合测试集
        test_signals = test_normal + test_fault if test_fault else test_normal
        test_labels = [0] * len(test_normal) + [1] * len(test_fault) if test_fault else [0] * len(test_normal)

        print(
            f"数据加载完成: 训练集有 {len(train_normal)} 个正常信号片段，测试集有 {len(test_normal)} 个正常信号片段和 {len(test_fault) if test_fault else 0} 个故障信号片段")

        return train_normal, segmented_fault_signals, test_signals, test_labels

    # 测试模式
    else:
        # 测试模式下检查是否有任何数据
        if not normal_files and not fault_files:
            print("错误: 测试模式下没有找到任何数据文件")
            return load_example_data()  # 如果测试模式下没有数据，返回示例数据

        # 在测试模式下，将所有文件直接作为测试数据
        test_signals = []
        test_labels = []

        # 处理正常数据文件
        for file in normal_files:
            try:
                signal_data, _ = data_loader.load_csv_data(file, signal_column=signal_column)
                signal_array = np.array(signal_data)
                segments = data_loader.segment_signals(signal_array, segment_length, overlap)

                # 将所有分段添加到测试数据中，标签为0(正常)
                test_signals.extend(segments)
                test_labels.extend([0] * len(segments))
            except Exception as e:
                print(f"加载文件 {file} 时出错: {e}")

        # 处理故障数据文件
        for file in fault_files:
            try:
                signal_data, _ = data_loader.load_csv_data(file, signal_column=signal_column)
                signal_array = np.array(signal_data)
                segments = data_loader.segment_signals(signal_array, segment_length, overlap)

                # 将所有分段添加到测试数据中，标签为1(故障)
                test_signals.extend(segments)
                test_labels.extend([1] * len(segments))
            except Exception as e:
                print(f"加载文件 {file} 时出错: {e}")

        print(f"测试数据加载完成: 共有 {len(test_signals)} 个信号片段，其中正常片段 {test_labels.count(0)} 个，故障片段 {test_labels.count(1)} 个")

        # 为了保持函数返回格式一致，对于测试模式，返回空的训练集和故障信号集
        return [], [], test_signals, test_labels


def train_model(args):
    """训练故障诊断模型"""
    print("\n" + "=" * 50)
    print("开始训练故障诊断模型")
    print("=" * 50)

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
            args.data_path,
            normal_pattern=DATA_CONFIG.get('normal_pattern', '*normal*.csv'),
            fault_pattern=DATA_CONFIG.get('fault_pattern', '*fault*.csv'),
            signal_column=DATA_CONFIG.get('signal_column', 'signal'),
            train_ratio=args.train_ratio,
            segment_length=DATA_CONFIG.get('segment_length', 1024),
            overlap=DATA_CONFIG.get('segment_overlap', 0.5),
            mode='train'  # 明确指定训练模式
        )


    # 预处理信号
    print("预处理信号...")
    normal_signals = preprocessor.process_signal_batch(normal_signals)

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
        plt.close(fig)

    # 保存模型
    print(f"保存模型到 {args.save_path}...")
    model_path = os.path.join(args.save_path, 'pca_model.pkl')
    pca_model.save_model(model_path)

    # 保存预处理器
    preprocessor.save_scaler(os.path.join(args.save_path, 'scaler.pkl'))

    print("\n" + "=" * 50)
    print(f"训练完成! 模型已保存到 {model_path}")
    print("=" * 50)


def test_model(args):
    """测试故障诊断模型"""
    print("\n" + "=" * 50)
    print("开始测试故障诊断模型")
    print("=" * 50)

    # 创建必要的目录
    os.makedirs(PATH_CONFIG['result_dir'], exist_ok=True)

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

    # 加载模型
    print(f"从 {args.load_path} 加载模型...")
    model_path = os.path.join(args.load_path, 'pca_model.pkl')
    pca_model.load_model(model_path)

    # 加载预处理器
    try:
        preprocessor.load_scaler(os.path.join(args.load_path, 'scaler.pkl'))
        print("成功加载标准化器")
    except Exception as e:
        print(f"警告: 无法加载标准化器: {e}，将使用原始特征")

    # 加载测试数据
    if args.use_example_data or args.data_path is None:
        print("加载示例数据...")
        _, _, test_signals, test_labels = load_example_data(
            signal_length=DATA_CONFIG['signal_length'],
            sampling_rate=DATA_CONFIG['sampling_rate']
        )
    else:
        # 加载真实数据 - 只关心测试部分
        _, _, test_signals, test_labels = load_real_data(
            args.data_path,
            normal_pattern=DATA_CONFIG.get('normal_pattern', '*normal*.csv'),
            fault_pattern=DATA_CONFIG.get('fault_pattern', '*fault*.csv'),
            signal_column=DATA_CONFIG.get('signal_column', 'signal'),
            train_ratio=args.train_ratio,
            segment_length=DATA_CONFIG.get('segment_length', 1024),
            overlap=DATA_CONFIG.get('segment_overlap', 0.5),
            mode='test'  # 明确指定测试模式
        )

    # 预处理测试信号
    print("预处理测试信号...")
    test_signals = preprocessor.process_signal_batch(test_signals)   #带通滤波

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
    print("执行诊断...")
    diagnosis_results = pca_model.diagnose(normalized_test_features)

    # 计算性能指标
    predicted_labels = diagnosis_results['Fault_Detected'].astype(int)
    test_labels_np = np.array(test_labels)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    # 初始化性能字典
    performance = {}

    # 计算基本指标，处理边缘情况
    try:
        performance['accuracy'] = accuracy_score(test_labels_np, predicted_labels)
    except Exception as e:
        print(f"计算准确率时出错: {e}")
        performance['accuracy'] = float('nan')

    try:
        performance['precision'] = precision_score(test_labels_np, predicted_labels, zero_division=0)
    except Exception as e:
        print(f"计算精确率时出错: {e}")
        performance['precision'] = float('nan')

    try:
        performance['recall'] = recall_score(test_labels_np, predicted_labels, zero_division=0)
    except Exception as e:
        print(f"计算召回率时出错: {e}")
        performance['recall'] = float('nan')

    try:
        performance['f1_score'] = f1_score(test_labels_np, predicted_labels, zero_division=0)
    except Exception as e:
        print(f"计算F1分数时出错: {e}")
        performance['f1_score'] = float('nan')

    # 计算AUC（如果有多于一个类）
    if len(np.unique(test_labels_np)) > 1:
        try:
            performance['auc'] = roc_auc_score(test_labels_np, diagnosis_results['Fault_Score'])
        except Exception as e:
            print(f"计算AUC时出错: {e}")
            performance['auc'] = float('nan')

    print("\n======== 性能评估 ========")
    print(f"准确率: {performance.get('accuracy', 'N/A'):.4f}")
    print(f"精确率: {performance.get('precision', 'N/A'):.4f}")
    print(f"召回率: {performance.get('recall', 'N/A'):.4f}")
    print(f"F1分数: {performance.get('f1_score', 'N/A'):.4f}")
    if 'auc' in performance:
        print(f"AUC: {performance.get('auc', 'N/A'):.4f}")

    # 诊断示例
    print("\n======== 诊断示例 ========")

    # 安全地获取一个正常信号进行诊断
    normal_result = None
    normal_plot_result = None

    if 0 in test_labels:
        normal_idx = test_labels.index(0)
        normal_signal = test_signals[normal_idx]

        # 绘制正常信号波形
        if args.visualize:
            fig = visualizer.plot_signal(normal_signal, sampling_rate, title='正常信号波形')
            plt.savefig(os.path.join(PATH_CONFIG['result_dir'], f'normal_signal.{VIZ_CONFIG["save_format"]}'))
            plt.close(fig)

            fig = visualizer.plot_spectrum(normal_signal, sampling_rate, title='正常信号频谱')
            plt.savefig(os.path.join(PATH_CONFIG['result_dir'], f'normal_spectrum.{VIZ_CONFIG["save_format"]}'))
            plt.close(fig)

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
    else:
        print("\n测试数据中没有正常信号样本，跳过正常信号诊断。")

    # 初始化故障诊断结果变量
    fault_result = None
    fault_plot_result = None
    fault_features_df = None

    predicted_labels = predicted_labels.tolist()

    # 诊断一个故障信号(如果有)
    # if 1 in predicted_labels:
    #     fault_idx = predicted_labels.index(1)
    #     fault_signal = test_signals[fault_idx]
    if 1 in predicted_labels:
        # 找出所有故障样本的索引
        fault_indices = [i for i, label in enumerate(predicted_labels) if label == 1]

        # 随机选择一个故障样本
        fault_idx = random.choice(fault_indices)
        fault_signal = test_signals[fault_idx]

        # 绘制故障信号波形
        if args.visualize:
            fig = visualizer.plot_signal(fault_signal, sampling_rate, title='故障信号波形')
            plt.savefig(os.path.join(PATH_CONFIG['result_dir'], f'fault_signal.{VIZ_CONFIG["save_format"]}'))
            plt.close(fig)

            fig = visualizer.plot_spectrum(fault_signal, sampling_rate, title='故障信号频谱')
            plt.savefig(os.path.join(PATH_CONFIG['result_dir'], f'fault_spectrum.{VIZ_CONFIG["save_format"]}'))
            plt.close(fig)

            fig = visualizer.plot_time_frequency(fault_signal, sampling_rate, title='故障信号时频分析')
            plt.savefig(os.path.join(PATH_CONFIG['result_dir'], f'fault_timefreq.{VIZ_CONFIG["save_format"]}'))
            plt.close(fig)

        # 提取特征并诊断
        fault_features = feature_extractor.extract_all_features(fault_signal)
        fault_features_df = pd.DataFrame([fault_features])
        print(fault_features_df)

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
    else:
        print("\n测试数据中没有故障信号样本，跳过故障信号诊断。")

    # 可视化结果
    if args.visualize:
        print("\n生成可视化结果...")
        save_format = VIZ_CONFIG.get('save_format', 'png')
        result_dir = PATH_CONFIG['result_dir']

        # 绘制特征分布
        fig = visualizer.plot_features(test_features_df, title='测试信号特征分布')
        plt.savefig(os.path.join(result_dir, f'feature_distribution.{save_format}'))
        plt.close(fig)

        # 绘制特征相关性
        fig = visualizer.plot_feature_correlation(test_features_df)
        plt.savefig(os.path.join(result_dir, f'feature_correlation.{save_format}'))
        plt.close(fig)

        # 使用PCA model的方法绘制PCA空间
        fig = pca_model.plot_pca_space(normalized_test_features, test_labels)
        plt.savefig(os.path.join(result_dir, f'pca_space.{save_format}'))
        plt.close(fig)

        # 使用PCA model的方法绘制T2和SPE得分
        fig = pca_model.plot_scores(normalized_test_features, test_labels)
        plt.savefig(os.path.join(result_dir, f'scores.{save_format}'))
        plt.close(fig)

        # 使用PCA model的方法绘制混淆矩阵
        fig = pca_model.plot_confusion_matrix(normalized_test_features, test_labels)
        plt.savefig(os.path.join(result_dir, f'confusion_matrix.{save_format}'))
        plt.close(fig)

        # 绘制诊断结果摘要 (如果有故障结果)
        if fault_plot_result:
            fig = visualizer.plot_diagnosis_summary(fault_plot_result)
            plt.savefig(os.path.join(result_dir, f'diagnosis_summary.{save_format}'))
            plt.close(fig)

            # 如果检测到故障，绘制特征贡献图
            if fault_result and fault_result['Fault_Detected']:
                fig = visualizer.plot_feature_contributions(fault_plot_result)
                plt.savefig(os.path.join(result_dir, f'feature_contributions.{save_format}'))
                plt.close(fig)

                # 使用PCA model的方法绘制特征贡献
                if fault_features_df is not None:
                    fig = pca_model.plot_contribution(fault_features_df)
                    plt.savefig(os.path.join(result_dir, f'pca_contribution.{save_format}'))
                    plt.close(fig)

        print(f"可视化结果已保存到 {result_dir}")

    print("\n" + "=" * 50)
    print("测试完成!")
    print("=" * 50)


def main():
    """主函数，根据命令行参数决定运行模式"""
    parser = argparse.ArgumentParser(description='故障诊断系统')
    parser.add_argument('--mode', choices=['train', 'test', 'demo'], default='train',
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

    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'test':
        test_model(args)
    elif args.mode == 'demo':
        # 演示模式：先训练然后测试
        train_model(args)
        test_model(args)


if __name__ == '__main__':
    # 设置测试参数
    test_data_dir = r'E:\Github\machinery_fault_diagnosis\test_data'
    train_data_dir = r'E:\Github\machinery_fault_diagnosis\train_data'

    model_dir = './models'

    # 确保输出目录存在
    os.makedirs('./diagnosis_results', exist_ok=True)

    print("=" * 50)
    print(f"加载模型: {model_dir}")
    print(f"测试数据目录: {test_data_dir}")
    print("=" * 50)

    # 更新配置
    DATA_CONFIG.update({
        'normal_pattern': 'normal*.csv',
        'fault_pattern': 'fault*.csv',
        'sampling_rate': 10000,
        'segment_length': 8192,
        'segment_overlap': 0.5
    })

    # 设置命令行参数为测试模式
    import sys


    # sys.argv = [
    #     sys.argv[0],
    #     '--mode', 'train',
    #     '--data_path', train_data_dir,
    #     '--load_path', './models',
    #     '--save_path', './models'
    # ]

    sys.argv = [
        sys.argv[0],
        '--mode', 'test',
        '--data_path', test_data_dir,
        '--load_path', './models',
        '--save_path', './models'
    ]


    # 直接运行主函数(只进行测试)
    main()