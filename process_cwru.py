import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
import h5py


def process_cwru_data(data_path, output_csv, sampling_rate=12000, signal_length=2048):
    """
    处理CWRU轴承数据集并转换为CSV格式

    参数:
    data_path: CWRU数据文件所在目录
    output_csv: 输出CSV文件路径
    sampling_rate: 原始数据采样率(通常为12kHz或48kHz)
    signal_length: 每个样本段长度
    """
    processed_data = []
    labels = []

    # 遍历所有.mat文件
    for file in os.listdir(data_path):
        if file.endswith('.mat'):
            file_path = os.path.join(data_path, file)
            print(f"处理: {file}")

            try:
                # 尝试加载.mat文件
                try:
                    mat_data = loadmat(file_path)
                    # 查找振动数据变量(通常是DE_time或FE_time)
                    data_key = None
                    for key in mat_data.keys():
                        if isinstance(key, str) and ('DE_time' in key or 'FE_time' in key):
                            data_key = key
                            break

                    vibration_data = mat_data[data_key].flatten()
                except:
                    # 尝试用h5py处理新版本.mat文件
                    f = h5py.File(file_path, 'r')
                    for key in f.keys():
                        if 'DE_time' in key or 'FE_time' in key:
                            data_key = key
                            break
                    vibration_data = np.array(f[data_key]).flatten()

                # 确定故障类型
                fault_type = 0  # 默认为正常
                if 'normal' in file.lower():
                    fault_type = 0  # 正常
                elif 'ir' in file.lower():
                    fault_type = 1  # 内圈故障
                elif 'or' in file.lower():
                    fault_type = 2  # 外圈故障
                elif 'b' in file.lower():
                    fault_type = 3  # 滚动体故障

                # 将长信号分割成多个样本段
                num_samples = len(vibration_data) // signal_length
                for i in range(num_samples):
                    start_idx = i * signal_length
                    end_idx = start_idx + signal_length
                    signal_segment = vibration_data[start_idx:end_idx]
                    processed_data.append(signal_segment)
                    labels.append(fault_type)

                print(f"从文件{file}中提取了{num_samples}个样本")

            except Exception as e:
                print(f"处理文件{file}时出错: {e}")

    # 创建DataFrame并保存为CSV
    # 先创建信号数据列
    columns = [f'sample_{i}' for i in range(signal_length)]
    signal_df = pd.DataFrame(processed_data, columns=columns)

    # 添加标签列
    signal_df['label'] = labels

    # 保存为CSV
    signal_df.to_csv(output_csv, index=False)
    print(f"已保存处理后的数据到: {output_csv}")

    return signal_df


if __name__ == "__main__":
    data_path = r"E:\cwru.data\Data\CWRU\12DriveEndFault\1730"
    output_csv = r"E:\Github\Fault_diagnosis_system\data\raw\cwru_processed.csv"
    process_cwru_data(data_path, output_csv)